// Phase-1 gate (c): numerics dry run for the fattn KV-path redesign.
//
// Loads the BMO_ATTN_DUMP capture (temporal layer 0, frame 300, -c 256:
// q / attn_bias / x / K cache / V cache, all real) and runs BOTH attention
// tails on the CUDA backend on identical inputs:
//   path A (current chain):  cast(V,f32) -> mul_mat(K,q) ->
//                            soft_max_ext(+bias,scale) -> cont(transpose(V))
//                            -> mul_mat            (K read via mmvq q8_1)
//   path B (candidate):      ggml_flash_attn_ext(q, K, V, mask_f16, scale)
//                            (fattn-vec q4_0/q4_0, D=128, native layout)
// Reports rel_l2(B vs A) — the calibration number for the integration gate —
// plus rel_l2(A vs live x) as capture/harness validation (expect ~0), and
// per-call timings of both tails as a perf preview.
//
// Build: tools/build_attn_fattn_check.sh   Usage: attn_fattn_check <dump_dir>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

struct DumpTensor {
    int type;
    int64_t ne[4];
    size_t nb[4];
    size_t nbytes;
    std::vector<uint8_t> data;
};

static std::map<std::string, DumpTensor> load_dump(const std::string & dir) {
    std::map<std::string, DumpTensor> out;
    FILE * meta = fopen((dir + "/meta.txt").c_str(), "r");
    if (!meta) { fprintf(stderr, "no meta.txt in %s\n", dir.c_str()); exit(1); }
    char name[64];
    DumpTensor t;
    while (fscanf(meta, "%63s type=%d ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu nbytes=%zu\n",
            name, &t.type,
            (long long *)&t.ne[0], (long long *)&t.ne[1], (long long *)&t.ne[2], (long long *)&t.ne[3],
            &t.nb[0], &t.nb[1], &t.nb[2], &t.nb[3], &t.nbytes) == 11) {
        FILE * f = fopen((dir + "/" + name + ".bin").c_str(), "rb");
        if (!f) { fprintf(stderr, "missing %s.bin\n", name); exit(1); }
        t.data.resize(t.nbytes);
        if (fread(t.data.data(), 1, t.nbytes, f) != t.nbytes) { fprintf(stderr, "short read %s\n", name); exit(1); }
        fclose(f);
        out[name] = t;
        printf("loaded %-10s type=%d ne=[%lld,%lld,%lld,%lld] nbytes=%zu\n",
            name, t.type, (long long)t.ne[0], (long long)t.ne[1], (long long)t.ne[2], (long long)t.ne[3], t.nbytes);
    }
    fclose(meta);
    return out;
}

// Gather a (possibly strided) f32 dump into contiguous order.
static std::vector<float> to_contig_f32(const DumpTensor & t) {
    std::vector<float> out((size_t)t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t.ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t.ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t.ne[1]; i1++)
    for (int64_t i0 = 0; i0 < t.ne[0]; i0++) {
        size_t off = i0 * t.nb[0] + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
        float v;
        memcpy(&v, t.data.data() + off, 4);
        out[idx++] = v;
    }
    return out;
}

struct DiffStats { double rel_l2; double max_abs; };
static DiffStats diff(const std::vector<float> & a, const std::vector<float> & b) {
    double e = 0, r = 0, m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - (double)b[i];
        e += d * d; r += (double)b[i] * (double)b[i];
        m = std::max(m, std::fabs(d));
    }
    return { std::sqrt(e) / std::sqrt(r), m };
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <dump_dir>\n", argv[0]); return 1; }
    auto dump = load_dump(argv[1]);
    DumpTensor & dq = dump["q"], & dbias = dump["attn_bias"], & dx = dump["x"],
               & dk = dump["keys"], & dv = dump["values"];

    const int D   = (int)dq.ne[0];
    const int H   = (int)dq.ne[2];
    const int CTX = (int)dk.ne[1];
    const float scale = 1.0f / sqrtf((float)D);
    printf("D=%d H=%d CTX=%d scale=%f k/v type=%d\n", D, H, CTX, scale, dk.type);

    auto q_host    = to_contig_f32(dq);     // [D, 1, H, 1]
    auto bias_host = to_contig_f32(dbias);  // [CTX, 1]
    auto x_live    = to_contig_f32(dx);     // [D, 1, H, 1]

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "cuda init failed\n"); return 1; }

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead() * 2, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, H, 1);
    ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CTX, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, CTX, 1);
    ggml_tensor * K    = ggml_new_tensor_4d(ctx, (ggml_type)dk.type, D, CTX, H, 1);
    ggml_tensor * V    = ggml_new_tensor_4d(ctx, (ggml_type)dv.type, D, CTX, H, 1);

    // path A — the current chain, verbatim semantics
    ggml_tensor * v32 = ggml_cast(ctx, V, GGML_TYPE_F32);
    ggml_tensor * kq  = ggml_mul_mat(ctx, K, q);
    kq = ggml_soft_max_ext(ctx, kq, bias, scale, 0.0f);
    ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v32));
    ggml_tensor * xA = ggml_mul_mat(ctx, vt, kq);
    ggml_set_output(xA);

    // path B — flash_attn_ext on the native q4_0 caches
    ggml_tensor * xB = ggml_flash_attn_ext(ctx, q, K, V, mask, scale, 0.0f, 0.0f);
    ggml_set_output(xB);

    ggml_cgraph * gA = ggml_new_graph(ctx);
    ggml_build_forward_expand(gA, xA);
    ggml_cgraph * gB = ggml_new_graph(ctx);
    ggml_build_forward_expand(gB, xB);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    // reserve for the union by allocating gA then gB is not supported by one
    // gallocr; use two.
    ggml_gallocr_t gallocB = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!inbuf) { fprintf(stderr, "input alloc failed\n"); return 1; }
    if (!ggml_gallocr_alloc_graph(galloc, gA)) { fprintf(stderr, "alloc A failed\n"); return 1; }
    if (!ggml_gallocr_alloc_graph(gallocB, gB)) { fprintf(stderr, "alloc B failed\n"); return 1; }

    ggml_backend_tensor_set(q, q_host.data(), 0, q_host.size() * 4);
    ggml_backend_tensor_set(bias, bias_host.data(), 0, bias_host.size() * 4);
    std::vector<ggml_fp16_t> mask_host(bias_host.size());
    for (size_t i = 0; i < bias_host.size(); i++) mask_host[i] = ggml_fp32_to_fp16(bias_host[i]);
    ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(K, dk.data.data(), 0, dk.nbytes);
    ggml_backend_tensor_set(V, dv.data.data(), 0, dv.nbytes);

    if (ggml_backend_graph_compute(backend, gA) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute A failed\n"); return 1; }
    if (ggml_backend_graph_compute(backend, gB) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute B failed\n"); return 1; }

    std::vector<float> a_out(ggml_nelements(xA)), b_out(ggml_nelements(xB));
    ggml_backend_tensor_get(xA, a_out.data(), 0, a_out.size() * 4);
    ggml_backend_tensor_get(xB, b_out.data(), 0, b_out.size() * 4);

    // xA is [D,1,H,1], xB is [D,H,1,1] — identical flat order (d fastest, then head).
    DiffStats val = diff(a_out, x_live);
    DiffStats cmp = diff(b_out, a_out);
    printf("\nVALIDATION rel_l2(pathA vs live x) = %.3e  max_abs = %.3e  (expect ~0)\n", val.rel_l2, val.max_abs);
    printf("GATE-CALIB rel_l2(fattn vs pathA)  = %.3e  max_abs = %.3e\n", cmp.rel_l2, cmp.max_abs);

    // per-head worst
    double worst = 0; int worst_h = -1;
    for (int h = 0; h < H; h++) {
        std::vector<float> ah(a_out.begin() + (size_t)h * D, a_out.begin() + (size_t)(h + 1) * D);
        std::vector<float> bh(b_out.begin() + (size_t)h * D, b_out.begin() + (size_t)(h + 1) * D);
        DiffStats dh = diff(bh, ah);
        if (dh.rel_l2 > worst) { worst = dh.rel_l2; worst_h = h; }
    }
    printf("per-head worst rel_l2 = %.3e (head %d)\n", worst, worst_h);

    // timing preview: 20 warmup + 100 timed each
    for (int i = 0; i < 20; i++) ggml_backend_graph_compute(backend, gA);
    ggml_backend_synchronize(backend);
    int64_t t0 = ggml_time_us();
    for (int i = 0; i < 100; i++) ggml_backend_graph_compute(backend, gA);
    ggml_backend_synchronize(backend);
    int64_t t1 = ggml_time_us();
    for (int i = 0; i < 20; i++) ggml_backend_graph_compute(backend, gB);
    ggml_backend_synchronize(backend);
    int64_t t2 = ggml_time_us();
    for (int i = 0; i < 100; i++) ggml_backend_graph_compute(backend, gB);
    ggml_backend_synchronize(backend);
    int64_t t3 = ggml_time_us();
    printf("\nTIMING (1 layer, CTX=%d): pathA %.3f ms   fattn %.3f ms   (%.1fx)\n",
        CTX, (t1 - t0) / 100e3, (t3 - t2) / 100e3, (double)(t1 - t0) / (t3 - t2));

    return 0;
}
