// Format-v2 decision bench, STEP 1: ggml's own mul_mat_vec_q path at the two
// real gating shapes, synthetic weights in candidate quant formats.
//
// Speed case only (Candidate B): weights are random floats quantized with
// ggml_quantize_chunk (real block layout, real scales), activations F32; the
// timed graph is ggml_mul_mat(W, x) on the CUDA backend, which dispatches
// quantize_q8_1 on x followed by mul_mat_vec_q<type> — exactly what
// production would run. ms/call = median of 100 individually wall-timed
// graph computes (launch overhead included, matching the bmo_kernel_bench
// convention). Effective GB/s = (weight row bytes * rows + x + y) / time.
// No rel_l2 here: format accuracy is the H100's axis; this measures speed.
//
// Usage: mmvq_shape_bench   (shapes/types fixed: the decision matrix)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

struct Shape { const char * name; int64_t rows, cols; };
static const Shape SHAPES[] = {
    { "gating_in  (22528x4096)", 22528, 4096 },
    { "gating_out (4096x11264)",  4096, 11264 },
};
static const ggml_type TYPES[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0 };

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "cuda init failed\n"); return 1; }

    printf("mmvq_shape_bench — ggml mul_mat_vec_q (incl. q8_1 act quant) at gating shapes\n");
    printf("%-26s %-6s %10s %10s %14s\n", "shape", "type", "ms/call", "GB/s", "weight_MiB");

    for (const Shape & sh : SHAPES) {
        // synthetic source weights, fixed seed per shape
        std::vector<float> wf((size_t) sh.rows * sh.cols);
        std::mt19937 rng(0x676d6d76u ^ (unsigned) sh.rows);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto & v : wf) v = dist(rng);

        for (ggml_type type : TYPES) {
            size_t row_bytes = ggml_row_size(type, sh.cols);
            std::vector<uint8_t> wq(row_bytes * sh.rows);
            ggml_quantize_chunk(type, wf.data(), wq.data(), 0,
                sh.rows, sh.cols, nullptr);

            ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, type, sh.cols, sh.rows);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.cols, 1);
            ggml_tensor * y = ggml_mul_mat(ctx, W, x);
            ggml_set_output(y);
            ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, y);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!buf || !ggml_gallocr_alloc_graph(galloc, g)) {
                fprintf(stderr, "alloc failed\n"); return 1;
            }
            ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
            std::vector<float> xh(sh.cols);
            for (auto & v : xh) v = dist(rng);
            ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * 4);

            for (int i = 0; i < 20; i++) ggml_backend_graph_compute(backend, g);
            ggml_backend_synchronize(backend);
            std::vector<double> ms(100);
            for (int i = 0; i < 100; i++) {
                int64_t t0 = ggml_time_us();
                ggml_backend_graph_compute(backend, g);
                ggml_backend_synchronize(backend);
                ms[i] = (ggml_time_us() - t0) / 1e3;
            }
            std::sort(ms.begin(), ms.end());
            double med = ms[50];
            double bytes = (double) wq.size() + sh.cols * 4.0 + sh.rows * 4.0;
            printf("%-26s %-6s %10.4f %10.1f %14.1f\n",
                sh.name, ggml_type_name(type), med, bytes / (med / 1e3) / 1e9,
                wq.size() / 1024.0 / 1024.0);
            fflush(stdout);

            ggml_gallocr_free(galloc);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    return 0;
}
