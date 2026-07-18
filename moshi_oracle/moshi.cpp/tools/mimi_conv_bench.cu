// Standalone microbenchmark for mimi decode's conv_transpose_1d kernel (sm_87).
//
// The four SEANet decoder transposed convs (mimi.decoder.model.{2,5,8,11})
// are 126.5 ms of the 133.4 ms t_mimi_dec phase — 40.5% of ALL GPU time in
// the config-B nsys capture. The shapes below are certified two ways:
//   1. source: moshi/models/lm_default.h decoder construction +
//      moshi/modules/conv.h streaming length arithmetic (Lin doubles from the
//      12.5 Hz frame via upsample.convtr, then x8 x6 x5 x4 to 1920 samples);
//   2. graph: the nsys cuda_gpu_trace of the real 300-frame run shows exactly
//      4 distinct conv_transpose_1d_kernel grids {48,102,243,481}x256, one
//      call per module per frame, matching output_size = Lout_raw*Cout =
//      {12288, 26112, 62080, 123136} for the shapes below.
// The grouped upsample convtr (groups=512) never reaches this kernel — it is
// lowered to mul/concat in conv.h — so it is out of scope here.
//
// Reference = V0, a verbatim copy of the existing ggml kernel
// (ggml-cuda/conv-transpose-1d.cu conv_transpose_1d_kernel) at its original
// launch config. That kernel loops every (cin, i) pair per output element and
// branches away non-contributing taps, so it does ~Lin/ntaps times the
// necessary work (~240x waste at model_11's Lin=480) — the candidates below
// compute only the <= ceil(K/S) = 2 contributing taps per output element.
// A CPU double-precision reference guards V0 itself (harness sanity).
//
// Gate per shape: rel_l2 < 1e-5 vs V0 on identical inputs/weights
// (summation-order changes expected; bit-identity is NOT required here —
// model-level gates come at integration). Weights are BF16-truncated F32 to
// mirror the deployed codec (BF16 checkpoints loaded into the F32-only op);
// the op precision itself is untouched — F32 in, F32 out, like the graph.
// Reported per variant per shape: ms/call (median of 100, cudaEvent) and
// effective GB/s = useful bytes (W + X + OUT, F32) / time.
//
// Usage: mimi_conv_bench   (no arguments; shapes are the certified set)

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CU_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// ============================================================================
// Certified shapes. Lout_raw = (Lin-1)*S + K is the raw kernel output length;
// the streaming window trim (-PT) happens outside the op and is not benched.
// in_graph_ms: per-call average from the config-B nsys trace (272 calls each,
// ~3% nsys inflation) — the number the candidate has to beat.
// ============================================================================
struct ConvShape {
    const char * name;
    int cin, cout, k, s, lin;
    float in_graph_ms;
};

static const ConvShape SHAPES[] = {
    { "model_2  [1024->512 k16 s8  L2  ]", 1024, 512, 16, 8,   2,   1.281f },
    { "model_5  [ 512->256 k12 s6  L16 ]",  512, 256, 12, 6,  16,   3.611f },
    { "model_8  [ 256->128 k10 s5  L96 ]",  256, 128, 10, 5,  96,  21.153f },
    { "model_11 [ 128-> 64 k8  s4  L480]",  128,  64,  8, 4, 480, 100.484f },
};
static const int N_SHAPES = 4;
static const float IN_GRAPH_TOTAL_MS = 126.529f;  // sum of the above
static const float T_MIMI_DEC_MS     = 133.365f;  // config-B phase baseline

// ============================================================================
// V0 — verbatim copy of ggml-cuda/conv-transpose-1d.cu conv_transpose_1d_kernel
// (the reference; keep in sync with that file).
// ============================================================================
#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 256

static __global__ void v0_ggml_kernel(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0,
        const int dst_ne0,
        const float * src0, const float * src1, float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value = src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

// ============================================================================
// Candidate (a) — transposed-conv-as-GEMM, two kernels.
// Layouts (all as in the graph): W[cin][cout][j] = cin*K*Cout + cout*K + j,
// so W is a row-major [Cin, M=K*Cout] matrix with NO reshape needed at all —
// kernel 1 is a plain P = W^T X GEMM (mmv-style: coalesced over m, columns
// tiled in registers), kernel 2 gathers the <= 2 (i, j=idx-i*S) taps of each
// output element from P. P is [Lin, M] so both kernels write/read coalesced
// over m.
// ============================================================================
#define VA_COLS 8

static __global__ void va_gemm_kernel(
        const int M, const int cin_n, const int lin,
        const float * __restrict__ W, const float * __restrict__ X,
        float * __restrict__ P) {
    const int m    = threadIdx.x + blockIdx.x * blockDim.x;
    const int col0 = blockIdx.y * VA_COLS;
    if (m >= M) {
        return;
    }

    float acc[VA_COLS] = { 0 };
    for (int cin = 0; cin < cin_n; cin++) {
        const float w = W[(int64_t) cin * M + m];
        const float * xr = X + (int64_t) cin * lin + col0;
#pragma unroll
        for (int c = 0; c < VA_COLS; c++) {
            if (col0 + c < lin) {
                acc[c] += w * xr[c];
            }
        }
    }
#pragma unroll
    for (int c = 0; c < VA_COLS; c++) {
        if (col0 + c < lin) {
            P[(int64_t) (col0 + c) * M + m] = acc[c];
        }
    }
}

static __global__ void va_combine_kernel(
        const int k, const int s0, const int cout_n, const int lin, const int lout,
        const float * __restrict__ P, float * __restrict__ dst) {
    const int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= lout * cout_n) {
        return;
    }
    const int cout = g / lout;
    const int idx  = g % lout;
    const int M    = k * cout_n;

    const int i_lo = (idx >= k) ? (idx - k) / s0 + 1 : 0;
    const int i_hi = min(idx / s0, lin - 1);

    float acc = 0;
    for (int i = i_lo; i <= i_hi; i++) {
        acc += P[(int64_t) i * M + cout * k + (idx - i * s0)];
    }
    dst[g] = acc;
}

// ============================================================================
// Candidate (b) — direct fused gather kernel, one launch. Each thread owns one
// output element (same global index mapping as V0, so dst writes stay
// coalesced) and sums only its contributing taps: i in [i_lo, i_hi] (at most
// ceil(K/S) = 2 for all four shapes), j = idx - i*S. Per cin, the two W taps
// sit S floats apart inside one 64 B (cin, cout) row — consecutive threads
// read consecutive j, so W traffic is coalesced runs and the row is L1-hot
// across the block; X reads are broadcast. No shared memory, no atomics.
// ============================================================================
static __global__ void vb_fused_kernel(
        const int k, const int s0, const int cin_n, const int cout_n,
        const int lin, const int lout,
        const float * __restrict__ W, const float * __restrict__ X,
        float * __restrict__ dst) {
    const int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= lout * cout_n) {
        return;
    }
    const int cout = g / lout;
    const int idx  = g % lout;

    const int i_lo = (idx >= k) ? (idx - k) / s0 + 1 : 0;
    const int i_hi = min(idx / s0, lin - 1);
    const int kc   = k * cout_n;

    float acc = 0;
    for (int cin = 0; cin < cin_n; cin++) {
        const float * wr = W + (int64_t) cin * kc + cout * k;
        const float * xr = X + (int64_t) cin * lin;
        for (int i = i_lo; i <= i_hi; i++) {
            acc += wr[idx - i * s0] * xr[i];
        }
    }
    dst[g] = acc;
}

// ============================================================================
// Harness
// ============================================================================

// BF16 truncation (round-to-nearest-even like the checkpoint conversion).
static float bf16_trunc(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t lsb = (bits >> 16) & 1;
    bits = (bits + 0x7fff + lsb) & 0xffff0000u;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

static void cpu_reference(const ConvShape &sh, const std::vector<float> &W,
        const std::vector<float> &X, std::vector<double> &ref) {
    const int lout = (sh.lin - 1) * sh.s + sh.k;
    ref.assign((size_t) lout * sh.cout, 0.0);
    for (int cin = 0; cin < sh.cin; cin++) {
        for (int cout = 0; cout < sh.cout; cout++) {
            const float * wr = W.data() + (size_t) cin * sh.k * sh.cout + (size_t) cout * sh.k;
            for (int i = 0; i < sh.lin; i++) {
                const double x = X[(size_t) cin * sh.lin + i];
                for (int j = 0; j < sh.k; j++) {
                    ref[(size_t) cout * lout + i * sh.s + j] += (double) wr[j] * x;
                }
            }
        }
    }
}

struct DiffStats {
    double max_abs;
    double rel_l2;
};

template <typename TRef>
static DiffStats diff_stats(const std::vector<float> &got, const std::vector<TRef> &ref) {
    double err = 0, refsq = 0, max_abs = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = (double) got[i] - (double) ref[i];
        err += d * d;
        refsq += (double) ref[i] * (double) ref[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    return { max_abs, std::sqrt(err) / std::sqrt(refsq) };
}

template <typename LaunchFn>
static float time_median_ms(LaunchFn launch) {
    const int warmup = 20, iters = 100;
    for (int i = 0; i < warmup; i++) {
        launch();
    }
    CU_CHECK(cudaDeviceSynchronize());
    cudaEvent_t ea, eb;
    CU_CHECK(cudaEventCreate(&ea));
    CU_CHECK(cudaEventCreate(&eb));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; i++) {
        CU_CHECK(cudaEventRecord(ea));
        launch();
        CU_CHECK(cudaEventRecord(eb));
        CU_CHECK(cudaEventSynchronize(eb));
        CU_CHECK(cudaEventElapsedTime(&ms[i], ea, eb));
    }
    cudaEventDestroy(ea);
    cudaEventDestroy(eb);
    std::sort(ms.begin(), ms.end());
    return ms[iters / 2];
}

int main() {
    printf("mimi_conv_bench — conv_transpose_1d candidates vs existing ggml kernel\n");
    printf("shapes: certified from lm_default.h + nsys grid evidence {48,102,243,481}x256\n\n");
    printf("%-36s %10s %10s %8s %14s %12s  %s\n",
        "shape", "variant", "ms/call", "GB/s", "max_abs_diff", "rel_l2(V0)", "gate(<1e-5)");

    const char * variant_names[3] = { "V0_ggml", "Va_gemm", "Vb_fused" };
    double sum_ms[3] = { 0, 0, 0 };
    bool all_pass = true;

    for (int si = 0; si < N_SHAPES; si++) {
        const ConvShape &sh = SHAPES[si];
        const int lout = (sh.lin - 1) * sh.s + sh.k;
        const int M = sh.k * sh.cout;
        const size_t w_n = (size_t) sh.cin * sh.cout * sh.k;
        const size_t x_n = (size_t) sh.cin * sh.lin;
        const size_t o_n = (size_t) sh.cout * lout;
        const double useful_gb = (double) (w_n + x_n + o_n) * 4 / 1e9;

        // Identical inputs/weights for every variant; weights BF16-truncated.
        std::mt19937 rng(0x6d696d69u + si);  // "mimi"
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        std::vector<float> W(w_n), X(x_n);
        for (auto &v : W) v = bf16_trunc(dist(rng));
        for (auto &v : X) v = dist(rng);

        std::vector<double> cpu_ref;
        cpu_reference(sh, W, X, cpu_ref);

        float *d_w, *d_x, *d_out, *d_p;
        CU_CHECK(cudaMalloc(&d_w, w_n * 4));
        CU_CHECK(cudaMalloc(&d_x, x_n * 4));
        CU_CHECK(cudaMalloc(&d_out, o_n * 4));
        CU_CHECK(cudaMalloc(&d_p, (size_t) M * sh.lin * 4));
        CU_CHECK(cudaMemcpy(d_w, W.data(), w_n * 4, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(d_x, X.data(), x_n * 4, cudaMemcpyHostToDevice));

        const int out_blocks = ((int) o_n + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1)
                / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;

        auto launch_v0 = [&]() {
            v0_ggml_kernel<<<out_blocks, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE>>>(
                sh.s, (int) o_n, sh.k, sh.cout, sh.cin, sh.lin, lout, d_w, d_x, d_out);
        };
        auto launch_va = [&]() {
            dim3 grid((M + 255) / 256, (sh.lin + VA_COLS - 1) / VA_COLS);
            va_gemm_kernel<<<grid, 256>>>(M, sh.cin, sh.lin, d_w, d_x, d_p);
            va_combine_kernel<<<out_blocks, 256>>>(sh.k, sh.s, sh.cout, sh.lin, lout, d_p, d_out);
        };
        auto launch_vb = [&]() {
            vb_fused_kernel<<<out_blocks, 256>>>(
                sh.k, sh.s, sh.cin, sh.cout, sh.lin, lout, d_w, d_x, d_out);
        };

        // V0 first: output IS the reference for the gate; CPU double guards V0.
        std::vector<float> v0_out(o_n), out(o_n);
        launch_v0();
        CU_CHECK(cudaDeviceSynchronize());
        CU_CHECK(cudaGetLastError());
        CU_CHECK(cudaMemcpy(v0_out.data(), d_out, o_n * 4, cudaMemcpyDeviceToHost));
        DiffStats v0_vs_cpu = diff_stats(v0_out, cpu_ref);
        if (v0_vs_cpu.rel_l2 > 1e-6) {
            fprintf(stderr, "HARNESS FAIL: V0 vs CPU double rel_l2=%.3e on %s\n",
                v0_vs_cpu.rel_l2, sh.name);
            return 1;
        }

        for (int vi = 0; vi < 3; vi++) {
            CU_CHECK(cudaMemset(d_out, 0xff, o_n * 4));  // poison: catch unwritten output
            if (vi == 0) launch_v0();
            if (vi == 1) launch_va();
            if (vi == 2) launch_vb();
            CU_CHECK(cudaDeviceSynchronize());
            CU_CHECK(cudaGetLastError());
            CU_CHECK(cudaMemcpy(out.data(), d_out, o_n * 4, cudaMemcpyDeviceToHost));
            DiffStats ds = diff_stats(out, v0_out);
            bool pass = ds.rel_l2 < 1e-5;
            all_pass = all_pass && pass;

            float med = 0;
            if (vi == 0) med = time_median_ms(launch_v0);
            if (vi == 1) med = time_median_ms(launch_va);
            if (vi == 2) med = time_median_ms(launch_vb);
            sum_ms[vi] += med;

            printf("%-36s %10s %10.4f %8.1f %14.3e %12.3e  %s\n",
                vi == 0 ? sh.name : "", variant_names[vi], med, useful_gb / (med / 1e3),
                ds.max_abs, ds.rel_l2, pass ? "PASS" : "FAIL");
        }
        printf("%-36s %10s %10s   (in-graph V0: %.3f ms; V0 vs CPU-double rel_l2 %.2e)\n",
            "", "", "", sh.in_graph_ms, v0_vs_cpu.rel_l2);

        CU_CHECK(cudaFree(d_w));
        CU_CHECK(cudaFree(d_x));
        CU_CHECK(cudaFree(d_out));
        CU_CHECK(cudaFree(d_p));
    }

    printf("\nper-frame totals (sum of 4 shapes, 1 call each):\n");
    for (int vi = 0; vi < 3; vi++) {
        // Project t_mimi_dec by scaling the in-graph conv time by the bench
        // ratio (bench absolute times differ from in-graph: no nsys, no
        // concurrent graph work), then swapping it into the phase baseline.
        double ratio = sum_ms[vi] / sum_ms[0];
        double proj = T_MIMI_DEC_MS - IN_GRAPH_TOTAL_MS + IN_GRAPH_TOTAL_MS * ratio;
        printf("  %-10s %8.4f ms   speedup_vs_V0 %7.2fx   projected t_mimi_dec %7.2f ms\n",
            variant_names[vi], sum_ms[vi], sum_ms[0] / sum_ms[vi], proj);
    }
    printf("\nGATE: %s\n", all_pass ? "ALL PASS (rel_l2 < 1e-5 vs V0 on every shape)" : "FAIL");
    return all_pass ? 0 : 1;
}
