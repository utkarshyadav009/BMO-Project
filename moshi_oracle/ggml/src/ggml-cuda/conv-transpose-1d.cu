#include "conv-transpose-1d.cuh"

// Rewritten 2026-07-17 — transposed-conv-as-GEMM, selected by
// moshi.cpp/tools/mimi_conv_bench.cu (102.8x vs the previous kernel on the
// four mimi SEANet decoder shapes, sm_87). The previous kernel gave every
// output element a full (cin x Lin) loop and branched away non-contributing
// taps, doing ~Lin/ceil(K/s0) times the necessary work: 240x waste at mimi's
// model_11 shape (Lin=480), 126.5 ms/frame of a 133.4 ms t_mimi_dec phase.
//
// Two launches, no atomics:
//   1. P = W^T X. W's ggml layout [K, Cout, Cin] is already a row-major
//      [Cin, M=K*Cout] matrix — no reshape needed. P[i][m] =
//      sum_cin W[cin*M + m] * X[cin*Lin + i]; threads run over m (coalesced
//      W read and P write), a register tile of CONV_TRANSPOSE_1D_GEMM_COLS
//      columns amortizes each W element over up to that many i.
//   2. Gather: dst[cout][idx] = sum of its <= ceil(K/s0) contributing taps
//      P[i][cout*K + (idx - i*s0)], i in [i_lo, i_hi]. Same global-index
//      mapping as the previous kernel, so dst writes stay coalesced.
//
// Precision: F32 in / F32 out, unchanged. Summation order differs from the
// previous kernel (per-shape rel_l2 vs CPU double ~5e-7, same order as
// before; bench gate rel_l2 < 1e-5 vs the previous kernel PASSes all four
// mimi shapes).

#define CONV_TRANSPOSE_1D_GEMM_COLS 8

static __global__ void conv_transpose_1d_gemm_kernel(
        const int M, const int cin_n, const int lin,
        const float * __restrict__ W, const float * __restrict__ X,
        float * __restrict__ P) {
    const int m    = threadIdx.x + blockIdx.x * blockDim.x;
    const int col0 = blockIdx.y * CONV_TRANSPOSE_1D_GEMM_COLS;
    if (m >= M) {
        return;
    }

    float acc[CONV_TRANSPOSE_1D_GEMM_COLS] = { 0 };
    for (int cin = 0; cin < cin_n; cin++) {
        const float w = W[(int64_t) cin * M + m];
        const float * xr = X + (int64_t) cin * lin + col0;
#pragma unroll
        for (int c = 0; c < CONV_TRANSPOSE_1D_GEMM_COLS; c++) {
            if (col0 + c < lin) {
                acc[c] += w * xr[c];
            }
        }
    }
#pragma unroll
    for (int c = 0; c < CONV_TRANSPOSE_1D_GEMM_COLS; c++) {
        if (col0 + c < lin) {
            P[(int64_t) (col0 + c) * M + m] = acc[c];
        }
    }
}

static __global__ void conv_transpose_1d_combine_kernel(
        const int k, const int s0, const int cout_n, const int lin, const int lout,
        const int output_size,
        const float * __restrict__ P, float * __restrict__ dst) {
    const int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= output_size) {
        return;
    }
    const int cout = g / lout;
    const int idx  = g % lout;
    const int M    = k * cout_n;

    // Contributing inputs: i with i*s0 <= idx < i*s0 + k, clamped to [0, lin).
    const int i_lo = (idx >= k) ? (idx - k) / s0 + 1 : 0;
    const int i_hi = min(idx / s0, lin - 1);

    float acc = 0;
    for (int i = i_lo; i <= i_hi; i++) {
        acc += P[(int64_t) i * M + cout * k + (idx - i * s0)];
    }
    dst[g] = acc;
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    // The previous kernel silently mis-indexed the weight for batched input;
    // this one makes the (never exercised) limit explicit.
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];

    const int k      = src0->ne[0];
    const int cout_n = src0->ne[1];
    const int cin_n  = src0->ne[2];
    const int lin    = src1->ne[0];
    const int lout   = dst->ne[0];
    const int M      = k * cout_n;

    const int64_t output_size = ggml_nelements(dst);

    ggml_cuda_pool_alloc<float> P(ctx.pool(), (int64_t) M * lin);

    const dim3 gemm_grid(
        (M + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE,
        (lin + CONV_TRANSPOSE_1D_GEMM_COLS - 1) / CONV_TRANSPOSE_1D_GEMM_COLS);
    conv_transpose_1d_gemm_kernel<<<gemm_grid, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        M, cin_n, lin, src0_d, src1_d, P.get());

    const int num_blocks = ((int) output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1)
            / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_combine_kernel<<<num_blocks, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        k, s0, cout_n, lin, lout, (int) output_size, P.get(), dst_d);
}
