#include "conv-transpose-1d.cuh"

// Rewritten 2026-07-17 — direct fused gather kernel, selected by
// moshi.cpp/tools/mimi_conv_bench.cu (25.0x vs the previous kernel on the
// four mimi SEANet decoder shapes, sm_87). The previous kernel gave every
// output element a full (cin x Lin) loop and branched away non-contributing
// taps, doing ~Lin/ceil(K/s0) times the necessary work: 240x waste at mimi's
// model_11 shape (Lin=480), 126.5 ms/frame of a 133.4 ms t_mimi_dec phase.
//
// Each thread owns one output element (same global-index mapping as the
// previous kernel, so dst writes stay coalesced) and visits only its
// contributing taps: i in [i_lo, i_hi] (<= ceil(K/s0), = 2 for all four mimi
// shapes), j = idx - i*s0. No shared memory, no atomics, one launch.
//
// BIT-EXACTNESS CONSTRAINT (do not "optimize" this away): the accumulation
// runs cin-outer / i-inner ascending with a single accumulator — the exact
// FMA order of the previous kernel restricted to its contributing taps — so
// the result is bit-identical to the previous kernel (bench max_abs_diff
// 0.0e+00 on all four mimi shapes; the integration gate requires end-to-end
// decoded-waveform rel_l2 < 1e-4 vs the old build). The faster GEMM+gather
// formulation (102.8x per-op, Va_gemm in the bench, briefly integrated as
// b0e525b) reorders the summation: its per-op rel_l2 ~6e-7 — harmless in
// isolation — amplifies through the streaming decoder (resnet chains,
// 250-frame-context transformer, conv tail-state feedback) to ~5.6e-4
// end-to-end over 1250 frames and FAILS that gate. Both t_mimi_dec outcomes
// (~8.5 ms GEMM vs ~12 ms fused) are far below the 55 ms integration
// target, so the bit-exact variant wins.

static __global__ void conv_transpose_1d_kernel(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0,
        const int dst_ne0,
        const float * __restrict__ src0, const float * __restrict__ src1,
        float * __restrict__ dst) {
    const int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    const int out_index = global_index / dst_ne0;
    const int idx       = global_index % dst_ne0;

    // Contributing inputs: i with i*s0 <= idx < i*s0 + K, clamped to [0, Lin).
    const int i_lo = (idx >= src0_ne0) ? (idx - src0_ne0) / s0 + 1 : 0;
    const int i_hi = min(idx / s0, src1_ne0 - 1);

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        const int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        const int input_offset = src1_ne0 * c;

        for (int i = i_lo; i <= i_hi; i++) {
            const float kernel_weight = src0[kernel_offset + idx - i * s0];
            const float input_value = src1[input_offset + i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0,
        const int dst_ne0,
        const float * src0, const float * src1, float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<<<num_blocks, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        s0, output_size,
        src0_ne0, src0_ne1, src0_ne2,
        src1_ne0,
        dst_ne0,
        src0, src1, dst);
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

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_cuda(s0, (int) output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0],
        dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}
