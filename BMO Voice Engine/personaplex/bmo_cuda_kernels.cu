#include "bmo.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace {

__device__ inline float fp16_to_fp32_device(ggml_fp16_t value) {
    __half_raw raw;
    raw.x = value;
    const __half half_value = *reinterpret_cast<const __half *>(&raw);
    return __half2float(half_value);
}

__global__ void unpack_kernel(
    const uint8_t * packed_weights,
    const uint8_t * packed_mask,
    int rows,
    int cols,
    int n_2bit_bytes,
    int n_4bit_bytes,
    int n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const int32_t * block_offset,
    const ggml_fp16_t * fp16_values,
    int block_size,
    float * out_w) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (pos >= total) {
        return;
    }

    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    const int block_idx = pos / block_size;
    const int in_block = pos - block_idx * block_size;
    const uint8_t mbyte = packed_mask[block_idx / 4];
    const uint8_t tier = (mbyte >> ((block_idx % 4) * 2)) & 0x3;
    const int off = block_offset[block_idx];

    float v = 0.0f;
    if (tier == 0) {
        v = fp16_to_fp32_device(fp16_values[off + in_block]);
    } else if (tier == 1) {
        const uint8_t q = stream8[off + in_block];
        v = ((float) q - zp_int8) * scale_int8;
    } else if (tier == 2) {
        const int idx4 = off + in_block;
        const uint8_t b = stream4[idx4 / 2];
        const uint8_t q = (idx4 % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
        v = ((float) q - zp_int4) * scale_int4;
    } else {
        const int idx2 = off + in_block;
        const uint8_t b = stream2[idx2 / 4];
        const uint8_t q = (b >> ((idx2 % 4) * 2)) & 0x3;
        v = ((float) q - zp_low) * scale_low;
    }
    out_w[pos] = v;
}

} // namespace

// V2: 8 rows per block, 1 warp per row, blockDim.x = 256 fixed.
template <int ROWS_PER_BLOCK = 8>
__global__ void fused_dequant_matvec_kernel_v2(
    const uint8_t * __restrict__ pw,
    const uint8_t * __restrict__ pm,
    const __half * __restrict__ fp16_vals,
    int rows, int cols, int block_size,
    int n_2bit_bytes, int n_4bit_bytes,
    float scale_low, float scale_int4, float scale_int8,
    float zp_low, float zp_int4, float zp_int8,
    const float * __restrict__ x,
    float * __restrict__ y) {
    static_assert(ROWS_PER_BLOCK == 8, "v2 expects 8 rows/block (256 threads)");

    const int tid = threadIdx.x;
    const int row_in_block = tid >> 5; // tid / 32
    const int lane = tid & 31;         // tid % 32
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;

    if (row >= rows) return;

    const int blocks_per_row = cols / block_size; // assumes cols % 32 == 0

    extern __shared__ uint8_t smem_raw[];
    uint8_t * s_tier_all = smem_raw;
    int * s_off_all = reinterpret_cast<int *>(s_tier_all + ROWS_PER_BLOCK * blocks_per_row);

    uint8_t * s_tier = s_tier_all + row_in_block * blocks_per_row;
    int * s_off = s_off_all + row_in_block * blocks_per_row;

    // ---- Load tiers for this row (32 threads cooperate, stride 32) ----
    for (int b = lane; b < blocks_per_row; b += 32) {
        const int64_t b_global = (int64_t) row * blocks_per_row + b;
        const uint8_t mbyte = pm[b_global >> 2];
        s_tier[b] = (mbyte >> ((b_global & 3) * 2)) & 0x3;
    }
    __syncthreads();

    // ---- Prefix scan: lane 0 of each row, sequential ----
    if (lane == 0) {
        int c2 = 0, c4 = 0, c8 = 0, c16 = 0;
        for (int i = 0; i < blocks_per_row; ++i) {
            const uint8_t t = s_tier[i];
            if (t == 0) {
                s_off[i] = c16;
                c16 += block_size;
            } else if (t == 1) {
                s_off[i] = c8;
                c8 += block_size;
            } else if (t == 2) {
                s_off[i] = c4;
                c4 += block_size;
            } else {
                s_off[i] = c2;
                c2 += block_size;
            }
        }
    }
    __syncwarp();

    // ---- Matvec: 32 threads stride through cols ----
    const uint8_t * stream8 = pw + n_2bit_bytes + n_4bit_bytes;
    const uint8_t * stream4 = pw + n_2bit_bytes;
    const uint8_t * stream2 = pw;

    float acc = 0.0f;
    const int n_iters = cols >> 5; // cols / 32

#pragma unroll 4
    for (int k = 0; k < n_iters; ++k) {
        const int c = (k << 5) + lane;
        const uint8_t tier = s_tier[k];
        const int off = s_off[k];

        float w;
        if (tier == 0) {
            w = __half2float(fp16_vals[off + lane]);
        } else if (tier == 1) {
            const uint8_t q = stream8[off + lane];
            w = ((float) q - zp_int8) * scale_int8;
        } else if (tier == 2) {
            const int idx = off + lane;
            const uint8_t bb = stream4[idx >> 1];
            const uint8_t q = (idx & 1) ? ((bb >> 4) & 0xF) : (bb & 0xF);
            w = ((float) q - zp_int4) * scale_int4;
        } else {
            const int idx = off + lane;
            const uint8_t bb = stream2[idx >> 2];
            const uint8_t q = (bb >> ((idx & 3) * 2)) & 0x3;
            w = ((float) q - zp_low) * scale_low;
        }
        acc += w * x[c];
    }

    // ---- Pure warp-shuffle reduction ----
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) y[row] = acc;
}

void launch_unpack_kernel_streamed(
    const void * packed_weights,
    const void * packed_mask,
    const void * fp16_values,
    const int32_t * block_offset,
    int32_t rows,
    int32_t cols,
    int32_t block_size,
    int32_t n_2bit_bytes,
    int32_t n_4bit_bytes,
    int32_t n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    float * out_w) {
    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;

    unpack_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint8_t *>(packed_weights),
        reinterpret_cast<const uint8_t *>(packed_mask),
        rows,
        cols,
        n_2bit_bytes,
        n_4bit_bytes,
        n_8bit_bytes,
        scale_low,
        scale_int4,
        scale_int8,
        zp_low,
        zp_int4,
        zp_int8,
        block_offset,
        reinterpret_cast<const ggml_fp16_t *>(fp16_values),
        block_size > 0 ? block_size : 32,
        out_w);
}

__global__ void rmsnorm_kernel(
    const float * __restrict__ x,
    const float * __restrict__ weight,
    float eps, int n_embd, float * __restrict__ y) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        const float v = x[i];
        sum_sq += v * v;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    float total_sq = 0.0f;
    if (warp_id == 0) {
        total_sq = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            total_sq += __shfl_down_sync(0xff, total_sq, offset);
        }
        if (lane == 0) warp_sums[0] = total_sq;
    }
    __syncthreads();
    total_sq = warp_sums[0];

    const float scale = rsqrtf(total_sq / (float) n_embd + eps);

    for (int i = tid; i < n_embd; i += blockDim.x) {
        y[i] = x[i] * scale * weight[i];
    }
}

void launch_rmsnorm(
    const float * x_dev,
    const float * weight_dev,
    float eps,
    int n_embd,
    float * y_dev,
    void * stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    rmsnorm_kernel<<<1, 256, 0, s>>>(x_dev, weight_dev, eps, n_embd, y_dev);
}

void launch_fused_dequant_matvec(
    const void * pw,
    const void * pm,
    const void * fp16_vals,
    int rows,
    int cols,
    int block_size,
    int n_2bit_bytes,
    int n_4bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const float * x,
    float * y) {
    constexpr int ROWS_PER_BLOCK = 8;
    const int blocks_per_row = cols / block_size;
    const size_t smem_bytes =
        (size_t) ROWS_PER_BLOCK * blocks_per_row * sizeof(uint8_t)
        + (size_t) ROWS_PER_BLOCK * blocks_per_row * sizeof(int);

    const int n_blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    fused_dequant_matvec_kernel_v2<ROWS_PER_BLOCK><<<n_blocks, 256, smem_bytes>>>(
        reinterpret_cast<const uint8_t *>(pw),
        reinterpret_cast<const uint8_t *>(pm),
        reinterpret_cast<const __half *>(fp16_vals),
        rows,
        cols,
        block_size,
        n_2bit_bytes,
        n_4bit_bytes,
        scale_low,
        scale_int4,
        scale_int8,
        zp_low,
        zp_int4,
        zp_int8,
        x,
        y);
}

#ifndef BMO_JETSON
void launch_unpack_kernel(
    const device_packed_t * dp,
    int32_t rows,
    int32_t cols,
    int32_t n_2bit_bytes,
    int32_t n_4bit_bytes,
    int32_t n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    float * out_w) {
    if (!dp || !dp->is_valid) {
        throw std::runtime_error("launch_unpack_kernel: invalid device_packed_t");
    }

    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;

    unpack_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint8_t *>(dp->packed_weights),
        reinterpret_cast<const uint8_t *>(dp->packed_mask),
        rows,
        cols,
        n_2bit_bytes,
        n_4bit_bytes,
        n_8bit_bytes,
        scale_low,
        scale_int4,
        scale_int8,
        zp_low,
        zp_int4,
        zp_int8,
        reinterpret_cast<const int32_t *>(dp->block_offset),
        reinterpret_cast<const ggml_fp16_t *>(dp->fp16_values),
        dp->block_size > 0 ? dp->block_size : 32,
        out_w);
}
#endif
