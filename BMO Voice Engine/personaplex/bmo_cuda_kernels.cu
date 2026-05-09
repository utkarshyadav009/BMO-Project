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

__global__ void fused_dequant_matvec_kernel_no_offsets(
    const uint8_t * __restrict__ pw,
    const uint8_t * __restrict__ pm,
    const __half * __restrict__ fp16_vals,
    int rows, int cols, int block_size,
    int n_2bit_bytes, int n_4bit_bytes,
    int row_block_start_global,
    int c2_base_global, int c4_base_global, int c8_base_global, int c16_base_global,
    float scale_low, float scale_int4, float scale_int8,
    float zp_low, float zp_int4, float zp_int8,
    const float * __restrict__ x,
    float * __restrict__ y) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    (void) row_block_start_global;
    (void) c2_base_global;
    (void) c4_base_global;
    (void) c8_base_global;
    (void) c16_base_global;

    const int blocks_per_row = cols / block_size;
    const int64_t global_block_start = (int64_t) row * blocks_per_row;

    __shared__ int32_t s_off[256];
    __shared__ uint8_t s_tier[256];

    if (tid < blocks_per_row) {
        const int64_t b_global = global_block_start + tid;
        const uint8_t mbyte = pm[b_global >> 2];
        const uint8_t tier = (mbyte >> ((b_global & 3) * 2)) & 0x3;
        s_tier[tid] = tier;
    }
    __syncthreads();

    if (tid == 0) {
        int32_t c2 = 0, c4 = 0, c8 = 0, c16 = 0;
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
    __syncthreads();

    float acc = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const int block_within_row = c / block_size;
        const int in_block = c - block_within_row * block_size;
        const uint8_t tier = s_tier[block_within_row];
        const int off = s_off[block_within_row];

        float w;
        if (tier == 0) {
            w = __half2float(fp16_vals[off + in_block]);
        } else if (tier == 1) {
            uint8_t q = pw[n_2bit_bytes + n_4bit_bytes + off + in_block];
            w = ((float) q - zp_int8) * scale_int8;
        } else if (tier == 2) {
            int idx = off + in_block;
            uint8_t bb = pw[n_2bit_bytes + (idx >> 1)];
            uint8_t q = (idx & 1) ? ((bb >> 4) & 0xF) : (bb & 0xF);
            w = ((float) q - zp_int4) * scale_int4;
        } else {
            int idx = off + in_block;
            uint8_t bb = pw[idx >> 2];
            uint8_t q = (bb >> ((idx & 3) * 2)) & 0x3;
            w = ((float) q - zp_low) * scale_low;
        }
        acc += w * x[c];
    }

    __shared__ float sdata[256];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = sdata[0];
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
    fused_dequant_matvec_kernel_no_offsets<<<rows, 256>>>(
        reinterpret_cast<const uint8_t *>(pw),
        reinterpret_cast<const uint8_t *>(pm),
        reinterpret_cast<const __half *>(fp16_vals),
        rows,
        cols,
        block_size,
        n_2bit_bytes,
        n_4bit_bytes,
        0, 0, 0, 0, 0,
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
