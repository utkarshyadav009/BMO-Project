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
    const int32_t * idx2_start,
    const int32_t * idx4_start,
    const int32_t * idx8_start,
    float * out_w) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }

    int idx2 = idx2_start[r];
    int idx4 = idx4_start[r];
    int idx8 = idx8_start[r];

    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    const int base_pos = r * cols;
    for (int c = 0; c < cols; ++c) {
        const int pos = base_pos + c;
        const uint8_t mbyte = packed_mask[pos / 4];
        const uint8_t tier = (mbyte >> ((pos % 4) * 2)) & 0x3;
        float v = 0.0f;
        if (tier >= 3) {
            const uint8_t b = stream2[idx2 / 4];
            const uint8_t q = (b >> ((idx2 % 4) * 2)) & 0x3;
            ++idx2;
            v = ((float) q - zp_low) * scale_low;
        } else if (tier == 2) {
            const uint8_t b = stream4[idx4 / 2];
            const uint8_t q = (idx4 % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
            ++idx4;
            v = ((float) q - zp_int4) * scale_int4;
        } else if (tier == 1) {
            const uint8_t q = stream8[idx8];
            ++idx8;
            v = ((float) q - zp_int8) * scale_int8;
        }
        out_w[pos] = v;
    }
}

__global__ void apply_fp16_overrides(
    const int32_t * fp16_idx,
    const ggml_fp16_t * fp16_vals,
    int64_t n_fp16,
    float * out_w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int64_t) i >= n_fp16) {
        return;
    }
    const int32_t pos = fp16_idx[i];
    out_w[pos] = fp16_to_fp32_device(fp16_vals[i]);
}

} // namespace

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

    const int threads = 128;
    const int blocks = (rows + threads - 1) / threads;

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
        reinterpret_cast<const int32_t *>(dp->idx2_start),
        reinterpret_cast<const int32_t *>(dp->idx4_start),
        reinterpret_cast<const int32_t *>(dp->idx8_start),
        out_w);

    if (dp->n_fp16 > 0 && dp->fp16_indices && dp->fp16_values) {
        const int fp16_threads = 128;
        const int fp16_blocks = (int) ((dp->n_fp16 + fp16_threads - 1) / fp16_threads);
        apply_fp16_overrides<<<fp16_blocks, fp16_threads>>>(
            reinterpret_cast<const int32_t *>(dp->fp16_indices),
            reinterpret_cast<const ggml_fp16_t *>(dp->fp16_values),
            dp->n_fp16,
            out_w);
    }
}