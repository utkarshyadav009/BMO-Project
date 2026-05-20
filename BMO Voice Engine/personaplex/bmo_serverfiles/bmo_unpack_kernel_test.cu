#include "bmo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

static int32_t read_scalar_i32(ggml_context * data_ctx, const std::string & name, int32_t fallback = 0) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name.c_str());
    if (!t || ggml_nbytes(t) < (int) sizeof(int32_t)) return fallback;
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context * data_ctx, const std::string & name, float fallback = 0.0f) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name.c_str());
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) return fallback;
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

static void unpack_blockwise_cpu(
    const uint8_t * packed_weights,
    const uint8_t * packed_mask,
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
    const ggml_fp16_t * fp16_values,
    float * out_w) {
    const int64_t total = (int64_t) rows * (int64_t) cols;
    const int64_t n_blocks = (total + block_size - 1) / block_size;
    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    int32_t c2 = 0;
    int32_t c4 = 0;
    int32_t c8 = 0;
    int32_t c16 = 0;
    for (int64_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t mbyte = packed_mask[(size_t) (block_idx / 4)];
        const uint8_t tier = unpack_u2_le(mbyte, (int) (block_idx % 4));
        if (tier == 0) {
            for (int32_t in_block = 0; in_block < block_size; ++in_block) {
                const int64_t pos = block_idx * block_size + in_block;
                if (pos >= total) break;
                out_w[(size_t) pos] = ggml_fp16_to_fp32(fp16_values[(size_t) c16 + (size_t) in_block]);
            }
            c16 += block_size;
        } else if (tier == 1) {
            for (int32_t in_block = 0; in_block < block_size; ++in_block) {
                const int64_t pos = block_idx * block_size + in_block;
                if (pos >= total) break;
                const uint8_t q = stream8[c8 + in_block];
                out_w[(size_t) pos] = ((float) q - zp_int8) * scale_int8;
            }
            c8 += block_size;
        } else if (tier == 2) {
            for (int32_t in_block = 0; in_block < block_size; ++in_block) {
                const int64_t pos = block_idx * block_size + in_block;
                if (pos >= total) break;
                const int32_t idx4 = c4 + in_block;
                const uint8_t b = stream4[idx4 / 2];
                const uint8_t q = (idx4 % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                out_w[(size_t) pos] = ((float) q - zp_int4) * scale_int4;
            }
            c4 += block_size;
        } else {
            for (int32_t in_block = 0; in_block < block_size; ++in_block) {
                const int64_t pos = block_idx * block_size + in_block;
                if (pos >= total) break;
                const int32_t idx2 = c2 + in_block;
                const uint8_t b = stream2[idx2 / 4];
                const uint8_t q = unpack_u2_le(b, (int) (idx2 % 4));
                out_w[(size_t) pos] = ((float) q - zp_low) * scale_low;
            }
            c2 += block_size;
        }
    }
}

__device__ inline float fp16_to_fp32_device(ggml_fp16_t value) {
    __half_raw raw;
    raw.x = value;
    const __half half_value = *reinterpret_cast<const __half *>(&raw);
    return __half2float(half_value);
}

__global__ void unpack_blockwise_kernel(
    const uint8_t * packed_weights,
    const uint8_t * packed_mask,
    int rows,
    int cols,
    int block_size,
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
    float * out_w) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (pos >= total) return;

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

template <typename T>
static void cuda_copy_to_device(T ** dst, const void * src, size_t bytes, const char * label) {
    if (bytes == 0) {
        *dst = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(dst), bytes);
    if (err != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc failed for ") + label + ": " + cudaGetErrorString(err));
    err = cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy failed for ") + label + ": " + cudaGetErrorString(err));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: bmo_unpack_kernel_test <weights.gguf> [packed_base]\n";
        return 1;
    }

    const std::string base = argc >= 3 ? argv[2] : "transformer_layers_0_gating_linear_in_weight";

    bmo_model model;
    bmo_context ctx;
    try {
        bmo_load_model(argv[1], model, ctx);
    } catch (const std::exception & ex) {
        std::cerr << "Failed to load model: " << ex.what() << "\n";
        return 2;
    }

    ggml_tensor * pw_t = ggml_get_tensor(model.wctx, (base + ".packed_weights").c_str());
    ggml_tensor * pm_t = ggml_get_tensor(model.wctx, (base + ".packed_mask").c_str());
    ggml_tensor * fv_t = ggml_get_tensor(model.wctx, (base + ".fp16_values").c_str());
    if (!pw_t || !pm_t || !fv_t) {
        std::cerr << "Missing SEPTQ v3 block-wise tensors for base: " << base << "\n";
        return 3;
    }

    const int32_t rows = read_scalar_i32(model.wctx, base + ".rows", 0);
    const int32_t cols = read_scalar_i32(model.wctx, base + ".cols", 0);
    const int32_t block_size = read_scalar_i32(model.wctx, base + ".block_size", 32);
    if (rows <= 0 || cols <= 0 || block_size <= 0) {
        std::cerr << "Invalid dimensions for " << base << " rows=" << rows << " cols=" << cols << " block_size=" << block_size << "\n";
        return 4;
    }

    const int32_t n_2bit_bytes = read_scalar_i32(model.wctx, base + ".n_2bit_bytes", 0);
    const int32_t n_4bit_bytes = read_scalar_i32(model.wctx, base + ".n_4bit_bytes", 0);
    const int32_t n_8bit_bytes = read_scalar_i32(model.wctx, base + ".n_8bit_bytes", 0);
    const float scale_low = read_scalar_f32(model.wctx, base + ".scale_low", 1.0f);
    const float scale_int4 = read_scalar_f32(model.wctx, base + ".scale_int4", 1.0f);
    const float scale_int8 = read_scalar_f32(model.wctx, base + ".scale_int8", 1.0f);
    const float zp_low = read_scalar_f32(model.wctx, base + ".zp_low", 1.5f);
    const float zp_int4 = read_scalar_f32(model.wctx, base + ".zp_int4", 7.5f);
    const float zp_int8 = read_scalar_f32(model.wctx, base + ".zp_int8", 127.5f);

    if (fv_t->type != GGML_TYPE_F16) {
        std::cerr << "Expected fp16_values to be GGML_TYPE_F16 for SEPTQ v3 test\n";
        return 5;
    }

    const auto * pw = reinterpret_cast<const uint8_t *>(pw_t->data);
    const auto * pm = reinterpret_cast<const uint8_t *>(pm_t->data);
    const auto * fv = reinterpret_cast<const ggml_fp16_t *>(fv_t->data);

    const int64_t total = (int64_t) rows * (int64_t) cols;
    const int64_t n_blocks = (total + block_size - 1) / block_size;
    std::vector<int32_t> block_offset((size_t) n_blocks, 0);
    int32_t c2 = 0;
    int32_t c4 = 0;
    int32_t c8 = 0;
    int32_t c16 = 0;
    for (int64_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t mbyte = pm[(size_t) block_idx / 4];
        const uint8_t tier = unpack_u2_le(mbyte, (int) (block_idx % 4));
        if (tier == 0) {
            block_offset[(size_t) block_idx] = c16;
            c16 += block_size;
        } else if (tier == 1) {
            block_offset[(size_t) block_idx] = c8;
            c8 += block_size;
        } else if (tier == 2) {
            block_offset[(size_t) block_idx] = c4;
            c4 += block_size;
        } else {
            block_offset[(size_t) block_idx] = c2;
            c2 += block_size;
        }
    }

    std::vector<float> cpu_out((size_t) total);
    unpack_blockwise_cpu(
        pw, pm, rows, cols, block_size, n_2bit_bytes, n_4bit_bytes, n_8bit_bytes,
        scale_low, scale_int4, scale_int8, zp_low, zp_int4, zp_int8,
        fv, cpu_out.data());

    uint8_t * d_pw = nullptr;
    uint8_t * d_pm = nullptr;
    int32_t * d_block_offset = nullptr;
    ggml_fp16_t * d_fv = nullptr;
    float * d_out = nullptr;

    try {
        cuda_copy_to_device(&d_pw, pw, (size_t) ggml_nbytes(pw_t), "packed_weights");
        cuda_copy_to_device(&d_pm, pm, (size_t) ggml_nbytes(pm_t), "packed_mask");
        cuda_copy_to_device(&d_block_offset, block_offset.data(), block_offset.size() * sizeof(int32_t), "block_offset");
        cuda_copy_to_device(&d_fv, fv, (size_t) ggml_nbytes(fv_t), "fp16_values");
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_out), (size_t) total * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc failed for output: ") + cudaGetErrorString(err));
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << "\n";
        return 6;
    }

    const int threads = 256;
    const int blocks = (int) ((total + threads - 1) / threads);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    unpack_blockwise_kernel<<<blocks, threads>>>(
        d_pw, d_pm, rows, cols, block_size, n_2bit_bytes, n_4bit_bytes, n_8bit_bytes,
        scale_low, scale_int4, scale_int8, zp_low, zp_int4, zp_int8,
        d_block_offset, d_fv, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err) << "\n";
        return 7;
    }

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    std::vector<float> gpu_out((size_t) total);
    err = cudaMemcpy(gpu_out.data(), d_out, (size_t) total * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy output failed: " << cudaGetErrorString(err) << "\n";
        return 8;
    }

    double max_abs_diff = 0.0;
    size_t mismatches = 0;
    for (int64_t pos = 0; pos < total; ++pos) {
        const double diff = std::abs((double) cpu_out[(size_t) pos] - (double) gpu_out[(size_t) pos]);
        max_abs_diff = std::max(max_abs_diff, diff);
        if (diff != 0.0) ++mismatches;
    }

    std::cout << "base=" << base << "\n";
    std::cout << "rows=" << rows << " cols=" << cols << " block_size=" << block_size << "\n";
    std::cout << "max_abs_diff=" << max_abs_diff << " mismatches=" << mismatches << "\n";
    std::cout << "kernel_ms=" << kernel_ms << "\n";
    std::cout << (mismatches == 0 ? "[PASS] block-wise CUDA unpack is bit-exact\n" : "[FAIL] block-wise CUDA unpack mismatch\n");

    cudaFree(d_pw);
    cudaFree(d_pm);
    cudaFree(d_block_offset);
    cudaFree(d_fv);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return mismatches == 0 ? 0 : 9;
}
