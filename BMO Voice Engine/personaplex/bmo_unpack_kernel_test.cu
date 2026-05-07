#include "bmo.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

static int32_t read_scalar_i32(ggml_context * data_ctx, const std::string & name, int32_t fallback = -1) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name.c_str());
    if (!t) return fallback;
    if (ggml_nbytes(t) < (int)sizeof(int32_t)) return fallback;
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context * data_ctx, const std::string & name, float fallback = 0.0f) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name.c_str());
    if (!t) return fallback;
    if (ggml_nbytes(t) < (int)sizeof(float)) return fallback;
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

// Copied CPU baseline unpack implementation (kept local to avoid linking)
static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

static void unpack_layer_to_f32_cpu(
    const uint8_t * packed_weights,
    const uint8_t * packed_mask,
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
    const int32_t * fp16_indices,
    int64_t n_fp16,
    const ggml_fp16_t * fp16_values,
    float * out_w) {
    const int64_t total = (int64_t) rows * (int64_t) cols;

    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    int64_t idx2 = 0;
    int64_t idx4 = 0;
    int64_t idx8 = 0;

    for (int64_t pos = 0; pos < total; ++pos) {
        const uint8_t mbyte = packed_mask[(size_t) (pos / 4)];
        const uint8_t tier = unpack_u2_le(mbyte, (int) (pos % 4));

        float v = 0.0f;
        if (tier >= 3) {
            const uint8_t b = stream2[(size_t) (idx2 / 4)];
            const uint8_t q = unpack_u2_le(b, (int) (idx2 % 4));
            ++idx2;
            v = ((float) q - zp_low) * scale_low;
        } else if (tier == 2) {
            const uint8_t b = stream4[(size_t) (idx4 / 2)];
            const uint8_t q = (idx4 % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
            ++idx4;
            v = ((float) q - zp_int4) * scale_int4;
        } else if (tier == 1) {
            const uint8_t q = stream8[(size_t) idx8];
            ++idx8;
            v = ((float) q - zp_int8) * scale_int8;
        }

        out_w[(size_t) pos] = v;
    }

    for (int64_t i = 0; i < n_fp16; ++i) {
        const int32_t pos = fp16_indices[i];
        if (pos >= 0 && (int64_t) pos < total) {
            out_w[(size_t) pos] = ggml_fp16_to_fp32(fp16_values[i]);
        }
    }
}

// CUDA kernel: one thread per row, sequential scan across columns
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
    __half * out_half) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    int idx2 = idx2_start[r];
    int idx4 = idx4_start[r];
    int idx8 = idx8_start[r];

    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    int base_pos = r * cols;
    for (int c = 0; c < cols; ++c) {
        int pos = base_pos + c;
        uint8_t mbyte = packed_mask[pos / 4];
        uint8_t tier = (mbyte >> ((pos % 4) * 2)) & 0x3;
        float v = 0.0f;
        if (tier >= 3) {
            uint8_t b = stream2[idx2 / 4];
            uint8_t q = (b >> ((idx2 % 4) * 2)) & 0x3;
            ++idx2;
            v = ((float) q - zp_low) * scale_low;
        } else if (tier == 2) {
            uint8_t b = stream4[idx4 / 2];
            uint8_t q = (idx4 % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
            ++idx4;
            v = ((float) q - zp_int4) * scale_int4;
        } else if (tier == 1) {
            uint8_t q = stream8[idx8];
            ++idx8;
            v = ((float) q - zp_int8) * scale_int8;
        } else {
            // tier 0 (fp16 outliers) leave as 0, will be overwritten by fp16 kernel
            v = 0.0f;
        }
        out_half[pos] = __float2half(v);
    }
}

// Kernel to apply fp16 outliers (host-provided floats)
__global__ void apply_fp16_overrides(const int32_t * fp16_idx, const float * fp16_vals, int64_t n_fp16, __half * out_half) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_fp16) return;
    int pos = fp16_idx[i];
    out_half[pos] = __float2half(fp16_vals[i]);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: bmo_unpack_kernel_test <weights.gguf>\n";
        return 1;
    }
    const char * fname = argv[1];

    bmo_model model;
    bmo_context ctx;

    try {
        bmo_load_model(fname, model, ctx);
    } catch (const std::exception & ex) {
        std::cerr << "Failed to load model: " << ex.what() << std::endl;
        return 2;
    }

    const std::string base = "transformer_layers_0_gating_linear_in_weight";

    ggml_tensor * pw_t = ggml_get_tensor(model.wctx, (base + ".packed_weights").c_str());
    ggml_tensor * pm_t = ggml_get_tensor(model.wctx, (base + ".packed_mask").c_str());
    ggml_tensor * fi_t = ggml_get_tensor(model.wctx, (base + ".fp16_indices").c_str());
    ggml_tensor * fv_t = ggml_get_tensor(model.wctx, (base + ".fp16_values").c_str());

    if (!pw_t || !pm_t) {
        std::cerr << "Missing packed tensors for base: " << base << std::endl;
        return 3;
    }

    const uint8_t * pw = reinterpret_cast<const uint8_t *>(pw_t->data);
    const uint8_t * pm = reinterpret_cast<const uint8_t *>(pm_t->data);

    const int32_t rows = read_scalar_i32(model.wctx, base + ".rows", 0);
    const int32_t cols = read_scalar_i32(model.wctx, base + ".cols", 0);
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Invalid rows/cols metadata for " << base << " rows=" << rows << " cols=" << cols << std::endl;
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

    int64_t n_fp16 = 0;
    const int32_t * fi = nullptr;
    std::vector<float> fp16_vals_float;
    if (fi_t && fv_t) {
        fi = reinterpret_cast<const int32_t *>(fi_t->data);
        n_fp16 = ggml_nbytes(fi_t) / sizeof(int32_t);
        // convert fp16 values to float on host
        if (fv_t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * f16 = reinterpret_cast<const ggml_fp16_t *>(fv_t->data);
            fp16_vals_float.resize((size_t) n_fp16);
            for (int64_t i = 0; i < n_fp16; ++i) fp16_vals_float[(size_t) i] = ggml_fp16_to_fp32(f16[i]);
        } else if (fv_t->type == GGML_TYPE_F32) {
            const float * f32 = reinterpret_cast<const float *>(fv_t->data);
            fp16_vals_float.assign(f32, f32 + n_fp16);
        } else {
            std::cerr << "Unsupported fp16_values type for " << base << std::endl;
            return 5;
        }
    }

    const int64_t total = (int64_t) rows * (int64_t) cols;
    std::vector<float> cpu_out((size_t) total, 0.0f);

    // Run CPU baseline
    unpack_layer_to_f32_cpu(pw, pm, rows, cols, n_2bit_bytes, n_4bit_bytes, n_8bit_bytes,
                            scale_low, scale_int4, scale_int8, zp_low, zp_int4, zp_int8,
                            fi, n_fp16, reinterpret_cast<const ggml_fp16_t *>(fv_t ? fv_t->data : nullptr), cpu_out.data());

    // Precompute per-row starts
    std::vector<int32_t> cnt2(rows), cnt4(rows), cnt8(rows);
    for (int r = 0; r < rows; ++r) {
        int64_t base_pos = (int64_t) r * cols;
        int c2 = 0, c4 = 0, c8 = 0;
        for (int c = 0; c < cols; ++c) {
            int64_t pos = base_pos + c;
            uint8_t mbyte = pm[pos / 4];
            uint8_t tier = (mbyte >> ((pos % 4) * 2)) & 0x3;
            if (tier >= 3) ++c2;
            else if (tier == 2) ++c4;
            else if (tier == 1) ++c8;
        }
        cnt2[r] = c2;
        cnt4[r] = c4;
        cnt8[r] = c8;
    }

    std::vector<int32_t> idx2_start(rows), idx4_start(rows), idx8_start(rows);
    int32_t acc2 = 0, acc4 = 0, acc8 = 0;
    for (int r = 0; r < rows; ++r) {
        idx2_start[r] = acc2;
        idx4_start[r] = acc4;
        idx8_start[r] = acc8;
        acc2 += cnt2[r];
        acc4 += cnt4[r];
        acc8 += cnt8[r];
    }

    // Allocate device buffers
    uint8_t * d_pw = nullptr;
    uint8_t * d_pm = nullptr;
    __half * d_out = nullptr;
    int32_t * d_idx2 = nullptr;
    int32_t * d_idx4 = nullptr;
    int32_t * d_idx8 = nullptr;
    int32_t * d_fp16_idx = nullptr;
    float * d_fp16_vals = nullptr;

    const size_t pw_bytes = (size_t) n_2bit_bytes + (size_t) n_4bit_bytes + (size_t) n_8bit_bytes;
    cudaError_t err;
    err = cudaMalloc(&d_pw, pw_bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_pw failed: " << cudaGetErrorString(err) << std::endl; return 10; }
    err = cudaMalloc(&d_pm, (size_t)((total + 3) / 4));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_pm failed: " << cudaGetErrorString(err) << std::endl; return 11; }
    err = cudaMalloc(&d_out, (size_t) total * sizeof(__half));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_out failed: " << cudaGetErrorString(err) << std::endl; return 12; }
    err = cudaMalloc(&d_idx2, (size_t) rows * sizeof(int32_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_idx2 failed: " << cudaGetErrorString(err) << std::endl; return 13; }
    err = cudaMalloc(&d_idx4, (size_t) rows * sizeof(int32_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_idx4 failed: " << cudaGetErrorString(err) << std::endl; return 14; }
    err = cudaMalloc(&d_idx8, (size_t) rows * sizeof(int32_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_idx8 failed: " << cudaGetErrorString(err) << std::endl; return 15; }

    // copy data
    err = cudaMemcpy(d_pw, pw, pw_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy pw failed: " << cudaGetErrorString(err) << std::endl; return 16; }
    err = cudaMemcpy(d_pm, pm, (size_t)((total + 3) / 4), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy pm failed: " << cudaGetErrorString(err) << std::endl; return 17; }
    err = cudaMemcpy(d_idx2, idx2_start.data(), (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy idx2 failed: " << cudaGetErrorString(err) << std::endl; return 18; }
    err = cudaMemcpy(d_idx4, idx4_start.data(), (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy idx4 failed: " << cudaGetErrorString(err) << std::endl; return 19; }
    err = cudaMemcpy(d_idx8, idx8_start.data(), (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy idx8 failed: " << cudaGetErrorString(err) << std::endl; return 20; }

    if (n_fp16 > 0) {
        err = cudaMalloc(&d_fp16_idx, (size_t) n_fp16 * sizeof(int32_t));
        if (err != cudaSuccess) { std::cerr << "cudaMalloc d_fp16_idx failed: " << cudaGetErrorString(err) << std::endl; return 21; }
        err = cudaMalloc(&d_fp16_vals, (size_t) n_fp16 * sizeof(float));
        if (err != cudaSuccess) { std::cerr << "cudaMalloc d_fp16_vals failed: " << cudaGetErrorString(err) << std::endl; return 22; }
        err = cudaMemcpy(d_fp16_idx, fi, (size_t) n_fp16 * sizeof(int32_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "cudaMemcpy fi failed: " << cudaGetErrorString(err) << std::endl; return 23; }
        err = cudaMemcpy(d_fp16_vals, fp16_vals_float.data(), (size_t) n_fp16 * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "cudaMemcpy fp16 vals failed: " << cudaGetErrorString(err) << std::endl; return 24; }
    }

    // Launch kernel
    const int threads = 128;
    const int blocks = (rows + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    unpack_kernel<<<blocks, threads>>>(d_pw, d_pm, rows, cols, n_2bit_bytes, n_4bit_bytes, n_8bit_bytes,
                                       scale_low, scale_int4, scale_int8, zp_low, zp_int4, zp_int8,
                                       d_idx2, d_idx4, d_idx8, d_out);
    if (n_fp16 > 0) {
        const int t2 = 128;
        const int b2 = (int)((n_fp16 + t2 - 1) / t2);
        apply_fp16_overrides<<<b2, t2>>>(d_fp16_idx, d_fp16_vals, n_fp16, d_out);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Copy back
    std::vector<__half> host_half((size_t) total);
    err = cudaMemcpy(host_half.data(), d_out, (size_t) total * sizeof(__half), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy out failed: " << cudaGetErrorString(err) << std::endl; return 25; }

    // Convert to float
    std::vector<float> gpu_out((size_t) total);
    for (int64_t i = 0; i < total; ++i) gpu_out[(size_t) i] = __half2float(host_half[(size_t) i]);

    // Compare
    double max_abs_diff = 0.0;
    size_t mismatch_tier2 = 0, mismatch_tier4 = 0, mismatch_tier8 = 0, mismatch_fp16 = 0;
    std::vector<std::pair<int64_t, std::pair<float,float>>> mismatches;
    for (int64_t pos = 0; pos < total; ++pos) {
        uint8_t mbyte = pm[pos / 4];
        uint8_t tier = (mbyte >> ((pos % 4) * 2)) & 0x3;
        float a = cpu_out[(size_t) pos];
        float b = gpu_out[(size_t) pos];
        double diff = std::abs((double)a - (double)b);
        if (diff > max_abs_diff) max_abs_diff = diff;
        bool mismatch = false;
        if (tier >= 3) {
            // Tier 2-bit
            if (a != b) { mismatch_tier2++; mismatch = true; }
        } else if (tier == 2) {
            if (a != b) { mismatch_tier4++; mismatch = true; }
        } else if (tier == 1) {
            if (a != b) { mismatch_tier8++; mismatch = true; }
        } else {
            if (diff > 1e-6) { mismatch_fp16++; mismatch = true; }
        }
        if (mismatch && mismatches.size() < 5) mismatches.push_back({pos, {a,b}});
    }

    std::cout << "max_abs_diff=" << max_abs_diff << std::endl;
    std::cout << "mismatch_counts: tier2(2bit)=" << mismatch_tier2
              << " tier4(4bit)=" << mismatch_tier4
              << " tier8(8bit)=" << mismatch_tier8
              << " fp16_outliers=" << mismatch_fp16 << std::endl;
    std::cout << "kernel_ms=" << kernel_ms << " ms" << std::endl;
    if (!mismatches.empty()) {
        std::cout << "First mismatches (pos cpu gpu):\n";
        for (auto &m : mismatches) {
            std::cout << " pos=" << m.first << " cpu=" << m.second.first << " gpu=" << m.second.second << "\n";
        }
    }

    // cleanup
    cudaFree(d_pw);
    cudaFree(d_pm);
    cudaFree(d_out);
    cudaFree(d_idx2);
    cudaFree(d_idx4);
    cudaFree(d_idx8);
    if (d_fp16_idx) cudaFree(d_fp16_idx);
    if (d_fp16_vals) cudaFree(d_fp16_vals);

    return 0;
}
