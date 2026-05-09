// bmo_fused_mat_test.cu
// Phase 3 Stage 3: Fused dequant + matmul prototype
//
// Validates whether fusing dequant into the matmul kernel beats the current
// "unpack to FP32 scratch then ggml mul_mat" path on Jetson Orin Nano.
//
// Architecture mirrors production JIT streaming:
//   1. memcpy packed_weights, packed_mask, fp16_values into stream buffer
//   2. JIT compute block_offset table into stream buffer
//   3. Launch fused kernel: <<<rows, 256>>>
//      - One block per output row
//      - 256 threads cooperate on dot product
//      - Each thread strides through cols, dequants on-the-fly, accumulates
//      - Block-wide reduction via shared memory, thread 0 writes y[row]
//
// Pass criteria:
//   - max relative error vs CPU reference < 1e-3
//   - median fused E2E (memcpy + JIT + kernel) on gating_linear_in:
//       < 30 ms  GREEN  (>= 3.7x vs 110.5 ms baseline) -> commit to refactor
//       < 60 ms  YELLOW (>= 1.8x)                       -> proceed cautiously
//       > 100 ms RED                                    -> investigate
//
// Usage:
//   ./bmo_fused_mat_test <weights.gguf> [packed_base]

#include "bmo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t _err = (expr);                                          \
        if (_err != cudaSuccess) {                                          \
            std::cerr << "CUDA error " << cudaGetErrorString(_err)          \
                      << " at " << __FILE__ << ":" << __LINE__              \
                      << " (" << #expr << ")\n";                            \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// GGUF scalar readers (mirror existing test code)
// ---------------------------------------------------------------------------

static int32_t read_scalar_i32(ggml_context * ctx, const std::string & name, int32_t fallback = 0) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t || ggml_nbytes(t) < (int) sizeof(int32_t)) return fallback;
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context * ctx, const std::string & name, float fallback = 0.0f) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) return fallback;
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// CPU reference: dense unpack + naive matmul
// Only used for correctness validation; not on the timed path.
// ---------------------------------------------------------------------------

static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

static void cpu_reference_matvec(
    const uint8_t * pw, const uint8_t * pm, const ggml_fp16_t * fv,
    int32_t rows, int32_t cols, int32_t block_size,
    int32_t n_2bit_bytes, int32_t n_4bit_bytes, int32_t /*n_8bit_bytes*/,
    float scale_low, float scale_int4, float scale_int8,
    float zp_low,    float zp_int4,    float zp_int8,
    const float * x, float * y_out)
{
    const int64_t total = (int64_t) rows * (int64_t) cols;
    const int64_t n_blocks = (total + block_size - 1) / block_size;

    const uint8_t * stream2 = pw;
    const uint8_t * stream4 = pw + n_2bit_bytes;
    const uint8_t * stream8 = pw + n_2bit_bytes + n_4bit_bytes;

    // Walk all blocks once, sequentially, to mirror tier-stream packing.
    int32_t c2 = 0, c4 = 0, c8 = 0, c16 = 0;
    std::vector<float> dense((size_t) total);
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t mbyte = pm[b / 4];
        const uint8_t tier  = unpack_u2_le(mbyte, (int)(b % 4));
        const int64_t base  = b * block_size;
        for (int32_t k = 0; k < block_size && base + k < total; ++k) {
            float v = 0.0f;
            if (tier == 0) {
                v = ggml_fp16_to_fp32(fv[c16 + k]);
            } else if (tier == 1) {
                uint8_t q = stream8[c8 + k];
                v = ((float) q - zp_int8) * scale_int8;
            } else if (tier == 2) {
                int idx = c4 + k;
                uint8_t bb = stream4[idx / 2];
                uint8_t q  = (idx % 2 == 0) ? (bb & 0x0F) : ((bb >> 4) & 0x0F);
                v = ((float) q - zp_int4) * scale_int4;
            } else {
                int idx = c2 + k;
                uint8_t bb = stream2[idx / 4];
                uint8_t q  = unpack_u2_le(bb, idx % 4);
                v = ((float) q - zp_low) * scale_low;
            }
            dense[(size_t)(base + k)] = v;
        }
        if      (tier == 0) c16 += block_size;
        else if (tier == 1) c8  += block_size;
        else if (tier == 2) c4  += block_size;
        else                c2  += block_size;
    }

    // Matvec
    for (int32_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const float * row = dense.data() + (size_t) r * cols;
        for (int32_t c = 0; c < cols; ++c) acc += row[c] * x[c];
        y_out[r] = acc;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2]
                        : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1, (size_t) std::floor(p * (v.size() - 1)));
    return v[i];
}

} // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: bmo_fused_mat_test <weights.gguf> [packed_base]\n";
        return 1;
    }
    const std::string base = argc >= 3
        ? argv[2]
        : "transformer_layers_0_gating_linear_in_weight";

    // Load model
    bmo_model model;
    bmo_context ctx;
    try {
        bmo_load_model(argv[1], model, ctx);
    } catch (const std::exception & ex) {
        std::cerr << "Failed to load model: " << ex.what() << "\n";
        return 2;
    }

    // Get tensors
    ggml_tensor * pw_t = ggml_get_tensor(model.wctx, (base + ".packed_weights").c_str());
    ggml_tensor * pm_t = ggml_get_tensor(model.wctx, (base + ".packed_mask").c_str());
    ggml_tensor * fv_t = ggml_get_tensor(model.wctx, (base + ".fp16_values").c_str());
    if (!pw_t || !pm_t || !fv_t) {
        std::cerr << "Missing SEPTQ v3 tensors for base: " << base << "\n";
        return 3;
    }

    const int32_t rows         = read_scalar_i32(model.wctx, base + ".rows", 0);
    const int32_t cols         = read_scalar_i32(model.wctx, base + ".cols", 0);
    const int32_t block_size   = read_scalar_i32(model.wctx, base + ".block_size", 32);
    const int32_t n_2bit_bytes = read_scalar_i32(model.wctx, base + ".n_2bit_bytes", 0);
    const int32_t n_4bit_bytes = read_scalar_i32(model.wctx, base + ".n_4bit_bytes", 0);
    const int32_t n_8bit_bytes = read_scalar_i32(model.wctx, base + ".n_8bit_bytes", 0);
    const float scale_low  = read_scalar_f32(model.wctx, base + ".scale_low",  1.0f);
    const float scale_int4 = read_scalar_f32(model.wctx, base + ".scale_int4", 1.0f);
    const float scale_int8 = read_scalar_f32(model.wctx, base + ".scale_int8", 1.0f);
    const float zp_low     = read_scalar_f32(model.wctx, base + ".zp_low",     1.5f);
    const float zp_int4    = read_scalar_f32(model.wctx, base + ".zp_int4",    7.5f);
    const float zp_int8    = read_scalar_f32(model.wctx, base + ".zp_int8",  127.5f);

    if (rows <= 0 || cols <= 0 || block_size != 32) {
        std::cerr << "Bad dims: rows=" << rows << " cols=" << cols
                  << " block_size=" << block_size << " (must be 32)\n";
        return 4;
    }

    const int64_t total    = (int64_t) rows * cols;
    const int64_t n_blocks = (total + block_size - 1) / block_size;

    std::cout << "[fused_test] base=" << base
              << " rows=" << rows << " cols=" << cols
              << " n_blocks=" << n_blocks << "\n";
    std::cout << "[fused_test] n_2bit=" << n_2bit_bytes
              << " n_4bit=" << n_4bit_bytes
              << " n_8bit=" << n_8bit_bytes << "\n";

    // Host source pointers (mmap-backed; same as production)
    const uint8_t   * h_pw = reinterpret_cast<const uint8_t *>(pw_t->data);
    const uint8_t   * h_pm = reinterpret_cast<const uint8_t *>(pm_t->data);
    const ggml_fp16_t * h_fv = reinterpret_cast<const ggml_fp16_t *>(fv_t->data);
    const size_t pw_size = ggml_nbytes(pw_t);
    const size_t pm_size = ggml_nbytes(pm_t);
    const size_t fv_size = ggml_nbytes(fv_t);

    // Stream buffer layout (same shape as production):
    //   [pw][pm][fv][pad-to-4][block_offset]
    const size_t bo_size = (size_t) n_blocks * sizeof(int32_t);
    size_t pre_bo_size = pw_size + pm_size + fv_size;
    pre_bo_size = (pre_bo_size + 3) & ~size_t(3);
    const size_t stream_bytes = pre_bo_size + bo_size;
    std::cout << "[fused_test] stream buffer = " << stream_bytes / (1024.0 * 1024.0)
              << " MB\n";

    if (!ctx.cuda_packed_stream_buffer || !ctx.cuda_packed_stream_buffer_dev) {
        std::cerr << "[fused_test] ctx.cuda_packed_stream_buffer not allocated (need bmo_load_model CUDA path)\n";
        return 5;
    }
    if (stream_bytes > ctx.cuda_packed_stream_buffer_bytes) {
        std::cerr << "[fused_test] stream payload " << stream_bytes
                  << " bytes exceeds ctx.cuda_packed_stream_buffer_bytes "
                  << ctx.cuda_packed_stream_buffer_bytes << "\n";
        return 5;
    }
    // Local pinned slab for x/y (model no longer allocates cuda_unpack_scratch on Jetson).
    const size_t y_al = ((size_t) rows * sizeof(float) + 63) & ~size_t(63);
    const size_t x_al = ((size_t) cols * sizeof(float) + 63) & ~size_t(63);
    const size_t vec_pin = y_al + x_al;
    void * vec_raw = nullptr;
    if (posix_memalign(&vec_raw, 64, vec_pin) != 0 || !vec_raw) {
        std::cerr << "[fused_test] posix_memalign vec failed\n";
        return 6;
    }
    CUDA_CHECK(cudaHostRegister(vec_raw, vec_pin, cudaHostRegisterMapped | cudaHostRegisterPortable));
    void * vec_dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&vec_dev, vec_raw, 0));

    // --- BYPASS NVMAP: Reuse the existing OS-allocated ctx stream buffer ---
    uint8_t * sb_host = reinterpret_cast<uint8_t *>(ctx.cuda_packed_stream_buffer);
    uint8_t * sb_dev  = reinterpret_cast<uint8_t *>(ctx.cuda_packed_stream_buffer_dev);

    // Host pointers for memcpy and JIT
    uint8_t * d_pw_in_sb = sb_host;
    uint8_t * d_pm_in_sb = sb_host + pw_size;
    ggml_fp16_t * d_fv_in_sb = reinterpret_cast<ggml_fp16_t *>(sb_host + pw_size + pm_size);
    int32_t * d_bo = reinterpret_cast<int32_t *>(sb_host + pre_bo_size);

    // Device pointers for kernel execution
    uint8_t * dev_pw = sb_dev;
    uint8_t * dev_pm = sb_dev + pw_size;
    const __half * dev_fv = reinterpret_cast<const __half *>(sb_dev + pw_size + pm_size);

    float * h_y = reinterpret_cast<float *>(vec_raw);
    float * h_x =
        reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(vec_raw) + y_al);
    float * d_y = reinterpret_cast<float *>(vec_dev);
    float * d_x =
        reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(vec_dev) + y_al);

    // Fill x with deterministic random data via host pointer
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int c = 0; c < cols; ++c) h_x[c] = dist(rng);

    // ---------- E2E iterations ----------
    const int N_ITERS = 12;        // discard first 2 as warmup
    const int N_WARMUP = 2;
    std::vector<double> e2e_ms;
    std::vector<double> kernel_ms_list;
    std::vector<double> memcpy_ms_list;
    std::vector<double> jit_ms_list;

    cudaEvent_t k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));

    for (int iter = 0; iter < N_ITERS; ++iter) {
        auto e2e_t0 = std::chrono::steady_clock::now();

        // 1. memcpy from mmap-backed host into stream buffer
        auto mcpy_t0 = std::chrono::steady_clock::now();
        std::memcpy(d_pw_in_sb, h_pw, pw_size);
        std::memcpy(d_pm_in_sb, h_pm, pm_size);
        std::memcpy(d_fv_in_sb, h_fv, fv_size);
        auto mcpy_t1 = std::chrono::steady_clock::now();

        // 2. JIT compute block_offset
        auto jit_t0 = std::chrono::steady_clock::now();
        {
            int32_t c2 = 0, c4 = 0, c8 = 0, c16 = 0;
            for (int b = 0; b < (int) n_blocks; ++b) {
                const uint8_t mbyte = d_pm_in_sb[b >> 2];
                const uint8_t tier  = (mbyte >> ((b & 3) * 2)) & 0x3;
                if (tier == 0)      { d_bo[b] = c16; c16 += block_size; }
                else if (tier == 1) { d_bo[b] = c8;  c8  += block_size; }
                else if (tier == 2) { d_bo[b] = c4;  c4  += block_size; }
                else                { d_bo[b] = c2;  c2  += block_size; }
            }
        }
        auto jit_t1 = std::chrono::steady_clock::now();

        // 3. Launch fused kernel (implemented in bmo_cuda_kernels.cu)
        CUDA_CHECK(cudaEventRecord(k_start));
        launch_fused_dequant_matvec(
            dev_pw,
            dev_pm,
            dev_fv,
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
            d_x,
            d_y);
        CUDA_CHECK(cudaEventRecord(k_stop));
        CUDA_CHECK(cudaEventSynchronize(k_stop));

        auto e2e_t1 = std::chrono::steady_clock::now();

        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            std::cerr << "Kernel failed at iter " << iter << ": "
                      << cudaGetErrorString(kerr) << "\n";
            cudaHostUnregister(vec_raw);
            std::free(vec_raw);
            cudaEventDestroy(k_start);
            cudaEventDestroy(k_stop);
            return 7;
        }

        float k_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&k_ms, k_start, k_stop));

        const double mcpy_ms = std::chrono::duration<double, std::milli>(mcpy_t1 - mcpy_t0).count();
        const double jit_ms  = std::chrono::duration<double, std::milli>(jit_t1  - jit_t0 ).count();
        const double e2e_ms_v = std::chrono::duration<double, std::milli>(e2e_t1 - e2e_t0).count();

        if (iter >= N_WARMUP) {
            e2e_ms.push_back(e2e_ms_v);
            kernel_ms_list.push_back((double) k_ms);
            memcpy_ms_list.push_back(mcpy_ms);
            jit_ms_list.push_back(jit_ms);
        }
        std::cout << "[fused_test] iter=" << iter
                  << " mcpy=" << mcpy_ms << "ms"
                  << " jit="  << jit_ms  << "ms"
                  << " kern=" << k_ms    << "ms"
                  << " e2e="  << e2e_ms_v << "ms"
                  << (iter < N_WARMUP ? " (warmup)" : "") << "\n";
    }

    // ---------- Correctness vs CPU reference ----------
    std::cout << "[fused_test] computing CPU reference (slow)...\n";
    std::vector<float> y_ref((size_t) rows);
    cpu_reference_matvec(
        h_pw, h_pm, h_fv,
        rows, cols, block_size,
        n_2bit_bytes, n_4bit_bytes, n_8bit_bytes,
        scale_low, scale_int4, scale_int8,
        zp_low,    zp_int4,    zp_int8,
        h_x, y_ref.data());

    double max_abs = 0.0, max_rel = 0.0, ref_norm = 0.0, err_norm = 0.0;
    for (int r = 0; r < rows; ++r) {
        const double a = (double) y_ref[r];
        const double b = (double) h_y[r]; // Read from host pointer alias
        const double diff = std::abs(a - b);
        const double denom = std::max(1e-6, std::abs(a));
        max_abs   = std::max(max_abs, diff);
        max_rel   = std::max(max_rel, diff / denom);
        ref_norm += a * a;
        err_norm += diff * diff;
    }
    const double l2_rel = std::sqrt(err_norm) / std::max(1e-12, std::sqrt(ref_norm));

    // ---------- Report ----------
    const double k_med   = median(kernel_ms_list);
    const double e2e_med = median(e2e_ms);
    const double e2e_p95 = percentile(e2e_ms, 0.95);
    const double mcpy_med = median(memcpy_ms_list);
    const double jit_med  = median(jit_ms_list);

    std::cout << "\n========== FUSED DEQUANT + MATVEC RESULTS ==========\n";
    std::cout << "matrix:           " << base << " (" << rows << "x" << cols << ")\n";
    std::cout << "iters:            " << e2e_ms.size() << " (after " << N_WARMUP << " warmup)\n";
    std::cout << "kernel median:    " << k_med   << " ms\n";
    std::cout << "memcpy median:    " << mcpy_med << " ms (" << pre_bo_size / (1024.0*1024.0) << " MB)\n";
    std::cout << "jit_offset median:" << jit_med  << " ms\n";
    std::cout << "E2E median:       " << e2e_med << " ms\n";
    std::cout << "E2E p95:          " << e2e_p95 << " ms\n";
    std::cout << "max abs diff:     " << max_abs << "\n";
    std::cout << "max rel diff:     " << max_rel << "\n";
    std::cout << "L2 rel error:     " << l2_rel  << "\n";

    // Pass/fail gates
    bool correct = (l2_rel < 1e-3);
    std::cout << "\ncorrectness:      "
              << (correct ? "[PASS] (L2 rel < 1e-3)" : "[FAIL]") << "\n";

    const double baseline_ms = 110.5; // current unpack-only on gating_linear_in
    const double speedup = baseline_ms / e2e_med;
    std::cout << "vs 110.5ms unpack baseline: " << speedup << "x speedup\n";

    if (!correct) {
        std::cout << "verdict: RED — math broken, do not refactor\n";
    } else if (e2e_med < 30.0) {
        std::cout << "verdict: GREEN — commit to full fused refactor\n";
    } else if (e2e_med < 60.0) {
        std::cout << "verdict: YELLOW — proceed cautiously, profile kernel for next opt\n";
    } else if (e2e_med < 100.0) {
        std::cout << "verdict: ORANGE — speedup smaller than hoped; investigate before refactor\n";
    } else {
        std::cout << "verdict: RED — fused not faster; bottleneck is elsewhere\n";
    }
    std::cout << "====================================================\n";

    cudaHostUnregister(vec_raw);
    std::free(vec_raw);

    cudaEventDestroy(k_start);
    cudaEventDestroy(k_stop);
    return correct ? 0 : 9;
}