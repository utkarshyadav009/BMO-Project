// Register-pressure + numerical gate for Path B per-element tier prototype vs production v2.
//
// Usage:
//   ./bmo_proto_test <weights.gguf> [packed_base] [rows_per_block]
//
// Example:
//   ./bmo_proto_test model.gguf transformer_layers_0_self_attn_out_proj_weight 8

#include "bmo.h"
#include "bmo_proto_kernels.h"

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

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << cudaGetErrorString(_err) << " at " << __FILE__ \
                      << ":" << __LINE__ << " (" << #expr << ")\n";                     \
            std::exit(2);                                                                \
        }                                                                                \
    } while (0)

namespace {

static int32_t read_scalar_i32(ggml_context *ctx, const std::string &name, int32_t fallback = 0) {
    ggml_tensor *t = ggml_get_tensor(ctx, name.c_str());
    if (!t || ggml_nbytes(t) < (int) sizeof(int32_t)) {
        return fallback;
    }
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context *ctx, const std::string &name, float fallback = 0.0f) {
    ggml_tensor *t = ggml_get_tensor(ctx, name.c_str());
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) {
        return fallback;
    }
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

// Expand per-block mask to per-element 2-bit packed buffer (4 elements / byte).
static void expand_block_mask_to_element_mask(
    const uint8_t *pm_block,
    int32_t rows,
    int32_t cols,
    int32_t block_size,
    std::vector<uint8_t> *out_elem) {
    const int64_t total = (int64_t) rows * cols;
    out_elem->assign((size_t)((total + 3) / 4), 0);
    for (int64_t pos = 0; pos < total; ++pos) {
        const int64_t bidx = pos / (int64_t) block_size;
        const uint8_t mbyte = pm_block[(size_t)(bidx / 4)];
        const uint8_t tier = unpack_u2_le(mbyte, (int) (bidx % 4));
        const size_t byte_i = (size_t)(pos / 4);
        const int shift = (int) ((pos & 3) * 2);
        (*out_elem)[byte_i] = (uint8_t)(((*out_elem)[byte_i] & (uint8_t) ~(0x3 << shift)) | (tier << shift));
    }
}

static void jit_row_tier_bases(
    const uint8_t *pm_row_major_block_mask,
    int32_t rows,
    int32_t cols,
    int32_t block_size,
    std::vector<int32_t> *row_c2,
    std::vector<int32_t> *row_c4,
    std::vector<int32_t> *row_c8,
    std::vector<int32_t> *row_c16) {
    const int64_t total = (int64_t) rows * cols;
    const int64_t n_blocks = (total + block_size - 1) / block_size;
    const int32_t blocks_per_row = cols / block_size;

    row_c2->assign((size_t) rows, 0);
    row_c4->assign((size_t) rows, 0);
    row_c8->assign((size_t) rows, 0);
    row_c16->assign((size_t) rows, 0);

    int32_t c2 = 0, c4 = 0, c8 = 0, c16 = 0;
    for (int64_t b = 0; b < n_blocks; ++b) {
        if (blocks_per_row > 0 && (b % blocks_per_row) == 0) {
            const int r = (int) (b / blocks_per_row);
            if (r >= 0 && r < rows) {
                (*row_c2)[(size_t) r] = c2;
                (*row_c4)[(size_t) r] = c4;
                (*row_c8)[(size_t) r] = c8;
            }
        }
        const uint8_t mbyte = pm_row_major_block_mask[(size_t)(b / 4)];
        const uint8_t tier = unpack_u2_le(mbyte, (int) (b % 4));
        if (tier == 0) {
            c16 += block_size;
        } else if (tier == 1) {
            c8 += block_size;
        } else if (tier == 2) {
            c4 += block_size;
        } else {
            c2 += block_size;
        }
    }

    for (int32_t r = 0; r < rows; ++r) {
        const int32_t rb2 = (*row_c2)[(size_t) r];
        const int32_t rb4 = (*row_c4)[(size_t) r];
        const int32_t rb8 = (*row_c8)[(size_t) r];
        const int32_t blocks_before = r * blocks_per_row;
        (*row_c16)[(size_t) r] =
            (blocks_before - rb2 / block_size - rb4 / block_size - rb8 / block_size) * block_size;
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <weights.gguf> [packed_base] [rows_per_block]\n"
                  << "  packed_base default: transformer_layers_0_self_attn_out_proj_weight\n"
                  << "  rows_per_block: 8 (default) or 4\n";
        return 1;
    }

    const std::string base = argc >= 3 ? argv[2] : "transformer_layers_0_self_attn_out_proj_weight";
    const int rows_per_block = (argc >= 4) ? std::atoi(argv[3]) : 8;
    if (rows_per_block != 8 && rows_per_block != 4) {
        std::cerr << "rows_per_block must be 4 or 8\n";
        return 1;
    }

    bmo_model model;
    bmo_context ctx;
    try {
        bmo_load_model(argv[1], model, ctx);
    } catch (const std::exception &ex) {
        std::cerr << "Failed to load model: " << ex.what() << "\n";
        return 2;
    }

    ggml_tensor *pw_t = ggml_get_tensor(model.wctx, (base + ".packed_weights").c_str());
    ggml_tensor *pm_t = ggml_get_tensor(model.wctx, (base + ".packed_mask").c_str());
    ggml_tensor *fv_t = ggml_get_tensor(model.wctx, (base + ".fp16_values").c_str());
    if (!pw_t || !pm_t || !fv_t) {
        std::cerr << "Missing SEPTQ tensors for base: " << base << "\n";
        return 3;
    }

    const int32_t rows = read_scalar_i32(model.wctx, base + ".rows", 0);
    const int32_t cols = read_scalar_i32(model.wctx, base + ".cols", 0);
    const int32_t block_size = read_scalar_i32(model.wctx, base + ".block_size", 32);
    const int32_t n_2bit_bytes = read_scalar_i32(model.wctx, base + ".n_2bit_bytes", 0);
    const int32_t n_4bit_bytes = read_scalar_i32(model.wctx, base + ".n_4bit_bytes", 0);
    const int32_t n_8bit_bytes = read_scalar_i32(model.wctx, base + ".n_8bit_bytes", 0);
    (void) n_8bit_bytes;

    const float scale_low = read_scalar_f32(model.wctx, base + ".scale_low", 1.0f);
    const float scale_int4 = read_scalar_f32(model.wctx, base + ".scale_int4", 1.0f);
    const float scale_int8 = read_scalar_f32(model.wctx, base + ".scale_int8", 1.0f);
    const float zp_low = read_scalar_f32(model.wctx, base + ".zp_low", 1.5f);
    const float zp_int4 = read_scalar_f32(model.wctx, base + ".zp_int4", 7.5f);
    const float zp_int8 = read_scalar_f32(model.wctx, base + ".zp_int8", 127.5f);

    const int64_t total = (int64_t) rows * cols;
    const int64_t n_blocks = (total + block_size - 1) / block_size;
    const int32_t blocks_per_row = cols / block_size;

    if (rows <= 0 || cols <= 0 || block_size != 32 || blocks_per_row <= 0 ||
        (int64_t) rows * (int64_t) blocks_per_row != n_blocks) {
        std::cerr << "Bad geometry rows=" << rows << " cols=" << cols << " block_size=" << block_size
                  << " blocks_per_row=" << blocks_per_row << " n_blocks=" << n_blocks << "\n";
        return 4;
    }

    std::cout << "[proto_test] base=" << base << " rows=" << rows << " cols=" << cols
              << " rows_per_block=" << rows_per_block << "\n";

    const uint8_t *h_pm = reinterpret_cast<const uint8_t *>(pm_t->data);
    std::vector<uint8_t> pm_elem;
    expand_block_mask_to_element_mask(h_pm, rows, cols, block_size, &pm_elem);

    std::vector<int32_t> row_c2, row_c4, row_c8, row_c16;
    jit_row_tier_bases(h_pm, rows, cols, block_size, &row_c2, &row_c4, &row_c8, &row_c16);

    const size_t pw_bytes = ggml_nbytes(pw_t);
    const size_t fv_bytes = ggml_nbytes(fv_t);
    const size_t pm_elem_bytes = pm_elem.size();
    const size_t row_tbl = (size_t) rows * sizeof(int32_t);
    const size_t x_bytes = (size_t) cols * sizeof(float);
    const size_t y_bytes = (size_t) rows * sizeof(float);

    uint8_t *d_pw = nullptr;
    uint8_t *d_pm_elem = nullptr;
    uint8_t *d_pm_block = nullptr;
    __half *d_fv = nullptr;
    int32_t *d_rc2 = nullptr;
    int32_t *d_rc4 = nullptr;
    int32_t *d_rc8 = nullptr;
    int32_t *d_rc16 = nullptr;
    float *d_x = nullptr;
    float *d_y_v2 = nullptr;
    float *d_y_proto = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pw, pw_bytes));
    CUDA_CHECK(cudaMalloc(&d_pm_elem, pm_elem_bytes));
    CUDA_CHECK(cudaMalloc(&d_pm_block, ggml_nbytes(pm_t)));
    CUDA_CHECK(cudaMalloc(&d_fv, fv_bytes));
    CUDA_CHECK(cudaMalloc(&d_rc2, row_tbl));
    CUDA_CHECK(cudaMalloc(&d_rc4, row_tbl));
    CUDA_CHECK(cudaMalloc(&d_rc8, row_tbl));
    CUDA_CHECK(cudaMalloc(&d_rc16, row_tbl));
    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_y_v2, y_bytes));
    CUDA_CHECK(cudaMalloc(&d_y_proto, y_bytes));

    CUDA_CHECK(cudaMemcpy(d_pw, pw_t->data, pw_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pm_elem, pm_elem.data(), pm_elem_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pm_block, pm_t->data, ggml_nbytes(pm_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fv, fv_t->data, fv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rc2, row_c2.data(), row_tbl, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rc4, row_c4.data(), row_tbl, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rc8, row_c8.data(), row_tbl, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rc16, row_c16.data(), row_tbl, cudaMemcpyHostToDevice));

    std::vector<float> h_x((size_t) cols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int c = 0; c < cols; ++c) {
        h_x[(size_t) c] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));

    launch_fused_dequant_matvec(
        d_pw,
        d_pm_block,
        d_fv,
        d_rc2,
        d_rc4,
        d_rc8,
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
        d_y_v2,
        nullptr);

    launch_fused_dequant_matvec_proto(
        d_pw,
        d_pm_elem,
        d_fv,
        d_rc2,
        d_rc4,
        d_rc8,
        d_rc16,
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
        d_y_proto,
        rows_per_block,
        nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> y_v2((size_t) rows);
    std::vector<float> y_proto((size_t) rows);
    CUDA_CHECK(cudaMemcpy(y_v2.data(), d_y_v2, y_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_proto.data(), d_y_proto, y_bytes, cudaMemcpyDeviceToHost));

    bool bit_exact = false;
    bool compared = false;
    int first_mismatch = -1;
    if (rows_per_block == 8) {
        compared = true;
        bit_exact = true;
        for (int r = 0; r < rows; ++r) {
            if (y_v2[(size_t) r] != y_proto[(size_t) r]) {
                bit_exact = false;
                first_mismatch = r;
                break;
            }
        }
    }

    std::cout << "\n========== BMO PROTO (Path B register de-risk) ==========\n";
    if (!compared) {
        std::cout << "bit_exact vs fused_dequant_matvec_v2: N/A (production launch is ROWS_PER_BLOCK=8 only; "
                     "re-run with rows_per_block 8 to compare)\n";
    } else {
        std::cout << "bit_exact (float == vs v2): " << (bit_exact ? "YES" : "NO") << "\n";
        if (!bit_exact && first_mismatch >= 0) {
            std::cout << "first mismatch row " << first_mismatch << " v2=" << y_v2[(size_t) first_mismatch]
                      << " proto=" << y_proto[(size_t) first_mismatch] << "\n";
        }
    }
    std::cout << "Next: cuobjdump --dump-resource-usage on libbmo_proto.so (Linux) or bmo_proto.dll "
                 "(Windows) for register/thread at sm_87.\n";
    std::cout << "===========================================================\n";

    cudaFree(d_pw);
    cudaFree(d_pm_elem);
    cudaFree(d_pm_block);
    cudaFree(d_fv);
    cudaFree(d_rc2);
    cudaFree(d_rc4);
    cudaFree(d_rc8);
    cudaFree(d_rc16);
    cudaFree(d_x);
    cudaFree(d_y_v2);
    cudaFree(d_y_proto);

    if (!compared) {
        return 0;
    }
    return bit_exact ? 0 : 9;
}
