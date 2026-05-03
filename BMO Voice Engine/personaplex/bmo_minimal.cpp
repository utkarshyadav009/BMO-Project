#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
}

// Small helper: throw with context if required tensor is missing.
static ggml_tensor * require_tensor(ggml_context * data_ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name.c_str());
    if (!t) {
        throw std::runtime_error("Missing tensor in GGUF: " + name);
    }
    return t;
}

static float read_scalar_f32(ggml_context * data_ctx, const std::string & name, float default_value = 0.0f) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name.c_str());
    if (!t) {
        return default_value;
    }
    if (ggml_nbytes(t) < sizeof(float)) {
        throw std::runtime_error("Tensor too small for scalar f32: " + name);
    }

    // Exporter stores scales as f32 scalars; be permissive and read first float-sized bytes.
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

static int32_t read_scalar_i32(ggml_context * data_ctx, const std::string & name) {
    ggml_tensor * t = require_tensor(data_ctx, name);
    if (ggml_nbytes(t) < sizeof(int32_t)) {
        throw std::runtime_error("Tensor too small for scalar i32: " + name);
    }
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static std::vector<uint8_t> tensor_as_u8(ggml_tensor * t) {
    const size_t n = ggml_nbytes(t);
    std::vector<uint8_t> out(n);
    std::memcpy(out.data(), t->data, n);
    return out;
}

static std::vector<int32_t> tensor_as_i32(ggml_tensor * t) {
    const size_t n = ggml_nbytes(t) / sizeof(int32_t);
    std::vector<int32_t> out(n);
    std::memcpy(out.data(), t->data, n * sizeof(int32_t));
    return out;
}

static std::vector<float> tensor_as_f16_or_f32_to_f32(ggml_tensor * t) {
    const int64_t n_elem = ggml_nelements(t);
    std::vector<float> out((size_t) n_elem);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t) n_elem * sizeof(float));
        return out;
    }

    if (t->type == GGML_TYPE_F16) {
        // ggml_fp16_t is available in ggml headers.
        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(t->data);
        for (int64_t i = 0; i < n_elem; ++i) {
            out[(size_t) i] = ggml_fp16_to_fp32(src[i]);
        }
        return out;
    }

    throw std::runtime_error("Unsupported fp16_values tensor type. Expected F16/F32.");
}

// Little-endian 2-bit extractor for value stream packed 4 values per byte:
// value[0] in bits 0..1, value[1] in bits 2..3, value[2] in bits 4..5, value[3] in bits 6..7.
static inline uint8_t unpack_u2_le(const uint8_t byte, const int lane /*0..3*/) {
    return (byte >> (lane * 2)) & 0x3;
}

// Rebuild dense F32 weight [rows x cols] from packed blobs.
static void unpack_layer_to_f32(
    const std::vector<uint8_t> & packed_weights,
    const std::vector<uint8_t> & packed_mask,
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
    const std::vector<int32_t> & fp16_indices,
    const std::vector<float> & fp16_values,
    float * out_w // size rows*cols
) {
    const int64_t total = (int64_t) rows * (int64_t) cols;

    // Streams inside packed_weights: [2-bit bytes][4-bit bytes][8-bit bytes]
    const uint8_t * stream2 = packed_weights.data();
    const uint8_t * stream4 = packed_weights.data() + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights.data() + n_2bit_bytes + n_4bit_bytes;

    int64_t idx2 = 0; // number of consumed 2-bit values
    int64_t idx4 = 0; // number of consumed 4-bit values
    int64_t idx8 = 0; // number of consumed 8-bit values

    // First pass: fill quantized tiers according to packed tier mask.
    for (int64_t pos = 0; pos < total; ++pos) {
        // Tier mask unpack follows the same little-endian convention (4 uint2 per byte).
        const uint8_t mbyte = packed_mask[(size_t) (pos / 4)];
        const uint8_t tier  = unpack_u2_le(mbyte, (int) (pos % 4));

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
        } else {
            // Tier 0 values are exact and restored in second pass via fp16 indices/values.
            v = 0.0f;
        }

        out_w[(size_t) pos] = v;
    }

    // Second pass: overwrite tier-0 positions with exact fp16 values.
    if (fp16_indices.size() != fp16_values.size()) {
        throw std::runtime_error("fp16_indices and fp16_values size mismatch");
    }

    for (size_t i = 0; i < fp16_indices.size(); ++i) {
        const int32_t pos = fp16_indices[i];
        if (pos < 0 || (int64_t) pos >= total) {
            throw std::runtime_error("fp16 index out of range");
        }
        out_w[(size_t) pos] = fp16_values[i];
    }

    // Sanity-check stream usage against metadata counts.
    const int64_t used2_bytes = (idx2 + 3) / 4;
    const int64_t used4_bytes = (idx4 + 1) / 2;
    const int64_t used8_bytes = idx8;

    if (used2_bytes != n_2bit_bytes || used4_bytes != n_4bit_bytes || used8_bytes != n_8bit_bytes) {
        throw std::runtime_error("Packed stream usage mismatch. Check mask/packing metadata.");
    }
}

int main(int argc, char ** argv) {
    try {
        const std::string gguf_path = (argc >= 2) ? argv[1] : "bmo_weights.gguf";
        const std::string layer_base = (argc >= 3) ? argv[2] : "transformer_layers_0_gating_linear_in";

        // Load GGUF.
        ggml_context * data_ctx = nullptr;
        gguf_init_params params = {
            /*.no_alloc =*/ false,
            /*.ctx =*/ &data_ctx,
        };

        gguf_context * gctx = gguf_init_from_file(gguf_path.c_str(), params);
        if (!gctx) {
            throw std::runtime_error("Failed to load GGUF file: " + gguf_path);
        }
        if (!data_ctx) {
            throw std::runtime_error("GGUF loaded but data context is null");
        }

        // Retrieve required tensors for this single packed layer.
        auto * t_packed_weights = require_tensor(data_ctx, layer_base + ".packed_weights");
        auto * t_packed_mask    = require_tensor(data_ctx, layer_base + ".packed_mask");
        auto * t_fp16_idx       = require_tensor(data_ctx, layer_base + ".fp16_indices");
        auto * t_fp16_vals      = require_tensor(data_ctx, layer_base + ".fp16_values");

        const int32_t rows = read_scalar_i32(data_ctx, layer_base + ".rows");
        const int32_t cols = read_scalar_i32(data_ctx, layer_base + ".cols");

        const int32_t n_2bit_bytes = read_scalar_i32(data_ctx, layer_base + ".n_2bit_bytes");
        const int32_t n_4bit_bytes = read_scalar_i32(data_ctx, layer_base + ".n_4bit_bytes");
        const int32_t n_8bit_bytes = read_scalar_i32(data_ctx, layer_base + ".n_8bit_bytes");

        const float scale_low  = read_scalar_f32(data_ctx, layer_base + ".scale_low", 1.0f);
        const float scale_int4 = read_scalar_f32(data_ctx, layer_base + ".scale_int4", 1.0f);
        const float scale_int8 = read_scalar_f32(data_ctx, layer_base + ".scale_int8", 1.0f);

        // Zero points may be absent in current export format; default to 0.
        const float zp_low  = read_scalar_f32(data_ctx, layer_base + ".zp_low", 0.0f);
        const float zp_int4 = read_scalar_f32(data_ctx, layer_base + ".zp_int4", 0.0f);
        const float zp_int8 = read_scalar_f32(data_ctx, layer_base + ".zp_int8", 0.0f);

        std::vector<uint8_t> packed_weights = tensor_as_u8(t_packed_weights);
        std::vector<uint8_t> packed_mask    = tensor_as_u8(t_packed_mask);
        std::vector<int32_t> fp16_indices   = tensor_as_i32(t_fp16_idx);
        std::vector<float> fp16_values      = tensor_as_f16_or_f32_to_f32(t_fp16_vals);

        const int64_t total = (int64_t) rows * (int64_t) cols;
        std::vector<float> w_dense((size_t) total, 0.0f);

        // Naive correctness kernel: unpack packed streams into transient F32 matrix.
        unpack_layer_to_f32(
            packed_weights,
            packed_mask,
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
            fp16_indices,
            fp16_values,
            w_dense.data());

        // Build minimal ggml graph for y = W * x where x is all ones.
        // In ggml, a matrix with shape [rows, cols] is represented as ne0=cols, ne1=rows.
        const size_t graph_mem = (size_t) 64 * 1024 * 1024;
        std::vector<uint8_t> graph_buf(graph_mem);
        ggml_init_params gparams = {
            /*.mem_size   =*/ graph_mem,
            /*.mem_buffer =*/ graph_buf.data(),
            /*.no_alloc   =*/ false,
        };

        ggml_context * ctx = ggml_init(gparams);
        if (!ctx) {
            throw std::runtime_error("Failed to create ggml compute context");
        }

        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 1);

        std::memcpy(W->data, w_dense.data(), (size_t) total * sizeof(float));

        float * x_ptr = reinterpret_cast<float *>(x->data);
        for (int32_t i = 0; i < cols; ++i) {
            x_ptr[i] = 1.0f;
        }

        ggml_tensor * y = ggml_mul_mat(ctx, W, x);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        // CPU compute path.
        const int n_threads = 8;
        const ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_graph_compute_with_ctx failed");
        }

        // Print first 20 output values for Python/C++ visual comparison.
        float * y_ptr = reinterpret_cast<float *>(y->data);
        const int to_print = std::min<int>(20, rows);

        std::cout << "C++ output first " << to_print << " values:" << std::endl;
        for (int i = 0; i < to_print; ++i) {
            std::cout << y_ptr[i];
            if (i + 1 < to_print) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        ggml_free(ctx);
        gguf_free(gctx);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "[bmo_minimal] ERROR: " << ex.what() << std::endl;
        return 1;
    }
}
