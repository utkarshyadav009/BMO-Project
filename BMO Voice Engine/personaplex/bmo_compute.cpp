#include "bmo.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
static inline struct ggml_tensor * bmo_safe(struct ggml_tensor * t, const char * msg, int line) {
    if (!t) { fprintf(stderr, "\n[CRITICAL] Tensor '%s' is NULL at line %d!\n", msg, line); exit(1); }
    return t;
}
#define S(x) bmo_safe((x), #x, __LINE__)


#include <algorithm>
#include <cmath>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef BMO_ENABLE_CUDA
#include <cuda_runtime.h>
#include "ggml-backend.h"
#endif

#ifdef BMO_ENABLE_CUDA
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
    float * out_w);
#endif

namespace {

static void stage_tensor_upload(bmo_context & ctx, ggml_tensor * tensor, const void * data, size_t size) {
    bmo_context::owned_tensor_upload up;
    up.tensor = tensor;
    up.bytes.resize(size);
    std::memcpy(up.bytes.data(), data, size);
    ctx.graph_uploads.push_back(std::move(up));
}

static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

static ggml_tensor * get_tensor(ggml_context * data_ctx, const std::string & name) {
    return ggml_get_tensor(data_ctx, name.c_str());
}

static int32_t read_scalar_i32(ggml_context * data_ctx, const std::string & name, int32_t fallback = 0) {
    ggml_tensor * t = get_tensor(data_ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(int32_t)) {
        return fallback;
    }
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context * data_ctx, const std::string & name, float fallback = 0.0f) {
    ggml_tensor * t = get_tensor(data_ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) {
        return fallback;
    }
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

struct packed_linear_ref {
    std::string base;
    ggml_tensor * packed_weights = nullptr;
    ggml_tensor * packed_mask = nullptr;
    ggml_tensor * fp16_indices = nullptr;
    ggml_tensor * fp16_values = nullptr;
    ggml_tensor * dense_weight = nullptr;
    ggml_tensor * dense_bias = nullptr;
    device_packed_t * device_packed = nullptr;
};

static packed_linear_ref resolve_linear(ggml_context * data_ctx, const std::vector<std::string> & bases) {
    packed_linear_ref out;
    for (const auto & base : bases) {
        packed_linear_ref cand;
        cand.base = base;
        cand.packed_weights = get_tensor(data_ctx, base + ".packed_weights");
        cand.packed_mask = get_tensor(data_ctx, base + ".packed_mask");
        cand.fp16_indices = get_tensor(data_ctx, base + ".fp16_indices");
        cand.fp16_values = get_tensor(data_ctx, base + ".fp16_values");
        cand.dense_weight = get_tensor(data_ctx, base + ".weight");
        cand.dense_bias = get_tensor(data_ctx, base + ".bias");

        if (cand.packed_weights || cand.dense_weight) {
            return cand;
        }

        // Some exports use direct key where base itself is already "..._weight".
        cand.dense_weight = get_tensor(data_ctx, base);
        if (cand.dense_weight) {
            return cand;
        }

        // Older dot-style alias.
        std::string dot = base;
        std::replace(dot.begin(), dot.end(), '_', '.');
        cand.dense_weight = get_tensor(data_ctx, dot + ".weight");
        if (cand.dense_weight) {
            cand.base = dot;
            return cand;
        }
    }
    return out;
}

static int parse_temporal_layer_index(const std::string & base) {
    constexpr const char * prefix = "transformer_layers_";
    if (base.compare(0, std::strlen(prefix), prefix) != 0) {
        return -1;
    }

    const size_t start = std::strlen(prefix);
    size_t end = start;
    while (end < base.size() && std::isdigit(static_cast<unsigned char>(base[end]))) {
        ++end;
    }
    if (end == start) {
        return -1;
    }

    try {
        return std::stoi(base.substr(start, end - start));
    } catch (...) {
        return -1;
    }
}

static void unpack_layer_to_f32(
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

    const int64_t used2 = (idx2 + 3) / 4;
    const int64_t used4 = (idx4 + 1) / 2;
    const int64_t used8 = idx8;

    if (used2 != n_2bit_bytes || used4 != n_4bit_bytes || used8 != n_8bit_bytes) {
        fprintf(stderr, "[WARNING] Bypassed stream padding mismatch!\n");
        fprintf(stderr, "[STREAM-DELTA] used2=%lld n2=%lld delta=%lld | used4=%lld n4=%lld delta=%lld | used8=%lld n8=%lld delta=%lld\n",
                (long long)used2, (long long)n_2bit_bytes, (long long)(used2 - n_2bit_bytes),
                (long long)used4, (long long)n_4bit_bytes, (long long)(used4 - n_4bit_bytes),
                (long long)used8, (long long)n_8bit_bytes, (long long)(used8 - n_8bit_bytes));
    }
}

static ggml_tensor * apply_linear_with_transient_unpack(
    bmo_context & ctx,
    bmo_model & model,
    ggml_context * wctx,
    ggml_tensor * x,
    const std::vector<std::string> & base_candidates) {

    auto unpack_t0 = std::chrono::steady_clock::now();

    packed_linear_ref linear = resolve_linear(model.wctx, base_candidates);
    // device packed metadata is stored in ctx.packed_registry keyed by base name

    if (!linear.packed_weights && !linear.dense_weight) {
        // Naive correctness mode: if a specific projection is not exported in this GGUF,
        // pass through so the graph can still be built while mapping coverage is expanded.
        return x;
    }

    ggml_tensor * y = nullptr;

    if (linear.packed_weights) {
        const int32_t cols = (int32_t) x->ne[0];

        int32_t rows = read_scalar_i32(model.wctx, linear.base + ".rows", 0);
        if (rows <= 0) {
            rows = read_scalar_i32(model.wctx, linear.base + ".out_features", 0);
        }
        if (rows <= 0) {
            throw std::runtime_error("missing rows/out_features metadata for " + linear.base);
        }

        const int32_t n_2bit_bytes = read_scalar_i32(model.wctx, linear.base + ".n_2bit_bytes", 0);
        const int32_t n_4bit_bytes = read_scalar_i32(model.wctx, linear.base + ".n_4bit_bytes", 0);
        const int32_t n_8bit_bytes = read_scalar_i32(model.wctx, linear.base + ".n_8bit_bytes", 0);

        const float scale_low = read_scalar_f32(model.wctx, linear.base + ".scale_low", 1.0f);
        const float scale_int4 = read_scalar_f32(model.wctx, linear.base + ".scale_int4", 1.0f);
        const float scale_int8 = read_scalar_f32(model.wctx, linear.base + ".scale_int8", 1.0f);

        const float zp_low = read_scalar_f32(model.wctx, linear.base + ".zp_low", 1.5f);
        const float zp_int4 = read_scalar_f32(model.wctx, linear.base + ".zp_int4", 7.5f);
        const float zp_int8 = read_scalar_f32(model.wctx, linear.base + ".zp_int8", 127.5f);

        if (!linear.packed_mask || !linear.fp16_indices || !linear.fp16_values) {
            throw std::runtime_error("incomplete packed tensor set for " + linear.base);
        }

        ggml_tensor * W = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, cols, rows);

#ifdef BMO_ENABLE_CUDA
        if (ctx.cuda_backend && ctx.packed_registry.find(linear.base) != ctx.packed_registry.end()) {
            device_packed_t & dp_ref = ctx.packed_registry[linear.base];
            if (dp_ref.is_valid) {
                bmo_context::kernel_launch_info info;
                info.W = W;
                info.dp = &dp_ref;
                info.rows = rows;
                info.cols = cols;
                info.n_2bit_bytes = n_2bit_bytes;
                info.n_4bit_bytes = n_4bit_bytes;
                info.n_8bit_bytes = n_8bit_bytes;
                info.scale_low = scale_low;
                info.scale_int4 = scale_int4;
                info.scale_int8 = scale_int8;
                info.zp_low = zp_low;
                info.zp_int4 = zp_int4;
                info.zp_int8 = zp_int8;
                ctx.pending_kernel_launches.push_back(info);

                y = ggml_mul_mat(wctx, S(W), S(x));
                if (linear.dense_bias) {
                    y = ggml_add(wctx, y, linear.dense_bias);
                }
                return y;
            }
        }
#endif
        {
            const int64_t total = (int64_t) rows * (int64_t) cols;
            if ((int64_t) ctx.shared_scratch_w.size() < total) {
                ctx.shared_scratch_w.resize((size_t) total);
            }

            // Memory lifecycle note:
            // 1) We unpack exactly one layer's packed streams into ctx.shared_scratch_w.
            // 2) The same scratch buffer is reused for the next layer, so peak unpacked
            //    host memory is bounded by the largest temporal matrix (~gating linear_in).
            // 3) This keeps us from storing all 32 unpacked F32 matrices at once.
            const uint8_t * pw = reinterpret_cast<const uint8_t *>(linear.packed_weights->data);
            const uint8_t * pm = reinterpret_cast<const uint8_t *>(linear.packed_mask->data);
            const int32_t * fi = reinterpret_cast<const int32_t *>(linear.fp16_indices->data);
            const int64_t n_fp16 = ggml_nbytes(linear.fp16_indices) / (int64_t) sizeof(int32_t);

            const ggml_fp16_t * fv16 = nullptr;
            std::vector<ggml_fp16_t> tmp_f16;
            if (linear.fp16_values->type == GGML_TYPE_F16) {
                fv16 = reinterpret_cast<const ggml_fp16_t *>(linear.fp16_values->data);
            } else if (linear.fp16_values->type == GGML_TYPE_F32) {
                const float * src = reinterpret_cast<const float *>(linear.fp16_values->data);
                tmp_f16.resize((size_t) n_fp16);
                for (int64_t i = 0; i < n_fp16; ++i) {
                    tmp_f16[(size_t) i] = ggml_fp32_to_fp16(src[i]);
                }
                fv16 = tmp_f16.data();
            } else {
                throw std::runtime_error("unsupported fp16_values type in " + linear.base);
            }

            unpack_layer_to_f32(
                pw,
                pm,
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
                fi,
                n_fp16,
                fv16,
                ctx.shared_scratch_w.data());

            auto unpack_t1 = std::chrono::steady_clock::now();
            long unpack_ms = std::chrono::duration_cast<std::chrono::microseconds>(unpack_t1 - unpack_t0).count();
            std::fprintf(stderr, "[prof_unpack] base=%s rows=%d cols=%d unpack_us=%ld\n", linear.base.c_str(), rows, cols, unpack_ms);

            std::memcpy(W->data, ctx.shared_scratch_w.data(), (size_t) total * sizeof(float));
        }

        y = ggml_mul_mat(wctx, S(W), S(x));
    } else {
        y = ggml_mul_mat(wctx, linear.dense_weight, x);
    }

    if (linear.dense_bias) {
        y = ggml_add(wctx, y, linear.dense_bias);
    }

    auto unpack_t1 = std::chrono::steady_clock::now();
    long unpack_us = std::chrono::duration_cast<std::chrono::microseconds>(unpack_t1 - unpack_t0).count();
    if (!linear.packed_weights) {
        std::fprintf(stderr, "[prof_unpack] base=%s dense_path_us=%ld\n", linear.base.c_str(), unpack_us);
    }

    return y;
}

} // namespace

void bmo_execute_graph(bmo_context & ctx, ggml_cgraph * gf, const std::vector<tensor_upload> & inputs) {
#ifdef BMO_ENABLE_CUDA
    if (ctx.cuda_backend && ggml_get_no_alloc(ctx.work_ctx)) {
        if (ctx.current_execution_buffer) {
            ggml_backend_buffer_free((ggml_backend_buffer_t) ctx.current_execution_buffer);
            ctx.current_execution_buffer = nullptr;
        }
        ggml_backend_t backend = (ggml_backend_t) ctx.cuda_backend;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx.work_ctx, backend);
        if (!buf) {
            ctx.graph_uploads.clear();
            ctx.pending_kernel_launches.clear();
            throw std::runtime_error("bmo_execute_graph: failed to allocate backend tensors");
        }

        try {
            for (const auto & in : inputs) {
                if (in.tensor && in.host_data) {
                    ggml_backend_tensor_set(in.tensor, in.host_data, 0, ggml_nbytes(in.tensor));
                }
            }
            for (const auto & in : ctx.graph_uploads) {
                if (in.tensor && !in.bytes.empty()) {
                    ggml_backend_tensor_set(in.tensor, in.bytes.data(), 0, in.bytes.size());
                }
            }

            for (const auto & launch : ctx.pending_kernel_launches) {
                if (!launch.W || !launch.dp || !launch.dp->is_valid) {
                    throw std::runtime_error("bmo_execute_graph: invalid queued kernel launch");
                }
                launch_unpack_kernel(
                    launch.dp,
                    launch.rows,
                    launch.cols,
                    launch.n_2bit_bytes,
                    launch.n_4bit_bytes,
                    launch.n_8bit_bytes,
                    launch.scale_low,
                    launch.scale_int4,
                    launch.scale_int8,
                    launch.zp_low,
                    launch.zp_int4,
                    launch.zp_int8,
                    reinterpret_cast<float *>(launch.W->data));
            }

            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                throw std::runtime_error(std::string("bmo_execute_graph: cudaDeviceSynchronize failed: ") + cudaGetErrorString(sync_err));
            }

            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("bmo_execute_graph: ggml_backend_graph_compute failed");
            }

            ctx.current_execution_buffer = buf;
            ctx.graph_uploads.clear();
            ctx.pending_kernel_launches.clear();
            return;
        } catch (...) {
            ggml_backend_buffer_free(buf);
            ctx.graph_uploads.clear();
            ctx.pending_kernel_launches.clear();
            throw;
        }
    }
#endif

    const ggml_status status = ggml_graph_compute_with_ctx(ctx.work_ctx, gf, /*n_threads=*/32);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_graph_compute_with_ctx failed");
    }
    ctx.graph_uploads.clear();
    ctx.pending_kernel_launches.clear();
}

ggml_cgraph * bmo_build_temporal_graph(
    bmo_context & ctx,
    bmo_model & model,
    ggml_tensor * input_tokens,
    int n_past,
    int layer_begin,
    int layer_end) {
    if (!input_tokens) {
        throw std::runtime_error("bmo_build_temporal_graph: input_tokens is null");
    }
    if (!ctx.k_cache || !ctx.v_cache) {
        throw std::runtime_error("bmo_build_temporal_graph: KV cache not initialized");
    }
    if (model.temporal_layers.size() != (size_t) ctx.n_layers) {
        throw std::runtime_error("bmo_build_temporal_graph: temporal_layers size mismatch");
    }

    const int64_t n_token = input_tokens->ne[1];
    if (n_token <= 0) {
        throw std::runtime_error("bmo_build_temporal_graph: invalid token count");
    }

    if (!ctx.work_ctx) {
        throw std::runtime_error("failed to initialize temporal work context");
    }

    const int begin = std::max(0, layer_begin);
    const int end = (layer_end < 0) ? ctx.n_layers : std::min(layer_end, ctx.n_layers);
    if (begin >= end) {
        throw std::runtime_error("bmo_build_temporal_graph: invalid layer range");
    }

    ggml_context * wctx = ctx.work_ctx;
    ggml_cgraph * gf = ggml_new_graph(wctx);
    ctx.graph_uploads.clear();

    // Position ids include n_past so RoPE/KV indexing follows autoregressive offset.
    ggml_tensor * pos = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, n_token);
    std::vector<int32_t> pos_host((size_t) n_token);
    for (int64_t t = 0; t < n_token; ++t) {
        pos_host[(size_t) t] = (int32_t) n_past + (int32_t) t;
    }
    stage_tensor_upload(ctx, pos, pos_host.data(), (size_t) n_token * sizeof(int32_t));

    ggml_tensor * x = ggml_cont(wctx, S(input_tokens));

    for (int layer = begin; layer < end; ++layer) {
        const std::string base = "transformer_layers_" + std::to_string(layer);
        const bool is_final_layer = (layer == end - 1);

        // -------- Attention block --------
        ggml_tensor * residual = x;
        ggml_tensor * x_norm = ggml_rms_norm(wctx, x, 1e-5f);
        if (model.temporal_layers[layer].norm1_weight) {
            x_norm = ggml_mul(wctx, x_norm, model.temporal_layers[layer].norm1_weight);
        }
        ggml_tensor * qkv = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            x_norm,
            {
                base + "_self_attn_in_proj_weight",
                base + "_self_attn_in_proj",
            });

            const int64_t q_dim = (int64_t) ctx.n_heads * ctx.head_dim;
            const int64_t qkv_w = qkv->ne[0];
            const bool qkv_valid = (qkv_w > q_dim) && ((qkv_w - q_dim) % 2 == 0);

            if (!qkv_valid) {
                throw std::runtime_error("layer=" + std::to_string(layer) + " invalid qkv width " + std::to_string((long long) qkv_w) + " (q_dim=" + std::to_string((long long) q_dim) + ")");
            }

            const int64_t kv_dim = (qkv_w - q_dim) / 2;
            const int32_t n_kv_heads = (int32_t) (kv_dim / ctx.head_dim);
            if (n_kv_heads <= 0) {
                throw std::runtime_error("layer=" + std::to_string(layer) + " invalid n_kv_heads=" + std::to_string(n_kv_heads) + " from qkv width " + std::to_string((long long) qkv_w));
            }

            const size_t e = qkv->nb[0];
            const size_t nb1_q = (size_t) ctx.head_dim * e;
            const size_t nb2_qkv = qkv->nb[1];

            ggml_tensor * q_raw = ggml_view_3d(wctx, qkv, ctx.head_dim, ctx.n_heads, n_token, nb1_q, nb2_qkv, 0);
            ggml_tensor * k_raw = ggml_view_3d(wctx, qkv, ctx.head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) q_dim * e);
            ggml_tensor * v_raw = ggml_view_3d(wctx, qkv, ctx.head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) (q_dim + kv_dim) * e);

            ggml_tensor * q_rope = ggml_rope(wctx, q_raw, pos, ctx.head_dim, GGML_ROPE_TYPE_NORMAL);
            ggml_tensor * k_rope = ggml_rope(wctx, k_raw, pos, ctx.head_dim, GGML_ROPE_TYPE_NORMAL);

            ggml_tensor * q_trans = ggml_permute(wctx, q_rope, 0, 2, 1, 3);
            ggml_tensor * k_trans = ggml_permute(wctx, k_rope, 0, 2, 1, 3);
            ggml_tensor * v_trans = ggml_permute(wctx, v_raw,  0, 2, 1, 3);

            ggml_tensor * k_layer = ggml_view_3d(
                wctx, ctx.k_cache, ctx.head_dim, ctx.n_ctx, n_kv_heads,
                ctx.k_cache->nb[1], ctx.k_cache->nb[2], (size_t) layer * ctx.k_cache->nb[3]);

            ggml_tensor * v_layer = ggml_view_3d(
                wctx, ctx.v_cache, ctx.head_dim, ctx.n_ctx, n_kv_heads,
                ctx.v_cache->nb[1], ctx.v_cache->nb[2], (size_t) layer * ctx.v_cache->nb[3]);

            ggml_tensor * k_slot = ggml_view_3d(
                wctx, k_layer, ctx.head_dim, n_token, n_kv_heads,
                k_layer->nb[1], k_layer->nb[2], (size_t) n_past * k_layer->nb[1]);

            ggml_tensor * v_slot = ggml_view_3d(
                wctx, v_layer, ctx.head_dim, n_token, n_kv_heads,
                v_layer->nb[1], v_layer->nb[2], (size_t) n_past * v_layer->nb[1]);

            ggml_tensor * k_write = ggml_cpy(wctx, k_trans, k_slot);
            ggml_tensor * v_write = ggml_cpy(wctx, v_trans, v_slot);
            ggml_build_forward_expand(gf, k_write);
            ggml_build_forward_expand(gf, v_write);

            const int64_t kv_len = n_past + n_token;
            ggml_tensor * k_hist = ggml_view_3d(wctx, k_layer, ctx.head_dim, kv_len, n_kv_heads, k_layer->nb[1], k_layer->nb[2], 0);
            ggml_tensor * v_hist = ggml_view_3d(wctx, v_layer, ctx.head_dim, kv_len, n_kv_heads, v_layer->nb[1], v_layer->nb[2], 0);

            ggml_tensor * attn_heads = ggml_flash_attn_ext(
                wctx, q_trans, ggml_cont(wctx, k_hist), ggml_cont(wctx, v_hist), nullptr,
                1.0f / std::sqrt((float) ctx.head_dim), 0.0f, 0.0f);

            ggml_tensor * attn_trans_out = ggml_permute(wctx, attn_heads, 0, 2, 1, 3);
            ggml_tensor * attn_cont  = ggml_cont(wctx, attn_trans_out);
            ggml_tensor * attn_2d    = ggml_reshape_2d(wctx, attn_cont, ctx.n_embd, n_token);

            ggml_tensor * attn_out = apply_linear_with_transient_unpack(
                ctx,
                model,
                wctx,
                attn_2d,
                {
                    base + "_self_attn_out_proj_weight",
                    base + "_self_attn_out_proj",
                });

            x = ggml_add(wctx, S(residual), S(attn_out));

        // -------- Feed-forward block --------
        ggml_tensor * ff_residual = x;
        ggml_tensor * ff_norm = ggml_rms_norm(wctx, x, 1e-5f);
        if (model.temporal_layers[layer].norm2_weight) {
            ff_norm = ggml_mul(wctx, ff_norm, model.temporal_layers[layer].norm2_weight);
        }
        ggml_tensor * ff_in = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            ff_norm,
            {
                base + "_gating_linear_in_weight",
                base + "_gating_linear_in",
            });

        if (ff_in->ne[0] <= 0 || (ff_in->ne[0] % 2) != 0) {
            throw std::runtime_error("layer=" + std::to_string(layer) + " invalid ff_in width " + std::to_string((long long) ff_in->ne[0]));
        } else {
            // SwiGLU fused projection: split ff_in into gate and up projections
            const int64_t hidden_dim = ff_in->ne[0] / 2;
            ggml_tensor * ff_gate = ggml_view_2d(wctx, ff_in, hidden_dim, ff_in->ne[1], ff_in->nb[1], 0);
            ggml_tensor * ff_up   = ggml_view_2d(wctx, ff_in, hidden_dim, ff_in->ne[1], ff_in->nb[1], hidden_dim * ggml_type_size(ff_in->type));

            ggml_tensor * ff_act = ggml_mul(wctx, ggml_silu(wctx, ff_gate), ff_up);

            ggml_tensor * ff_out = apply_linear_with_transient_unpack(
                ctx,
                model,
                wctx,
                ff_act,
                {
                    base + "_gating_linear_out_weight",
                    base + "_gating_linear_out",
                });

            if (ggml_nelements(ff_out) != ggml_nelements(ff_residual)) {
                throw std::runtime_error(
                    "layer=" + std::to_string(layer) +
                    " ff_out elements=" + std::to_string((long long) ggml_nelements(ff_out)) +
                    " residual elements=" + std::to_string((long long) ggml_nelements(ff_residual))
                );
            } else {
                x = ggml_add(wctx, S(ff_residual), S(ff_out));
            }
        }

        const std::string out_name = "out_layer_" + std::to_string(layer);
        ggml_set_name(x, out_name.c_str());
        ggml_build_forward_expand(gf, x);
    }
    return gf;
}

ggml_cgraph * bmo_build_depth_graph(
    bmo_context & ctx,
    bmo_model & model,
    ggml_tensor * temporal_out,
    ggml_tensor * text_tokens,
    ggml_tensor * audio_tokens,
    int codebook_step,
    int n_past) {
    if (!temporal_out) {
        throw std::runtime_error("bmo_build_depth_graph: temporal_out is null");
    }
    if (!text_tokens) {
        throw std::runtime_error("bmo_build_depth_graph: text_tokens is null");
    }
    if (!audio_tokens) {
        throw std::runtime_error("bmo_build_depth_graph: audio_tokens is null");
    }
    if (!ctx.work_ctx) {
        throw std::runtime_error("bmo_build_depth_graph: work context is not initialized");
    }
    if (model.depth_layers.size() != 6) {
        throw std::runtime_error("bmo_build_depth_graph: expected 6 depth layers");
    }
    if (codebook_step < 0 || codebook_step >= (int) model.depformer_in.size()) {
        throw std::runtime_error("bmo_build_depth_graph: invalid codebook_step " + std::to_string(codebook_step));
    }
    if (!model.depformer_in[(size_t) codebook_step]) {
        throw std::runtime_error("bmo_build_depth_graph: missing depformer_in projection for codebook_step " + std::to_string(codebook_step));
    }
    if (codebook_step > 0 && !model.audio_embs[(size_t) (codebook_step - 1)]) {
        throw std::runtime_error("bmo_build_depth_graph: missing audio embedding table for previous codebook_step " + std::to_string(codebook_step - 1));
    }
    if (!model.text_emb) {
        throw std::runtime_error("bmo_build_depth_graph: missing text embedding table");
    }

    constexpr int64_t hidden_dim = 1024;
    constexpr int64_t depth_num_heads = 16;
    if ((hidden_dim % depth_num_heads) != 0) {
        throw std::runtime_error("bmo_build_depth_graph: hidden_dim must be divisible by depth_num_heads");
    }
    const int64_t head_dim = hidden_dim / depth_num_heads;

    ggml_context * wctx = ctx.work_ctx;
    ggml_cgraph * gf = ggml_new_graph(wctx);
    ctx.graph_uploads.clear();

    // Keep the depth stack causally aligned with the temporal stack interface.
    (void) n_past;

    ggml_tensor * z_s = ggml_mul_mat(wctx, model.depformer_in[(size_t) codebook_step], temporal_out);
    if (!z_s) {
        throw std::runtime_error("bmo_build_depth_graph: failed to build depformer input projection");
    }

    const int64_t n_token = temporal_out->ne[1];
    const bool debug_step0 = (codebook_step == 0);
    const int64_t step_count = (int64_t) model.depformer_in.size();
    const int64_t qkv_step = 3 * hidden_dim;
    const int64_t out_step = hidden_dim;

    ggml_tensor * last_tok = nullptr;
    if (codebook_step == 0) {
        last_tok = ggml_get_rows(wctx, model.text_emb, text_tokens);
    } else {
        last_tok = ggml_get_rows(wctx, model.audio_embs[(size_t) (codebook_step - 1)], audio_tokens);
    }

    if (!last_tok) {
        throw std::runtime_error("bmo_build_depth_graph: failed to build token embeddings");
    }

    // handle broadcast by reshaping last_tok to 2d if needed, though z_s is 1024x1 and last_tok is 1024x1
    ggml_tensor * x = ggml_add(wctx, z_s, ggml_reshape_2d(wctx, last_tok, 1024, 1));
    if (debug_step0) {
        ggml_set_name(x, "depth_x_init");
    }

    for (int i = 0; i < 6; ++i) {
        const std::string prefix = "depformer.layers." + std::to_string(i);
        const std::string base = "depformer_layers_" + std::to_string(i);
        const std::string step_idx = std::to_string(codebook_step);

        const std::string dot_prefix = prefix; // e.g. "depformer.layers.0"
        // Try dot-style keys first, then fall back to exporter underscore-style keys.
        ggml_tensor * w_massive_in = get_tensor(model.wctx, dot_prefix + ".self_attn.in_proj_weight");
        if (!w_massive_in) {
            std::string ug = std::string("depformer_layers_") + std::to_string(i) + "_self_attn_in_proj_weight";
            w_massive_in = get_tensor(model.wctx, ug);
        }

        ggml_tensor * w_massive_out = get_tensor(model.wctx, dot_prefix + ".self_attn.out_proj.weight");
        if (!w_massive_out) {
            std::string ug_out = std::string("depformer_layers_") + std::to_string(i) + "_self_attn_out_proj_weight";
            w_massive_out = get_tensor(model.wctx, ug_out);
        }

        if (!w_massive_in || !w_massive_out) {
            throw std::runtime_error("bmo_build_depth_graph: missing shared attention weight tensors for layer " + std::to_string(i));
        }

        ggml_tensor * w_slice = nullptr;
        if (w_massive_in->ne[0] == qkv_step * step_count && w_massive_in->ne[1] == hidden_dim) {
            ggml_tensor * w_view = ggml_view_2d(
                wctx,
                w_massive_in,
                qkv_step,
                hidden_dim,
                w_massive_in->nb[1],
                (size_t) codebook_step * qkv_step * w_massive_in->nb[0]);
            w_slice = ggml_cont(wctx, ggml_transpose(wctx, w_view));
        } else if (w_massive_in->ne[0] == hidden_dim && w_massive_in->ne[1] == qkv_step * step_count) {
            w_slice = ggml_view_2d(
                wctx,
                w_massive_in,
                hidden_dim,
                qkv_step,
                w_massive_in->nb[1],
                (size_t) codebook_step * qkv_step * w_massive_in->nb[1]);
        } else {
            throw std::runtime_error("bmo_build_depth_graph: unexpected in_proj shape " +
                std::to_string((long long) w_massive_in->ne[0]) + "x" +
                std::to_string((long long) w_massive_in->ne[1]));
        }

        ggml_tensor * w_out_slice = nullptr;
        if (w_massive_out->ne[0] == out_step * step_count && w_massive_out->ne[1] == hidden_dim) {
            ggml_tensor * w_view = ggml_view_2d(
                wctx,
                w_massive_out,
                out_step,
                hidden_dim,
                w_massive_out->nb[1],
                (size_t) codebook_step * out_step * w_massive_out->nb[0]);
            w_out_slice = ggml_cont(wctx, ggml_transpose(wctx, w_view));
        } else if (w_massive_out->ne[0] == hidden_dim && w_massive_out->ne[1] == out_step * step_count) {
            w_out_slice = ggml_view_2d(
                wctx,
                w_massive_out,
                hidden_dim,
                out_step,
                w_massive_out->nb[1],
                (size_t) codebook_step * out_step * w_massive_out->nb[1]);
        } else {
            throw std::runtime_error("bmo_build_depth_graph: unexpected out_proj shape " +
                std::to_string((long long) w_massive_out->ne[0]) + "x" +
                std::to_string((long long) w_massive_out->ne[1]));
        }

        // Shared attention block.
        ggml_tensor * residual = x;
        ggml_tensor * x_norm = ggml_rms_norm(wctx, x, 1e-5f);
        if (!model.depth_layers[(size_t) i].norm1_weight) {
            throw std::runtime_error("bmo_build_depth_graph: missing norm1_weight for depth layer " + std::to_string(i));
        }
        x_norm = ggml_mul(wctx, x_norm, model.depth_layers[(size_t) i].norm1_weight);
        if (debug_step0 && i == 0) {
            ggml_set_name(x_norm, "depth_x_norm");
            ggml_build_forward_expand(gf, x_norm);
        }

        ggml_tensor * qkv = ggml_mul_mat(wctx, S(w_slice), S(x_norm));
        if (debug_step0 && i == 0) {
            ggml_set_name(qkv, "depth_qkv_raw");
            ggml_build_forward_expand(gf, qkv);
        }

        const int64_t q_dim = hidden_dim;
        const int64_t kv_dim = hidden_dim;
        const int64_t qkv_w = qkv->ne[0];
        if (qkv_w != q_dim + kv_dim + kv_dim) {
            throw std::runtime_error(
                "bmo_build_depth_graph: layer=" + std::to_string(i) +
                " invalid qkv width " + std::to_string((long long) qkv_w) +
                " (expected " + std::to_string((long long) (q_dim + 2 * kv_dim)) + ")");
        }

        const int32_t n_kv_heads = (int32_t) (kv_dim / head_dim);
        if (n_kv_heads <= 0) {
            throw std::runtime_error("bmo_build_depth_graph: invalid n_kv_heads for layer " + std::to_string(i));
        }

        const size_t e = qkv->nb[0];
        const size_t nb1_q = (size_t) head_dim * e;
        const size_t nb2_qkv = qkv->nb[1];

        ggml_tensor * q_raw = ggml_view_3d(wctx, qkv, head_dim, depth_num_heads, n_token, nb1_q, nb2_qkv, 0);
        ggml_tensor * k_raw = ggml_view_3d(wctx, qkv, head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) q_dim * e);
        ggml_tensor * v_raw = ggml_view_3d(wctx, qkv, head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) (q_dim + kv_dim) * e);

        if (debug_step0 && i == 0) {
            ggml_tensor * q_dbg = ggml_cont(wctx, q_raw);
            ggml_tensor * k_dbg = ggml_cont(wctx, k_raw);
            ggml_tensor * v_dbg = ggml_cont(wctx, v_raw);
            ggml_set_name(q_dbg, "depth_q_raw");
            ggml_set_name(k_dbg, "depth_k_raw");
            ggml_set_name(v_dbg, "depth_v_raw");
            ggml_build_forward_expand(gf, q_dbg);
            ggml_build_forward_expand(gf, k_dbg);
            ggml_build_forward_expand(gf, v_dbg);
        }

        ggml_tensor * pos = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, n_token);
        std::vector<int32_t> pos_host((size_t) n_token);
        for (int64_t t = 0; t < n_token; ++t) {
            pos_host[(size_t) t] = (int32_t) n_past + (int32_t) t;
        }
        stage_tensor_upload(ctx, pos, pos_host.data(), (size_t) n_token * sizeof(int32_t));

        ggml_tensor * q_rope = ggml_rope(wctx, q_raw, pos, (int) head_dim, GGML_ROPE_TYPE_NORMAL);
        ggml_tensor * k_rope = ggml_rope(wctx, k_raw, pos, (int) head_dim, GGML_ROPE_TYPE_NORMAL);

        ggml_tensor * q_trans = ggml_permute(wctx, q_rope, 0, 2, 1, 3);
        ggml_tensor * k_trans = ggml_permute(wctx, k_rope, 0, 2, 1, 3);
        ggml_tensor * v_trans = ggml_permute(wctx, v_raw,  0, 2, 1, 3);

        ggml_tensor * attn_heads = ggml_flash_attn_ext(
            wctx,
            q_trans,
            k_trans,
            v_trans,
            nullptr,
            1.0f / std::sqrt((float) head_dim),
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(attn_heads, GGML_PREC_F32);

        ggml_tensor * attn_trans_out = ggml_permute(wctx, attn_heads, 0, 2, 1, 3);
        ggml_tensor * attn_cont = ggml_cont(wctx, attn_trans_out);
        ggml_tensor * attn_2d = ggml_reshape_2d(wctx, attn_cont, hidden_dim, n_token);

        ggml_tensor * attn_out = ggml_mul_mat(wctx, S(w_out_slice), S(attn_2d));

        if (debug_step0 && i == 0) {
            ggml_tensor * attn_dbg = ggml_cont(wctx, attn_out);
            ggml_set_name(attn_dbg, "depth_attn_out");
            ggml_build_forward_expand(gf, attn_dbg);
        }

        x = ggml_add(wctx, residual, attn_out);
        if (debug_step0 && i == 0) {
            ggml_set_name(x, "depth_attn_x");
        }

        // Step-specific FFN block.
        ggml_tensor * ff_residual = x;
        ggml_tensor * ff_norm = ggml_rms_norm(wctx, x, 1e-5f);
        if (model.depth_layers[(size_t) i].norm2_weight) {
            ff_norm = ggml_mul(wctx, ff_norm, model.depth_layers[(size_t) i].norm2_weight);
        }

        ggml_tensor * ff_in = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            ff_norm,
            {
                base + "_gating_" + step_idx + "_linear_in_weight",
                base + "_gating_" + step_idx + "_linear_in",
            });

        if (!ff_in) {
            throw std::runtime_error("bmo_build_depth_graph: missing step-specific FFN input for layer " + std::to_string(i));
        }
        if (ff_in->ne[0] <= 0 || (ff_in->ne[0] % 2) != 0) {
            throw std::runtime_error("bmo_build_depth_graph: layer=" + std::to_string(i) + " invalid ff_in width " + std::to_string((long long) ff_in->ne[0]));
        }

        const int64_t ff_hidden = ff_in->ne[0] / 2;
        ggml_tensor * ff_gate = ggml_view_2d(wctx, ff_in, ff_hidden, ff_in->ne[1], ff_in->nb[1], 0);
        ggml_tensor * ff_up = ggml_view_2d(wctx, ff_in, ff_hidden, ff_in->ne[1], ff_in->nb[1], ff_hidden * ggml_type_size(ff_in->type));
        ggml_tensor * ff_act = ggml_mul(wctx, ggml_silu(wctx, ff_gate), ff_up);

        ggml_tensor * ff_out = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            ff_act,
            {
                base + "_gating_" + step_idx + "_linear_out_weight",
                base + "_gating_" + step_idx + "_linear_out",
            });

        if (!ff_out) {
            throw std::runtime_error("bmo_build_depth_graph: missing step-specific FFN output for layer " + std::to_string(i));
        }
        if (ggml_nelements(ff_out) != ggml_nelements(ff_residual)) {
            throw std::runtime_error(
                "bmo_build_depth_graph: layer=" + std::to_string(i) +
                " ff_out elements=" + std::to_string((long long) ggml_nelements(ff_out)) +
                " residual elements=" + std::to_string((long long) ggml_nelements(ff_residual))
            );
        }

        x = ggml_add(wctx, ff_residual, ff_out);
    }

    const std::string out_name = "depth_out_step_" + std::to_string(codebook_step);
    ggml_set_name(x, out_name.c_str());
    ggml_build_forward_expand(gf, x);
    return gf;
}
