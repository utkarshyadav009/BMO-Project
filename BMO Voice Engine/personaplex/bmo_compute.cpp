#include "bmo.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
        throw std::runtime_error("packed stream usage mismatch while unpacking temporal weights");
    }
}

static ggml_tensor * apply_linear_with_transient_unpack(
    bmo_context & ctx,
    bmo_model & model,
    ggml_context * wctx,
    ggml_tensor * x,
    const std::vector<std::string> & base_candidates) {

    packed_linear_ref linear = resolve_linear(model.wctx, base_candidates);

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

        const float zp_low = read_scalar_f32(model.wctx, linear.base + ".zp_low", 0.0f);
        const float zp_int4 = read_scalar_f32(model.wctx, linear.base + ".zp_int4", 0.0f);
        const float zp_int8 = read_scalar_f32(model.wctx, linear.base + ".zp_int8", 0.0f);

        if (!linear.packed_mask || !linear.fp16_indices || !linear.fp16_values) {
            throw std::runtime_error("incomplete packed tensor set for " + linear.base);
        }

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

        ggml_tensor * W = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, cols, rows);
        std::memcpy(W->data, ctx.shared_scratch_w.data(), (size_t) total * sizeof(float));

        y = ggml_mul_mat(wctx, W, x);
    } else {
        y = ggml_mul_mat(wctx, linear.dense_weight, x);
    }

    if (linear.dense_bias) {
        y = ggml_add(wctx, y, linear.dense_bias);
    }

    return y;
}

} // namespace

ggml_cgraph * bmo_build_temporal_graph(bmo_context & ctx, bmo_model & model, ggml_tensor * input_tokens, int n_past) {
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

    // Rebuild work context per call so all intermediate tensors are scoped to this pass.
    if (ctx.work_ctx) {
        ggml_free(ctx.work_ctx);
        ctx.work_ctx = nullptr;
    }

    // Naive correctness graph memory. This can be tuned once kernel behavior is validated.
    if (ctx.work_mem.empty()) {
        ctx.work_mem.resize((size_t) 1024 * 1024 * 1024); // 1 GiB
    }

    ggml_init_params p = {
        ctx.work_mem.size(),
        ctx.work_mem.data(),
        false,
    };
    ctx.work_ctx = ggml_init(p);
    if (!ctx.work_ctx) {
        throw std::runtime_error("failed to initialize temporal work context");
    }

    ggml_context * wctx = ctx.work_ctx;
    ggml_cgraph * gf = ggml_new_graph(wctx);

    // Position ids include n_past so RoPE/KV indexing follows autoregressive offset.
    ggml_tensor * pos = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, n_token);
    int32_t * pos_ptr = reinterpret_cast<int32_t *>(pos->data);
    for (int64_t t = 0; t < n_token; ++t) {
        pos_ptr[t] = (int32_t) n_past + (int32_t) t;
    }

    ggml_tensor * x = ggml_cont(wctx, input_tokens);

    for (int layer = 0; layer < ctx.n_layers; ++layer) {
        const std::string base = "transformer_layers_" + std::to_string(layer);

        // -------- Attention block --------
        ggml_tensor * residual = x;
        ggml_tensor * x_norm = ggml_rms_norm(wctx, x, 1e-5f);

        ggml_tensor * qkv = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            x_norm,
            {
                base + "_self_attn_in_proj_weight",
                base + "_self_attn_in_proj",
            });

        ggml_tensor * q = nullptr;
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;

        if (qkv->ne[0] == (int64_t) (3 * ctx.n_embd)) {
            const size_t e = qkv->nb[0];
            q = ggml_view_2d(wctx, qkv, ctx.n_embd, n_token, qkv->nb[1], 0);
            k = ggml_view_2d(wctx, qkv, ctx.n_embd, n_token, qkv->nb[1], (size_t) ctx.n_embd * e);
            v = ggml_view_2d(wctx, qkv, ctx.n_embd, n_token, qkv->nb[1], (size_t) (2 * ctx.n_embd) * e);
        } else {
            // Fallback when exporter emits non-fused attn-in shape.
            q = qkv;
            k = qkv;
            v = qkv;
        }

        ggml_tensor * q3 = ggml_reshape_3d(wctx, q, ctx.head_dim, ctx.n_heads, n_token);
        ggml_tensor * k3 = ggml_reshape_3d(wctx, k, ctx.head_dim, ctx.n_heads, n_token);
        ggml_tensor * v3 = ggml_reshape_3d(wctx, v, ctx.head_dim, ctx.n_heads, n_token);

        ggml_tensor * q_rope = ggml_rope(wctx, q3, pos, ctx.head_dim, GGML_ROPE_TYPE_NORMAL);
        ggml_tensor * k_rope = ggml_rope(wctx, k3, pos, ctx.head_dim, GGML_ROPE_TYPE_NORMAL);

        // KV views for one temporal layer in [head_dim, n_ctx, n_heads] layout.
        ggml_tensor * k_layer = ggml_view_3d(
            wctx,
            ctx.k_cache,
            ctx.head_dim,
            ctx.n_ctx,
            ctx.n_heads,
            ctx.k_cache->nb[1],
            ctx.k_cache->nb[2],
            (size_t) layer * ctx.k_cache->nb[3]);

        ggml_tensor * v_layer = ggml_view_3d(
            wctx,
            ctx.v_cache,
            ctx.head_dim,
            ctx.n_ctx,
            ctx.n_heads,
            ctx.v_cache->nb[1],
            ctx.v_cache->nb[2],
            (size_t) layer * ctx.v_cache->nb[3]);

        // Write current token block at n_past offset.
        ggml_tensor * k_slot = ggml_view_3d(
            wctx,
            k_layer,
            ctx.head_dim,
            n_token,
            ctx.n_heads,
            k_layer->nb[1],
            k_layer->nb[2],
            (size_t) n_past * k_layer->nb[1]);

        ggml_tensor * v_slot = ggml_view_3d(
            wctx,
            v_layer,
            ctx.head_dim,
            n_token,
            ctx.n_heads,
            v_layer->nb[1],
            v_layer->nb[2],
            (size_t) n_past * v_layer->nb[1]);

        ggml_tensor * k_write = ggml_cpy(wctx, k_rope, k_slot);
        ggml_tensor * v_write = ggml_cpy(wctx, v3, v_slot);
        ggml_build_forward_expand(gf, k_write);
        ggml_build_forward_expand(gf, v_write);

        const int64_t kv_len = n_past + n_token;
        ggml_tensor * k_hist = ggml_view_3d(wctx, k_layer, ctx.head_dim, kv_len, ctx.n_heads, k_layer->nb[1], k_layer->nb[2], 0);
        ggml_tensor * v_hist = ggml_view_3d(wctx, v_layer, ctx.head_dim, kv_len, ctx.n_heads, v_layer->nb[1], v_layer->nb[2], 0);

        // Attention kernel: use flash-attn path if available in this ggml build.
        ggml_tensor * attn_heads = ggml_flash_attn_ext(
            wctx,
            q_rope,
            k_hist,
            v_hist,
            nullptr,
            1.0f / std::sqrt((float) ctx.head_dim),
            0.0f,
            0.0f);

        ggml_tensor * attn_2d = ggml_reshape_2d(wctx, attn_heads, ctx.n_embd, n_token);

        ggml_tensor * attn_out = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            attn_2d,
            {
                base + "_self_attn_out_proj_weight",
                base + "_self_attn_out_proj",
            });

        x = ggml_add(wctx, residual, attn_out);

        // -------- Feed-forward block --------
        ggml_tensor * ff_residual = x;
        ggml_tensor * ff_norm = ggml_rms_norm(wctx, x, 1e-5f);

        ggml_tensor * ff_in = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            ff_norm,
            {
                base + "_gating_linear_in_weight",
                base + "_gating_linear_in",
            });

        ggml_tensor * ff_act = ggml_silu(wctx, ff_in);

        ggml_tensor * ff_out = apply_linear_with_transient_unpack(
            ctx,
            model,
            wctx,
            ff_act,
            {
                base + "_gating_linear_out_weight",
                base + "_gating_linear_out",
            });

        x = ggml_add(wctx, ff_residual, ff_out);
    }

    ggml_build_forward_expand(gf, x);
    return gf;
}
