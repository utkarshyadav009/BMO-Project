// bmo.cpp - model loader and KV cache allocator

#include "bmo.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <unordered_set>

extern "C" {
#include "ggml.h"
#include "gguf.h"
}

// Helper: read scalar int32 stored as a 1-element tensor in the GGUF data ctx
static int32_t read_scalar_i32(ggml_context * data_ctx, const char * name, int32_t fallback = -1) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name);
    if (!t) return fallback;
    if (ggml_nbytes(t) < (int)sizeof(int32_t)) return fallback;
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context * data_ctx, const char * name, float fallback = 0.0f) {
    ggml_tensor * t = ggml_get_tensor(data_ctx, name);
    if (!t) return fallback;
    if (ggml_nbytes(t) < (int)sizeof(float)) return fallback;
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

// Map layer tensors by conventional names. This helper will attempt to resolve
// both our temporal packed names and common unquantized names.
static void map_layer_tensors(ggml_context * data_ctx, bmo_layer & layer, const std::string & base) {
    if (!layer.packed_weights) layer.packed_weights = ggml_get_tensor(data_ctx, (base + ".packed_weights").c_str());
    if (!layer.packed_mask)    layer.packed_mask    = ggml_get_tensor(data_ctx, (base + ".packed_mask").c_str());
    if (!layer.scale_low)      layer.scale_low      = ggml_get_tensor(data_ctx, (base + ".scale_low").c_str());
    if (!layer.scale_int4)     layer.scale_int4     = ggml_get_tensor(data_ctx, (base + ".scale_int4").c_str());
    if (!layer.scale_int8)     layer.scale_int8     = ggml_get_tensor(data_ctx, (base + ".scale_int8").c_str());
    if (!layer.fp16_indices)   layer.fp16_indices   = ggml_get_tensor(data_ctx, (base + ".fp16_indices").c_str());
    if (!layer.fp16_values)    layer.fp16_values    = ggml_get_tensor(data_ctx, (base + ".fp16_values").c_str());

    // If temporal packed artifacts not found, try classical weight/bias names
    if (!layer.packed_weights) {
        if (!layer.weight) layer.weight = ggml_get_tensor(data_ctx, (base + ".weight").c_str());
        if (!layer.bias)   layer.bias   = ggml_get_tensor(data_ctx, (base + ".bias").c_str());
    }

    // Attention / FFN optional subcomponents
    if (!layer.wq) layer.wq = ggml_get_tensor(data_ctx, (base + ".wq").c_str());
    if (!layer.wk) layer.wk = ggml_get_tensor(data_ctx, (base + ".wk").c_str());
    if (!layer.wv) layer.wv = ggml_get_tensor(data_ctx, (base + ".wv").c_str());
    if (!layer.wo) layer.wo = ggml_get_tensor(data_ctx, (base + ".wo").c_str());

    if (!layer.ffn_in)  layer.ffn_in  = ggml_get_tensor(data_ctx, (base + ".ffn_in").c_str());
    if (!layer.ffn_out) layer.ffn_out = ggml_get_tensor(data_ctx, (base + ".ffn_out").c_str());

    // Exported names may use base ending with _weight and still append dotted payload keys,
    // e.g. transformer_layers_0_gating_linear_in_weight.packed_weights.
    if (!layer.packed_weights && base.size() >= 7 && base.compare(base.size() - 7, 7, "_weight") == 0) {
        if (!layer.packed_weights) layer.packed_weights = ggml_get_tensor(data_ctx, (base + ".packed_weights").c_str());
        if (!layer.packed_mask)    layer.packed_mask    = ggml_get_tensor(data_ctx, (base + ".packed_mask").c_str());
        if (!layer.scale_low)      layer.scale_low      = ggml_get_tensor(data_ctx, (base + ".scale_low").c_str());
        if (!layer.scale_int4)     layer.scale_int4     = ggml_get_tensor(data_ctx, (base + ".scale_int4").c_str());
        if (!layer.scale_int8)     layer.scale_int8     = ggml_get_tensor(data_ctx, (base + ".scale_int8").c_str());
        if (!layer.fp16_indices)   layer.fp16_indices   = ggml_get_tensor(data_ctx, (base + ".fp16_indices").c_str());
        if (!layer.fp16_values)    layer.fp16_values    = ggml_get_tensor(data_ctx, (base + ".fp16_values").c_str());

        // Also allow direct weight/bias using this base.
        if (!layer.weight) {
            layer.weight = ggml_get_tensor(data_ctx, (base + ".weight").c_str());
        }
        if (!layer.bias) {
            layer.bias = ggml_get_tensor(data_ctx, (base + ".bias").c_str());
        }
    }
}

static void add_tensor_bytes_unique(ggml_tensor * t, std::unordered_set<const void *> & seen, size_t & total_bytes) {
    if (!t) return;
    if (seen.insert((const void *) t).second) {
        total_bytes += (size_t) ggml_nbytes(t);
    }
}

static void add_layer_bytes_unique(const bmo_layer & L, std::unordered_set<const void *> & seen, size_t & total_bytes) {
    std::vector<ggml_tensor *> toks = {
        L.packed_weights, L.packed_mask, L.scale_low, L.scale_int4, L.scale_int8,
        L.fp16_indices, L.fp16_values, L.weight, L.bias, L.wq, L.wk, L.wv, L.wo,
        L.ffn_in, L.ffn_out
    };
    for (auto * t : toks) {
        add_tensor_bytes_unique(t, seen, total_bytes);
    }
}

void bmo_load_model(const char * fname, bmo_model & model, bmo_context & ctx) {
    // Load GGUF and obtain data ggml_context
    ggml_context * data_ctx = nullptr;
    gguf_init_params params = { /* no_alloc */ false, /* ctx */ &data_ctx };
    gguf_context * gctx = gguf_init_from_file(fname, params);
    if (!gctx) {
        throw std::runtime_error(std::string("Failed to open GGUF: ") + fname);
    }
    if (!data_ctx) {
        throw std::runtime_error("GGUF: data context not returned");
    }

    model.gctx = gctx;
    model.wctx = data_ctx;

    // Hardcoded Moshi 5.8B architecture.
    ctx.n_layers = 32;
    ctx.n_heads = 16;
    ctx.n_embd = 2048;
    ctx.head_dim = 128;

    // Keep n_ctx from file if present, otherwise 0 until KV init sets it.
    ctx.n_ctx = read_scalar_i32(data_ctx, "n_ctx", 0);

    // Temporal stack (32 layers)
    model.temporal_layers.resize((size_t) ctx.n_layers);
    for (int i = 0; i < ctx.n_layers; ++i) {
        std::string idx = std::to_string(i);
        std::string prefix = "transformer_layers_" + idx;
        bmo_layer & L = model.temporal_layers[(size_t) i];
        L.name = prefix;

        std::vector<std::string> bases = {
            prefix,
            prefix + "_gating_linear_in",
            prefix + "_gating_linear_out",
            prefix + "_gating_linear_in_weight",
            prefix + "_gating_linear_out_weight",
            prefix + "_self_attn_in_proj",
            prefix + "_self_attn_out_proj",
            prefix + "_self_attn_in_proj_weight",
            prefix + "_self_attn_out_proj_weight",
        };
        for (auto &b : bases) {
            map_layer_tensors(data_ctx, L, b);
        }
    }

    // Depth stack (6 layers)
    model.depth_layers.resize((size_t) 6);
    for (int i = 0; i < 6; ++i) {
        std::string idx = std::to_string(i);
        std::string prefix = "depformer_layers_" + idx;
        bmo_layer & L = model.depth_layers[(size_t) i];
        L.name = prefix;

        std::vector<std::string> bases = {
            prefix,
            prefix + "_self_attn_in_proj",
            prefix + "_self_attn_out_proj",
            prefix + "_self_attn_in_proj_weight",
            prefix + "_self_attn_out_proj_weight",
        };
        for (auto &b : bases) {
            map_layer_tensors(data_ctx, L, b);
        }
    }

    // Audio codebook embeddings and depformer input projections (16 each)
    model.audio_embs.assign((size_t) 16, nullptr);
    model.depformer_in.assign((size_t) 16, nullptr);
    for (int i = 0; i < 16; ++i) {
        std::string idx = std::to_string(i);
        model.audio_embs[(size_t) i] = ggml_get_tensor(data_ctx, ("emb." + idx + ".weight").c_str());
        model.depformer_in[(size_t) i] = ggml_get_tensor(data_ctx, ("depformer_in." + idx + ".weight").c_str());
    }

    // Text embedding and projection
    model.text_emb = ggml_get_tensor(data_ctx, "text_emb.weight");
    model.text_linear = ggml_get_tensor(data_ctx, "text_linear.weight");

    // Global embeddings / head lookups
    model.token_embedding = ggml_get_tensor(data_ctx, "token_embedding");
    model.output_head = ggml_get_tensor(data_ctx, "output_head");

    // Compute total weights bytes across all mapped groups.
    size_t total_bytes = 0;
    std::unordered_set<const void *> seen;

    for (const auto & L : model.temporal_layers) {
        add_layer_bytes_unique(L, seen, total_bytes);
    }
    for (const auto & L : model.depth_layers) {
        add_layer_bytes_unique(L, seen, total_bytes);
    }

    for (auto * t : model.audio_embs) {
        add_tensor_bytes_unique(t, seen, total_bytes);
    }
    for (auto * t : model.depformer_in) {
        add_tensor_bytes_unique(t, seen, total_bytes);
    }

    add_tensor_bytes_unique(model.text_emb, seen, total_bytes);
    add_tensor_bytes_unique(model.text_linear, seen, total_bytes);
    add_tensor_bytes_unique(model.token_embedding, seen, total_bytes);
    add_tensor_bytes_unique(model.output_head, seen, total_bytes);

    ctx.weights_bytes = total_bytes;

    std::cout << "[bmo_load_model] Loaded model '" << fname << "'\n";
    std::cout << "[bmo_load_model] n_layers=" << ctx.n_layers << " n_heads=" << ctx.n_heads << " n_embd=" << ctx.n_embd << " n_ctx=" << ctx.n_ctx << "\n";
    std::cout << "[bmo_load_model] Total weight bytes: " << (double) total_bytes / (1024.0 * 1024.0) << " MB\n";
}

void bmo_init_kv_cache(bmo_context & ctx, int32_t n_ctx) {
    if (ctx.n_heads <= 0 || ctx.head_dim <= 0 || ctx.n_layers <= 0) {
        throw std::runtime_error("KV cache init requires valid n_layers, n_heads and head_dim in context");
    }

    ctx.n_ctx = n_ctx;

    // Estimate required memory: two caches (k and v) stored as f16
    const int64_t elems_per_layer = (int64_t) n_ctx * (int64_t) ctx.n_heads * (int64_t) ctx.head_dim;
    const size_t bytes_per_layer = (size_t) elems_per_layer * sizeof(ggml_fp16_t) * 2; // k + v
    const size_t total_bytes = bytes_per_layer * (size_t) ctx.n_layers;

    // Allocate a KV ggml_context with this memory size + small slack
    const size_t alloc_size = total_bytes + (1 << 20);
    std::unique_ptr<uint8_t[]> mem(new uint8_t[alloc_size]);

    ggml_init_params iparams = { (size_t) alloc_size, mem.get(), /*no_alloc*/ false };
    ggml_context * kv_ctx = ggml_init(iparams);
    if (!kv_ctx) throw std::runtime_error("Failed to initialize KV ggml_context");

    // Create k_cache and v_cache as 4D tensors: (head_dim, n_ctx, n_heads, n_layers)
    ctx.kv_ctx = kv_ctx;
    ctx.k_cache = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, ctx.head_dim, n_ctx, ctx.n_heads, ctx.n_layers);
    ctx.v_cache = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, ctx.head_dim, n_ctx, ctx.n_heads, ctx.n_layers);

    ctx.kv_bytes = (size_t) ggml_nbytes(ctx.k_cache) + (size_t) ggml_nbytes(ctx.v_cache);

    std::cout << "[bmo_init_kv_cache] Allocated KV cache: " << (double) ctx.kv_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "[bmo_init_kv_cache] per-layer estimate: " << (double) bytes_per_layer / (1024.0 * 1024.0) << " MB\n";

    // Note: mem buffer is owned by this function local unique_ptr; we must ensure
    // the lifetime of the buffer outlives kv_ctx. For simplicity we leak it
    // intentionally here for the life of the program (acceptable for a process
    // that keeps kv_ctx for entire runtime). If desired, make it a field on
    // bmo_context and manage lifetime explicitly.
    (void) mem.release();
}
