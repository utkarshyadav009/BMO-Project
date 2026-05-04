// bmo.h - core data structures for BMO model runtime
#pragma once

#include <string>
#include <vector>

extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
}

// A single layer which may be either a temporal (multi-tier quantized)
// or a standard unquantized depth layer. Fields not used for a given
// layer type are left nullptr.
struct bmo_layer {
    std::string name;

    // Temporal multi-tier (packed) artifacts
    ggml_tensor * packed_weights = nullptr; // packed 2/4/8 stream (bytes)
    ggml_tensor * packed_mask = nullptr;    // uint8 packed mask (4 x uint2 per byte)
    ggml_tensor * scale_low = nullptr;      // scalar f32 or f16
    ggml_tensor * scale_int4 = nullptr;
    ggml_tensor * scale_int8 = nullptr;
    ggml_tensor * fp16_indices = nullptr;   // int32 indices
    ggml_tensor * fp16_values = nullptr;    // f16 or f32 values

    // Unquantized artifacts (dense weights)
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;

    // Optional attention / ffn sub-weights for downstream compute
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;

    ggml_tensor * ffn_in = nullptr;
    ggml_tensor * ffn_out = nullptr;

    // Learned RMSNorm scale weights (gamma)
    ggml_tensor * norm1_weight = nullptr;  // attention pre-norm
    ggml_tensor * norm2_weight = nullptr;  // FFN pre-norm

    bmo_layer() = default;
};

// Model container: arrays of layers and global heads/embeddings
struct bmo_model {
    std::vector<bmo_layer> temporal_layers;
    std::vector<bmo_layer> depth_layers;

    // Codebook embedding tables and depformer input projections
    std::vector<ggml_tensor *> audio_embs;
    std::vector<ggml_tensor *> depformer_in;

    ggml_tensor * text_emb = nullptr;
    ggml_tensor * text_linear = nullptr;

    // embeddings and head tensors (may be stored as ggml tensors inside the
    // weights ggml_context that gguf provides)
    ggml_tensor * token_embedding = nullptr;
    ggml_tensor * output_head = nullptr;

    // Keep a reference to the gguf/ggml data context in which weight tensors live.
    gguf_context * gctx = nullptr;     // owns the data memory
    ggml_context * wctx = nullptr;     // ggml context used by gguf to store weight tensors
};

// Runtime context that holds allocation contexts and runtime params (KV cache, etc.)
struct bmo_context {
    // GGML contexts
    ggml_context * kv_ctx = nullptr;   // dedicated KV cache context

    // Model hyperparameters filled at load time
    int32_t n_ctx = 0;
    int32_t n_layers = 0;
    int32_t n_heads = 0;
    int32_t n_embd = 0;
    int32_t head_dim = 0;

    // Physical KV cache tensors allocated in kv_ctx
    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;

    // Track memory usage
    size_t weights_bytes = 0;
    size_t kv_bytes = 0;
    
    // Inference compute arenas
    ggml_context * work_ctx = nullptr;
    std::vector<uint8_t> work_mem;
    std::vector<float> shared_scratch_w;
};

// Loader and allocator APIs
void bmo_load_model(const char * fname, bmo_model & model, bmo_context & ctx);
void bmo_init_kv_cache(bmo_context & ctx, int32_t n_ctx);

// Compute graph builder
struct ggml_cgraph * bmo_build_temporal_graph(
    bmo_context & ctx,
    bmo_model & model,
    struct ggml_tensor * input_tokens,
    int n_past,
    int layer_begin = 0,
    int layer_end = -1);
