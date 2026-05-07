// bmo.h - core data structures for BMO model runtime
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

// Device-side packed tensor metadata (for CUDA unpacking)
struct device_packed_t {
    void * packed_weights = nullptr;      // device ptr to packed 2/4/8 streams
    void * packed_mask = nullptr;         // device ptr to tier mask
    void * fp16_indices = nullptr;        // device ptr to fp16 index array
    void * fp16_values = nullptr;         // device ptr to fp16 value array
    void * idx2_start = nullptr;          // device ptr to tier2 per-row start offsets
    void * idx4_start = nullptr;          // device ptr to tier4 per-row start offsets
    void * idx8_start = nullptr;          // device ptr to tier8 per-row start offsets
    int32_t rows = 0;
    int32_t cols = 0;
    int64_t n_fp16 = 0;                   // number of fp16 overrides
    bool is_valid = false;                // flag indicating successful allocation
};

struct tensor_upload {
    ggml_tensor * tensor = nullptr;
    const void * host_data = nullptr;
};

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

    // Note: device-side packed pointers are now kept in the runtime registry
    // keyed by the matrix base name. Do not store device pointers here.

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

    // Optional memory-mapped GGUF backing (Linux).
    void * gguf_mmap = nullptr;
    size_t gguf_mmap_size = 0;
};

// Runtime context that holds allocation contexts and runtime params (KV cache, etc.)
struct bmo_context {
    // GGML contexts
    ggml_context * kv_ctx = nullptr;   // dedicated KV cache context

    // Per-matrix kernel launch info: collected during graph building, executed post-graph
    struct kernel_launch_info {
        ggml_tensor * W = nullptr;                   // graph tensor whose storage is materialized at execution time
        device_packed_t * dp = nullptr;              // persistent device-side packed metadata
        int32_t rows = 0, cols = 0;
        int32_t n_2bit_bytes = 0, n_4bit_bytes = 0, n_8bit_bytes = 0;
        float scale_low = 1.0f, scale_int4 = 1.0f, scale_int8 = 1.0f;
        float zp_low = 1.5f, zp_int4 = 7.5f, zp_int8 = 127.5f;
    };
    std::vector<kernel_launch_info> pending_kernel_launches;

    // CUDA backend (if available)
    void * cuda_backend = nullptr;     // ggml_backend_t (opaque, to avoid ggml-backend.h)
    void * kv_backend_buffer = nullptr; // ggml_backend_buffer_t for KV tensors
    void * cuda_unpack_scratch = nullptr; // float* device scratch for unpacked matrix
    size_t cuda_unpack_scratch_bytes = 0;
    void * current_execution_buffer = nullptr; // ggml_backend_buffer_t for active graph

    // Registry mapping a matrix base name (e.g. "transformer_layers_0_self_attn_in_proj_weight")
    // to device-side packed metadata allocated by bmo_prepare_device_packed_tensors.
    std::unordered_map<std::string, device_packed_t> packed_registry;

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
    struct owned_tensor_upload {
        ggml_tensor * tensor = nullptr;
        std::vector<uint8_t> bytes;
    };
    std::vector<owned_tensor_upload> graph_uploads;
};

// Loader and allocator APIs
void bmo_load_model(const char * fname, bmo_model & model, bmo_context & ctx);
void bmo_init_kv_cache(bmo_context & ctx, int32_t n_ctx);
void bmo_prepare_device_packed_tensors(bmo_model & model, bmo_context & ctx);
void bmo_free_cuda_resources(bmo_context & ctx);

// Compute graph builder
struct ggml_cgraph * bmo_build_temporal_graph(
    bmo_context & ctx,
    bmo_model & model,
    struct ggml_tensor * input_tokens,
    int n_past,
    int layer_begin = 0,
    int layer_end = -1);

struct ggml_cgraph * bmo_build_depth_graph(
    bmo_context & ctx,
    bmo_model & model,
    struct ggml_tensor * temporal_out,
    struct ggml_tensor * text_tokens,
    struct ggml_tensor * audio_tokens,
    int codebook_step,
    int n_past);

void bmo_execute_graph(bmo_context & ctx, struct ggml_cgraph * gf, const std::vector<tensor_upload> & inputs = {});
