// bmo.cpp - model loader and KV cache allocator

#include "bmo.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <unordered_set>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef BMO_ENABLE_CUDA
#include <cuda_runtime.h>
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif

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
        L.ffn_in, L.ffn_out, L.norm1_weight, L.norm2_weight
    };
    for (auto * t : toks) {
        add_tensor_bytes_unique(t, seen, total_bytes);
    }
}

void bmo_load_model(const char * fname, bmo_model & model, bmo_context & ctx) {
    // Load GGUF and obtain data ggml_context
    ggml_context * data_ctx = nullptr;
#ifdef _WIN32
    gguf_init_params params = { /* no_alloc */ false, /* ctx */ &data_ctx };
    gguf_context * gctx = gguf_init_from_file(fname, params);
#else
    gguf_init_params params = { /* no_alloc */ true, /* ctx */ &data_ctx };
    gguf_context * gctx = gguf_init_from_file(fname, params);
#endif
    if (!gctx) {
        throw std::runtime_error(std::string("Failed to open GGUF: ") + fname);
    }
    if (!data_ctx) {
        throw std::runtime_error("GGUF: data context not returned");
    }

    model.gctx = gctx;
    model.wctx = data_ctx;

#ifndef _WIN32
    {
        int fd = ::open(fname, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(std::string("Failed to open GGUF for mmap: ") + fname);
        }
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error(std::string("Failed to stat GGUF: ") + fname);
        }
        const size_t file_size = (size_t) st.st_size;
        void * base = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) {
            throw std::runtime_error(std::string("Failed to mmap GGUF: ") + fname);
        }

        const size_t data_offset = gguf_get_data_offset(gctx);
        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        for (int64_t tid = 0; tid < n_tensors; ++tid) {
            const char * name = gguf_get_tensor_name(gctx, tid);
            ggml_tensor * t = ggml_get_tensor(data_ctx, name);
            if (!t) {
                continue;
            }
            const size_t offs = gguf_get_tensor_offset(gctx, tid);
            t->data = (uint8_t *) base + data_offset + offs;
        }

        model.gguf_mmap = base;
        model.gguf_mmap_size = file_size;
    }
#endif

    // Hardcoded Moshi 5.8B architecture.
    ctx.n_layers = 32;
    ctx.n_heads = 32;
    ctx.n_embd = 4096;
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

        // Map learned RMSNorm scale weights (norm1 = attention pre-norm, norm2 = FFN pre-norm)
        L.norm1_weight = ggml_get_tensor(data_ctx, (prefix + "_norm1_weight").c_str());
        L.norm2_weight = ggml_get_tensor(data_ctx, (prefix + "_norm2_weight").c_str());
    }

    int64_t max_scratch = 0;
    for (int i = 0; i < ctx.n_layers; ++i) {
        std::string prefix = "transformer_layers_" + std::to_string(i);
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
        for (const auto & b : bases) {
            int32_t rows = read_scalar_i32(data_ctx, (b + ".rows").c_str(), 0);
            if (rows <= 0) rows = read_scalar_i32(data_ctx, (b + ".out_features").c_str(), 0);
            int32_t cols = read_scalar_i32(data_ctx, (b + ".cols").c_str(), 0);
            if (rows > 0 && cols > 0) {
                int64_t total = (int64_t) rows * (int64_t) cols;
                if (total > max_scratch) max_scratch = total;
            }
        }
    }
    if (max_scratch > 0) {
        ctx.shared_scratch_w.resize((size_t) max_scratch);
        std::cout << "[bmo_load_model] Dynamically allocated shared_scratch_w: " << (max_scratch * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";
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

        L.norm1_weight = ggml_get_tensor(data_ctx, (prefix + "_norm1_weight").c_str());
        if (!L.norm1_weight) {
            L.norm1_weight = ggml_get_tensor(data_ctx, ("depformer.layers." + idx + ".norm1.alpha").c_str());
        }
        L.norm2_weight = ggml_get_tensor(data_ctx, (prefix + "_norm2_weight").c_str());
        if (!L.norm2_weight) {
            L.norm2_weight = ggml_get_tensor(data_ctx, ("depformer.layers." + idx + ".norm2.alpha").c_str());
        }
    }

    // Audio codebook embeddings and depformer input projections (16 each)
    model.audio_embs.assign((size_t) 16, nullptr);
    model.depformer_in.assign((size_t) 16, nullptr);
    for (int i = 0; i < 16; ++i) {
        std::string idx = std::to_string(i);
        model.audio_embs[(size_t) i] = ggml_get_tensor(data_ctx, ("depformer_emb." + idx + ".weight").c_str());
        model.depformer_in[(size_t) i] = ggml_get_tensor(data_ctx, ("depformer_in." + idx + ".weight").c_str());
    }

    // Text embedding and projection
    model.text_emb = ggml_get_tensor(data_ctx, "depformer_text_emb.weight");
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
    
    // If no temporal packed artifacts were discovered during mapping, print
    // a short listing of GGUF tensor names to aid debugging of naming variants.
    int found_packed = 0;
    for (const auto & L : model.temporal_layers) if (L.packed_weights) ++found_packed;
    if (found_packed == 0) {
        std::cerr << "[bmo_load_model] Warning: no packed_weights mapped for any temporal layer\n";
        if (gctx) {
            const int64_t n_tensors = gguf_get_n_tensors(gctx);
            std::cerr << "[bmo_load_model] GGUF tensor count=" << n_tensors << ". Listing first 200 names:\n";
            for (int64_t tid = 0; tid < n_tensors && tid < 200; ++tid) {
                const char * name = gguf_get_tensor_name(gctx, tid);
                if (name) std::cerr << "  " << tid << ": " << name << "\n";
            }
        }
    }

    // Allocate and transfer packed tensors to GPU if CUDA is enabled
    bmo_prepare_device_packed_tensors(model, ctx);
    
    std::cout << "[bmo_load_model] n_layers=" << ctx.n_layers << " n_heads=" << ctx.n_heads << " n_embd=" << ctx.n_embd << " n_ctx=" << ctx.n_ctx << "\n";
    std::cout << "[bmo_load_model] Total weight bytes: " << (double) total_bytes / (1024.0 * 1024.0) << " MB\n";
}

#ifdef BMO_ENABLE_CUDA
// Forward declaration for CUDA helper
extern "C" {
    void launch_unpack_kernel(
        const void * packed_weights,
        const void * packed_mask,
        const void * fp16_indices,
        const void * fp16_values,
        const int * idx2_start,
        const int * idx4_start,
        const int * idx8_start,
        int rows,
        int cols,
        int64_t n_fp16,
        int n_2bit_bytes,
        int n_4bit_bytes,
        int n_8bit_bytes,
        float scale_low,
        float scale_int4,
        float scale_int8,
        float zp_low,
        float zp_int4,
        float zp_int8,
        float * out_w);
}
#endif

void bmo_prepare_device_packed_tensors(bmo_model & model, bmo_context & ctx) {
#ifndef BMO_ENABLE_CUDA
    std::cerr << "[bmo_prepare_device_packed_tensors] CUDA not enabled; skipping GPU allocation\n";
    return;
#endif

#ifdef BMO_ENABLE_CUDA
    if (!ctx.cuda_backend) {
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) {
            std::cerr << "[bmo_prepare_device_packed_tensors] failed to initialize CUDA backend; skipping\n";
            return;
        }
        ctx.cuda_backend = backend;
    }

    // Quick inventory: count layers that have packed artifacts mapped
    int total_layers = (int) model.temporal_layers.size();
    int have_packed = 0;
    for (size_t _i = 0; _i < model.temporal_layers.size(); ++_i) {
        if (model.temporal_layers[_i].packed_weights) ++have_packed;
    }
    std::cout << "[bmo_prepare_device_packed_tensors] layers=" << total_layers << " have_packed=" << have_packed << "\n";

    // For each temporal layer, allocate and transfer packed tensors to device
    for (size_t i = 0; i < model.temporal_layers.size(); ++i) {
        bmo_layer & layer = model.temporal_layers[i];
        
        // Skip if no packed weights (dense-only layer)
        if (!layer.packed_weights || !layer.packed_mask || !layer.fp16_indices || !layer.fp16_values) {
            continue;
        }
        
        // Extract metadata from existing layer tensors
        // For rows: use packed_mask dimensions (it should have the shape info we need)
        // packed_mask is uint8 with 4 uint2 quants per output channel (row)
        int32_t rows = (int32_t) layer.packed_mask->ne[0];  // First dimension of packed_mask
        int32_t cols = (int32_t) layer.packed_weights->ne[0]; // packed_weights is 1D
        
        if (rows <= 0 || cols <= 0) {
            std::cerr << "[bmo_prepare_device_packed_tensors] Warning: invalid dims for layer " << i 
                      << " (rows=" << rows << " cols=" << cols << "); skipping\n";
            continue;
        }
        
        // Byte counts derived from packed layout
        int32_t n_2bit_bytes = (int32_t) (layer.packed_mask->ne[0] * layer.packed_mask->ne[1]) / 4;  // 2 bits per element
        int32_t n_4bit_bytes = 0;  // Not explicitly tracked in layer; derive from remaining packed_weights
        int32_t n_8bit_bytes = 0;
        
        // Read quantization scales from tensors (if available) or use defaults
        float scale_low = 1.0f;
        float scale_int4 = 1.0f;
        float scale_int8 = 1.0f;
        
        if (layer.scale_low && layer.scale_low->data) {
            scale_low = *((float*) layer.scale_low->data);
        }
        if (layer.scale_int4 && layer.scale_int4->data) {
            scale_int4 = *((float*) layer.scale_int4->data);
        }
        // Note: scale_int8 may not be available in layer structure
        
        float zp_low = 1.5f;
        float zp_int4 = 7.5f;
        float zp_int8 = 127.5f;
        
        int64_t n_fp16 = ggml_nbytes(layer.fp16_indices) / (int64_t) sizeof(int32_t);
        
        std::cout << "[bmo_prepare_device_packed_tensors] Layer " << i 
                  << ": rows=" << rows << " cols=" << cols 
                  << " n_2bit=" << n_2bit_bytes << " n_4bit=" << n_4bit_bytes 
                  << " n_8bit=" << n_8bit_bytes << " n_fp16=" << n_fp16 << "\n";
        
        // Allocate device buffers via cudaMalloc
        const uint8_t * h_pw = reinterpret_cast<const uint8_t *>(layer.packed_weights->data);
        const uint8_t * h_pm = reinterpret_cast<const uint8_t *>(layer.packed_mask->data);
        const int32_t * h_fi = reinterpret_cast<const int32_t *>(layer.fp16_indices->data);
        const ggml_fp16_t * h_fv = nullptr;
        
        // Handle fp16_values type conversion if needed
        std::vector<ggml_fp16_t> tmp_fv16;
        if (layer.fp16_values->type == GGML_TYPE_F16) {
            h_fv = reinterpret_cast<const ggml_fp16_t *>(layer.fp16_values->data);
        } else if (layer.fp16_values->type == GGML_TYPE_F32) {
            const float * src = reinterpret_cast<const float *>(layer.fp16_values->data);
            tmp_fv16.resize((size_t) n_fp16);
            for (int64_t j = 0; j < n_fp16; ++j) {
                tmp_fv16[(size_t) j] = ggml_fp32_to_fp16(src[j]);
            }
            h_fv = tmp_fv16.data();
        } else {
            std::cerr << "[bmo_prepare_device_packed_tensors] Warning: unsupported fp16_values type in layer " << i << "; skipping\n";
            continue;
        }
        
        // Compute idx*_start prefix sum arrays on host
        std::vector<int32_t> h_idx2_start, h_idx4_start, h_idx8_start;
        const uint8_t * pm_ptr = h_pm;
        
        h_idx2_start.resize((size_t) rows, 0);
        h_idx4_start.resize((size_t) rows, 0);
        h_idx8_start.resize((size_t) rows, 0);
        
        int32_t cnt2 = 0, cnt4 = 0, cnt8 = 0;
        for (int r = 0; r < rows; ++r) {
            h_idx2_start[(size_t) r] = cnt2;
            h_idx4_start[(size_t) r] = cnt4;
            h_idx8_start[(size_t) r] = cnt8;
            
            for (int c = 0; c < cols; ++c) {
                // Peek at tier for this (r, c) element to build prefix sum
                int byte_idx = r * cols + c; // or based on packed storage layout
                uint8_t tier = (pm_ptr[byte_idx / 4] >> ((byte_idx % 4) * 2)) & 0x3;
                if (tier == 0) cnt2 += 1;
                else if (tier == 1) cnt4 += 1;
                else if (tier == 2) cnt8 += 1;
            }
        }
        
        // CudaMalloc device buffers
        void * d_pw = nullptr, * d_pm = nullptr, * d_fi = nullptr, * d_fv = nullptr;
        void * d_i2s = nullptr, * d_i4s = nullptr, * d_i8s = nullptr;
        
        size_t pw_bytes = (size_t) ggml_nbytes(layer.packed_weights);
        size_t pm_bytes = (size_t) ggml_nbytes(layer.packed_mask);
        size_t fi_bytes = (size_t) ggml_nbytes(layer.fp16_indices);
        size_t fv_bytes = (size_t) ggml_nbytes(layer.fp16_values);
        
        cudaError_t err = cudaMalloc(&d_pw, pw_bytes);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc packed_weights failed: " << cudaGetErrorString(err) << "\n";
            return;
        }
        
        err = cudaMalloc(&d_pm, pm_bytes);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc packed_mask failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_pw);
            return;
        }
        
        err = cudaMalloc(&d_fi, fi_bytes);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc fp16_indices failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_pw);
            cudaFree(d_pm);
            return;
        }
        
        err = cudaMalloc(&d_fv, fv_bytes);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc fp16_values failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_pw);
            cudaFree(d_pm);
            cudaFree(d_fi);
            return;
        }
        
        err = cudaMalloc(&d_i2s, (size_t) rows * sizeof(int32_t));
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc idx2_start failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_pw);
            cudaFree(d_pm);
            cudaFree(d_fi);
            cudaFree(d_fv);
            return;
        }
        
        err = cudaMalloc(&d_i4s, (size_t) rows * sizeof(int32_t));
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc idx4_start failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_pw);
            cudaFree(d_pm);
            cudaFree(d_fi);
            cudaFree(d_fv);
            cudaFree(d_i2s);
            return;
        }
        
        err = cudaMalloc(&d_i8s, (size_t) rows * sizeof(int32_t));
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc idx8_start failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_pw);
            cudaFree(d_pm);
            cudaFree(d_fi);
            cudaFree(d_fv);
            cudaFree(d_i2s);
            cudaFree(d_i4s);
            return;
        }
        
        // CudaMemcpy host -> device
        err = cudaMemcpy(d_pw, h_pw, pw_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy packed_weights failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_pw);
            cudaFree(d_pm);
            cudaFree(d_fi);
            cudaFree(d_fv);
            cudaFree(d_i2s);
            cudaFree(d_i4s);
            cudaFree(d_i8s);
            return;
        }
        
        err = cudaMemcpy(d_pm, h_pm, pm_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy packed_mask failed: " << cudaGetErrorString(err) << "\n";
            goto cleanup_after_pm_fail;
        }
        
        err = cudaMemcpy(d_fi, h_fi, fi_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy fp16_indices failed: " << cudaGetErrorString(err) << "\n";
            goto cleanup_after_fi_fail;
        }
        
        err = cudaMemcpy(d_fv, h_fv, fv_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy fp16_values failed: " << cudaGetErrorString(err) << "\n";
            goto cleanup_after_fv_fail;
        }
        
        err = cudaMemcpy(d_i2s, h_idx2_start.data(), (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy idx2_start failed: " << cudaGetErrorString(err) << "\n";
            goto cleanup_after_i2s_fail;
        }
        
        err = cudaMemcpy(d_i4s, h_idx4_start.data(), (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy idx4_start failed: " << cudaGetErrorString(err) << "\n";
            goto cleanup_after_i4s_fail;
        }
        
        err = cudaMemcpy(d_i8s, h_idx8_start.data(), (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy idx8_start failed: " << cudaGetErrorString(err) << "\n";
            goto cleanup_after_i8s_fail;
        }
        
        // Store device pointers in layer's device_packed struct
        layer.device_packed.packed_weights = d_pw;
        layer.device_packed.packed_mask = d_pm;
        layer.device_packed.fp16_indices = d_fi;
        layer.device_packed.fp16_values = d_fv;
        layer.device_packed.idx2_start = d_i2s;
        layer.device_packed.idx4_start = d_i4s;
        layer.device_packed.idx8_start = d_i8s;
        layer.device_packed.n_fp16 = n_fp16;
        layer.device_packed.is_valid = true;
        
        std::cout << "[bmo_prepare_device_packed_tensors] Layer " << i << " GPU allocation successful\n";
        continue;
        
        // Cleanup label hierarchy for error handling
        cleanup_after_i8s_fail:
            cudaFree(d_i8s);
        cleanup_after_i4s_fail:
            cudaFree(d_i4s);
        cleanup_after_i2s_fail:
            cudaFree(d_i2s);
        cleanup_after_fv_fail:
            cudaFree(d_fv);
        cleanup_after_fi_fail:
            cudaFree(d_fi);
        cleanup_after_pm_fail:
            cudaFree(d_pm);
            cudaFree(d_pw);
    }
#endif
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
