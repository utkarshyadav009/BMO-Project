// bmo.cpp - model loader and KV cache allocator

#include "bmo.h"

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <unistd.h>
#include <vector>

#ifdef BMO_ENABLE_CUDA
#include <cuda_runtime.h>
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif

void bmo_print_mem_diag(const std::string & phase) {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long mem_avail = 0;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::sscanf(line.c_str(), "MemAvailable: %ld kB", &mem_avail);
            break;
        }
    }

    std::ifstream status("/proc/self/status");
    long vm_rss = 0, vm_lck = 0;
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &vm_rss);
        } else if (line.find("VmLck:") == 0) {
            std::sscanf(line.c_str(), "VmLck: %ld kB", &vm_lck);
        }
    }
    std::fprintf(stderr,
                 "[mem_diag] %-20s | MemAvail: %4ld MB | VmRSS: %4ld MB | VmLck: %4ld MB\n",
                 phase.c_str(),
                 mem_avail / 1024,
                 vm_rss / 1024,
                 vm_lck / 1024);
}

namespace {

static ggml_tensor * get_tensor(ggml_context * data_ctx, const std::string & name) {
    return ggml_get_tensor(data_ctx, name.c_str());
}

static int32_t read_scalar_i32(ggml_context * data_ctx, const char * name, int32_t fallback = 0) {
    ggml_tensor * t = get_tensor(data_ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(int32_t)) {
        return fallback;
    }
    int32_t out = fallback;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

static void add_tensor_bytes_unique(ggml_tensor * t, std::unordered_set<const void *> & seen, size_t & total_bytes) {
    if (!t || !t->data) {
        return;
    }
    const void * key = t->data;
    if (seen.insert(key).second) {
        total_bytes += (size_t) ggml_nbytes(t);
    }
}

static void add_layer_bytes_unique(const bmo_layer & L, std::unordered_set<const void *> & seen, size_t & total_bytes) {
    add_tensor_bytes_unique(L.packed_weights, seen, total_bytes);
    add_tensor_bytes_unique(L.packed_mask, seen, total_bytes);
    add_tensor_bytes_unique(L.scale_low, seen, total_bytes);
    add_tensor_bytes_unique(L.scale_int4, seen, total_bytes);
    add_tensor_bytes_unique(L.scale_int8, seen, total_bytes);
    add_tensor_bytes_unique(L.fp16_indices, seen, total_bytes);
    add_tensor_bytes_unique(L.fp16_values, seen, total_bytes);
    add_tensor_bytes_unique(L.weight, seen, total_bytes);
    add_tensor_bytes_unique(L.bias, seen, total_bytes);
    add_tensor_bytes_unique(L.wq, seen, total_bytes);
    add_tensor_bytes_unique(L.wk, seen, total_bytes);
    add_tensor_bytes_unique(L.wv, seen, total_bytes);
    add_tensor_bytes_unique(L.wo, seen, total_bytes);
    add_tensor_bytes_unique(L.ffn_in, seen, total_bytes);
    add_tensor_bytes_unique(L.ffn_out, seen, total_bytes);
    add_tensor_bytes_unique(L.norm1_weight, seen, total_bytes);
    add_tensor_bytes_unique(L.norm2_weight, seen, total_bytes);
}

#ifdef BMO_ENABLE_CUDA
static float read_scalar_f32_gguf(ggml_context * ctx, const std::string & name, float fallback = 0.0f) {
    ggml_tensor * t = get_tensor(ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) {
        return fallback;
    }
    float out = fallback;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

static void free_device_packed_owned_buffers(device_packed_t & dp) {
#ifdef BMO_JETSON
    // canonical_base_host is freed in bmo_free_cuda_resources (needs access to ctx).
#endif
#ifndef BMO_JETSON
    if (dp.packed_weights) cudaFree(dp.packed_weights);
    if (dp.packed_mask) cudaFree(dp.packed_mask);
    if (dp.fp16_values) cudaFree(dp.fp16_values);
    if (dp.fp16_indices) cudaFree(dp.fp16_indices);
    if (dp.block_offset) cudaFree(dp.block_offset);
#endif
    dp = device_packed_t{};
}
#endif

} // namespace

void bmo_load_model(const char * fname, bmo_model & model, bmo_context & ctx) {
    ctx.streaming_big_pool = nullptr;
    ctx.streaming_big_pool_size = 0;
    ctx.streaming_big_pool_registered = false;
    ctx.streaming_scalar_pool = nullptr;
    ctx.streaming_scalar_pool_size = 0;

    // 1. Init without mmap
    ggml_context * data_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ &data_ctx,
    };

    gguf_context * gctx = gguf_init_from_file(fname, params);
    if (!gctx || !data_ctx) {
        throw std::runtime_error("Failed to parse GGUF");
    }
    model.gctx = gctx;
    model.wctx = data_ctx;

    // 2. Open file for reading
    int fd = open(fname, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("Failed to open GGUF");
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);

    // 3. Measure pools
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    const size_t SCALAR_MAX = 4096;
    const size_t ALIGN = 64;
    auto round_up = [](size_t x, size_t a) -> size_t { return (x + a - 1) & ~(a - 1); };

    size_t scalar_total = 0;
    size_t big_total = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(data_ctx, gguf_get_tensor_name(gctx, i));
        if (!t) continue;
        size_t nb = (size_t) ggml_nbytes(t);
        if (nb <= SCALAR_MAX) scalar_total += round_up(nb, ALIGN);
        else                  big_total += round_up(nb, ALIGN);
    }

    // 4. Allocate pools
    void * scalar_base = nullptr;
    if (scalar_total > 0) {
        if (posix_memalign(&scalar_base, ALIGN, scalar_total) != 0) {
            close(fd);
            throw std::runtime_error("Failed to allocate scalar pool");
        }
    }

    void * big_base = nullptr;
    if (big_total > 0) {
        const size_t PAGE = 4096;
        size_t big_aligned = round_up(big_total, PAGE);
        if (posix_memalign(&big_base, PAGE, big_aligned) != 0) {
            close(fd);
            if (scalar_base) std::free(scalar_base);
            throw std::runtime_error("Failed to allocate big pool");
        }
#ifdef BMO_JETSON
        if (cudaHostRegister(big_base, big_aligned, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
            ctx.streaming_big_pool_registered = true;
        } else {
            close(fd);
            if (scalar_base) std::free(scalar_base);
            std::free(big_base);
            throw std::runtime_error("cudaHostRegister failed for big pool");
        }
#endif
        ctx.streaming_big_pool = big_base;
        ctx.streaming_big_pool_size = big_aligned;
    }
    ctx.streaming_scalar_pool = scalar_base;
    ctx.streaming_scalar_pool_size = scalar_total;

    // 5. Read data from disk
    const size_t data_offset = gguf_get_data_offset(gctx);
    size_t scalar_used = 0;
    size_t big_used = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(data_ctx, gguf_get_tensor_name(gctx, i));
        if (!t) continue;

        size_t nb = (size_t) ggml_nbytes(t);
        size_t aligned_nb = round_up(nb, ALIGN);
        off_t file_off = (off_t) (data_offset + gguf_get_tensor_offset(gctx, i));

        void * dst = nullptr;
        if (nb <= SCALAR_MAX) {
            dst = (uint8_t *) scalar_base + scalar_used;
            scalar_used += aligned_nb;
        } else {
            dst = (uint8_t *) big_base + big_used;
            big_used += aligned_nb;
        }

        size_t remaining = nb;
        uint8_t * out = (uint8_t *) dst;
        while (remaining > 0) {
            ssize_t r = pread(fd, out, remaining, file_off);
            if (r <= 0) {
                close(fd);
                throw std::runtime_error("pread failed while loading GGUF tensor payload");
            }
            out += (size_t) r;
            file_off += r;
            remaining -= (size_t) r;
        }
        t->data = dst; // Patch tensor
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);

    ctx.n_layers = read_scalar_i32(data_ctx, "n_layers", 0);
    if (ctx.n_layers <= 0) ctx.n_layers = read_scalar_i32(data_ctx, "n_layer", 32);

    ctx.n_heads = read_scalar_i32(data_ctx, "n_heads", 0);
    if (ctx.n_heads <= 0) ctx.n_heads = read_scalar_i32(data_ctx, "n_head", 0);

    ctx.n_embd = read_scalar_i32(data_ctx, "n_embd", 0);
    if (ctx.n_embd <= 0) ctx.n_embd = read_scalar_i32(data_ctx, "hidden_size", 0);

    ctx.n_ctx = read_scalar_i32(data_ctx, "n_ctx", 0);
    if (ctx.n_ctx <= 0) ctx.n_ctx = read_scalar_i32(data_ctx, "context_length", 2048);

    ctx.head_dim = read_scalar_i32(data_ctx, "head_dim", 0);
    if (ctx.head_dim <= 0) ctx.head_dim = read_scalar_i32(data_ctx, "n_embd_head_k", 0);

    // Infer missing temporal dimensions from packed QKV metadata in layer 0.
    {
        const std::string qkv0 = "transformer_layers_0_self_attn_in_proj_weight";
        const int32_t qkv_rows = read_scalar_i32(data_ctx, (qkv0 + ".rows").c_str(), 0);
        const int32_t qkv_cols = read_scalar_i32(data_ctx, (qkv0 + ".cols").c_str(), 0);

        if (ctx.n_embd <= 0 && qkv_cols > 0) {
            ctx.n_embd = qkv_cols;
        }
        if (ctx.n_embd <= 0 && qkv_rows > 0 && (qkv_rows % 3) == 0) {
            ctx.n_embd = qkv_rows / 3;
        }
        if (ctx.head_dim <= 0 && ctx.n_heads > 0 && ctx.n_embd > 0) {
            ctx.head_dim = ctx.n_embd / ctx.n_heads;
        }
        if (ctx.n_heads <= 0 && ctx.head_dim > 0 && ctx.n_embd > 0 && (ctx.n_embd % ctx.head_dim) == 0) {
            ctx.n_heads = ctx.n_embd / ctx.head_dim;
        }
    }

    if (ctx.n_heads <= 0) ctx.n_heads = 32;
    if (ctx.n_embd <= 0) ctx.n_embd = 4096;
    if (ctx.head_dim <= 0 && ctx.n_heads > 0) {
        ctx.head_dim = ctx.n_embd / ctx.n_heads;
    }

    model.temporal_layers.resize((size_t) ctx.n_layers);
    for (int i = 0; i < ctx.n_layers; ++i) {
        auto & layer = model.temporal_layers[(size_t) i];
        std::string base = "transformer_layers_" + std::to_string(i);
        layer.name = base;
        layer.packed_weights = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.packed_weights").c_str());
        layer.packed_mask = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.packed_mask").c_str());
        layer.fp16_indices = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.fp16_indices").c_str());
        layer.fp16_values = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.fp16_values").c_str());
        layer.norm1_weight = ggml_get_tensor(data_ctx, (base + "_attn_norm.weight").c_str());
        layer.norm2_weight = ggml_get_tensor(data_ctx, (base + "_ffn_norm.weight").c_str());
        if (!layer.norm1_weight) layer.norm1_weight = ggml_get_tensor(data_ctx, (base + "_norm1.weight").c_str());
        if (!layer.norm2_weight) layer.norm2_weight = ggml_get_tensor(data_ctx, (base + "_norm2.weight").c_str());
    }

    constexpr int kDepthLayers = 6;
    model.depth_layers.resize(kDepthLayers);
    for (int i = 0; i < kDepthLayers; ++i) {
        auto & layer = model.depth_layers[(size_t) i];
        std::string base = "depformer_layers_" + std::to_string(i);
        layer.name = base;
        layer.norm1_weight = ggml_get_tensor(data_ctx, ("depformer.layers." + std::to_string(i) + ".norm1.weight").c_str());
        layer.norm2_weight = ggml_get_tensor(data_ctx, ("depformer.layers." + std::to_string(i) + ".norm2.weight").c_str());
        if (!layer.norm1_weight) layer.norm1_weight = ggml_get_tensor(data_ctx, (base + "_norm1_weight").c_str());
        if (!layer.norm2_weight) layer.norm2_weight = ggml_get_tensor(data_ctx, (base + "_norm2_weight").c_str());
    }

    constexpr int kDepformerCodebooks = 16;
    model.audio_embs.resize(kDepformerCodebooks, nullptr);
    model.depformer_in.resize(kDepformerCodebooks, nullptr);
    for (int i = 0; i < kDepformerCodebooks; ++i) {
        std::string idx = std::to_string(i);
        model.audio_embs[(size_t) i] = ggml_get_tensor(data_ctx, ("depformer_emb." + idx + ".weight").c_str());
        model.depformer_in[(size_t) i] = ggml_get_tensor(data_ctx, ("depformer_in." + idx + ".weight").c_str());
    }

    model.text_emb = ggml_get_tensor(data_ctx, "depformer_text_emb.weight");
    model.text_linear = ggml_get_tensor(data_ctx, "text_linear.weight");
    model.token_embedding = ggml_get_tensor(data_ctx, "token_embedding");
    model.output_head = ggml_get_tensor(data_ctx, "output_head");

    // NOTE: token_embedding can belong to a different module width than temporal
    // transformer blocks; do not overwrite temporal n_embd from that tensor.

    size_t total_bytes = 0;
    std::unordered_set<const void *> seen;
    for (const auto & L : model.temporal_layers) add_layer_bytes_unique(L, seen, total_bytes);
    for (const auto & L : model.depth_layers) add_layer_bytes_unique(L, seen, total_bytes);
    for (auto * t : model.audio_embs) add_tensor_bytes_unique(t, seen, total_bytes);
    for (auto * t : model.depformer_in) add_tensor_bytes_unique(t, seen, total_bytes);
    add_tensor_bytes_unique(model.text_emb, seen, total_bytes);
    add_tensor_bytes_unique(model.text_linear, seen, total_bytes);
    add_tensor_bytes_unique(model.token_embedding, seen, total_bytes);
    add_tensor_bytes_unique(model.output_head, seen, total_bytes);
    ctx.weights_bytes = total_bytes;

    std::cout << "[bmo_load_model] Loaded model '" << fname << "'\n";
    bmo_prepare_device_packed_tensors(model, ctx);
    std::cout << "[bmo_load_model] n_layers=" << ctx.n_layers << " n_heads=" << ctx.n_heads << " n_embd=" << ctx.n_embd << " n_ctx=" << ctx.n_ctx << "\n";
    std::cout << "[bmo_load_model] Total weight bytes: " << (double) total_bytes / (1024.0 * 1024.0) << " MB\n";
}

void bmo_prepare_device_packed_tensors(bmo_model & model, bmo_context & ctx) {
#ifndef BMO_ENABLE_CUDA
    std::cerr << "[bmo_prepare_device_packed_tensors] CUDA not enabled; skipping GPU allocation\n";
    return;
#endif

#ifdef BMO_ENABLE_CUDA
    bmo_print_mem_diag("Start Prepare");
    size_t max_unpack_elems = 0;

    // Rebuild packed registry if called repeatedly; do not free streaming pools here.
    for (auto & kv : ctx.packed_registry) {
        free_device_packed_owned_buffers(kv.second);
    }
    ctx.packed_registry.clear();

    if (!ctx.cuda_backend) {
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) {
            std::cerr << "[bmo_prepare_device_packed_tensors] failed to initialize CUDA backend; skipping\n";
            return;
        }
        ctx.cuda_backend = backend;
    }

#ifdef BMO_JETSON
    // Opt-in to 96KB shared memory for our V1 optimized kernel.
    bmo_cuda_configure_fused_kernel_v1();
    bmo_print_mem_diag("After Stream Alloc");
#endif

    // Process all packed temporal matrices for each temporal layer.
    for (int i = 0; i < ctx.n_layers; ++i) {
        std::string prefix = "transformer_layers_" + std::to_string(i);
        std::vector<std::string> matrices = {
            prefix + "_self_attn_in_proj_weight",
            prefix + "_self_attn_out_proj_weight",
            prefix + "_gating_linear_in_weight",
            prefix + "_gating_linear_out_weight"
        };

        for (const std::string & base : matrices) {
            ggml_tensor * pw = ggml_get_tensor(model.wctx, (base + ".packed_weights").c_str());
            if (!pw) continue; // Not packed for this matrix

            ggml_tensor * pm = ggml_get_tensor(model.wctx, (base + ".packed_mask").c_str());
            ggml_tensor * fv = ggml_get_tensor(model.wctx, (base + ".fp16_values").c_str());

            // 1. Read exact dimensions from GGUF scalars
            int32_t rows = read_scalar_i32(model.wctx, (base + ".rows").c_str(), 0);
            if (rows <= 0) rows = read_scalar_i32(model.wctx, (base + ".out_features").c_str(), 0);
            int32_t cols = read_scalar_i32(model.wctx, (base + ".cols").c_str(), 0);

            if (rows <= 0 || cols <= 0) {
                std::cerr << "[bmo_prepare_device_packed_tensors] Invalid dims for " << base << "\n";
                continue;
            }

            int32_t block_size = read_scalar_i32(model.wctx, (base + ".block_size").c_str(), 0);
            int32_t n_blocks = read_scalar_i32(model.wctx, (base + ".n_blocks").c_str(), 0);
            if (block_size <= 0) block_size = 32;
            if (n_blocks <= 0) n_blocks = (rows * cols + block_size - 1) / block_size;

            if (!pm || !fv) {
                std::cerr << "[bmo_prepare_device_packed_tensors] Missing block-wise packed tensors for " << base << "; skipping\n";
                continue;
            }

            int64_t n_fp16 = 0;
            if (fv->type == GGML_TYPE_F16) {
                n_fp16 = ggml_nbytes(fv) / sizeof(ggml_fp16_t);
            } else if (fv->type == GGML_TYPE_F32) {
                n_fp16 = ggml_nbytes(fv) / sizeof(float);
            } else {
                std::cerr << "[bmo_prepare_device_packed_tensors] Unsupported fp16_values type for " << base << "; skipping\n";
                continue;
            }

            // 3. Allocate and Copy to CUDA
            device_packed_t dp;
            dp.rows = rows;
            dp.cols = cols;
            dp.block_size = block_size;
            dp.n_blocks = n_blocks;
            dp.n_fp16 = n_fp16;
            dp.is_blockwise = true;

            size_t pw_bytes = (size_t) ggml_nbytes(pw);
            size_t pm_bytes = (size_t) ggml_nbytes(pm);

#ifdef BMO_JETSON
            const int32_t n_blocks_from_mask = static_cast<int32_t>(pm_bytes * 4);
            n_blocks = n_blocks_from_mask;
            dp.n_blocks = n_blocks_from_mask;
#else
            const uint8_t * pm_host = reinterpret_cast<const uint8_t *>(pm->data);
            std::vector<int32_t> block_offset((size_t) n_blocks, 0);
            int32_t c2 = 0;
            int32_t c4 = 0;
            int32_t c8 = 0;
            int32_t c16 = 0;
            for (int32_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
                const uint8_t mbyte = pm_host[(size_t) block_idx / 4];
                const uint8_t tier = unpack_u2_le(mbyte, block_idx % 4);
                if (tier == 0) {
                    block_offset[(size_t) block_idx] = c16;
                    c16 += block_size;
                } else if (tier == 1) {
                    block_offset[(size_t) block_idx] = c8;
                    c8 += block_size;
                } else if (tier == 2) {
                    block_offset[(size_t) block_idx] = c4;
                    c4 += block_size;
                } else {
                    block_offset[(size_t) block_idx] = c2;
                    c2 += block_size;
                }
            }
#endif

            cudaError_t err = cudaSuccess;

#ifdef BMO_JETSON
            if (fv->type == GGML_TYPE_F32) {
                std::cerr << "[bmo_prepare_device_packed_tensors] Jetson fused path requires fp16_values to be F16 for " << base << "\n";
                continue;
            }
            dp.host_packed_weights = pw->data;
            dp.pw_size = pw_bytes;
            dp.host_packed_mask = pm->data;
            dp.pm_size = pm_bytes;
            dp.host_fp16_values = fv->data;
            dp.fv_size = (size_t) ggml_nbytes(fv);

            dp.n_2bit_bytes = read_scalar_i32(model.wctx, (base + ".n_2bit_bytes").c_str(), 0);
            dp.n_4bit_bytes = read_scalar_i32(model.wctx, (base + ".n_4bit_bytes").c_str(), 0);
            dp.n_8bit_bytes = read_scalar_i32(model.wctx, (base + ".n_8bit_bytes").c_str(), 0);
            dp.scale_low = read_scalar_f32_gguf(model.wctx, base + ".scale_low", 1.0f);
            dp.scale_int4 = read_scalar_f32_gguf(model.wctx, base + ".scale_int4", 1.0f);
            dp.scale_int8 = read_scalar_f32_gguf(model.wctx, base + ".scale_int8", 1.0f);
            dp.zp_low = read_scalar_f32_gguf(model.wctx, base + ".zp_low", 1.5f);
            dp.zp_int4 = read_scalar_f32_gguf(model.wctx, base + ".zp_int4", 7.5f);
            dp.zp_int8 = read_scalar_f32_gguf(model.wctx, base + ".zp_int8", 127.5f);

            dp.canonical_base_host = nullptr;
            dp.canonical_base = dp.host_packed_weights;
            dp.canonical_pw = dp.host_packed_weights;
            dp.canonical_pm = dp.host_packed_mask;
            dp.canonical_fv =
                dp.host_fp16_values ? reinterpret_cast<ggml_fp16_t *>(dp.host_fp16_values) : nullptr;

            void * pw_dev = nullptr;
            void * pm_dev = nullptr;
            void * fv_dev = nullptr;
            cudaError_t pw_map = cudaHostGetDevicePointer(&pw_dev, dp.host_packed_weights, 0);
            cudaError_t pm_map = cudaHostGetDevicePointer(&pm_dev, dp.host_packed_mask, 0);
            cudaError_t fv_map = cudaSuccess;
            if (dp.host_fp16_values) {
                fv_map = cudaHostGetDevicePointer(&fv_dev, dp.host_fp16_values, 0);
            }
            if (pw_map == cudaSuccess && pm_map == cudaSuccess && fv_map == cudaSuccess) {
                dp.canonical_pw_dev = pw_dev;
                dp.canonical_pm_dev = pm_dev;
                dp.canonical_fv_dev = fv_dev;
                dp.preloaded = true;
            } else {
                std::cerr << "[bmo_prepare_device_packed_tensors] cudaHostGetDevicePointer canonical map failed for "
                          << base << " pw=" << cudaGetErrorString(pw_map)
                          << " pm=" << cudaGetErrorString(pm_map)
                          << " fv=" << cudaGetErrorString(fv_map) << "\n";
                dp.preloaded = false;
            }
#else
            err = cudaMalloc(&dp.packed_weights, pw_bytes);
            if (err != cudaSuccess) { std::cerr << "cudaMalloc packed_weights failed: " << cudaGetErrorString(err) << "\n"; continue; }
            err = cudaMemcpy(dp.packed_weights, pw->data, pw_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { std::cerr << "cudaMemcpy packed_weights failed: " << cudaGetErrorString(err) << "\n"; free_device_packed_owned_buffers(dp); continue; }

            err = cudaMalloc(&dp.packed_mask, pm_bytes);
            if (err != cudaSuccess) { std::cerr << "cudaMalloc packed_mask failed: " << cudaGetErrorString(err) << "\n"; free_device_packed_owned_buffers(dp); continue; }
            err = cudaMemcpy(dp.packed_mask, pm->data, pm_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { std::cerr << "cudaMemcpy packed_mask failed: " << cudaGetErrorString(err) << "\n"; free_device_packed_owned_buffers(dp); continue; }
#endif

#ifndef BMO_JETSON
            err = cudaMalloc(&dp.block_offset, block_offset.size() * sizeof(int32_t));
            if (err != cudaSuccess) { std::cerr << "cudaMalloc block_offset failed: " << cudaGetErrorString(err) << "\n"; free_device_packed_owned_buffers(dp); continue; }
            err = cudaMemcpy(dp.block_offset, block_offset.data(), block_offset.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { std::cerr << "cudaMemcpy block_offset failed: " << cudaGetErrorString(err) << "\n"; free_device_packed_owned_buffers(dp); continue; }

            // fp16 values: may be stored as f32 in gguf; convert if needed
            if (fv->type == GGML_TYPE_F32) {
                int64_t nf = n_fp16;
                std::vector<ggml_fp16_t> tmp_f16((size_t) nf);
                const float * src = reinterpret_cast<const float *>(fv->data);
                for (int64_t j = 0; j < nf; ++j) tmp_f16[(size_t) j] = ggml_fp32_to_fp16(src[j]);
                size_t fv_bytes = (size_t) nf * sizeof(ggml_fp16_t);
                err = cudaMalloc(&dp.fp16_values, fv_bytes);
                if (err != cudaSuccess) { std::cerr << "cudaMalloc fp16_values failed: " << cudaGetErrorString(err) << "\n"; goto cleanup_partial_dp; }
                err = cudaMemcpy(dp.fp16_values, tmp_f16.data(), fv_bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) { std::cerr << "cudaMemcpy fp16_values failed: " << cudaGetErrorString(err) << "\n"; goto cleanup_partial_dp; }
            } else {
                size_t fv_bytes = (size_t) ggml_nbytes(fv);
                err = cudaMalloc(&dp.fp16_values, fv_bytes);
                if (err != cudaSuccess) { std::cerr << "cudaMalloc fp16_values failed: " << cudaGetErrorString(err) << "\n"; goto cleanup_partial_dp; }
                err = cudaMemcpy(dp.fp16_values, fv->data, fv_bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) { std::cerr << "cudaMemcpy fp16_values failed: " << cudaGetErrorString(err) << "\n"; goto cleanup_partial_dp; }
            }
#endif

            dp.is_valid = true;
            ctx.packed_registry[base] = dp;
#ifndef BMO_JETSON
            max_unpack_elems = std::max(max_unpack_elems, (size_t) rows * (size_t) cols);
#endif
            std::cout << "[bmo_prepare_device_packed_tensors] registered " << base << " rows=" << rows << " cols=" << cols << " n_fp16=" << n_fp16 << "\n";
            continue;

            cleanup_partial_dp:
                free_device_packed_owned_buffers(dp);
                continue;
        }
    }

#ifdef BMO_JETSON
    if (!ctx.packed_registry.empty()) {
        ctx.cuda_fused_output_buffer = nullptr;
        ctx.cuda_fused_output_buffer_dev = nullptr;
        ctx.cuda_fused_output_buffer_bytes = 0;
        ctx.cuda_fused_output_owns_raw_malloc = false;

        ctx.cuda_fused_input_buffer = nullptr;
        ctx.cuda_fused_input_buffer_dev = nullptr;
        ctx.cuda_fused_input_owns_raw_malloc = false;

        size_t out_bytes = 22528ULL * sizeof(float);
        out_bytes = (out_bytes + 63) & ~size_t(63);
        void * out_raw = nullptr;
        if (posix_memalign(&out_raw, 64, out_bytes) == 0 && out_raw) {
            cudaError_t out_reg =
                cudaHostRegister(out_raw, out_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (out_reg == cudaSuccess) {
                if (cudaHostGetDevicePointer(&ctx.cuda_fused_output_buffer_dev, out_raw, 0) == cudaSuccess) {
                    ctx.cuda_fused_output_buffer = out_raw;
                    ctx.cuda_fused_output_buffer_bytes = out_bytes;
                    ctx.cuda_fused_output_owns_raw_malloc = true;
                    std::cout << "[bmo_jetson] fused_output_buffer="
                              << (double) out_bytes / (1024.0 * 1024.0) << " MB pinned mapped\n";
                } else {
                    cudaHostUnregister(out_raw);
                    std::free(out_raw);
                    std::cerr << "[bmo_jetson] cudaHostGetDevicePointer fused_output failed\n";
                }
            } else {
                std::cerr << "[bmo_jetson] cudaHostRegister fused_output failed: " << cudaGetErrorString(out_reg)
                          << "\n";
                std::free(out_raw);
            }
        } else {
            std::cerr << "[bmo_jetson] posix_memalign fused_output failed\n";
        }

        size_t in_bytes = 11264ULL * sizeof(float);
        in_bytes = (in_bytes + 63) & ~size_t(63);
        void * in_raw = nullptr;
        if (posix_memalign(&in_raw, 64, in_bytes) == 0 && in_raw) {
            cudaError_t in_reg =
                cudaHostRegister(in_raw, in_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (in_reg == cudaSuccess) {
                if (cudaHostGetDevicePointer(&ctx.cuda_fused_input_buffer_dev, in_raw, 0) == cudaSuccess) {
                    ctx.cuda_fused_input_buffer = in_raw;
                    ctx.cuda_fused_input_owns_raw_malloc = true;
                    std::cout << "[bmo_jetson] fused_input_buffer="
                              << (double) in_bytes / (1024.0 * 1024.0) << " MB pinned mapped\n";
                } else {
                    cudaHostUnregister(in_raw);
                    std::free(in_raw);
                    std::cerr << "[bmo_jetson] cudaHostGetDevicePointer fused_input failed\n";
                }
            } else {
                std::cerr << "[bmo_jetson] cudaHostRegister fused_input failed: " << cudaGetErrorString(in_reg)
                          << "\n";
                std::free(in_raw);
            }
        } else {
            std::cerr << "[bmo_jetson] posix_memalign fused_input failed\n";
        }
    }
#endif
#ifndef BMO_JETSON
    const size_t scratch_bytes = max_unpack_elems * sizeof(float);
    if (scratch_bytes > 0) {
        cudaError_t err = cudaMalloc(&ctx.cuda_unpack_scratch, scratch_bytes);
        ctx.cuda_unpack_scratch_dev = nullptr;
        if (err == cudaSuccess) {
            ctx.cuda_unpack_scratch_dev = ctx.cuda_unpack_scratch;
            ctx.cuda_unpack_scratch_bytes = scratch_bytes;
            std::cout << "[bmo_prepare_device_packed_tensors] cuda_unpack_scratch="
                      << (double) scratch_bytes / (1024.0 * 1024.0) << " MB\n";
        } else {
            ctx.cuda_unpack_scratch = nullptr;
            ctx.cuda_unpack_scratch_dev = nullptr;
            ctx.cuda_unpack_scratch_bytes = 0;
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc scratch failed: "
                      << cudaGetErrorString(err) << "\n";
        }
    }
#endif
    bmo_print_mem_diag("End Prepare");
#endif
}

void bmo_free_cuda_resources(bmo_context & ctx) {
#ifdef BMO_ENABLE_CUDA
    for (auto & kv : ctx.packed_registry) {
        device_packed_t & dp = kv.second;
#ifdef BMO_JETSON
        if (dp.canonical_base_host) {
            cudaHostUnregister(dp.canonical_base_host);
            std::free(dp.canonical_base_host);
            dp.canonical_base_host = nullptr;
        }
#endif
        free_device_packed_owned_buffers(dp);
    }
    ctx.packed_registry.clear();

#ifndef BMO_JETSON
    if (ctx.cuda_unpack_scratch) {
        cudaFree(ctx.cuda_unpack_scratch);
        ctx.cuda_unpack_scratch = nullptr;
    }
    ctx.cuda_unpack_scratch_dev = nullptr;
    ctx.cuda_unpack_scratch_bytes = 0;
    ctx.cuda_unpack_scratch_managed = false;
    ctx.cuda_unpack_scratch_owns_raw_malloc = false;
#endif
#ifdef BMO_JETSON
    if (ctx.cuda_fused_output_buffer && ctx.cuda_fused_output_owns_raw_malloc) {
        cudaHostUnregister(ctx.cuda_fused_output_buffer);
        std::free(ctx.cuda_fused_output_buffer);
        ctx.cuda_fused_output_buffer = nullptr;
    }
    ctx.cuda_fused_output_buffer_dev = nullptr;
    ctx.cuda_fused_output_buffer_bytes = 0;
    ctx.cuda_fused_output_owns_raw_malloc = false;
    if (ctx.cuda_fused_input_buffer && ctx.cuda_fused_input_owns_raw_malloc) {
        cudaHostUnregister(ctx.cuda_fused_input_buffer);
        std::free(ctx.cuda_fused_input_buffer);
        ctx.cuda_fused_input_buffer = nullptr;
    }
    ctx.cuda_fused_input_buffer_dev = nullptr;
    ctx.cuda_fused_input_owns_raw_malloc = false;
    if (ctx.streaming_big_pool_registered && ctx.streaming_big_pool) {
        cudaHostUnregister(ctx.streaming_big_pool);
        ctx.streaming_big_pool_registered = false;
    }
    if (ctx.streaming_big_pool) {
        std::free(ctx.streaming_big_pool);
        ctx.streaming_big_pool = nullptr;
    }
    ctx.streaming_big_pool_size = 0;
    if (ctx.streaming_scalar_pool) {
        std::free(ctx.streaming_scalar_pool);
        ctx.streaming_scalar_pool = nullptr;
    }
    ctx.streaming_scalar_pool_size = 0;
#endif
#else
    (void) ctx;
#endif
}

void bmo_init_kv_cache(bmo_context & ctx, int32_t n_ctx) {
    if (ctx.n_heads <= 0 || ctx.head_dim <= 0 || ctx.n_layers <= 0) {
        throw std::runtime_error("KV cache init requires valid n_layers, n_heads and head_dim in context");
    }

#ifdef BMO_JETSON
    if (n_ctx > 128) {
        std::cout << "[bmo_jetson] Overriding n_ctx from " << n_ctx << " to 128 to prevent OOM\n";
        n_ctx = 128;
    }
#endif
    ctx.n_ctx = n_ctx;

    // Estimate required memory: two caches (k and v) stored as f16
    const int64_t elems_per_layer = (int64_t) n_ctx * (int64_t) ctx.n_heads * (int64_t) ctx.head_dim;
    const size_t bytes_per_layer = (size_t) elems_per_layer * sizeof(ggml_fp16_t) * 2; // k + v
    const size_t total_bytes = bytes_per_layer * (size_t) ctx.n_layers;

    // If reinitializing KV cache, free previous allocations first.
    if (ctx.kv_ctx) {
        ggml_free(ctx.kv_ctx);
        ctx.kv_ctx = nullptr;
    }
    ctx.kv_mem.reset();

    // Keep KV cache in host memory. The CUDA backend is used only for SEPTQ unpacking.
    const size_t alloc_size = total_bytes + (1 << 20);
    ctx.kv_mem.reset(new uint8_t[alloc_size]);
    ggml_init_params iparams = {
        /*.mem_size   =*/ alloc_size,
        /*.mem_buffer =*/ ctx.kv_mem.get(),
        /*.no_alloc   =*/ false
    };
    ggml_context * kv_ctx = ggml_init(iparams);
    if (!kv_ctx) throw std::runtime_error("Failed to initialize KV ggml_context");

    ctx.kv_ctx = kv_ctx;
    ctx.k_cache = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, ctx.head_dim, n_ctx, ctx.n_heads, ctx.n_layers);
    ctx.v_cache = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, ctx.head_dim, n_ctx, ctx.n_heads, ctx.n_layers);
    if (!ctx.k_cache || !ctx.v_cache) {
        throw std::runtime_error("Failed to create KV tensors");
    }

    ctx.kv_bytes = (size_t) ggml_nbytes(ctx.k_cache) + (size_t) ggml_nbytes(ctx.v_cache);

    std::cout << "[bmo_init_kv_cache] Allocated KV cache: " << (double) ctx.kv_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "[bmo_init_kv_cache] per-layer estimate: " << (double) bytes_per_layer / (1024.0 * 1024.0) << " MB\n";

}
