#include "bmo.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    const char * fname = (argc >= 2) ? argv[1] : "bmo_weights.gguf";

    bmo_model model;
    bmo_context ctx;

    try {
        std::cout << "[bmo_main] === BMO Temporal Forward Pass Test ===\n";
        std::cout << "[bmo_main] Loading model from: " << fname << "\n";
        bmo_load_model(fname, model, ctx);

        std::cout << "[bmo_main] weights_bytes = " << (double) ctx.weights_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << "[bmo_main] n_layers=" << ctx.n_layers << " n_heads=" << ctx.n_heads 
                  << " n_embd=" << ctx.n_embd << " head_dim=" << ctx.head_dim << "\n";

        // Initialize KV cache for 2048 tokens
        std::cout << "[bmo_main] Initializing KV cache for 2048 tokens...\n";
        bmo_init_kv_cache(ctx, 2048);
        std::cout << "[bmo_main] kv_bytes      = " << (double) ctx.kv_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        // ========== Initialize compute arenas ==========
        std::cout << "[bmo_main] Initializing compute arenas...\n";


        // Allocate work memory for graph context (1 GB)
        const size_t work_mem_size = (size_t) 1024 * 1024 * 1024;
        ctx.work_mem.resize(work_mem_size);
        std::cout << "[bmo_main] Allocated work_mem: " 
                  << (double) work_mem_size / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        ggml_init_params wp = {
            work_mem_size,
            ctx.work_mem.data(),
            false,
        };
        ctx.work_ctx = ggml_init(wp);
        if (!ctx.work_ctx) {
            throw std::runtime_error("Failed to initialize work context");
        }

        ggml_tensor * input_tokens = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
        if (!input_tokens || !input_tokens->data) {
            throw std::runtime_error("Failed to allocate input_tokens tensor");
        }
        float * input_ptr = reinterpret_cast<float *>(input_tokens->data);
        for (int32_t i = 0; i < ctx.n_embd; ++i) {
            input_ptr[i] = 1.0f;
        }

        std::cout << "[bmo_main] Running Layer-0 sub-layer audit...\n";
        ggml_cgraph * gf = bmo_build_temporal_graph(ctx, model, input_tokens, 0, 0, 1);
        if (!gf) {
            throw std::runtime_error("Failed to build temporal graph");
        }

        std::cout << "[bmo_main] Layer 0 graph has " << ggml_graph_n_nodes(gf) << " nodes\n";
        const ggml_status status = ggml_graph_compute_with_ctx(ctx.work_ctx, gf, 8);
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "[bmo_main] ggml_graph_compute_with_ctx failed with status "
                      << (int) status << "\n";
            throw std::runtime_error("Graph compute failed");
        }

        struct named_dump {
            const char * tensor_name;
            const char * path;
        };

        const named_dump dumps[] = {
            {"l0_norm1", "cpp_l0_norm1.bin"},
            {"l0_attn",  "cpp_l0_attn.bin"},
            {"l0_norm2", "cpp_l0_norm2.bin"},
            {"l0_ffn",   "cpp_l0_ffn.bin"},
        };

        for (const auto & d : dumps) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, d.tensor_name);
            if (!t || !t->data) {
                throw std::runtime_error(std::string("Missing graph tensor: ") + d.tensor_name);
            }

            FILE * out_file = fopen(d.path, "wb");
            if (!out_file) {
                throw std::runtime_error(std::string("Failed to open ") + d.path + " for writing");
            }

            fwrite(t->data, 1, ggml_nbytes(t), out_file);
            fclose(out_file);
            std::cout << "[bmo_main] Dumped " << d.tensor_name << " (" << ggml_nbytes(t)
                      << " bytes) to " << d.path << "\n";
        }

        std::cout << "[SUCCESS] Layer-0 sub-layer audit completed!\n";
        // ========== Cleanup ==========
        std::cout << "[bmo_main] Cleaning up...\n";
        if (ctx.work_ctx) {
            ggml_free(ctx.work_ctx);
            ctx.work_ctx = nullptr;
        }
        std::cout << "[bmo_main] Test completed successfully!\n";

        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "[bmo_main] ERROR: " << ex.what() << std::endl;
        if (ctx.work_ctx) {
            ggml_free(ctx.work_ctx);
        }
        return 1;
    }
}
