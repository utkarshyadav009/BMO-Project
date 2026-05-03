#include "bmo.h"
#include <iostream>
#include <cstring>
#include <algorithm>

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

        // Resize shared scratch buffer for unpacked weights (200 MB of F32)
        const size_t scratch_elems = (200 * 1024 * 1024) / sizeof(float);
        ctx.shared_scratch_w.resize(scratch_elems);
        std::cout << "[bmo_main] Allocated shared_scratch_w: " 
                  << (double)(scratch_elems * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";

        // Allocate work memory for graph context (1 GB)
        const size_t work_mem_size = (size_t) 1024 * 1024 * 1024;
        ctx.work_mem.resize(work_mem_size);
        std::cout << "[bmo_main] Allocated work_mem: " 
                  << (double) work_mem_size / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        // Initialize work context
        ggml_init_params wp = {
            work_mem_size,
            ctx.work_mem.data(),
            false,
        };
        ctx.work_ctx = ggml_init(wp);
        if (!ctx.work_ctx) {
            throw std::runtime_error("Failed to initialize work context");
        }
        std::cout << "[bmo_main] work_ctx initialized successfully\n";

        // ========== Create dummy input ==========
        std::cout << "[bmo_main] Creating dummy input tensor [1, n_embd]...\n";
        ggml_tensor * input_tokens = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
        if (!input_tokens) {
            throw std::runtime_error("Failed to allocate input_tokens tensor");
        }

        // Fill with 1.0f
        float * input_ptr = reinterpret_cast<float *>(input_tokens->data);
        for (int32_t i = 0; i < ctx.n_embd; ++i) {
            input_ptr[i] = 1.0f;
        }
        std::cout << "[bmo_main] Input tensor created and filled with 1.0f\n";

        // ========== Build temporal graph ==========
        std::cout << "[bmo_main] Building temporal compute graph (n_past=0)...\n";
        ggml_cgraph * gf = bmo_build_temporal_graph(ctx, model, input_tokens, 0);
        if (!gf) {
            throw std::runtime_error("Failed to build temporal graph");
        }
        std::cout << "[bmo_main] Temporal graph built successfully (" << gf->n_nodes << " nodes)\n";

        // ========== Execute graph ==========
        std::cout << "[bmo_main] Executing temporal graph (8 threads)...\n";
        const ggml_status status = ggml_graph_compute_with_ctx(ctx.work_ctx, gf, 8);
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "[bmo_main] ggml_graph_compute_with_ctx failed with status " << (int) status << "\n";
            throw std::runtime_error("Graph compute failed");
        }
        std::cout << "[SUCCESS] Temporal forward pass completed!\n";

        // ========== Extract and print output ==========
        std::cout << "[bmo_main] Extracting output tensor...\n";
        ggml_tensor * output = gf->nodes[gf->n_nodes - 1];
        if (!output || !output->data) {
            throw std::runtime_error("Output tensor is null or has no data");
        }

        float * out_ptr = reinterpret_cast<float *>(output->data);
        const int to_print = std::min<int>(10, (int) ggml_nelements(output));

        std::cout << "[bmo_main] Output tensor shape: [";
        for (int i = 0; i < 4; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << output->ne[i];
        }
        std::cout << "]\n";

        std::cout << "[bmo_main] First " << to_print << " output values:\n";
        for (int i = 0; i < to_print; ++i) {
            std::cout << "  [" << i << "] = " << out_ptr[i] << "\n";
        }

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
