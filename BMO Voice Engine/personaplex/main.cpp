#include "bmo.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void reset_work_ctx(bmo_context & ctx, size_t work_mem_size) {
    if (ctx.work_ctx) {
        ggml_free(ctx.work_ctx);
        ctx.work_ctx = nullptr;
    }

    ggml_init_params wp = {
        work_mem_size,
        ctx.work_mem.data(),
        false,
    };
    ctx.work_ctx = ggml_init(wp);
    if (!ctx.work_ctx) {
        throw std::runtime_error("Failed to initialize work context");
    }
}

static void dump_tensor(const char * path, ggml_tensor * t) {
    if (!t || !t->data) {
        throw std::runtime_error(std::string("Missing graph tensor for dump: ") + path);
    }

    FILE * out_file = fopen(path, "wb");
    if (!out_file) {
        throw std::runtime_error(std::string("Failed to open ") + path + " for writing");
    }

    fwrite(t->data, 1, ggml_nbytes(t), out_file);
    fclose(out_file);
}

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

        std::cout << "[bmo_main] Initializing KV cache for 2048 tokens...\n";
        bmo_init_kv_cache(ctx, 2048);
        std::cout << "[bmo_main] kv_bytes      = " << (double) ctx.kv_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        std::cout << "[bmo_main] Initializing compute arenas...\n";
        const size_t work_mem_size = (size_t) 1024 * 1024 * 1024;
        ctx.work_mem.resize(work_mem_size);
        std::cout << "[bmo_main] Allocated work_mem: "
                  << (double) work_mem_size / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        reset_work_ctx(ctx, work_mem_size);

        ggml_tensor * temporal_out = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
        ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
        ggml_tensor * audio_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
        if (!temporal_out || !temporal_out->data || !text_tokens || !text_tokens->data || !audio_tokens || !audio_tokens->data) {
            throw std::runtime_error("Failed to allocate depth validation tensors");
        }

        std::fill_n(reinterpret_cast<float *>(temporal_out->data), (size_t) ggml_nelements(temporal_out), 1.0f);
        reinterpret_cast<int32_t *>(text_tokens->data)[0] = 0;
        reinterpret_cast<int32_t *>(audio_tokens->data)[0] = 0;

        ggml_cgraph * gf = bmo_build_depth_graph(ctx, model, temporal_out, text_tokens, audio_tokens, 0, 0);
        if (!gf) {
            throw std::runtime_error("Failed to build depth graph");
        }

        std::cout << "[bmo_main] Depth graph has " << ggml_graph_n_nodes(gf) << " nodes\n";

        const ggml_status status = ggml_graph_compute_with_ctx(ctx.work_ctx, gf, 1);
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "[bmo_main] ggml_graph_compute_with_ctx failed with status "
                      << (int) status << " for depth graph\n";
            throw std::runtime_error("Depth graph compute failed");
        }

        ggml_tensor * out_tensor = ggml_graph_get_tensor(gf, "depth_out_step_0");
        if (!out_tensor || !out_tensor->data) {
            throw std::runtime_error("Missing graph tensor: depth_out_step_0");
        }

        const std::string dump_path = "cpp_depth_out.bin";
        dump_tensor(dump_path.c_str(), out_tensor);
        std::cout << "[bmo_main] Dumped depth output (" << ggml_nbytes(out_tensor)
                  << " bytes) to " << dump_path << "\n";

        std::cout << "[SUCCESS] Depth-step 0 validation completed!\n";
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
            ctx.work_ctx = nullptr;
        }
        return 1;
    }
}
