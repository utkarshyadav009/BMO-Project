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

        std::vector<float> layer_input((size_t) ctx.n_embd, 1.0f);

        std::cout << "[bmo_main] Running 32-layer cascade...\n";
        for (int layer = 0; layer < ctx.n_layers; ++layer) {
            reset_work_ctx(ctx, work_mem_size);

            ggml_tensor * input_tokens = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
            if (!input_tokens || !input_tokens->data) {
                throw std::runtime_error("Failed to allocate input_tokens tensor");
            }
            std::memcpy(input_tokens->data, layer_input.data(), layer_input.size() * sizeof(float));

            ggml_cgraph * gf = bmo_build_temporal_graph(ctx, model, input_tokens, 0, layer, layer + 1);
            if (!gf) {
                throw std::runtime_error("Failed to build temporal graph");
            }

            const std::string out_name = "out_layer_" + std::to_string(layer);
            std::cout << "[bmo_main] Layer " << layer << " graph has " << ggml_graph_n_nodes(gf) << " nodes\n";

            const ggml_status status = ggml_graph_compute_with_ctx(ctx.work_ctx, gf, 8);
            if (status != GGML_STATUS_SUCCESS) {
                std::cerr << "[bmo_main] ggml_graph_compute_with_ctx failed with status "
                          << (int) status << " at layer " << layer << "\n";
                throw std::runtime_error("Graph compute failed");
            }

            ggml_tensor * out_tensor = ggml_graph_get_tensor(gf, out_name.c_str());
            if (!out_tensor || !out_tensor->data) {
                throw std::runtime_error("Missing graph tensor: " + out_name);
            }

            const std::string dump_path = "cpp_out_layer_" + std::to_string(layer) + ".bin";
            dump_tensor(dump_path.c_str(), out_tensor);

            std::cout << "[bmo_main] Dumped layer " << layer << " (" << ggml_nbytes(out_tensor)
                      << " bytes) to " << dump_path << "\n";

            layer_input.resize((size_t) ggml_nelements(out_tensor));
            std::memcpy(layer_input.data(), out_tensor->data, ggml_nbytes(out_tensor));
        }

        std::cout << "[SUCCESS] 32-layer cascade completed!\n";
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
