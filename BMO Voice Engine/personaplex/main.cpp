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
    std::string fname = "bmo_weights.gguf";
    std::string mode = "depth_cascade"; // default to depth for backwards compat
    bool debug_dumps = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--debug-dumps") {
            debug_dumps = true;
        } else if (arg[0] != '-') {
            fname = arg;
        }
    }

    bmo_model model;
    bmo_context ctx;

    try {
        std::cout << "[bmo_main] === BMO Forward Pass Test ===\n";
        std::cout << "[bmo_main] Loading model from: " << fname << "\n";
        bmo_load_model(fname.c_str(), model, ctx);

        std::cout << "[bmo_main] weights_bytes = " << (double) ctx.weights_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << "[bmo_main] n_layers=" << ctx.n_layers << " n_heads=" << ctx.n_heads
                  << " n_embd=" << ctx.n_embd << " head_dim=" << ctx.head_dim << "\n";

        std::cout << "[bmo_main] Initializing KV cache for 2048 tokens...\n";
        bmo_init_kv_cache(ctx, 2048);
        std::cout << "[bmo_main] kv_bytes      = " << (double) ctx.kv_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        std::cout << "[bmo_main] Initializing compute arenas...\n";
        const size_t work_mem_size = (size_t) 2048ULL * 1024 * 1024;
        ctx.work_mem.resize(work_mem_size);
        std::cout << "[bmo_main] Allocated work_mem: "
                  << (double) work_mem_size / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        if (mode == "temporal_cascade") {
            std::cout << "[bmo_main] Running Temporal Cascade...\n";
            
            // Initialize the cascade input with ones, matching verify_all_layers.py
            std::vector<float> current_x(ctx.n_embd, 1.0f);

            for (int i = 0; i < ctx.n_layers; ++i) {
                reset_work_ctx(ctx, work_mem_size);
                
                ggml_tensor * layer_in = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                std::memcpy(layer_in->data, current_x.data(), ctx.n_embd * sizeof(float));

                // Build ONLY layer i (from i to i+1)
                ggml_cgraph * gf = bmo_build_temporal_graph(ctx, model, layer_in, 0, i, i + 1);
                
                const ggml_status status = ggml_graph_compute_with_ctx(ctx.work_ctx, gf, 1);
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("Temporal graph compute failed at layer " + std::to_string(i));
                }

                std::string out_name = "out_layer_" + std::to_string(i);
                ggml_tensor * layer_out = ggml_graph_get_tensor(gf, out_name.c_str());
                if (!layer_out) {
                    throw std::runtime_error("Missing graph tensor: " + out_name);
                }

                // Copy output back to current_x for the next layer's input
                std::memcpy(current_x.data(), layer_out->data, ctx.n_embd * sizeof(float));

                if (debug_dumps) {
                    std::string dump_path = "cpp_out_layer_" + std::to_string(i) + ".bin";
                    dump_tensor(dump_path.c_str(), layer_out);
                    std::cout << "[bmo_main] Dumped " << dump_path << "\n";
                }
            }
            std::cout << "[SUCCESS] Temporal validation cascade completed!\n";

        } else if (mode == "depth_cascade") {
            std::cout << "[bmo_main] Running Depth Cascade (Step 0)...\n";
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
                throw std::runtime_error("Depth graph compute failed");
            }

            ggml_tensor * out_tensor = ggml_graph_get_tensor(gf, "depth_out_step_0");
            if (!out_tensor || !out_tensor->data) {
                throw std::runtime_error("Missing graph tensor: depth_out_step_0");
            }

            if (debug_dumps) {
                const std::string dump_path = "cpp_depth_out.bin";
                dump_tensor(dump_path.c_str(), out_tensor);
                std::cout << "[bmo_main] Dumped depth output (" << ggml_nbytes(out_tensor)
                          << " bytes) to " << dump_path << "\n";

                auto dump_named = [&](const char * name, const char * path) {
                    ggml_tensor * t = ggml_graph_get_tensor(gf, name);
                    if (!t || !t->data) {
                        std::cout << "[bmo_main] Missing debug tensor: " << name << "\n";
                        return;
                    }
                    dump_tensor(path, t);
                    std::cout << "[bmo_main] Dumped " << name << " to " << path << "\n";
                };

                dump_named("depth_x_init", "cpp_depth_x_init.bin");
                dump_named("depth_x_norm", "cpp_depth_x_norm.bin");
                dump_named("depth_qkv_raw", "cpp_depth_qkv_raw.bin");
                dump_named("depth_q_raw", "cpp_depth_q_raw.bin");
                dump_named("depth_k_raw", "cpp_depth_k_raw.bin");
                dump_named("depth_v_raw", "cpp_depth_v_raw.bin");
                dump_named("depth_attn_out", "cpp_depth_attn_out.bin");
                dump_named("depth_attn_x", "cpp_depth_attn_x.bin");

                // DEBUG: Build separate graph for intermediate values
                {
                    std::cout << "[bmo_main] Computing debug intermediate values...\n";
                    reset_work_ctx(ctx, (size_t) 2048ULL * 1024 * 1024);
                    ggml_context * dbg_ctx = ctx.work_ctx;
                    ggml_cgraph * dbg_gf = ggml_new_graph(dbg_ctx);

                    // Recreate input tensors
                    ggml_tensor * dbg_temporal_out = ggml_new_tensor_2d(dbg_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                    ggml_tensor * dbg_text_tokens = ggml_new_tensor_1d(dbg_ctx, GGML_TYPE_I32, 1);
                    std::fill_n(reinterpret_cast<float *>(dbg_temporal_out->data), (size_t) ggml_nelements(dbg_temporal_out), 1.0f);
                    reinterpret_cast<int32_t *>(dbg_text_tokens->data)[0] = 0;

                    // Compute z_s (depformer_in projection)
                    ggml_tensor * dbg_z_s = ggml_mul_mat(dbg_ctx, model.depformer_in[0], dbg_temporal_out);
                    ggml_set_name(dbg_z_s, "debug_z_s");

                    // Get text embedding
                    ggml_tensor * dbg_text_emb = ggml_get_rows(dbg_ctx, model.text_emb, dbg_text_tokens);
                    ggml_set_name(dbg_text_emb, "debug_text_emb");

                    // Add them
                    ggml_tensor * dbg_x_init = ggml_add(dbg_ctx, dbg_z_s, ggml_reshape_2d(dbg_ctx, dbg_text_emb, 1024, 1));
                    ggml_set_name(dbg_x_init, "debug_x_init");

                    ggml_build_forward_expand(dbg_gf, dbg_z_s);
                    ggml_build_forward_expand(dbg_gf, dbg_text_emb);
                    ggml_build_forward_expand(dbg_gf, dbg_x_init);

                    ggml_graph_compute_with_ctx(dbg_ctx, dbg_gf, 1);

                    // Dump debug values
                    ggml_tensor * out_z_s = ggml_graph_get_tensor(dbg_gf, "debug_z_s");
                    ggml_tensor * out_text_emb = ggml_graph_get_tensor(dbg_gf, "debug_text_emb");
                    ggml_tensor * out_x_init = ggml_graph_get_tensor(dbg_gf, "debug_x_init");

                    if (out_z_s) dump_tensor("cpp_debug_z_s.bin", out_z_s);
                    if (out_text_emb) dump_tensor("cpp_debug_text_emb.bin", out_text_emb);
                    if (out_x_init) dump_tensor("cpp_debug_x_init.bin", out_x_init);

                    std::cout << "[bmo_main] Dumped debug tensors (z_s, text_emb, x_init)\n";
                }
            }

            std::cout << "[SUCCESS] Depth-step 0 validation completed!\n";
        } else if (mode == "stress_test") {
            std::cout << "[bmo_main] Running Stress Test (100 iterations)...\n";

            std::vector<float> temporal_state(ctx.n_embd, 1.0f);
            for (int iter = 0; iter < 100; ++iter) {
                if ((iter % 5) == 0) {
                    std::cout << "[bmo_main] Stress iteration " << (iter + 1) << "/100\n";
                }
                // Temporal pass over all layers, one layer at a time, to keep the
                // transient graph well within the 2 GB work arena.
                for (int layer = 0; layer < ctx.n_layers; ++layer) {
                    reset_work_ctx(ctx, work_mem_size);
                    ggml_tensor * layer_in = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                    std::memcpy(layer_in->data, temporal_state.data(), ctx.n_embd * sizeof(float));

                    ggml_cgraph * temporal_gf = bmo_build_temporal_graph(ctx, model, layer_in, 0, layer, layer + 1);
                    const ggml_status temporal_status = ggml_graph_compute_with_ctx(ctx.work_ctx, temporal_gf, 1);
                    if (temporal_status != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("Stress test: temporal graph compute failed at iteration " + std::to_string(iter) + " layer " + std::to_string(layer));
                    }

                    const std::string temporal_out_name = "out_layer_" + std::to_string(layer);
                    ggml_tensor * temporal_out = ggml_graph_get_tensor(temporal_gf, temporal_out_name.c_str());
                    if (!temporal_out || !temporal_out->data) {
                        throw std::runtime_error("Stress test: missing temporal output at iteration " + std::to_string(iter) + " layer " + std::to_string(layer));
                    }
                    std::memcpy(temporal_state.data(), temporal_out->data, ctx.n_embd * sizeof(float));

                    ggml_graph_clear(temporal_gf);
                    ggml_free(ctx.work_ctx);
                    ctx.work_ctx = nullptr;
                }

                // Depth pass for each codebook step.
                for (int step = 0; step < 16; ++step) {
                    reset_work_ctx(ctx, work_mem_size);

                    ggml_tensor * depth_in = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
                    ggml_tensor * audio_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
                    if (!depth_in || !depth_in->data || !text_tokens || !text_tokens->data || !audio_tokens || !audio_tokens->data) {
                        throw std::runtime_error("Stress test: failed to allocate depth inputs at iteration " + std::to_string(iter));
                    }

                    std::memcpy(depth_in->data, temporal_state.data(), ctx.n_embd * sizeof(float));
                    reinterpret_cast<int32_t *>(text_tokens->data)[0] = 0;
                    reinterpret_cast<int32_t *>(audio_tokens->data)[0] = 0;

                    ggml_cgraph * depth_gf = bmo_build_depth_graph(ctx, model, depth_in, text_tokens, audio_tokens, step, 0);
                    const ggml_status depth_status = ggml_graph_compute_with_ctx(ctx.work_ctx, depth_gf, 1);
                    if (depth_status != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("Stress test: depth graph compute failed at iteration " + std::to_string(iter));
                    }

                    ggml_graph_clear(depth_gf);
                    ggml_free(ctx.work_ctx);
                    ctx.work_ctx = nullptr;
                }
            }

            std::cout << "[SUCCESS] Stress test completed!\n";
        } else {
            throw std::runtime_error("Unknown mode: " + mode);
        }
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
