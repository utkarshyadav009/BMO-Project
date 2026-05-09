// bmo_api.cpp - Implementation of the BMO C-ABI bridge.
//
// Wraps bmo_model + bmo_context behind an opaque handle and serialises all
// per-handle entry points through a mutex so the resulting libbmo.so can be
// safely shared across Python threads.

#include "bmo_api.h"

#include "bmo.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct bmo_handle {
    bmo_model   model;
    bmo_context ctx;
    int         pos = 0;
    std::string last_error;
    std::mutex  mu;
};

namespace {

constexpr size_t kDefaultWorkMem =
#ifdef BMO_JETSON
    (size_t) 1024ULL * 1024 * 1024;   // 1 GB on Jetson Orin Nano (8 GB unified)
#else
    (size_t) 2048ULL * 1024 * 1024;   // 2 GB on discrete-GPU hosts
#endif

constexpr int32_t kDefaultKvCtx = 2048;

void set_err(bmo_handle_t * h, const std::string & m) {
    if (h) h->last_error = m;
}

} // namespace

extern "C" {

bmo_handle_t * bmo_init(const char * gguf_path, int n_ctx) {
    if (!gguf_path) {
        std::fprintf(stderr, "[bmo_api] bmo_init: gguf_path is null\n");
        return nullptr;
    }

    auto h = std::make_unique<bmo_handle_t>();
    auto init_cleanup = [](bmo_handle_t * hh) {
        if (!hh) return;
        try {
            if (hh->ctx.work_ctx) {
                ggml_free(hh->ctx.work_ctx);
                hh->ctx.work_ctx = nullptr;
            }
            if (hh->model.wctx) {
                ggml_free(hh->model.wctx);
                hh->model.wctx = nullptr;
            }
            if (hh->model.gctx) {
                gguf_free(hh->model.gctx);
                hh->model.gctx = nullptr;
            }
            bmo_free_cuda_resources(hh->ctx);
        } catch (...) {
            // best-effort cleanup; never throw from the failure path
        }
    };

    try {
        bmo_load_model(gguf_path, h->model, h->ctx);

        // bmo_init_kv_cache takes (ctx, n_ctx) -- the n_ctx caller value is
        // clamped internally on Jetson.
        const int32_t kv_ctx = (n_ctx > 0) ? (int32_t) n_ctx : kDefaultKvCtx;
        bmo_init_kv_cache(h->ctx, kv_ctx);

        h->ctx.work_mem.resize(kDefaultWorkMem);
        h->pos = 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "[bmo_api] init failed: %s\n", ex.what());
        init_cleanup(h.get());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "[bmo_api] init failed: unknown exception\n");
        init_cleanup(h.get());
        return nullptr;
    }

    return h.release();
}

void bmo_free(bmo_handle_t * h) {
    if (!h) return;
    try {
        if (h->ctx.work_ctx) {
            ggml_free(h->ctx.work_ctx);
            h->ctx.work_ctx = nullptr;
        }
        if (h->model.wctx) {
            ggml_free(h->model.wctx);
            h->model.wctx = nullptr;
        }
        if (h->model.gctx) {
            gguf_free(h->model.gctx);
            h->model.gctx = nullptr;
        }
        bmo_free_cuda_resources(h->ctx);
    } catch (...) {
        // Destructors must not throw across the C-ABI boundary; swallow.
    }
    delete h;
}

void bmo_reset(bmo_handle_t * h) {
    if (!h) return;
    std::lock_guard<std::mutex> lk(h->mu);
    h->pos = 0;
    if (h->ctx.k_cache && h->ctx.k_cache->data) {
        std::memset(h->ctx.k_cache->data, 0, (size_t) ggml_nbytes(h->ctx.k_cache));
    }
    if (h->ctx.v_cache && h->ctx.v_cache->data) {
        std::memset(h->ctx.v_cache->data, 0, (size_t) ggml_nbytes(h->ctx.v_cache));
    }
    h->last_error.clear();
}

int bmo_get_n_layers    (bmo_handle_t * h) { return h ? h->ctx.n_layers         : 0; }
int bmo_get_n_embd      (bmo_handle_t * h) { return h ? h->ctx.n_embd           : 0; }
int bmo_get_n_codebooks (bmo_handle_t * h) { return h ? h->ctx.num_codebooks    : 0; }
int bmo_get_dep_q       (bmo_handle_t * h) { return h ? h->ctx.dep_q            : 0; }
int bmo_get_text_vocab  (bmo_handle_t * h) { return h ? h->ctx.text_vocab_size  : 0; }
int bmo_get_audio_vocab (bmo_handle_t * h) { return h ? h->ctx.audio_vocab_size : 0; }

const char * bmo_last_error(bmo_handle_t * h) {
    return (!h || h->last_error.empty()) ? nullptr : h->last_error.c_str();
}

int bmo_forward_temporal(
    bmo_handle_t * h,
    const int32_t * input_tokens,
    int num_codebooks,
    int pos,
    float * out_transformer,
    float * out_text_logits) {
    if (!h || !input_tokens || !out_transformer || !out_text_logits) {
        set_err(h, "bmo_forward_temporal: null pointer argument");
        return 1;
    }
    if (num_codebooks <= 0 || num_codebooks > h->ctx.num_codebooks) {
        set_err(h, "bmo_forward_temporal: invalid num_codebooks");
        return 2;
    }
    if (pos < 0) {
        set_err(h, "bmo_forward_temporal: pos must be non-negative");
        return 2;
    }

    std::lock_guard<std::mutex> lk(h->mu);
    try {
        bmo_reset_work_ctx(h->ctx);

        ggml_tensor * layer_in = bmo_embed_input_tokens(
            h->ctx, h->model, input_tokens, num_codebooks);

        ggml_cgraph * gf = bmo_build_temporal_graph(
            h->ctx, h->model, layer_in, pos, /*layer_begin=*/0, /*layer_end=*/h->ctx.n_layers);
        if (!gf) {
            set_err(h, "bmo_forward_temporal: failed to build temporal graph");
            return 4;
        }

        bmo_execute_graph(h->ctx, gf, /*inputs=*/{});

        ggml_tensor * t_out = ggml_graph_get_tensor(gf, "transformer_out");
        ggml_tensor * t_lgt = ggml_graph_get_tensor(gf, "text_logits");
        if (!t_out || !t_out->data) {
            set_err(h, "bmo_forward_temporal: transformer_out tensor missing");
            return 3;
        }
        if (!t_lgt || !t_lgt->data) {
            set_err(h, "bmo_forward_temporal: text_logits tensor missing (model.text_linear may be null)");
            return 3;
        }

        const size_t hidden_bytes = (size_t) h->ctx.n_embd * sizeof(float);
        const size_t logits_bytes = (size_t) h->ctx.text_vocab_size * sizeof(float);
        if ((size_t) ggml_nbytes(t_out) < hidden_bytes ||
            (size_t) ggml_nbytes(t_lgt) < logits_bytes) {
            set_err(h, "bmo_forward_temporal: output tensor smaller than expected");
            return 5;
        }

        std::memcpy(out_transformer, t_out->data, hidden_bytes);
        std::memcpy(out_text_logits, t_lgt->data, logits_bytes);

        h->pos = pos + 1;
        h->last_error.clear();
        return 0;
    } catch (const std::exception & ex) {
        set_err(h, ex.what());
        return 9;
    } catch (...) {
        set_err(h, "bmo_forward_temporal: unknown exception");
        return 9;
    }
}

int bmo_forward_depth(
    bmo_handle_t * h,
    int cb_index,
    int32_t prev_token,
    const float * transformer_out,
    float * out_audio_logits) {
    // Argument validation runs even in the stub so callers get useful errors
    // for obvious misuse (null handle, bad cb_index, etc.) before they hit the
    // "not implemented" message.
    if (!h || !transformer_out || !out_audio_logits) {
        set_err(h, "bmo_forward_depth: null pointer argument");
        return 1;
    }
    if (cb_index < 0 || cb_index >= h->ctx.dep_q) {
        set_err(h, "bmo_forward_depth: cb_index out of range");
        return 2;
    }
    (void) prev_token;

    // Phase 4.2 stub. The temporal path (bmo_forward_temporal) is fully wired,
    // but the depformer path needs:
    //   * a per-frame depformer KV cache (reset at cb_index == 0, advanced at
    //     each subsequent cb_index call),
    //   * an audio-logits head: out = linears[cb_index] @ depformer_out,
    //   * input embedding selection between depformer_text_emb (cb_index==0)
    //     and depformer_emb[cb_index-1] (cb_index>0).
    //
    // bmo_build_depth_graph already covers most of the layer math but does not
    // yet plumb a KV cache or compute audio_logits. That work is scheduled for
    // Phase 4.3.
    set_err(h,
            "bmo_forward_depth: not implemented yet (Phase 4.3); the symbol "
            "is exported only so the Python ctypes wrapper can bind it.");
    return 10;
}

} // extern "C"
