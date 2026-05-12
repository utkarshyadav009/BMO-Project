// Compares ggml_rope (GGML_ROPE_TYPE_NORMAL, freq_base 10000) on a contiguous
// [head_dim, n_heads, n_token] tensor against launch_rope_interleaved invoked
// once per token (n_token=1) with pos_base = absolute position.

#include "bmo.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cassert>

extern "C" {
#include "ggml.h"
}

static void fill_tensor_deterministic(ggml_tensor * x, int head_dim, int n_heads, int n_token) {
    // Contiguous ggml layout ne[0]=head_dim, ne[1]=n_heads, ne[2]=n_token:
    //   offset(d,h,t) = d + head_dim * (h + n_heads * t)
    const float k = 0.01f;
    float * base = (float *) x->data;
    for (int t = 0; t < n_token; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const int64_t idx_global =
                    (int64_t) d + (int64_t) head_dim * ((int64_t) h + (int64_t) n_heads * (int64_t) t);
                base[(size_t) d + (size_t) head_dim * ((size_t) h + (size_t) n_heads * (size_t) t)] =
                    std::sin(k * (float) idx_global);
            }
        }
    }
}

static void linearize_ggml_f32(
    const ggml_tensor * t, int head_dim, int n_heads, int n_token, std::vector<float> & out) {
    out.resize((size_t) head_dim * (size_t) n_heads * (size_t) n_token);
    const float * base = (const float *) t->data;
    size_t w = 0;
    for (int t = 0; t < n_token; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                out[w++] = base[(size_t) d + (size_t) head_dim * ((size_t) h + (size_t) n_heads * (size_t) t)];
            }
        }
    }
}

static void stats_compare(
    const std::vector<float> & a,
    const std::vector<float> & b,
    double & max_abs,
    double & mean_abs,
    double & cosine) {
    const size_t n = a.size();
    assert(n == b.size());
    max_abs = 0.0;
    mean_abs = 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = (double) a[i];
        const double db = (double) b[i];
        const double diff = std::fabs(da - db);
        max_abs = std::max(max_abs, diff);
        mean_abs += diff;
        dot += da * db;
        na += da * da;
        nb += db * db;
    }
    mean_abs /= (double) n;
    cosine = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

int main() {
    constexpr int head_dim = 128;
    constexpr int n_heads = 32;
    constexpr int n_token = 8;
    constexpr float theta_base = 10000.0f;

    ggml_init_params gp = {
        /*.mem_size   =*/ 64u * 1024u * 1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * wctx = ggml_init(gp);
    if (!wctx) {
        std::fprintf(stderr, "rope_consistency_test: ggml_init failed\n");
        return 2;
    }

    ggml_tensor * x = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, head_dim, n_heads, n_token);
    ggml_tensor * pos = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, n_token);
    if (!x || !pos || !x->data || !pos->data) {
        std::fprintf(stderr, "rope_consistency_test: tensor alloc failed\n");
        ggml_free(wctx);
        return 2;
    }

    fill_tensor_deterministic(x, head_dim, n_heads, n_token);
    int32_t * pos_data = (int32_t *) pos->data;
    for (int i = 0; i < n_token; ++i) {
        pos_data[i] = i;
    }

    // Copy input before rope (ggml_rope may write in-place into the first arg's buffer chain).
    std::vector<float> x_copy((size_t) head_dim * (size_t) n_heads * (size_t) n_token);
    linearize_ggml_f32(x, head_dim, n_heads, n_token, x_copy);
    std::memcpy(x->data, x_copy.data(), x_copy.size() * sizeof(float));

    ggml_tensor * y_ggml = ggml_rope(wctx, x, pos, head_dim, GGML_ROPE_TYPE_NORMAL);
    ggml_cgraph * gf = ggml_new_graph(wctx);
    ggml_build_forward_expand(gf, y_ggml);

    const ggml_status st = ggml_graph_compute_with_ctx(wctx, gf, 32);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "rope_consistency_test: ggml_graph_compute_with_ctx failed (%d)\n", (int) st);
        ggml_free(wctx);
        return 2;
    }

    std::vector<float> out_ggml;
    linearize_ggml_f32(y_ggml, head_dim, n_heads, n_token, out_ggml);

    // --- CUDA path: one launch_rope_interleaved per time step ---
    const size_t slice_elems = (size_t) head_dim * (size_t) n_heads;
    const size_t slice_bytes = slice_elems * sizeof(float);
    float * x_dev = nullptr;
    float * y_dev = nullptr;
    cudaError_t ce = cudaMalloc(&x_dev, slice_bytes);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "rope_consistency_test: cudaMalloc x_dev: %s\n", cudaGetErrorString(ce));
        ggml_free(wctx);
        return 2;
    }
    ce = cudaMalloc(&y_dev, slice_bytes);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "rope_consistency_test: cudaMalloc y_dev: %s\n", cudaGetErrorString(ce));
        cudaFree(x_dev);
        ggml_free(wctx);
        return 2;
    }

    std::vector<float> out_cuda(out_ggml.size());
    for (int t = 0; t < n_token; ++t) {
        const float * host_slice = x_copy.data() + (size_t) t * slice_elems;
        ce = cudaMemcpy(x_dev, host_slice, slice_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "rope_consistency_test: cudaMemcpy H2D: %s\n", cudaGetErrorString(ce));
            cudaFree(x_dev);
            cudaFree(y_dev);
            ggml_free(wctx);
            return 2;
        }
        launch_rope_interleaved(x_dev, n_heads, head_dim, 1, t, theta_base, y_dev, nullptr);
        ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "rope_consistency_test: launch_rope_interleaved error: %s\n", cudaGetErrorString(ce));
            cudaFree(x_dev);
            cudaFree(y_dev);
            ggml_free(wctx);
            return 2;
        }
        float * host_out_slice = out_cuda.data() + (size_t) t * slice_elems;
        ce = cudaMemcpy(host_out_slice, y_dev, slice_bytes, cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "rope_consistency_test: cudaMemcpy D2H: %s\n", cudaGetErrorString(ce));
            cudaFree(x_dev);
            cudaFree(y_dev);
            ggml_free(wctx);
            return 2;
        }
    }
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaDeviceSynchronize();

    double max_abs = 0.0, mean_abs = 0.0, cosv = 0.0;
    stats_compare(out_ggml, out_cuda, max_abs, mean_abs, cosv);

    std::printf("rope_consistency_test: head_dim=%d n_heads=%d n_token=%d\n", head_dim, n_heads, n_token);
    std::printf("  max_abs_diff=%.8e mean_abs_diff=%.8e cosine=%.12f\n", max_abs, mean_abs, cosv);

    constexpr double thresh = 1e-5;
    const int pass = max_abs < thresh ? 0 : 1;
    if (pass != 0) {
        std::printf("  FAIL (threshold max_abs_diff < %.1e)\n", thresh);
    } else {
        std::printf("  PASS\n");
    }

    ggml_free(wctx);
    return pass;
}
