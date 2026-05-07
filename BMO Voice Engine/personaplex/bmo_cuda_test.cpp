#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    // Small test: compute y = W * x
    const size_t mem_size = 64 * 1024 * 1024; // 64 MB
    std::vector<uint8_t> mem_buf(mem_size);

    ggml_init_params params = {
        mem_size,
        mem_buf.data(),
        false,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return 1;
    }

    const int cols = 8;
    const int rows = 16;

    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 1);

    float * Wd = (float *)W->data;
    float * xd = (float *)x->data;

    for (int i = 0; i < cols * rows; ++i) Wd[i] = (float)(i % 7 + 1) * 0.001f;
    for (int i = 0; i < cols; ++i) xd[i] = 1.0f;

    ggml_tensor * y = ggml_mul_mat(ctx, W, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    const int n_threads = 1;
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_graph_compute_with_ctx failed\n");
        ggml_free(ctx);
        return 2;
    }

    float * yd = (float *)y->data;
    printf("y first 8: ");
    for (int i = 0; i < 8 && i < rows; ++i) {
        printf("%g ", yd[i]);
    }
    printf("\n");

    ggml_free(ctx);
    return 0;
}
