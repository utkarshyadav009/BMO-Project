#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void fill_inputs(float * w, float * x, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        w[i] = 0.001f * (float) ((i % 7) + 1);
    }
    for (int i = 0; i < cols; ++i) {
        x[i] = 1.0f;
    }
}

static float max_abs_diff(const float * a, const float * b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

int main() {
    const int cols = 8;
    const int rows = 16;
    const size_t mem_size = 64 * 1024 * 1024;

    std::vector<uint8_t> cpu_mem(mem_size);
    ggml_init_params cpu_params = { mem_size, cpu_mem.data(), false };
    ggml_context * cpu_ctx = ggml_init(cpu_params);
    if (!cpu_ctx) {
        fprintf(stderr, "Failed to init CPU ggml context\n");
        return 1;
    }

    ggml_tensor * cpu_W = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * cpu_x = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, cols, 1);
    ggml_tensor * cpu_y = ggml_mul_mat(cpu_ctx, cpu_W, cpu_x);
    ggml_cgraph * cpu_gf = ggml_new_graph(cpu_ctx);
    ggml_build_forward_expand(cpu_gf, cpu_y);

    fill_inputs((float *) cpu_W->data, (float *) cpu_x->data, rows, cols);
    if (ggml_graph_compute_with_ctx(cpu_ctx, cpu_gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "CPU ggml_graph_compute_with_ctx failed\n");
        ggml_free(cpu_ctx);
        return 2;
    }

    std::vector<float> cpu_out(rows);
    std::memcpy(cpu_out.data(), cpu_y->data, rows * sizeof(float));

    const int device = 0;
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(device);
    if (!cuda_backend) {
        fprintf(stderr, "Failed to init CUDA backend\n");
        ggml_free(cpu_ctx);
        return 3;
    }

    std::vector<uint8_t> cuda_mem(mem_size);
    ggml_init_params cuda_params = { mem_size, cuda_mem.data(), true };
    ggml_context * cuda_ctx = ggml_init(cuda_params);
    if (!cuda_ctx) {
        fprintf(stderr, "Failed to init CUDA ggml context\n");
        ggml_backend_free(cuda_backend);
        ggml_free(cpu_ctx);
        return 4;
    }

    ggml_tensor * cuda_W = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * cuda_x = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, cols, 1);
    ggml_tensor * cuda_y = ggml_mul_mat(cuda_ctx, cuda_W, cuda_x);
    ggml_cgraph * cuda_gf = ggml_new_graph(cuda_ctx);
    ggml_build_forward_expand(cuda_gf, cuda_y);

    ggml_backend_buffer_t cuda_buf = ggml_backend_alloc_ctx_tensors(cuda_ctx, cuda_backend);
    if (!cuda_buf) {
        fprintf(stderr, "Failed to allocate CUDA tensors\n");
        ggml_free(cuda_ctx);
        ggml_backend_free(cuda_backend);
        ggml_free(cpu_ctx);
        return 5;
    }

    std::vector<float> input_w(rows * cols);
    std::vector<float> input_x(cols);
    fill_inputs(input_w.data(), input_x.data(), rows, cols);
    ggml_backend_tensor_set(cuda_W, input_w.data(), 0, input_w.size() * sizeof(float));
    ggml_backend_tensor_set(cuda_x, input_x.data(), 0, input_x.size() * sizeof(float));

    if (ggml_backend_graph_compute(cuda_backend, cuda_gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA ggml_backend_graph_compute failed\n");
        ggml_backend_buffer_free(cuda_buf);
        ggml_free(cuda_ctx);
        ggml_backend_free(cuda_backend);
        ggml_free(cpu_ctx);
        return 6;
    }

    std::vector<float> cuda_out(rows);
    ggml_backend_tensor_get(cuda_y, cuda_out.data(), 0, cuda_out.size() * sizeof(float));

    const float diff = max_abs_diff(cpu_out.data(), cuda_out.data(), rows);
    printf("CPU y first 8: ");
    for (int i = 0; i < 8 && i < rows; ++i) {
        printf("%g ", cpu_out[i]);
    }
    printf("\nCUDA y first 8: ");
    for (int i = 0; i < 8 && i < rows; ++i) {
        printf("%g ", cuda_out[i]);
    }
    printf("\nmax_abs_diff=%g\n", diff);

    ggml_backend_buffer_free(cuda_buf);
    ggml_free(cuda_ctx);
    ggml_backend_free(cuda_backend);
    ggml_free(cpu_ctx);
    return diff <= 1e-3f ? 0 : 7;
}
