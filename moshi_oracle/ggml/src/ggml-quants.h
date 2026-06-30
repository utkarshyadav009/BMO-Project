#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void iq2xs_init_impl(enum ggml_type type);
GGML_API void iq2xs_free_impl(enum ggml_type type);
GGML_API void iq3xs_init_impl(int grid_size);
GGML_API void iq3xs_free_impl(int grid_size);

struct block_bmo_tier {
    int32_t rows;
    int32_t cols;
    int32_t n_tiles[4];      // n_fp16, n_int8, n_int4, n_int2
    int32_t tier_offsets[5];
    float scale_int8;
    float zp_int8;
    float scale_int4;
    float zp_int4;
    float scale_low;
    float zp_low;
    int32_t n_outliers;
    int32_t padding;

    int64_t dequantized_cpu_ptr; // Store the CPU pointer to dequantized float weights

    // Relative byte offsets from the start of this struct
    int64_t packed_weights_offset;
    int64_t tile_tiers_offset;
    int64_t outlier_indices_offset;
    int64_t outlier_values_offset;
    int64_t tile_stream_indices_offset;
};

GGML_API void dequantize_row_bmo_tier(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
