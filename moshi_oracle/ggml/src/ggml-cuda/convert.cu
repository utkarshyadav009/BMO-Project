#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

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

    int64_t dequantized_cpu_ptr;

    // Relative byte offsets from the start of this struct
    int64_t packed_weights_offset;
    int64_t tile_tiers_offset;
    int64_t outlier_indices_offset;
    int64_t outlier_values_offset;
    int64_t tile_stream_indices_offset;

    // Outliers are stored sorted by flat index (row-major); CSR-style
    // per-row ranges (int32 x (rows+1)) live at this offset.
    // MUST stay in sync with moshi.cpp/src/loader.h and
    // ggml/src/ggml-quants.h.
    int64_t outlier_row_starts_offset;

    // Band-major packed stream layout (assembled in loader.h): per 64-row
    // tile band, per tier, tiles in tile-column order. band_table_offset
    // points at int32 absolute byte offsets [n_bands*4 + 1] (end sentinel).
    // band_layout: 1 = row-minor [ir][pos][slice], 2 = tile-major
    // [pos][ir][slice]. tile_stream_indices hold each tile's position
    // within its band+tier list.
    int64_t band_table_offset;
    int32_t band_layout;
    int32_t reserved2;
};

// slice = one tile row (64 elements) of packed data, per tier
static __device__ __forceinline__ int bmo_slice_bytes(int tier) {
    return (tier == 0) ? 128 : (tier == 1) ? 64 : (tier == 2) ? 32 : 16;
}

template <typename T>
static __global__ void dequantize_row_bmo_tier_cuda_kernel(const void * vx, T * y, const int64_t k, const int32_t cols_arg) {
    const block_bmo_tier * header = (const block_bmo_tier *) vx;

    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / 64;
    const int32_t n_tiles_total = (rows / 64) * n_tiles_col;

    const int tile_idx = blockIdx.x;
    if (tile_idx >= n_tiles_total) return;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_pos = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->band_table_offset);

    const uint8_t tier = tile_tiers[tile_idx];
    const int32_t pos = tile_pos[tile_idx];

    const int tile_r = tile_idx / n_tiles_col;   // == band index
    const int tile_c = tile_idx % n_tiles_col;
    const int row_base = tile_r * 64;
    const int col_base = tile_c * 64;

    const int sb = bmo_slice_bytes(tier);
    const int tab_i = tile_r * 4 + tier;
    const char * tb = (const char *)header + band_tab[tab_i];
    const int n_bt = (band_tab[tab_i + 1] - band_tab[tab_i]) / (64 * sb);
    const bool tile_major = header->band_layout == 2;

    const int thread_id = threadIdx.x;
    const int elements_per_thread = 4096 / blockDim.x;

    for (int step = 0; step < elements_per_thread; ++step) {
        const int in_tile_idx = thread_id + step * blockDim.x;
        const int in_tile_r = in_tile_idx >> 6;
        const int in_tile_c = in_tile_idx & 63;

        const int r = row_base + in_tile_r;
        const int c = col_base + in_tile_c;
        const int64_t out_idx = (int64_t)r * cols + c;

        if (out_idx >= k) continue;

        const int slice = tile_major ? (pos * 64 + in_tile_r) : (in_tile_r * n_bt + pos);
        float val = 0.0f;

        if (tier == 0) {
            val = __half2float(((const half *)tb)[slice * 64 + in_tile_c]);
        } else if (tier == 1) {
            uint8_t q = ((const uint8_t *)tb)[slice * 64 + in_tile_c];
            val = ((float)q - header->zp_int8) * header->scale_int8;
        } else if (tier == 2) {
            uint8_t b = ((const uint8_t *)tb)[slice * 32 + (in_tile_c >> 1)];
            uint8_t q = (in_tile_c & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
            val = ((float)q - header->zp_int4) * header->scale_int4;
        } else {
            uint8_t b = ((const uint8_t *)tb)[slice * 16 + (in_tile_c >> 2)];
            uint8_t q = (b >> ((in_tile_c & 3) * 2)) & 0x03;
            val = ((float)q - header->zp_low) * header->scale_low;
        }

        y[out_idx] = (T)val;
    }
    GGML_UNUSED(cols_arg);
}

template <typename T>
static __global__ void apply_outliers_bmo_tier_cuda_kernel_impl(const void * vx, T * y) {
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t n_outliers = header->n_outliers;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_outliers) return;

    const int32_t * outlier_indices = (const int32_t *)((const char *)header + header->outlier_indices_offset);
    const half * outlier_values = (const half *)((const char *)header + header->outlier_values_offset);

    y[outlier_indices[idx]] = (T)__half2float(outlier_values[idx]);
}

// ============================================================================
// Fused GEMV kernels for BMO_TIER: dequantize-on-the-fly + dot product with
// in-kernel outlier correction (CSR per-row ranges; no second kernel, no
// atomics). Rewritten for Orin (sm_87) memory efficiency — developed and
// gated against tools/bmo_kernel_bench.cu (moshi.cpp repo): 7.5-7.8x over
// the previous one-block-per-row kernel, rel_l2 < 2e-6 vs FP32 reference.
//
// Two kernels over the two band-major payload layouts (built in loader.h):
//  * tile-major [pos][ir][slice], cols <= 8192: one block per 64-row band,
//    warps partition the band's tiles and compute all 64 rows of each tile
//    into 8 row-octet register accumulators; per-warp per-row partials
//    combine through a small shared reduction. Every weight load is
//    contiguous 128-512B; every x load is 64B contiguous broadcast.
//  * row-minor [ir][pos][slice], cols > 8192: one 512-thread block per
//    band, 16 warps x 4 consecutive rows; wins when the x vector exceeds
//    the L1 working set.
//
// No I2F in the hot loops: a quant q becomes an exact float via
// bits = 0x40000000 | (q << t), u = 2 + q/2^(22-t), and the affine remainder
// s*(2^(23-t)+zp)*sum(x) folds into per-tile-column x sums computed once per
// block (exact refactoring of w = (q-zp)*s; changes only summation order).
// ============================================================================

#define BMO_GEMV_BLOCK_SIZE 256
#define BMO_TILE_DIM 64
#define BMO_TILE_ELEMS 4096
#define BMO_RM_THREADS 512
#define BMO_RM_WARPS   (BMO_RM_THREADS / 32)

// exact float 2 + q/2^(22-sh_left) from the quant bits in w; sh_left is a
// compile-time constant after unrolling (negative = right shift)
static __device__ __forceinline__ float bmo_uq(uint32_t w, int sh_left, uint32_t mask) {
    const uint32_t m = (sh_left >= 0) ? (w << sh_left) : (w >> (-sh_left));
    return __int_as_float((m & mask) | 0x40000000u);
}

// scalar dequant of one element from the band-major streams (outlier base_w)
static __device__ __forceinline__ float bmo_dequant_one(
    const block_bmo_tier * header, const char * tb, int tier, int slice, int cin)
{
    if (tier == 3) {
        const uint8_t b = ((const uint8_t *)tb)[slice * 16 + (cin >> 2)];
        return ((float)((b >> ((cin & 3) * 2)) & 3) - header->zp_low) * header->scale_low;
    } else if (tier == 2) {
        const uint8_t b = ((const uint8_t *)tb)[slice * 32 + (cin >> 1)];
        return ((float)((cin & 1) ? (b >> 4) : (b & 0x0F)) - header->zp_int4) * header->scale_int4;
    } else if (tier == 1) {
        return ((float)((const uint8_t *)tb)[slice * 64 + cin] - header->zp_int8) * header->scale_int8;
    }
    return __half2float(((const half *)tb)[slice * 64 + cin]);
}

// ---------------------------------------------------------------------------
// tile-major kernel: grid = n_bands, block = 256 (8 warps). Warp w owns
// tile-list slots w, w+8, ...; lane = (row-in-octet rg, chunk sub); each
// lane's load covers 16 consecutive elements of one row.
// ---------------------------------------------------------------------------
static __global__ void __launch_bounds__(BMO_GEMV_BLOCK_SIZE, 5) mul_mat_vec_bmo_tier_tilemajor_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t  s_tiers[512];
    __shared__ uint16_t s_pos[512];
    __shared__ float    s_xsum[512];
    __shared__ uint16_t s_list[4 * 512];
    __shared__ int      s_cnt4[4];
    __shared__ float    s_part[8 * BMO_TILE_DIM];

    // Hard bounds check: a silent return would produce zeroed output rows
    // indistinguishable from correct computation. __trap() fires a GPU
    // illegal-instruction exception visible in cuda-memcheck / NSight.
    if (n_tiles_col > 512 || header->band_layout != 2) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO tile-major GEMV kernel: n_tiles_col=%d (max 512) band_layout=%d (need 2)\n",
                   n_tiles_col, header->band_layout);
        }
        __syncthreads();
        __trap(); // hard abort — never silently skip
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rg   = lane >> 2;
    const int sub  = lane & 3;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->band_table_offset) + band * 4;

    for (int t = threadIdx.x; t < n_tiles_col; t += blockDim.x) {
        s_tiers[t] = tile_tiers[band * n_tiles_col + t];
    }
    __syncthreads();

    if (warp == 0) {
        // deterministic ballot scan: lists ordered by tile column
        int cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;
        for (int b = 0; b < n_tiles_col; b += 32) {
            const int tc = b + lane;
            const int tier = (tc < n_tiles_col) ? (int)s_tiers[tc] : -1;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const unsigned m = __ballot_sync(0xFFFFFFFF, tier == t);
                const int cnt = (t == 0) ? cnt0 : (t == 1) ? cnt1 : (t == 2) ? cnt2 : cnt3;
                if (tier == t) {
                    const int pos = cnt + __popc(m & ((1u << lane) - 1));
                    s_list[t * 512 + pos] = (uint16_t)tc;
                    s_pos[tc] = (uint16_t)pos;
                }
                const int add = __popc(m);
                if (t == 0) cnt0 += add; else if (t == 1) cnt1 += add; else if (t == 2) cnt2 += add; else cnt3 += add;
            }
        }
        if (lane == 0) { s_cnt4[0] = cnt0; s_cnt4[1] = cnt1; s_cnt4[2] = cnt2; s_cnt4[3] = cnt3; }
    } else {
        // per-tile-column sums of x for the affine fold
        for (int tc = warp - 1; tc < n_tiles_col; tc += 7) {
            float s = x_vec[tc * BMO_TILE_DIM + lane] + x_vec[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    // acc[m]: partial for row 8*m + rg over this warp's tiles, tier scale
    // applied per tile
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float a_x2 = 0.0f, a_x4 = 0.0f, a_x8 = 0.0f;

    // ---- INT2: wv[8] preload; x in two half-tile register passes ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[3]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            uint32_t wv[8];
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                wv[m] = __ldcs(w32 + (i * BMO_TILE_DIM + 8 * m + rg) * 4 + sub);
            }
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jjh = 0; jjh < 2; ++jjh) {
                const float4 xq0 = xp[2 * jjh];
                const float4 xq1 = xp[2 * jjh + 1];
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    float part = 0.0f;
                    #pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        const float4 & xq = jj ? xq1 : xq0;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int sh = 21 - 2 * (4 * (2 * jjh + jj) + k);
                            const float xk = (k == 0) ? xq.x : (k == 1) ? xq.y : (k == 2) ? xq.z : xq.w;
                            part = fmaf(bmo_uq(wv[m], sh, 0x00600000u), xk, part);
                        }
                    }
                    acc[m] = fmaf(2.0f * s2, part, acc[m]);
                }
            }
            if (lane == 0) a_x2 += s_xsum[tc];
        }
    }

    // ---- INT4: w preloaded in m-halves; x reloaded per element-half ----
    {
        const int n = s_cnt4[2];
        const uint2 * w64 = (const uint2 *)((const char *)header + band_tab[2]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[2 * 512 + i];
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int mh = 0; mh < 2; ++mh) {
                uint2 w2v[4];
                #pragma unroll
                for (int mm = 0; mm < 4; ++mm) {
                    w2v[mm] = __ldcs(w64 + (i * BMO_TILE_DIM + 8 * (4 * mh + mm) + rg) * 4 + sub);
                }
                #pragma unroll
                for (int half_i = 0; half_i < 2; ++half_i) {
                    const float4 xq0 = xp[2 * half_i];
                    const float4 xq1 = xp[2 * half_i + 1];
                    #pragma unroll
                    for (int mm = 0; mm < 4; ++mm) {
                        const int m = 4 * mh + mm;
                        const uint32_t w = half_i ? w2v[mm].y : w2v[mm].x;
                        float part = 0.0f;
                        #pragma unroll
                        for (int jj = 0; jj < 2; ++jj) {
                            const float4 & xq = jj ? xq1 : xq0;
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const int sh = 19 - 4 * (4 * jj + k);
                                const float xk = (k == 0) ? xq.x : (k == 1) ? xq.y : (k == 2) ? xq.z : xq.w;
                                part = fmaf(bmo_uq(w, sh, 0x00780000u), xk, part);
                            }
                        }
                        acc[m] = fmaf(8.0f * s4, part, acc[m]);
                    }
                }
            }
            if (lane == 0) a_x4 += s_xsum[tc];
        }
    }

    // ---- INT8: element-halves; w as uint2 per (m, half) ----
    {
        const int n = s_cnt4[1];
        const uint2 * w64 = (const uint2 *)((const char *)header + band_tab[1]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[1 * 512 + i];
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int qh = 0; qh < 2; ++qh) {
                const float4 xq0 = xp[2 * qh];
                const float4 xq1 = xp[2 * qh + 1];
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const uint2 w2 = __ldcs(w64 + (i * BMO_TILE_DIM + 8 * m + rg) * 8 + 2 * sub + qh);
                    float part = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t w = q ? w2.y : w2.x;
                        const float4 & xq = q ? xq1 : xq0;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int sh = 15 - 8 * k;
                            const float xk = (k == 0) ? xq.x : (k == 1) ? xq.y : (k == 2) ? xq.z : xq.w;
                            part = fmaf(bmo_uq(w, sh, 0x007F8000u), xk, part);
                        }
                    }
                    acc[m] = fmaf(128.0f * s8, part, acc[m]);
                }
            }
            if (lane == 0) a_x8 += s_xsum[tc];
        }
    }

    // ---- FP16: x loaded per row-octet as float2 pairs (rare tier) ----
    {
        const int n = s_cnt4[0];
        const uint4 * w128 = (const uint4 *)((const char *)header + band_tab[0]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[0 * 512 + i];
            const float * xb = x_vec + tc * BMO_TILE_DIM + 16 * sub;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                const uint4 wa = __ldcs(w128 + (i * BMO_TILE_DIM + 8 * m + rg) * 8 + sub * 2);
                const uint4 wb = __ldcs(w128 + (i * BMO_TILE_DIM + 8 * m + rg) * 8 + sub * 2 + 1);
                const uint32_t ws[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
                float part = 0.0f;
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const float2 wf = __half22float2(*(const half2 *)&ws[q]);
                    const float2 xv2 = *(const float2 *)(xb + 2 * q);
                    part = fmaf(wf.x, xv2.x, part);
                    part = fmaf(wf.y, xv2.y, part);
                }
                acc[m] += part;
            }
        }
    }

    // row-independent -k2*sum(x) fold: each warp folded only ITS tiles'
    // xsum, so the cross-warp sum below carries the full fold exactly once
    const float fold = - (s2 * (4.0f   + z2)) * a_x2
                       - (s4 * (16.0f  + z4)) * a_x4
                       - (s8 * (256.0f + z8)) * a_x8;
    const float fold_l0 = __shfl_sync(0xFFFFFFFF, fold, 0);

    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        float v = acc[m];
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        if (sub == 0) s_part[warp * BMO_TILE_DIM + 8 * m + rg] = v + fold_l0;
    }
    __syncthreads();

    // final phase: warp w owns output rows 8w..8w+7; sum the 8 warps'
    // partials, apply this row's outlier corrections, write
    {
        const int ir = warp * 8 + rg;
        const int row = row_base + ir;
        const bool live = row < rows;

        float tot = s_part[sub * BMO_TILE_DIM + ir]
                  + s_part[(sub + 4) * BMO_TILE_DIM + ir];

        if (live) {
            const int32_t * row_starts = (const int32_t *)((const char *)header + header->outlier_row_starts_offset);
            const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
            const half * ov = (const half *)((const char *)header + header->outlier_values_offset);
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + sub; k < e1; k += 4) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int slice = (int)s_pos[tc] * BMO_TILE_DIM + ir;
                const char * tb = (const char *)header + band_tab[tier];
                const float base_w = bmo_dequant_one(header, tb, tier, slice, col & 63);
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        tot += __shfl_down_sync(0xFFFFFFFF, tot, 1);
        tot += __shfl_down_sync(0xFFFFFFFF, tot, 2);
        if (sub == 0 && live) y_out[row] = tot;
    }
    GGML_UNUSED(ncols);
}

// ---------------------------------------------------------------------------
// row-minor kernel: grid = n_bands, block = 512 (16 warps x 4 consecutive
// rows). Weight loads span 8 lane-quads' contiguous slices; used when the
// x vector is too large for the tile-major kernel's L1 reuse pattern.
// ---------------------------------------------------------------------------
static __global__ void __launch_bounds__(BMO_RM_THREADS, 2) mul_mat_vec_bmo_tier_rowminor_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t  s_tiers[512];
    __shared__ uint16_t s_pos[512];
    __shared__ float    s_xsum[512];
    __shared__ uint16_t s_list[4 * 512];
    __shared__ int      s_cnt4[4];

    if (n_tiles_col > 512 || header->band_layout != 1) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO row-minor GEMV kernel: n_tiles_col=%d (max 512) band_layout=%d (need 1)\n",
                   n_tiles_col, header->band_layout);
        }
        __syncthreads();
        __trap(); // hard abort — never silently skip
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->band_table_offset) + band * 4;

    for (int t = threadIdx.x; t < n_tiles_col; t += blockDim.x) {
        s_tiers[t] = tile_tiers[band * n_tiles_col + t];
    }
    __syncthreads();

    if (warp == 0) {
        int cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;
        for (int b = 0; b < n_tiles_col; b += 32) {
            const int tc = b + lane;
            const int tier = (tc < n_tiles_col) ? (int)s_tiers[tc] : -1;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const unsigned m = __ballot_sync(0xFFFFFFFF, tier == t);
                const int cnt = (t == 0) ? cnt0 : (t == 1) ? cnt1 : (t == 2) ? cnt2 : cnt3;
                if (tier == t) {
                    const int pos = cnt + __popc(m & ((1u << lane) - 1));
                    s_list[t * 512 + pos] = (uint16_t)tc;
                    s_pos[tc] = (uint16_t)pos;
                }
                const int add = __popc(m);
                if (t == 0) cnt0 += add; else if (t == 1) cnt1 += add; else if (t == 2) cnt2 += add; else cnt3 += add;
            }
        }
        if (lane == 0) { s_cnt4[0] = cnt0; s_cnt4[1] = cnt1; s_cnt4[2] = cnt2; s_cnt4[3] = cnt3; }
    } else {
        for (int tc = warp - 1; tc < n_tiles_col; tc += BMO_RM_WARPS - 1) {
            float s = x_vec[tc * BMO_TILE_DIM + lane] + x_vec[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    // warp handles 4 consecutive rows of the band
    const int r0 = row_base + warp * 4;
    if (r0 >= rows) return;
    const int ir0 = warp * 4;

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float accf[4] = {0, 0, 0, 0};
    float au2[4]  = {0, 0, 0, 0};
    float au4[4]  = {0, 0, 0, 0};
    float au8[4]  = {0, 0, 0, 0};
    float a_x2 = 0.0f, a_x4 = 0.0f, a_x8 = 0.0f;

    // ---- INT2: 8 tiles/step, 4 lanes/tile, 16 elems/lane/uint32, 4 rows ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[3]);
        const int g = lane >> 2, sub = lane & 3;
        for (int i = g; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = __ldcs(w32 + ((ir0 + r) * n + i) * 4 + sub);
            }
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 xv = xp[jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 21 - 2 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au2[r] = fmaf(bmo_uq(w[r], sh, 0x00600000u), xk, au2[r]);
                    }
                }
            }
            if (sub == 0) a_x2 += s_xsum[tc];
        }
    }

    // ---- INT4: 4 tiles/step, 8 lanes/tile, 8 elems/lane/uint32, 4 rows ----
    {
        const int n = s_cnt4[2];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[2]);
        const int g = lane >> 3, sub = lane & 7;
        for (int i = g; i < n; i += 4) {
            const int tc = s_list[2 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = __ldcs(w32 + ((ir0 + r) * n + i) * 8 + sub);
            }
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 8 * sub);
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const float4 xv = xp[jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 19 - 4 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au4[r] = fmaf(bmo_uq(w[r], sh, 0x00780000u), xk, au4[r]);
                    }
                }
            }
            if (sub == 0) a_x4 += s_xsum[tc];
        }
    }

    // ---- INT8: 2 tiles/step, 16 lanes/tile, 4 elems/lane/uint32, 4 rows ----
    {
        const int n = s_cnt4[1];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[1]);
        const int g = lane >> 4, sub = lane & 15;
        for (int i = g; i < n; i += 2) {
            const int tc = s_list[1 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = __ldcs(w32 + ((ir0 + r) * n + i) * 16 + sub);
            }
            const float4 xv = *(const float4 *)(x_vec + tc * BMO_TILE_DIM + 4 * sub);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int sh = 15 - 8 * k;
                const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    au8[r] = fmaf(bmo_uq(w[r], sh, 0x007F8000u), xk, au8[r]);
                }
            }
            if (sub == 0) a_x8 += s_xsum[tc];
        }
    }

    // ---- FP16: 1 tile/step, 2 elems/lane, 4 rows ----
    {
        const int n = s_cnt4[0];
        const half2 * f16 = (const half2 *)((const char *)header + band_tab[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const float2 xv = *(const float2 *)(x_vec + tc * BMO_TILE_DIM + 2 * lane);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float2 wf = __half22float2(f16[((ir0 + r) * n + i) * 32 + lane]);
                accf[r] = fmaf(wf.x, xv.x, accf[r]);
                accf[r] = fmaf(wf.y, xv.y, accf[r]);
            }
        }
    }

    const int32_t * row_starts = (const int32_t *)((const char *)header + header->outlier_row_starts_offset);
    const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
    const half * ov = (const half *)((const char *)header + header->outlier_values_offset);

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = r0 + r;
        if (row >= rows) break;
        float tot = accf[r]
            + (2.0f   * s2) * au2[r] - (s2 * (4.0f   + z2)) * a_x2
            + (8.0f   * s4) * au4[r] - (s4 * (16.0f  + z4)) * a_x4
            + (128.0f * s8) * au8[r] - (s8 * (256.0f + z8)) * a_x8;

        {
            const int ir = ir0 + r;
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + lane; k < e1; k += 32) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int nn = (band_tab[tier + 1] - band_tab[tier]) / (BMO_TILE_DIM * bmo_slice_bytes(tier));
                const int slice = ir * nn + (int)s_pos[tc];
                const char * tb = (const char *)header + band_tab[tier];
                const float base_w = bmo_dequant_one(header, tb, tier, slice, col & 63);
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xFFFFFFFF, tot, o);
        if (lane == 0) y_out[row] = tot;
    }
    GGML_UNUSED(ncols);
}

void mul_mat_vec_bmo_tier_cuda(
    const void * vx, const float * x_vec, float * y_out,
    const int32_t nrows, const int32_t ncols, const int32_t n_outliers, cudaStream_t stream)
{
    // Shape dispatch — MUST match the band layout rule in loader.h
    // build_custom_ffn_tensor (cols <= 8192 -> tile-major payload).
    // Outlier correction is fused in-kernel (CSR ranges); no second launch.
    const int n_blocks = (nrows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    if (ncols <= 8192) {
        mul_mat_vec_bmo_tier_tilemajor_kernel<<<n_blocks, BMO_GEMV_BLOCK_SIZE, 0, stream>>>(
            vx, x_vec, y_out, ncols);
    } else {
        mul_mat_vec_bmo_tier_rowminor_kernel<<<n_blocks, BMO_RM_THREADS, 0, stream>>>(
            vx, x_vec, y_out, ncols);
    }
    CUDA_CHECK(cudaGetLastError());
    GGML_UNUSED(n_outliers);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = blockIdx.y;
    const int64_t i02 = blockIdx.z % ne02;
    const int64_t i03 = blockIdx.z / ne02;

    const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    const int64_t ib = ibx0 + i00/qk; // block index
    const int64_t iqs = (i00%qk)/qr; // quant index
    const int64_t iybs = i00 - i00%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    float2 v;
    dequantize_kernel(vx, ib, iqs, v);

    const int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
    y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __half2float(x[i].data.d);     // RMS scale of Group 0
    float dmin = __half2float(x[i].data.dmin);  // RMS scale of Group 1
    const float centroids[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};

    __shared__ float s_mem[256];

    s_mem[128*n + l + 0]  = centroids[(q >> 0) & 3];
    s_mem[128*n + l + 32] = centroids[(q >> 2) & 3];
    s_mem[128*n + l + 64] = centroids[(q >> 4) & 3];
    s_mem[128*n + l + 96] = centroids[(q >> 6) & 3];

    __syncthreads();

    // Parallel in-place Walsh-Hadamard Transform of size 128 on group 0 and group 1
    for (int h = 1; h < 128; h *= 2) {
        int j_g0 = (tid / h) * (2 * h) + (tid % h);
        int j_g1 = j_g0 + 128;

        float a0 = s_mem[j_g0];
        float b0 = s_mem[j_g0 + h];
        s_mem[j_g0] = a0 + b0;
        s_mem[j_g0 + h] = a0 - b0;

        float a1 = s_mem[j_g1];
        float b1 = s_mem[j_g1 + h];
        s_mem[j_g1] = a1 + b1;
        s_mem[j_g1 + h] = a1 - b1;

        __syncthreads();
    }

    // Apply normalization, group scale & write out to destination
    float scale = (n == 0 ? dall : dmin) * 0.0883883476f;
    y[l+ 0] = ggml_cuda_cast<dst_t>(s_mem[128*n + l + 0] * scale);
    y[l+32] = ggml_cuda_cast<dst_t>(s_mem[128*n + l + 32] * scale);
    y[l+64] = ggml_cuda_cast<dst_t>(s_mem[128*n + l + 64] * scale);
    y[l+96] = ggml_cuda_cast<dst_t>(s_mem[128*n + l + 96] * scale);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), ne01, ne02*ne03);
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne02, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_bmo_tier_cuda_impl(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    int32_t cols = 4096;
    if (k == 46137344) {
        cols = 11264;
    }
    const int block_size = 256;
    const int grid_size = k / 4096;
    dequantize_row_bmo_tier_cuda_kernel<dst_t><<<grid_size, block_size, 0, stream>>>(vx, y, k, cols);
    CUDA_CHECK(cudaGetLastError());

    const int outlier_grid_size = 2048;
    apply_outliers_bmo_tier_cuda_kernel_impl<dst_t><<<outlier_grid_size, block_size, 0, stream>>>(vx, y);
    CUDA_CHECK(cudaGetLastError());
}

static void dequantize_row_bmo_tier_cuda(const void * vx, float * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_bmo_tier_cuda_impl<float>(vx, y, k, stream);
}

static void dequantize_row_bmo_tier_cuda_f16(const void * vx, half * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_bmo_tier_cuda_impl<half>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = blockIdx.y;
    const int64_t i02 = blockIdx.z % ne02;
    const int64_t i03 = blockIdx.z / ne02;

    const src_t * x = (const src_t *) vx;

    const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
    const int64_t iy = ((i03*ne02 + i02)*ne01 + i01)*ne00 + i00;
    y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, ne01, ne02*ne03);
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne02, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        case GGML_TYPE_BMO_TIER:
            return dequantize_row_bmo_tier_cuda_f16;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        case GGML_TYPE_BMO_TIER:
            return dequantize_row_bmo_tier_cuda;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
