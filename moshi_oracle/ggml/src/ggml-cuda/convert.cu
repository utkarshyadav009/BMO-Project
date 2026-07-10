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
};

template <typename T>
static __global__ void dequantize_row_bmo_tier_cuda_kernel(const void * vx, T * y, const int64_t k, const int32_t cols) {
    const block_bmo_tier * header = (const block_bmo_tier *) vx;

    const int32_t rows = header->rows;
    const int32_t n_tiles_col = cols / 64;
    const int32_t n_tiles_total = (rows / 64) * n_tiles_col;

    const int tile_idx = blockIdx.x;
    if (tile_idx >= n_tiles_total) return;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    const uint8_t tier = tile_tiers[tile_idx];
    const int32_t stream_idx = tile_stream_indices[tile_idx];

    const int tile_r = tile_idx / n_tiles_col;
    const int tile_c = tile_idx % n_tiles_col;
    const int row_base = tile_r * 64;
    const int col_base = tile_c * 64;

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

        float val = 0.0f;

        if (tier == 0) {
            // FP16 stream
            const half * raw_fp16 = (const half *)(pw + header->tier_offsets[0]);
            val = __half2float(raw_fp16[stream_idx * 4096 + in_tile_idx]);
        } else if (tier == 1) {
            // INT8 stream
            const uint8_t * raw_int8 = pw + header->tier_offsets[1];
            uint8_t q = raw_int8[stream_idx * 4096 + in_tile_idx];
            val = ((float)q - header->zp_int8) * header->scale_int8;
        } else if (tier == 2) {
            // INT4 stream
            const uint8_t * raw_int4 = pw + header->tier_offsets[2];
            int flat_idx = stream_idx * 4096 + in_tile_idx;
            uint8_t b = raw_int4[flat_idx / 2];
            uint8_t q = (flat_idx % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
            val = ((float)q - header->zp_int4) * header->scale_int4;
        } else if (tier == 3) {
            // INT2 stream
            const uint8_t * raw_int2 = pw + header->tier_offsets[3];
            int flat_idx = stream_idx * 4096 + in_tile_idx;
            uint8_t b = raw_int2[flat_idx / 4];
            uint8_t q = (flat_idx % 4 == 0) ? (b & 0x03) :
                        ((flat_idx % 4 == 1) ? ((b >> 2) & 0x03) :
                         ((flat_idx % 4 == 2) ? ((b >> 4) & 0x03) : ((b >> 6) & 0x03)));
            val = ((float)q - header->zp_low) * header->scale_low;
        }

        y[out_idx] = (T)val;
    }
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
// Fused GEMV kernel for BMO_TIER: dequantize-on-the-fly + dot product
// Eliminates the FP16 intermediate buffer entirely.
//
// Grid:  (nrows, 1, 1)   — one block per output row
// Block: (256, 1, 1)     — 256 threads per block = 8 warps
//
// Each block computes: y[row] = sum_over_cols( W[row, col] * x[col] ) + outlier_corrections
//
// The weight matrix is stored in 64x64 tiles. For a given output row, we iterate
// over the tile columns that span that row, dequantize each weight element
// on-the-fly from the packed stream, multiply by x[col], and accumulate.
// ============================================================================

#define BMO_GEMV_BLOCK_SIZE 256
#define BMO_TILE_DIM 64
#define BMO_TILE_ELEMS 4096

static __device__ __forceinline__ float bmo_dequant_element_fast(
    const uint8_t * __restrict__ pw,
    const int32_t * __restrict__ tier_offsets,
    const float * __restrict__ scales_and_zps,
    const uint8_t tier,
    const int32_t stream_idx,
    const int in_tile_idx)
{
    float val = 0.0f;
    if (tier == 0) {
        const half * raw_fp16 = (const half *)(pw + tier_offsets[0]);
        val = __half2float(raw_fp16[stream_idx * BMO_TILE_ELEMS + in_tile_idx]);
    } else if (tier == 1) {
        const uint8_t * raw_int8 = pw + tier_offsets[1];
        uint8_t q = raw_int8[stream_idx * BMO_TILE_ELEMS + in_tile_idx];
        val = ((float)q - scales_and_zps[1]) * scales_and_zps[0];
    } else if (tier == 2) {
        const uint8_t * raw_int4 = pw + tier_offsets[2];
        int flat_idx = stream_idx * BMO_TILE_ELEMS + in_tile_idx;
        uint8_t b = raw_int4[flat_idx >> 1];
        uint8_t q = (flat_idx & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
        val = ((float)q - scales_and_zps[3]) * scales_and_zps[2];
    } else { // tier == 3 (INT2)
        const uint8_t * raw_int2 = pw + tier_offsets[3];
        int flat_idx = stream_idx * BMO_TILE_ELEMS + in_tile_idx;
        uint8_t b = raw_int2[flat_idx >> 2];
        int shift = (flat_idx & 3) * 2;
        uint8_t q = (b >> shift) & 0x03;
        val = ((float)q - scales_and_zps[5]) * scales_and_zps[4];
    }
    return val;
}

static __global__ void mul_mat_vec_bmo_tier_cuda_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const int row = blockIdx.x;
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    // Allocate shared memory for caching tiles, tier offsets, scales, and zero-points
    // to avoid redundant global memory reads in the loop.
    // Size increased to 512 to support large hidden dimensions (up to cols = 32,768).
    __shared__ uint8_t s_tiers[512];
    __shared__ uint16_t s_stream_indices[512];
    __shared__ int32_t s_tier_offsets[5];
    __shared__ float s_scales_and_zps[6];

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    const int tile_row = row / BMO_TILE_DIM;
    const int in_tile_r = row % BMO_TILE_DIM;
    const int in_tile_row_base = in_tile_r * BMO_TILE_DIM;

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && row == 0) {
            printf("ERROR: BMO fused GEMV kernel exceeded maximum tile column bounds (%d > 512)\n", n_tiles_col);
        }
        return;
    }

    // Parallel load into shared memory
    for (int t = threadIdx.x; t < n_tiles_col; t += blockDim.x) {
        const int tile_idx = tile_row * n_tiles_col + t;
        s_tiers[t] = tile_tiers[tile_idx];
        s_stream_indices[t] = tile_stream_indices[tile_idx];
    }

    if (threadIdx.x < 5) {
        s_tier_offsets[threadIdx.x] = header->tier_offsets[threadIdx.x];
    }
    if (threadIdx.x == 5) {
        s_scales_and_zps[0] = header->scale_int8;
        s_scales_and_zps[1] = header->zp_int8;
        s_scales_and_zps[2] = header->scale_int4;
        s_scales_and_zps[3] = header->zp_int4;
        s_scales_and_zps[4] = header->scale_low;
        s_scales_and_zps[5] = header->zp_low;
    }
    __syncthreads();

    float thread_sum = 0.0f;

    // Each thread iterates over assigned columns across all tile columns
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int tile_c = col / BMO_TILE_DIM;
        const int in_tile_c = col % BMO_TILE_DIM;
        
        // Fast reads from shared memory
        const uint8_t tier = s_tiers[tile_c];
        const int32_t stream_idx = s_stream_indices[tile_c];
        const int in_tile_idx = in_tile_row_base + in_tile_c;

        float w = bmo_dequant_element_fast(pw, s_tier_offsets, s_scales_and_zps, tier, stream_idx, in_tile_idx);
        thread_sum += w * x_vec[col];
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    // Inter-warp reduction via shared memory
    __shared__ float warp_sums[BMO_GEMV_BLOCK_SIZE / 32];  // 8 warps

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        float val = (lane_id < (BMO_GEMV_BLOCK_SIZE / 32)) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = (BMO_GEMV_BLOCK_SIZE / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            y_out[row] = val;
        }
    }
}

// Separate kernel to apply outlier corrections to the GEMV output
static __global__ void apply_outliers_gemv_bmo_tier_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t n_outliers = header->n_outliers;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_outliers) return;

    const int32_t * outlier_indices = (const int32_t *)((const char *)header + header->outlier_indices_offset);
    const half * outlier_values = (const half *)((const char *)header + header->outlier_values_offset);

    // outlier_indices[idx] is a flat index into the weight matrix (row * cols + col)
    const int32_t flat_idx = outlier_indices[idx];
    const int32_t row = flat_idx / ncols;
    const int32_t col = flat_idx % ncols;
    const float outlier_w = __half2float(outlier_values[idx]);

    // Calculate base_w to subtract the base contribution from the tile streams
    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    const int32_t n_tiles_col = ncols / BMO_TILE_DIM;
    const int tile_row = row / BMO_TILE_DIM;
    const int tile_col = col / BMO_TILE_DIM;
    const int tile_idx = tile_row * n_tiles_col + tile_col;

    const uint8_t tier = tile_tiers[tile_idx];
    const int32_t stream_idx = tile_stream_indices[tile_idx];

    const int in_tile_r = row % BMO_TILE_DIM;
    const int in_tile_c = col % BMO_TILE_DIM;
    const int in_tile_idx = in_tile_r * BMO_TILE_DIM + in_tile_c;

    float base_w = 0.0f;
    if (tier == 0) {
        const half * raw_fp16 = (const half *)(pw + header->tier_offsets[0]);
        base_w = __half2float(raw_fp16[stream_idx * BMO_TILE_ELEMS + in_tile_idx]);
    } else if (tier == 1) {
        const uint8_t * raw_int8 = pw + header->tier_offsets[1];
        uint8_t q = raw_int8[stream_idx * BMO_TILE_ELEMS + in_tile_idx];
        base_w = ((float)q - header->zp_int8) * header->scale_int8;
    } else if (tier == 2) {
        const uint8_t * raw_int4 = pw + header->tier_offsets[2];
        int flat_idx_in = stream_idx * BMO_TILE_ELEMS + in_tile_idx;
        uint8_t b = raw_int4[flat_idx_in >> 1];
        uint8_t q = (flat_idx_in & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
        base_w = ((float)q - header->zp_int4) * header->scale_int4;
    } else { // tier == 3 (INT2)
        const uint8_t * raw_int2 = pw + header->tier_offsets[3];
        int flat_idx_in = stream_idx * BMO_TILE_ELEMS + in_tile_idx;
        uint8_t b = raw_int2[flat_idx_in >> 2];
        int shift = (flat_idx_in & 3) * 2;
        uint8_t q = (b >> shift) & 0x03;
        base_w = ((float)q - header->zp_low) * header->scale_low;
    }

    // Mathematically correct delta: (outlier_w - base_w) * x_vec[col]
    atomicAdd(&y_out[row], (outlier_w - base_w) * x_vec[col]);
}

void mul_mat_vec_bmo_tier_cuda(
    const void * vx, const float * x_vec, float * y_out,
    const int32_t nrows, const int32_t ncols, const int32_t n_outliers, cudaStream_t stream)
{
    // Launch main GEMV kernel: one block per row
    mul_mat_vec_bmo_tier_cuda_kernel<<<nrows, BMO_GEMV_BLOCK_SIZE, 0, stream>>>(
        vx, x_vec, y_out, ncols);
    CUDA_CHECK(cudaGetLastError());

    // Apply outlier corrections dynamically based on actual outlier count
    if (n_outliers > 0) {
        const int outlier_block_size = 256;
        const int outlier_grid_size = (n_outliers + outlier_block_size - 1) / outlier_block_size;
        apply_outliers_gemv_bmo_tier_kernel<<<outlier_grid_size, outlier_block_size, 0, stream>>>(
            vx, x_vec, y_out, ncols);
        CUDA_CHECK(cudaGetLastError());
    }
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
