// Standalone microbenchmark for mul_mat_vec_bmo_tier kernel variants (sm_87).
//
// Loads ONE real BMO_TIER tensor payload from the GGUF exactly as
// src/loader.h build_custom_ffn_tensor() assembles it (struct layout, field
// alignment and tile_stream_indices computation are replicated verbatim from
// there — keep in sync), builds a CPU FP32 reference via the same math as
// loader.h dequantize_ffn_cpu(), and reports per variant:
//     max_abs_diff, rel_l2 vs reference, ms/call (median of 100, cudaEvent),
//     achieved GB/s = shipped_payload_bytes / time.
// Gate for any variant: rel_l2 < 1e-5 (summation-order changes expected;
// bit-identity is NOT required here — model-level gates come at integration).
//
// Usage: bmo_kernel_bench <model.gguf> [tensor_base_name]
//   default tensor: transformer_layers_0_gating_linear_in_weight

#include "gguf.h"
#include "ggml.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#define CU_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// ============================================================================
// Shipped payload header — must match src/loader.h and ggml-cuda/convert.cu.
// ============================================================================
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

// The fused-outlier variant (V3) appends a CSR-style per-row outlier range
// table (int32 x (rows+1), outliers sorted by flat index) after the shipped
// payload; in the bench its offset rides in the unused dequantized_cpu_ptr
// header slot. Integration adds a real header field to loader.h/convert.cu.

// ============================================================================
// GGUF raw reading (metadata via libggml-base, bytes via fread — same file
// access pattern as loader.h read_raw_bytes_from_gguf_file()).
// ============================================================================
struct GgufReader {
    gguf_context * g = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE * f = nullptr;
    size_t data_off = 0;
    std::string path;

    bool open(const char * fname) {
        path = fname;
        gguf_init_params params;
        params.no_alloc = true;
        params.ctx = &meta_ctx;
        g = gguf_init_from_file(fname, params);
        if (!g) return false;
        f = fopen(fname, "rb");
        if (!f) return false;
        data_off = gguf_get_data_offset(g);
        return true;
    }

    bool tensor_bytes(const std::string & name, std::vector<uint8_t> & out) {
        int64_t tid = gguf_find_tensor(g, name.c_str());
        if (tid < 0) { out.clear(); return false; }
        size_t off = gguf_get_tensor_offset(g, tid);
        size_t nbytes = gguf_get_tensor_size(g, tid);
        out.resize(nbytes);
        if (nbytes == 0) return true;
        if (fseek(f, (long)(data_off + off), SEEK_SET) != 0) return false;
        return fread(out.data(), nbytes, 1, f) == 1;
    }

    int32_t scalar_i32(const std::string & name, int32_t def_val = 0) {
        std::vector<uint8_t> b;
        if (!tensor_bytes(name, b) || b.size() < 4) return def_val;
        int32_t v; memcpy(&v, b.data(), 4); return v;
    }
    float scalar_f32(const std::string & name, float def_val = 0.0f) {
        std::vector<uint8_t> b;
        if (!tensor_bytes(name, b) || b.size() < 4) return def_val;
        float v; memcpy(&v, b.data(), 4); return v;
    }
};

// ============================================================================
// Payload assembly — replicated from loader.h build_custom_ffn_tensor().
// ============================================================================
struct BmoTensor {
    block_bmo_tier header;
    std::vector<uint8_t> payload;      // shipped format
    std::vector<uint8_t> payload_v3;   // extended: outliers sorted + CSR row starts
    size_t shipped_bytes = 0;          // GB/s denominator (payload as shipped)
    // raw components kept for the CPU reference
    std::vector<uint8_t> pw, tt, oi, ov;
    // V6: band-major repacked payload, [ir][pos] within band (see build_repacked_payload)
    std::vector<uint8_t> payload_v6;
    // V9: band-major repacked payload, [pos][ir] within band (tile-major)
    std::vector<uint8_t> payload_v9;
};

// Repacked payload: shipped payload + CSR (as V3) + a per-band tier base
// table + the packed streams repacked band-major: for each 64-row tile band,
// each tier's tiles (in tile-column order, matching the kernel's list order)
// are stored either [in_tile_row][tile_position][row slice] (V6,
// tile_major=false) or [tile_position][in_tile_row][row slice] (V9,
// tile_major=true — one warp load spans 8 consecutive rows of one tile).
// Table entries are absolute byte offsets from payload start; the table's
// own offset rides in header.padding (bench-local; the integration header
// gets real fields).
static void build_repacked_payload(BmoTensor & t, bool tile_major, std::vector<uint8_t> & out) {
    const block_bmo_tier & h = t.header;
    const int32_t rows = h.rows, cols = h.cols;
    const int32_t n_tiles_col = cols / 64;
    const int32_t n_bands = rows / 64;
    const uint8_t * tt = t.tt.data();
    const uint8_t * pw_old = t.pw.data();

    // per-tier slice bytes for one 64-element row slice
    const int slice_bytes[4] = { 128, 64, 32, 16 }; // fp16, int8, int4, int2

    size_t tab_off = t.payload_v3.size(); // 16-aligned? v3 ends at CSR (rows+1)*4; align up
    tab_off = (tab_off + 15) & ~(size_t)15;
    size_t tab_bytes = (size_t)n_bands * 4 * sizeof(int32_t);
    size_t stream_off = (tab_off + tab_bytes + 15) & ~(size_t)15;

    // compute total repacked size = sum over bands/tiers of n_bt * 64 * slice
    std::vector<int32_t> band_tab(n_bands * 4);
    size_t cur = stream_off;
    std::vector<std::vector<uint16_t>> band_lists(4);
    std::vector<int32_t> stream_ctr(4, 0); // running stream index per tier (global tc order)
    // stream indices are assigned globally in tile-index order (loader logic),
    // so recompute them per tile as we walk bands in order.
    std::vector<uint16_t> tile_stream(n_bands * n_tiles_col);
    { int32_t p[4] = {0,0,0,0};
      for (int i = 0; i < n_bands * n_tiles_col; ++i) tile_stream[i] = (uint16_t)p[tt[i]]++; }

    for (int b = 0; b < n_bands; ++b) {
        for (int tier = 0; tier < 4; ++tier) {
            band_tab[b * 4 + tier] = (int32_t)cur;
            int n_bt = 0;
            for (int tc = 0; tc < n_tiles_col; ++tc) if (tt[b * n_tiles_col + tc] == tier) n_bt++;
            cur += (size_t)n_bt * 64 * slice_bytes[tier];
        }
        cur = (cur + 15) & ~(size_t)15; // keep every band 16-aligned (sizes already are)
    }

    out.assign(cur, 0);
    memcpy(out.data(), t.payload_v3.data(), t.payload_v3.size());
    block_bmo_tier * hb = (block_bmo_tier *)out.data();
    hb->padding = (int32_t)tab_off;
    memcpy(out.data() + tab_off, band_tab.data(), tab_bytes);

    // repack: dst[band][tier][ir][pos][slice] (or [pos][ir] for tile_major)
    //         <- src[stream(tile)][ir][slice]
    for (int b = 0; b < n_bands; ++b) {
        int pos[4] = {0, 0, 0, 0};
        int n_bt[4] = {0, 0, 0, 0};
        for (int tc = 0; tc < n_tiles_col; ++tc) n_bt[tt[b * n_tiles_col + tc]]++;
        for (int tc = 0; tc < n_tiles_col; ++tc) {
            const int tile_idx = b * n_tiles_col + tc;
            const int tier = tt[tile_idx];
            const int stream = tile_stream[tile_idx];
            const int sb = slice_bytes[tier];
            const uint8_t * src_tile = pw_old + h.tier_offsets[tier] + (size_t)stream * 64 * sb;
            uint8_t * dst_base = out.data() + band_tab[b * 4 + tier];
            for (int ir = 0; ir < 64; ++ir) {
                const size_t dst_slice = tile_major ? ((size_t)pos[tier] * 64 + ir)
                                                    : ((size_t)ir * n_bt[tier] + pos[tier]);
                memcpy(dst_base + dst_slice * sb, src_tile + (size_t)ir * sb, sb);
            }
            pos[tier]++;
        }
    }
}

static bool load_bmo_tensor(GgufReader & r, const std::string & base_name, BmoTensor & t) {
    if (gguf_find_tensor(r.g, (base_name + ".packed_weights").c_str()) < 0) {
        fprintf(stderr, "error: tensor %s.packed_weights not found in %s\n", base_name.c_str(), r.path.c_str());
        return false;
    }
    int32_t rows = r.scalar_i32(base_name + ".rows");
    int32_t cols = r.scalar_i32(base_name + ".cols");
    int32_t n_outliers = r.scalar_i32(base_name + ".n_outliers");

    block_bmo_tier & h = t.header;
    memset(&h, 0, sizeof(h));
    h.rows = rows;
    h.cols = cols;
    h.scale_int8 = r.scalar_f32(base_name + ".scale_int8", 1.0f);
    h.zp_int8    = r.scalar_f32(base_name + ".zp_int8", 0.0f);
    h.scale_int4 = r.scalar_f32(base_name + ".scale_int4", 1.0f);
    h.zp_int4    = r.scalar_f32(base_name + ".zp_int4", 0.0f);
    h.scale_low  = r.scalar_f32(base_name + ".scale_low", 1.0f);
    h.zp_low     = r.scalar_f32(base_name + ".zp_low", 0.0f);
    h.n_outliers = n_outliers;

    std::vector<uint8_t> n_tiles_buf, tier_offsets_buf;
    if (!r.tensor_bytes(base_name + ".packed_weights", t.pw)) return false;
    if (!r.tensor_bytes(base_name + ".tile_tiers", t.tt)) return false;
    r.tensor_bytes(base_name + ".n_tiles", n_tiles_buf);
    r.tensor_bytes(base_name + ".tier_offsets", tier_offsets_buf);
    r.tensor_bytes(base_name + ".outlier_indices", t.oi);
    r.tensor_bytes(base_name + ".outlier_values", t.ov);

    if (n_tiles_buf.size() >= 4 * sizeof(int32_t)) memcpy(h.n_tiles, n_tiles_buf.data(), 4 * sizeof(int32_t));
    if (tier_offsets_buf.size() >= 5 * sizeof(int32_t)) memcpy(h.tier_offsets, tier_offsets_buf.data(), 5 * sizeof(int32_t));

    // tile stream indices — identical computation to loader.h
    std::vector<uint16_t> tile_stream_indices(t.tt.size());
    int32_t ptrs[4] = {0, 0, 0, 0};
    for (size_t t_idx = 0; t_idx < t.tt.size(); ++t_idx) {
        uint8_t tier = t.tt[t_idx];
        tile_stream_indices[t_idx] = (uint16_t)ptrs[tier]++;
    }

    // offsets — identical arithmetic to loader.h
    size_t write_offset = sizeof(block_bmo_tier);
    h.packed_weights_offset = write_offset;
    write_offset += t.pw.size();
    write_offset = (write_offset + 3) & ~(size_t)3;
    h.tile_tiers_offset = write_offset;
    write_offset += t.tt.size();
    write_offset = (write_offset + 3) & ~(size_t)3;
    h.outlier_indices_offset = write_offset;
    write_offset += t.oi.size();
    write_offset = (write_offset + 3) & ~(size_t)3;
    h.outlier_values_offset = write_offset;
    write_offset += t.ov.size();
    write_offset = (write_offset + 15) & ~(size_t)15;
    h.tile_stream_indices_offset = write_offset;
    write_offset += tile_stream_indices.size() * sizeof(uint16_t);
    write_offset = (write_offset + 15) & ~(size_t)15;

    t.shipped_bytes = write_offset;
    t.payload.assign(write_offset, 0);
    memcpy(t.payload.data(), &h, sizeof(h));
    memcpy(t.payload.data() + h.packed_weights_offset, t.pw.data(), t.pw.size());
    memcpy(t.payload.data() + h.tile_tiers_offset, t.tt.data(), t.tt.size());
    if (!t.oi.empty()) memcpy(t.payload.data() + h.outlier_indices_offset, t.oi.data(), t.oi.size());
    if (!t.ov.empty()) memcpy(t.payload.data() + h.outlier_values_offset, t.ov.data(), t.ov.size());
    memcpy(t.payload.data() + h.tile_stream_indices_offset, tile_stream_indices.data(), tile_stream_indices.size() * sizeof(uint16_t));

    // ---- V3 payload: outliers sorted by flat index + CSR per-row starts ----
    {
        int32_t n = h.n_outliers;
        const int32_t * oi32 = (const int32_t *)t.oi.data();
        const uint16_t * ov16 = (const uint16_t *)t.ov.data();
        std::vector<int32_t> order(n);
        for (int32_t i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return oi32[a] < oi32[b]; });
        std::vector<int32_t> oi_sorted(n);
        std::vector<uint16_t> ov_sorted(n);
        for (int32_t i = 0; i < n; ++i) { oi_sorted[i] = oi32[order[i]]; ov_sorted[i] = ov16[order[i]]; }
        std::vector<int32_t> row_starts(rows + 1, 0);
        for (int32_t i = 0; i < n; ++i) row_starts[oi_sorted[i] / cols + 1]++;
        for (int32_t rr = 0; rr < rows; ++rr) row_starts[rr + 1] += row_starts[rr];

        size_t off3 = t.payload.size();                     // already 16-aligned
        t.payload_v3.assign(off3 + (size_t)(rows + 1) * 4, 0);
        memcpy(t.payload_v3.data(), t.payload.data(), t.payload.size());
        // In the bench the CSR offset rides in the dequantized_cpu_ptr slot
        // (unused on device). Integration adds a real header field instead.
        block_bmo_tier * hb = (block_bmo_tier *)t.payload_v3.data();
        hb->dequantized_cpu_ptr = (int64_t)off3; // V3: repurposed as outlier_row_starts_offset
        if (n > 0) {
            memcpy(t.payload_v3.data() + h.outlier_indices_offset, oi_sorted.data(), (size_t)n * 4);
            memcpy(t.payload_v3.data() + h.outlier_values_offset, ov_sorted.data(), (size_t)n * 2);
        }
        memcpy(t.payload_v3.data() + off3, row_starts.data(), (size_t)(rows + 1) * 4);
    }
    return true;
}

// ============================================================================
// CPU reference — same math as loader.h dequantize_ffn_cpu(), followed by a
// double-accumulated matvec.
// ============================================================================
static float fp16_to_f32(uint16_t x) {
    __half_raw hr; hr.x = x;
    return __half2float(*(const __half *)&hr);
}

static std::vector<float> dequant_reference(const BmoTensor & t) {
    const block_bmo_tier & h = t.header;
    const uint8_t * pw = t.pw.data();
    const uint8_t * tile_tiers = t.tt.data();
    int64_t total = (int64_t)h.rows * h.cols;
    std::vector<float> w(total, 0.0f);

    const int tile_size = 4096;
    int32_t n_tiles_col = h.cols / 64;
    int32_t n_tiles_total = (int32_t)(total / tile_size);
    int32_t ptrs[4] = {0, 0, 0, 0};

    for (int32_t t_idx = 0; t_idx < n_tiles_total; ++t_idx) {
        uint8_t tier = tile_tiers[t_idx];
        int32_t stream = ptrs[tier]++;
        int32_t row_start = (t_idx / n_tiles_col) * 64;
        int32_t col_start = (t_idx % n_tiles_col) * 64;
        for (int32_t i = 0; i < tile_size; ++i) {
            int32_t r = row_start + (i >> 6);
            int32_t c = col_start + (i & 63);
            int64_t flat = (int64_t)r * h.cols + c;
            int32_t sidx = stream * tile_size + i;
            float v = 0.0f;
            if (tier == 0) {
                const uint16_t * f16 = (const uint16_t *)(pw + h.tier_offsets[0]);
                v = fp16_to_f32(f16[sidx]);
            } else if (tier == 1) {
                v = ((float)(pw + h.tier_offsets[1])[sidx] - h.zp_int8) * h.scale_int8;
            } else if (tier == 2) {
                uint8_t b = (pw + h.tier_offsets[2])[sidx >> 1];
                uint8_t q = (sidx & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                v = ((float)q - h.zp_int4) * h.scale_int4;
            } else {
                uint8_t b = (pw + h.tier_offsets[3])[sidx >> 2];
                uint8_t q = (b >> ((sidx & 3) * 2)) & 0x03;
                v = ((float)q - h.zp_low) * h.scale_low;
            }
            w[flat] = v;
        }
    }
    if (h.n_outliers > 0 && !t.oi.empty() && !t.ov.empty()) {
        const int32_t * oi = (const int32_t *)t.oi.data();
        const uint16_t * ov = (const uint16_t *)t.ov.data();
        for (int32_t i = 0; i < h.n_outliers; ++i) {
            w[oi[i]] = fp16_to_f32(ov[i]);
        }
    }
    return w;
}

// ============================================================================
// V0 — verbatim copy of the current production kernel pair
// (ggml/src/ggml-cuda/convert.cu mul_mat_vec_bmo_tier_cuda_kernel +
// apply_outliers_gemv_bmo_tier_kernel). Baseline AND harness validation:
// if V0 fails the rel_l2 gate, the harness itself is wrong.
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

static __global__ void v0_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const int row = blockIdx.x;
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

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
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

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
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int tile_c = col / BMO_TILE_DIM;
        const int in_tile_c = col % BMO_TILE_DIM;
        const uint8_t tier = s_tiers[tile_c];
        const int32_t stream_idx = s_stream_indices[tile_c];
        const int in_tile_idx = in_tile_row_base + in_tile_c;
        float w = bmo_dequant_element_fast(pw, s_tier_offsets, s_scales_and_zps, tier, stream_idx, in_tile_idx);
        thread_sum += w * x_vec[col];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    __shared__ float warp_sums[BMO_GEMV_BLOCK_SIZE / 32];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = thread_sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (BMO_GEMV_BLOCK_SIZE / 32)) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = (BMO_GEMV_BLOCK_SIZE / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) y_out[row] = val;
    }
}

static __global__ void v0_outliers_kernel(
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

    const int32_t flat_idx = outlier_indices[idx];
    const int32_t row = flat_idx / ncols;
    const int32_t col = flat_idx % ncols;
    const float outlier_w = __half2float(outlier_values[idx]);

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    const int32_t n_tiles_col = ncols / BMO_TILE_DIM;
    const int tile_idx = (row / BMO_TILE_DIM) * n_tiles_col + (col / BMO_TILE_DIM);
    const uint8_t tier = tile_tiers[tile_idx];
    const int32_t stream_idx = tile_stream_indices[tile_idx];
    const int in_tile_idx = (row % BMO_TILE_DIM) * BMO_TILE_DIM + (col % BMO_TILE_DIM);

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
    } else {
        const uint8_t * raw_int2 = pw + header->tier_offsets[3];
        int flat_idx_in = stream_idx * BMO_TILE_ELEMS + in_tile_idx;
        uint8_t b = raw_int2[flat_idx_in >> 2];
        int shift = (flat_idx_in & 3) * 2;
        uint8_t q = (b >> shift) & 0x03;
        base_w = ((float)q - header->zp_low) * header->scale_low;
    }
    atomicAdd(&y_out[row], (outlier_w - base_w) * x_vec[col]);
}

static void launch_v0(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    v0_gemv_kernel<<<rows, BMO_GEMV_BLOCK_SIZE, 0, stream>>>(vx, x, y, cols);
    if (n_outliers > 0) {
        const int bs = 256;
        v0_outliers_kernel<<<(n_outliers + bs - 1) / bs, bs, 0, stream>>>(vx, x, y, cols);
    }
}

// ============================================================================
// Read-bandwidth ceiling probe: grid-stride uint4 xor-reduction over the
// payload. Establishes the achievable DRAM read GB/s on this part with the
// current clocks, contextualizing the 50%-of-102 gate.
// ============================================================================
static __global__ void bw_probe_kernel(const uint4 * __restrict__ p, size_t n_words, uint4 * out) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) r ^= __shfl_down_sync(0xFFFFFFFF, r, o);
    if ((threadIdx.x & 31) == 0 && r == 0xDEADBEEFu) out[0].x = r; // never true in practice; defeats DCE
}

// ============================================================================
// V1 — 8 rows/block, 1 warp/row (prior art: BMO Voice Engine/personaplex/
// bmo_cuda_kernels.cu fused_dequant_matvec_kernel_v2), metadata staged once
// per BLOCK: because 64 % 8 == 0, all 8 rows of a block share one tile row.
// Each lane handles 2 consecutive elements per 64-wide tile column.
// Outlier correction: unchanged separate kernel (folded in V3).
// ============================================================================
#define V1_ROWS_PER_BLOCK 8

static __global__ void v1_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t s_tiers[512];
    __shared__ uint16_t s_stream[512];

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int row_base = blockIdx.x * V1_ROWS_PER_BLOCK;
    const int tile_row = row_base / BMO_TILE_DIM; // shared by all 8 rows (64 % 8 == 0)

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    for (int t = threadIdx.x; t < n_tiles_col; t += blockDim.x) {
        const int tile_idx = tile_row * n_tiles_col + t;
        s_tiers[t] = tile_tiers[tile_idx];
        s_stream[t] = tile_stream_indices[tile_idx];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = row_base + warp;
    if (row >= rows) return;

    const int in_tile_r = row & (BMO_TILE_DIM - 1);

    const half    * raw_f16 = (const half *)(pw + header->tier_offsets[0]);
    const uint8_t * raw_i8  = pw + header->tier_offsets[1];
    const uint8_t * raw_i4  = pw + header->tier_offsets[2];
    const uint8_t * raw_i2  = pw + header->tier_offsets[3];
    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float acc = 0.0f;
    for (int tc = 0; tc < n_tiles_col; ++tc) {
        const int tier = s_tiers[tc];
        const int base = (int)s_stream[tc] * BMO_TILE_ELEMS + in_tile_r * BMO_TILE_DIM;
        const float2 xv = *(const float2 *)(x_vec + tc * BMO_TILE_DIM + 2 * lane);
        float w0, w1;
        if (tier == 3) {
            const uint8_t b = raw_i2[(base >> 2) + (lane >> 1)];
            const int sh = (lane & 1) * 4;
            w0 = ((float)((b >> sh) & 3)        - z2) * s2;
            w1 = ((float)((b >> (sh + 2)) & 3)  - z2) * s2;
        } else if (tier == 2) {
            const uint8_t b = raw_i4[(base >> 1) + lane];
            w0 = ((float)(b & 0x0F) - z4) * s4;
            w1 = ((float)(b >> 4)   - z4) * s4;
        } else if (tier == 1) {
            const uchar2 b2 = *(const uchar2 *)(raw_i8 + base + 2 * lane);
            w0 = ((float)b2.x - z8) * s8;
            w1 = ((float)b2.y - z8) * s8;
        } else {
            const half2 h2 = *(const half2 *)(raw_f16 + base + 2 * lane);
            const float2 wf = __half22float2(h2);
            w0 = wf.x; w1 = wf.y;
        }
        acc += w0 * xv.x + w1 * xv.y;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, o);
    if (lane == 0) y_out[row] = acc;
}

static void launch_v1(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + V1_ROWS_PER_BLOCK - 1) / V1_ROWS_PER_BLOCK;
    v1_gemv_kernel<<<n_blocks, 256, 0, stream>>>(vx, x, y, cols);
    if (n_outliers > 0) {
        const int bs = 256;
        v0_outliers_kernel<<<(n_outliers + bs - 1) / bs, bs, 0, stream>>>(vx, x, y, cols);
    }
}

// ============================================================================
// V2 — vectorized packed-stream loads + tier-phased warp-uniform dispatch.
//
// Key ideas on top of V1:
//  * Tile columns are grouped by tier into shared-memory lists (built once
//    per block by warp 0 with a deterministic ballot scan), so each phase
//    runs warp-uniform: no per-tile tier branching in the hot loop.
//  * Each lane loads one uint32 of packed data per step (16 int2 / 8 int4 /
//    4 int8 elements) instead of per-element byte extracts.
//  * No I2F conversions in the hot loop (I2F runs on the slow conversion
//    pipe). A quant q at bit position p becomes an exact float via
//    bits = 0x40000000 | (q << (23 - qbits)), giving u = 2 + q/2^qbits, so
//    w = (q - zp)*s = s*2^qbits*u - s*(2^(qbits+1) + zp). The per-element
//    FMA accumulates u*x; the affine remainder folds into per-tile column
//    sums of x (s_xsum, computed once per block), so per element the cost
//    is shift+LOP3+FFMA, all full-rate pipes, with no catastrophic
//    cancellation (u is O(1)).
//  * Outlier correction stays a separate kernel in V2 (folded in V3).
// ============================================================================

static __device__ __forceinline__ float v2_uq(uint32_t wbits, int shr, int shl, uint32_t mask) {
    // exact float 2 + q/2^qbits from q = (wbits >> shr) & (2^qbits - 1)
    return __int_as_float((((wbits >> shr) << shl) & mask) | 0x40000000u);
}

static __global__ void v2_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t  s_tiers[512];
    __shared__ uint16_t s_stream[512];
    __shared__ float    s_xsum[512];
    __shared__ uint16_t s_list[4 * 512];
    __shared__ int      s_cnt4[4];

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int row_base = blockIdx.x * V1_ROWS_PER_BLOCK;
    const int tile_row = row_base / BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    for (int t = threadIdx.x; t < n_tiles_col; t += blockDim.x) {
        const int tile_idx = tile_row * n_tiles_col + t;
        s_tiers[t] = tile_tiers[tile_idx];
        s_stream[t] = tile_stream_indices[tile_idx];
    }
    __syncthreads();

    if (warp == 0) {
        // deterministic ballot scan: lists are ordered by tile column
        int cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;
        for (int b = 0; b < n_tiles_col; b += 32) {
            const int tc = b + lane;
            const int tier = (tc < n_tiles_col) ? (int)s_tiers[tc] : -1;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const unsigned m = __ballot_sync(0xFFFFFFFF, tier == t);
                const int cnt = (t == 0) ? cnt0 : (t == 1) ? cnt1 : (t == 2) ? cnt2 : cnt3;
                if (tier == t) {
                    s_list[t * 512 + cnt + __popc(m & ((1u << lane) - 1))] = (uint16_t)tc;
                }
                const int add = __popc(m);
                if (t == 0) cnt0 += add; else if (t == 1) cnt1 += add; else if (t == 2) cnt2 += add; else cnt3 += add;
            }
        }
        if (lane == 0) { s_cnt4[0] = cnt0; s_cnt4[1] = cnt1; s_cnt4[2] = cnt2; s_cnt4[3] = cnt3; }
    } else {
        // warps 1..7 compute per-tile-column sums of x (used by the affine fold)
        for (int tc = warp - 1; tc < n_tiles_col; tc += 7) {
            float s = x_vec[tc * BMO_TILE_DIM + lane] + x_vec[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    const int row = row_base + warp;
    if (row >= rows) return;
    const int in_tile_r = row & (BMO_TILE_DIM - 1);

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float acc   = 0.0f; // fp16 tiles, direct
    float acc_u2 = 0.0f, acc_x2 = 0.0f;
    float acc_u4 = 0.0f, acc_x4 = 0.0f;
    float acc_u8 = 0.0f, acc_x8 = 0.0f;

    // ---- INT2 phase: 8 tiles/step, 4 lanes/tile, 16 elems/lane/uint32 ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[3]);
        const int g = lane >> 2, sub = lane & 3;
        for (int i = g; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            const uint32_t wbits = w32[(int)s_stream[tc] * 256 + in_tile_r * 4 + sub];
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 xv = xp[jj];
                acc_u2 = fmaf(v2_uq(wbits, 8 * jj + 0, 21, 0x00600000u), xv.x, acc_u2);
                acc_u2 = fmaf(v2_uq(wbits, 8 * jj + 2, 21, 0x00600000u), xv.y, acc_u2);
                acc_u2 = fmaf(v2_uq(wbits, 8 * jj + 4, 21, 0x00600000u), xv.z, acc_u2);
                acc_u2 = fmaf(v2_uq(wbits, 8 * jj + 6, 21, 0x00600000u), xv.w, acc_u2);
            }
            if (sub == 0) acc_x2 += s_xsum[tc];
        }
    }

    // ---- INT4 phase: 4 tiles/step, 8 lanes/tile, 8 elems/lane/uint32 ----
    {
        const int n = s_cnt4[2];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[2]);
        const int g = lane >> 3, sub = lane & 7;
        for (int i = g; i < n; i += 4) {
            const int tc = s_list[2 * 512 + i];
            const uint32_t wbits = w32[(int)s_stream[tc] * 512 + in_tile_r * 8 + sub];
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 8 * sub);
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const float4 xv = xp[jj];
                acc_u4 = fmaf(v2_uq(wbits, 16 * jj + 0,  19, 0x00780000u), xv.x, acc_u4);
                acc_u4 = fmaf(v2_uq(wbits, 16 * jj + 4,  19, 0x00780000u), xv.y, acc_u4);
                acc_u4 = fmaf(v2_uq(wbits, 16 * jj + 8,  19, 0x00780000u), xv.z, acc_u4);
                acc_u4 = fmaf(v2_uq(wbits, 16 * jj + 12, 19, 0x00780000u), xv.w, acc_u4);
            }
            if (sub == 0) acc_x4 += s_xsum[tc];
        }
    }

    // ---- INT8 phase: 2 tiles/step, 16 lanes/tile, 4 elems/lane/uint32 ----
    {
        const int n = s_cnt4[1];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[1]);
        const int g = lane >> 4, sub = lane & 15;
        for (int i = g; i < n; i += 2) {
            const int tc = s_list[1 * 512 + i];
            const uint32_t wbits = w32[(int)s_stream[tc] * 1024 + in_tile_r * 16 + sub];
            const float4 xv = *(const float4 *)(x_vec + tc * BMO_TILE_DIM + 4 * sub);
            acc_u8 = fmaf(v2_uq(wbits,  0, 15, 0x007F8000u), xv.x, acc_u8);
            acc_u8 = fmaf(v2_uq(wbits,  8, 15, 0x007F8000u), xv.y, acc_u8);
            acc_u8 = fmaf(v2_uq(wbits, 16, 15, 0x007F8000u), xv.z, acc_u8);
            acc_u8 = fmaf(v2_uq(wbits, 24, 15, 0x007F8000u), xv.w, acc_u8);
            if (sub == 0) acc_x8 += s_xsum[tc];
        }
    }

    // ---- FP16 phase: 1 tile/step, 2 elems/lane ----
    {
        const int n = s_cnt4[0];
        const half * f16 = (const half *)(pw + header->tier_offsets[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const half2 h2 = *(const half2 *)(f16 + (int)s_stream[tc] * BMO_TILE_ELEMS + in_tile_r * BMO_TILE_DIM + 2 * lane);
            const float2 wf = __half22float2(h2);
            const float2 xv = *(const float2 *)(x_vec + tc * BMO_TILE_DIM + 2 * lane);
            acc = fmaf(wf.x, xv.x, acc);
            acc = fmaf(wf.y, xv.y, acc);
        }
    }

    // affine fold: with mantissa placement q << t (t = 21/19/15) the loaded
    // float is u = 2 + q/2^(22-t), i.e. q = 2^(22-t) * (u - 2). Hence
    //   sum(w*x) = s*2^(22-t)*sum(u*x) - s*(2^(23-t) + zp)*sum(x)
    float acc_total = acc
        + (2.0f   * s2) * acc_u2 - (s2 * (4.0f   + z2)) * acc_x2
        + (8.0f   * s4) * acc_u4 - (s4 * (16.0f  + z4)) * acc_x4
        + (128.0f * s8) * acc_u8 - (s8 * (256.0f + z8)) * acc_x8;

    // ---- V3: fused per-row outlier correction (CSR ranges, sorted by flat idx)
    if (fuse_outliers) {
        const int32_t * row_starts = (const int32_t *)((const char *)header + header->dequantized_cpu_ptr);
        const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
        const half * ov = (const half *)((const char *)header + header->outlier_values_offset);
        const int e0 = row_starts[row], e1 = row_starts[row + 1];
        for (int k = e0 + lane; k < e1; k += 32) {
            const int col = oi[k] - row * cols;
            const int tc = col >> 6;
            const int tier = s_tiers[tc];
            const int sidx = (int)s_stream[tc] * BMO_TILE_ELEMS + in_tile_r * BMO_TILE_DIM + (col & 63);
            float base_w;
            if (tier == 3) {
                const uint8_t b = (pw + header->tier_offsets[3])[sidx >> 2];
                base_w = ((float)((b >> ((sidx & 3) * 2)) & 3) - z2) * s2;
            } else if (tier == 2) {
                const uint8_t b = (pw + header->tier_offsets[2])[sidx >> 1];
                base_w = ((float)((sidx & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
            } else if (tier == 1) {
                base_w = ((float)(pw + header->tier_offsets[1])[sidx] - z8) * s8;
            } else {
                base_w = __half2float(((const half *)(pw + header->tier_offsets[0]))[sidx]);
            }
            acc_total += (__half2float(ov[k]) - base_w) * x_vec[col];
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc_total += __shfl_down_sync(0xFFFFFFFF, acc_total, o);
    if (lane == 0) y_out[row] = acc_total;
}

static void launch_v2(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + V1_ROWS_PER_BLOCK - 1) / V1_ROWS_PER_BLOCK;
    v2_gemv_kernel<<<n_blocks, 256, 0, stream>>>(vx, x, y, cols, false);
    if (n_outliers > 0) {
        const int bs = 256;
        v0_outliers_kernel<<<(n_outliers + bs - 1) / bs, bs, 0, stream>>>(vx, x, y, cols);
    }
}

static void launch_v3(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + V1_ROWS_PER_BLOCK - 1) / V1_ROWS_PER_BLOCK;
    v2_gemv_kernel<<<n_blocks, 256, 0, stream>>>(vx, x, y, cols, true);
}

// ============================================================================
// V4 — V3 plus:
//  * single-shift unpack: shift amount is a compile-time constant after
//    unrolling, so each element costs SHF + LOP3 + FFMA (was 2 shifts).
//  * 2 rows per warp (16 rows/block): each x float4 load feeds FMAs for two
//    rows, halving x-vector L1 traffic; the per-tile sum-of-x fold is
//    row-independent so it is accumulated once for both rows.
//  * streaming-cache weight loads (__ldcs) so the one-pass weight stream
//    does not evict the hot x vector from L1.
// ============================================================================
#define V4_ROWS_PER_BLOCK 16

static __device__ __forceinline__ float v4_uq(uint32_t w, int sh_left, uint32_t mask) {
    // sh_left is compile-time constant after unroll; negative = right shift
    const uint32_t m = (sh_left >= 0) ? (w << sh_left) : (w >> (-sh_left));
    return __int_as_float((m & mask) | 0x40000000u);
}

static __global__ void __launch_bounds__(256, 4) v4_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t  s_tiers[512];
    __shared__ uint16_t s_stream[512];
    __shared__ float    s_xsum[512];
    __shared__ uint16_t s_list[4 * 512];
    __shared__ int      s_cnt4[4];

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int row_base = blockIdx.x * V4_ROWS_PER_BLOCK;
    const int tile_row = row_base / BMO_TILE_DIM; // 64 % 16 == 0: one tile row per block
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    for (int t = threadIdx.x; t < n_tiles_col; t += blockDim.x) {
        const int tile_idx = tile_row * n_tiles_col + t;
        s_tiers[t] = tile_tiers[tile_idx];
        s_stream[t] = tile_stream_indices[tile_idx];
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
                    s_list[t * 512 + cnt + __popc(m & ((1u << lane) - 1))] = (uint16_t)tc;
                }
                const int add = __popc(m);
                if (t == 0) cnt0 += add; else if (t == 1) cnt1 += add; else if (t == 2) cnt2 += add; else cnt3 += add;
            }
        }
        if (lane == 0) { s_cnt4[0] = cnt0; s_cnt4[1] = cnt1; s_cnt4[2] = cnt2; s_cnt4[3] = cnt3; }
    } else {
        for (int tc = warp - 1; tc < n_tiles_col; tc += 7) {
            float s = x_vec[tc * BMO_TILE_DIM + lane] + x_vec[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    const int rowA = row_base + warp;      // warp handles rows base+warp and base+warp+8
    const int rowB = row_base + warp + 8;
    if (rowA >= rows) return;
    const bool hasB = rowB < rows;
    const int irA = rowA & (BMO_TILE_DIM - 1);
    const int irB = rowB & (BMO_TILE_DIM - 1);

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float accA = 0.0f, accB = 0.0f;             // fp16 tiles, direct
    float aA_u2 = 0.0f, aB_u2 = 0.0f, a_x2 = 0.0f;
    float aA_u4 = 0.0f, aB_u4 = 0.0f, a_x4 = 0.0f;
    float aA_u8 = 0.0f, aB_u8 = 0.0f, a_x8 = 0.0f;

    // ---- INT2: 8 tiles/step, 4 lanes/tile, 16 elems/lane/uint32, 2 rows ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[3]);
        const int g = lane >> 2, sub = lane & 3;
        for (int i = g; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            const uint32_t * wb = w32 + (int)s_stream[tc] * 256 + sub;
            const uint32_t wA = __ldcs(wb + irA * 4);
            const uint32_t wB = hasB ? __ldcs(wb + irB * 4) : 0u;
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 xv = xp[jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 21 - 2 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    aA_u2 = fmaf(v4_uq(wA, sh, 0x00600000u), xk, aA_u2);
                    aB_u2 = fmaf(v4_uq(wB, sh, 0x00600000u), xk, aB_u2);
                }
            }
            if (sub == 0) a_x2 += s_xsum[tc];
        }
    }

    // ---- INT4: 4 tiles/step, 8 lanes/tile, 8 elems/lane/uint32, 2 rows ----
    {
        const int n = s_cnt4[2];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[2]);
        const int g = lane >> 3, sub = lane & 7;
        for (int i = g; i < n; i += 4) {
            const int tc = s_list[2 * 512 + i];
            const uint32_t * wb = w32 + (int)s_stream[tc] * 512 + sub;
            const uint32_t wA = __ldcs(wb + irA * 8);
            const uint32_t wB = hasB ? __ldcs(wb + irB * 8) : 0u;
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 8 * sub);
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const float4 xv = xp[jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 19 - 4 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    aA_u4 = fmaf(v4_uq(wA, sh, 0x00780000u), xk, aA_u4);
                    aB_u4 = fmaf(v4_uq(wB, sh, 0x00780000u), xk, aB_u4);
                }
            }
            if (sub == 0) a_x4 += s_xsum[tc];
        }
    }

    // ---- INT8: 2 tiles/step, 16 lanes/tile, 4 elems/lane/uint32, 2 rows ----
    {
        const int n = s_cnt4[1];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[1]);
        const int g = lane >> 4, sub = lane & 15;
        for (int i = g; i < n; i += 2) {
            const int tc = s_list[1 * 512 + i];
            const uint32_t * wb = w32 + (int)s_stream[tc] * 1024 + sub;
            const uint32_t wA = __ldcs(wb + irA * 16);
            const uint32_t wB = hasB ? __ldcs(wb + irB * 16) : 0u;
            const float4 xv = *(const float4 *)(x_vec + tc * BMO_TILE_DIM + 4 * sub);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int sh = 15 - 8 * k;
                const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                aA_u8 = fmaf(v4_uq(wA, sh, 0x007F8000u), xk, aA_u8);
                aB_u8 = fmaf(v4_uq(wB, sh, 0x007F8000u), xk, aB_u8);
            }
            if (sub == 0) a_x8 += s_xsum[tc];
        }
    }

    // ---- FP16: 1 tile/step, 2 elems/lane, 2 rows ----
    {
        const int n = s_cnt4[0];
        const half * f16 = (const half *)(pw + header->tier_offsets[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const half * tb = f16 + (int)s_stream[tc] * BMO_TILE_ELEMS + 2 * lane;
            const float2 xv = *(const float2 *)(x_vec + tc * BMO_TILE_DIM + 2 * lane);
            const float2 wfA = __half22float2(*(const half2 *)(tb + irA * BMO_TILE_DIM));
            accA = fmaf(wfA.x, xv.x, accA);
            accA = fmaf(wfA.y, xv.y, accA);
            if (hasB) {
                const float2 wfB = __half22float2(*(const half2 *)(tb + irB * BMO_TILE_DIM));
                accB = fmaf(wfB.x, xv.x, accB);
                accB = fmaf(wfB.y, xv.y, accB);
            }
        }
    }

    float totA = accA
        + (2.0f   * s2) * aA_u2 - (s2 * (4.0f   + z2)) * a_x2
        + (8.0f   * s4) * aA_u4 - (s4 * (16.0f  + z4)) * a_x4
        + (128.0f * s8) * aA_u8 - (s8 * (256.0f + z8)) * a_x8;
    float totB = accB
        + (2.0f   * s2) * aB_u2 - (s2 * (4.0f   + z2)) * a_x2
        + (8.0f   * s4) * aB_u4 - (s4 * (16.0f  + z4)) * a_x4
        + (128.0f * s8) * aB_u8 - (s8 * (256.0f + z8)) * a_x8;

    if (fuse_outliers) {
        const int32_t * row_starts = (const int32_t *)((const char *)header + header->dequantized_cpu_ptr);
        const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
        const half * ov = (const half *)((const char *)header + header->outlier_values_offset);
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int row = rr == 0 ? rowA : rowB;
            if (rr == 1 && !hasB) break;
            const int ir = rr == 0 ? irA : irB;
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            float corr = 0.0f;
            for (int k = e0 + lane; k < e1; k += 32) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int sidx = (int)s_stream[tc] * BMO_TILE_ELEMS + ir * BMO_TILE_DIM + (col & 63);
                float base_w;
                if (tier == 3) {
                    const uint8_t b = (pw + header->tier_offsets[3])[sidx >> 2];
                    base_w = ((float)((b >> ((sidx & 3) * 2)) & 3) - z2) * s2;
                } else if (tier == 2) {
                    const uint8_t b = (pw + header->tier_offsets[2])[sidx >> 1];
                    base_w = ((float)((sidx & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
                } else if (tier == 1) {
                    base_w = ((float)(pw + header->tier_offsets[1])[sidx] - z8) * s8;
                } else {
                    base_w = __half2float(((const half *)(pw + header->tier_offsets[0]))[sidx]);
                }
                corr += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
            if (rr == 0) totA += corr; else totB += corr;
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        totA += __shfl_down_sync(0xFFFFFFFF, totA, o);
        totB += __shfl_down_sync(0xFFFFFFFF, totB, o);
    }
    if (lane == 0) {
        y_out[rowA] = totA;
        if (hasB) y_out[rowB] = totB;
    }
}

static void launch_v4(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + V4_ROWS_PER_BLOCK - 1) / V4_ROWS_PER_BLOCK;
    v4_gemv_kernel<<<n_blocks, 256, 0, stream>>>(vx, x, y, cols, true);
}

// ============================================================================
// V5 — one block per full 64-row tile band (512 threads, 16 warps x 4 rows).
//
// V4's remaining bandwidth sin: a 16-row block consumes only 16/64 of every
// packed-weight DRAM sector it touches; the other 3 blocks covering the same
// tile band re-fetch the same sectors from other SMs (L2 cannot hold the
// stream), multiplying effective DRAM traffic ~2-4x. With the whole tile band
// in one block, every fetched sector is fully consumed exactly once, and the
// warp's 4 consecutive rows make each tile's slice a contiguous 64B read per
// lane quad. x float4 loads now feed FMAs for 4 rows.
// ============================================================================
#define V5_THREADS 512
#define V5_WARPS   (V5_THREADS / 32)

// MODE: 0 = normal; 1 = skip x loads (constant x, isolates weight-load+ALU
// path); 2 = skip weight loads (constant bits, isolates x-load+ALU path);
// 4 = uniform-address x loads (all lanes read x_vec[0..3] — guaranteed
// cache-resident; distinguishes x-memory-bound from FFMA/issue-bound).
// Modes 1/2/4 are diagnostics — output is intentionally wrong.
template <int MODE>
static __global__ void __launch_bounds__(V5_THREADS, 2) v5_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t  s_tiers[512];
    __shared__ uint16_t s_stream[512];
    __shared__ float    s_xsum[512];
    __shared__ uint16_t s_list[4 * 512];
    __shared__ int      s_cnt4[4];

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int tile_row = blockIdx.x;            // one block = one 64-row band
    const int row_base = tile_row * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const uint16_t * tile_stream_indices = (const uint16_t *)((const char *)header + header->tile_stream_indices_offset);
    const uint8_t * pw = (const uint8_t *)((const char *)header + header->packed_weights_offset);

    for (int t = threadIdx.x; t < n_tiles_col; t += blockDim.x) {
        const int tile_idx = tile_row * n_tiles_col + t;
        s_tiers[t] = tile_tiers[tile_idx];
        s_stream[t] = tile_stream_indices[tile_idx];
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
                    s_list[t * 512 + cnt + __popc(m & ((1u << lane) - 1))] = (uint16_t)tc;
                }
                const int add = __popc(m);
                if (t == 0) cnt0 += add; else if (t == 1) cnt1 += add; else if (t == 2) cnt2 += add; else cnt3 += add;
            }
        }
        if (lane == 0) { s_cnt4[0] = cnt0; s_cnt4[1] = cnt1; s_cnt4[2] = cnt2; s_cnt4[3] = cnt3; }
    } else {
        for (int tc = warp - 1; tc < n_tiles_col; tc += V5_WARPS - 1) {
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
    const int ir0 = warp * 4;                   // in-tile row of first handled row

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float acc[4]  = {0, 0, 0, 0};               // fp16 tiles, direct
    float au2[4]  = {0, 0, 0, 0};
    float au4[4]  = {0, 0, 0, 0};
    float au8[4]  = {0, 0, 0, 0};
    float a_x2 = 0.0f, a_x4 = 0.0f, a_x8 = 0.0f;

    // ---- INT2: 8 tiles/step, 4 lanes/tile, 16 elems/lane/uint32, 4 rows ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[3]);
        const int g = lane >> 2, sub = lane & 3;
        for (int i = g; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            const uint32_t * wb = w32 + (int)s_stream[tc] * 256 + ir0 * 4 + sub;
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) w[r] = (MODE == 2) ? 0x55555555u : __ldcs(wb + r * 4);
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                              : xp[(MODE == 4) ? 0 : jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 21 - 2 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au2[r] = fmaf(v4_uq(w[r], sh, 0x00600000u), xk, au2[r]);
                    }
                }
            }
            if (sub == 0) a_x2 += s_xsum[tc];
        }
    }

    // ---- INT4: 4 tiles/step, 8 lanes/tile, 8 elems/lane/uint32, 4 rows ----
    {
        const int n = s_cnt4[2];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[2]);
        const int g = lane >> 3, sub = lane & 7;
        for (int i = g; i < n; i += 4) {
            const int tc = s_list[2 * 512 + i];
            const uint32_t * wb = w32 + (int)s_stream[tc] * 512 + ir0 * 8 + sub;
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) w[r] = (MODE == 2) ? 0x55555555u : __ldcs(wb + r * 8);
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 8 * sub);
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                              : xp[(MODE == 4) ? 0 : jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 19 - 4 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au4[r] = fmaf(v4_uq(w[r], sh, 0x00780000u), xk, au4[r]);
                    }
                }
            }
            if (sub == 0) a_x4 += s_xsum[tc];
        }
    }

    // ---- INT8: 2 tiles/step, 16 lanes/tile, 4 elems/lane/uint32, 4 rows ----
    {
        const int n = s_cnt4[1];
        const uint32_t * w32 = (const uint32_t *)(pw + header->tier_offsets[1]);
        const int g = lane >> 4, sub = lane & 15;
        for (int i = g; i < n; i += 2) {
            const int tc = s_list[1 * 512 + i];
            const uint32_t * wb = w32 + (int)s_stream[tc] * 1024 + ir0 * 16 + sub;
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) w[r] = (MODE == 2) ? 0x55555555u : __ldcs(wb + r * 16);
            const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                          : (MODE == 4) ? *(const float4 *)x_vec
                                          : *(const float4 *)(x_vec + tc * BMO_TILE_DIM + 4 * sub);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int sh = 15 - 8 * k;
                const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    au8[r] = fmaf(v4_uq(w[r], sh, 0x007F8000u), xk, au8[r]);
                }
            }
            if (sub == 0) a_x8 += s_xsum[tc];
        }
    }

    // ---- FP16: 1 tile/step, 2 elems/lane, 4 rows ----
    {
        const int n = s_cnt4[0];
        const half * f16 = (const half *)(pw + header->tier_offsets[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const half * tb = f16 + (int)s_stream[tc] * BMO_TILE_ELEMS + ir0 * BMO_TILE_DIM + 2 * lane;
            const float2 xv = *(const float2 *)(x_vec + tc * BMO_TILE_DIM + 2 * lane);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float2 wf = __half22float2(*(const half2 *)(tb + r * BMO_TILE_DIM));
                acc[r] = fmaf(wf.x, xv.x, acc[r]);
                acc[r] = fmaf(wf.y, xv.y, acc[r]);
            }
        }
    }

    const int32_t * row_starts = fuse_outliers
        ? (const int32_t *)((const char *)header + header->dequantized_cpu_ptr) : nullptr;
    const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
    const half * ov = (const half *)((const char *)header + header->outlier_values_offset);

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = r0 + r;
        if (row >= rows) break;
        float tot = acc[r]
            + (2.0f   * s2) * au2[r] - (s2 * (4.0f   + z2)) * a_x2
            + (8.0f   * s4) * au4[r] - (s4 * (16.0f  + z4)) * a_x4
            + (128.0f * s8) * au8[r] - (s8 * (256.0f + z8)) * a_x8;

        if (fuse_outliers) {
            const int ir = ir0 + r;
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + lane; k < e1; k += 32) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int sidx = (int)s_stream[tc] * BMO_TILE_ELEMS + ir * BMO_TILE_DIM + (col & 63);
                float base_w;
                if (tier == 3) {
                    const uint8_t b = (pw + header->tier_offsets[3])[sidx >> 2];
                    base_w = ((float)((b >> ((sidx & 3) * 2)) & 3) - z2) * s2;
                } else if (tier == 2) {
                    const uint8_t b = (pw + header->tier_offsets[2])[sidx >> 1];
                    base_w = ((float)((sidx & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
                } else if (tier == 1) {
                    base_w = ((float)(pw + header->tier_offsets[1])[sidx] - z8) * s8;
                } else {
                    base_w = __half2float(((const half *)(pw + header->tier_offsets[0]))[sidx]);
                }
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xFFFFFFFF, tot, o);
        if (lane == 0) y_out[row] = tot;
    }
}

static void launch_v5(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v5_gemv_kernel<0><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, true);
}

static void launch_v5_dbg_nox(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v5_gemv_kernel<1><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

static void launch_v5_dbg_now(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v5_gemv_kernel<2><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

// ============================================================================
// V6 — V5 on the band-major repacked stream layout (see build_v6_payload).
// A warp step's weight loads become one contiguous 128B line instead of 8
// scattered 64B regions (8x fewer L1 wavefronts on the dominant int2 path),
// and every block's DRAM walk is sequential. Tile position in the per-band
// tier list IS the address, so the hot loop needs no stream-index lookups.
// ============================================================================
template <int MODE>
static __global__ void __launch_bounds__(V5_THREADS, 2) v6_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
{
    const block_bmo_tier * header = (const block_bmo_tier *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t  s_tiers[512];
    __shared__ uint16_t s_pos[512];   // tile position within its tier's band list
    __shared__ float    s_xsum[512];
    __shared__ uint16_t s_list[4 * 512];
    __shared__ int      s_cnt4[4];

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->padding) + band * 4;

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
        for (int tc = warp - 1; tc < n_tiles_col; tc += V5_WARPS - 1) {
            float s = x_vec[tc * BMO_TILE_DIM + lane] + x_vec[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    const int r0 = row_base + warp * 4;
    if (r0 >= rows) return;
    const int ir0 = warp * 4;

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float acc[4]  = {0, 0, 0, 0};
    float au2[4]  = {0, 0, 0, 0};
    float au4[4]  = {0, 0, 0, 0};
    float au8[4]  = {0, 0, 0, 0};
    float a_x2 = 0.0f, a_x4 = 0.0f, a_x8 = 0.0f;

    // ---- INT2: 8 tiles/step, 4 lanes/tile; slice bytes: (ir*n2 + i)*16 ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[3]);
        const int g = lane >> 2, sub = lane & 3;
        for (int i = g; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i) * 4 + sub);
            }
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                              : xp[(MODE == 4) ? 0 : jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 21 - 2 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au2[r] = fmaf(v4_uq(w[r], sh, 0x00600000u), xk, au2[r]);
                    }
                }
            }
            if (sub == 0) a_x2 += s_xsum[tc];
        }
    }

    // ---- INT4: 4 tiles/step, 8 lanes/tile; slice bytes: (ir*n4 + i)*32 ----
    {
        const int n = s_cnt4[2];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[2]);
        const int g = lane >> 3, sub = lane & 7;
        for (int i = g; i < n; i += 4) {
            const int tc = s_list[2 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i) * 8 + sub);
            }
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 8 * sub);
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                              : xp[(MODE == 4) ? 0 : jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 19 - 4 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au4[r] = fmaf(v4_uq(w[r], sh, 0x00780000u), xk, au4[r]);
                    }
                }
            }
            if (sub == 0) a_x4 += s_xsum[tc];
        }
    }

    // ---- INT8: 2 tiles/step, 16 lanes/tile; slice bytes: (ir*n8 + i)*64 ----
    {
        const int n = s_cnt4[1];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[1]);
        const int g = lane >> 4, sub = lane & 15;
        for (int i = g; i < n; i += 2) {
            const int tc = s_list[1 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i) * 16 + sub);
            }
            const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                          : (MODE == 4) ? *(const float4 *)x_vec
                                          : *(const float4 *)(x_vec + tc * BMO_TILE_DIM + 4 * sub);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int sh = 15 - 8 * k;
                const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    au8[r] = fmaf(v4_uq(w[r], sh, 0x007F8000u), xk, au8[r]);
                }
            }
            if (sub == 0) a_x8 += s_xsum[tc];
        }
    }

    // ---- FP16: 1 tile/step; slice bytes: (ir*nf + i)*128 ----
    {
        const int n = s_cnt4[0];
        const half2 * f16 = (const half2 *)((const char *)header + band_tab[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const float2 xv = *(const float2 *)(x_vec + tc * BMO_TILE_DIM + 2 * lane);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float2 wf = __half22float2(f16[((ir0 + r) * n + i) * 32 + lane]);
                acc[r] = fmaf(wf.x, xv.x, acc[r]);
                acc[r] = fmaf(wf.y, xv.y, acc[r]);
            }
        }
    }

    const int32_t * row_starts = fuse_outliers
        ? (const int32_t *)((const char *)header + header->dequantized_cpu_ptr) : nullptr;
    const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
    const half * ov = (const half *)((const char *)header + header->outlier_values_offset);

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = r0 + r;
        if (row >= rows) break;
        float tot = acc[r]
            + (2.0f   * s2) * au2[r] - (s2 * (4.0f   + z2)) * a_x2
            + (8.0f   * s4) * au4[r] - (s4 * (16.0f  + z4)) * a_x4
            + (128.0f * s8) * au8[r] - (s8 * (256.0f + z8)) * a_x8;

        if (fuse_outliers) {
            const int ir = ir0 + r;
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + lane; k < e1; k += 32) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int p = s_pos[tc];
                const int cin = col & 63;
                const char * tb = (const char *)header + band_tab[tier];
                float base_w;
                if (tier == 3) {
                    const int nn = s_cnt4[3];
                    const uint8_t b = ((const uint8_t *)tb)[(ir * nn + p) * 16 + (cin >> 2)];
                    base_w = ((float)((b >> ((cin & 3) * 2)) & 3) - z2) * s2;
                } else if (tier == 2) {
                    const int nn = s_cnt4[2];
                    const uint8_t b = ((const uint8_t *)tb)[(ir * nn + p) * 32 + (cin >> 1)];
                    base_w = ((float)((cin & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
                } else if (tier == 1) {
                    const int nn = s_cnt4[1];
                    base_w = ((float)((const uint8_t *)tb)[(ir * nn + p) * 64 + cin] - z8) * s8;
                } else {
                    const int nn = s_cnt4[0];
                    base_w = __half2float(((const half *)tb)[(ir * nn + p) * 64 + cin]);
                }
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xFFFFFFFF, tot, o);
        if (lane == 0) y_out[row] = tot;
    }
}

static void launch_v6(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v6_gemv_kernel<0><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, true);
}

static void launch_v6_dbg_nox(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v6_gemv_kernel<1><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

static void launch_v6_dbg_now(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v6_gemv_kernel<2><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

static void launch_v6_dbg_unix(const void * vx, const float * x, float * y,
                               int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v6_gemv_kernel<4><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

// ============================================================================
// V7 — V6 + software pipelining. The bisect showed weight-path and x-path
// times ADD instead of overlapping: each iteration is a serial chain
// (metadata -> address -> 4 DRAM loads -> 64 dependent FMAs) and ~6
// iterations/warp cannot hide ~600-cycle DRAM latency. Prefetch the next
// step's weight words (and next tile index) before consuming the current
// ones so loads stay in flight during compute.
// ============================================================================
template <int MODE>
static __global__ void __launch_bounds__(V5_THREADS, 2) v7_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
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

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->padding) + band * 4;

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
        for (int tc = warp - 1; tc < n_tiles_col; tc += V5_WARPS - 1) {
            float s = x_vec[tc * BMO_TILE_DIM + lane] + x_vec[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    const int r0 = row_base + warp * 4;
    if (r0 >= rows) return;
    const int ir0 = warp * 4;

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float acc[4]  = {0, 0, 0, 0};
    float au2[4]  = {0, 0, 0, 0};
    float au4[4]  = {0, 0, 0, 0};
    float au8[4]  = {0, 0, 0, 0};
    float a_x2 = 0.0f, a_x4 = 0.0f, a_x8 = 0.0f;

    // ---- INT2, software-pipelined ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[3]);
        const int g = lane >> 2, sub = lane & 3;
        int i = g;
        uint32_t w[4];
        int tc = 0;
        if (i < n) {
            tc = s_list[3 * 512 + i];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i) * 4 + sub);
            }
        }
        for (; i < n; ) {
            const int i_next = i + 8;
            const uint32_t wc0 = w[0], wc1 = w[1], wc2 = w[2], wc3 = w[3];
            const int tc_cur = tc;
            if (i_next < n) {
                tc = s_list[3 * 512 + i_next];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i_next) * 4 + sub);
                }
            }
            const uint32_t wcur[4] = { wc0, wc1, wc2, wc3 };
            const float4 * xp = (const float4 *)(x_vec + tc_cur * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 21 - 2 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au2[r] = fmaf(v4_uq(wcur[r], sh, 0x00600000u), xk, au2[r]);
                    }
                }
            }
            if (sub == 0) a_x2 += s_xsum[tc_cur];
            i = i_next;
        }
    }

    // ---- INT4, software-pipelined ----
    {
        const int n = s_cnt4[2];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[2]);
        const int g = lane >> 3, sub = lane & 7;
        int i = g;
        uint32_t w[4];
        int tc = 0;
        if (i < n) {
            tc = s_list[2 * 512 + i];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i) * 8 + sub);
            }
        }
        for (; i < n; ) {
            const int i_next = i + 4;
            const uint32_t wc0 = w[0], wc1 = w[1], wc2 = w[2], wc3 = w[3];
            const int tc_cur = tc;
            if (i_next < n) {
                tc = s_list[2 * 512 + i_next];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i_next) * 8 + sub);
                }
            }
            const uint32_t wcur[4] = { wc0, wc1, wc2, wc3 };
            const float4 * xp = (const float4 *)(x_vec + tc_cur * BMO_TILE_DIM + 8 * sub);
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 19 - 4 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au4[r] = fmaf(v4_uq(wcur[r], sh, 0x00780000u), xk, au4[r]);
                    }
                }
            }
            if (sub == 0) a_x4 += s_xsum[tc_cur];
            i = i_next;
        }
    }

    // ---- INT8, software-pipelined ----
    {
        const int n = s_cnt4[1];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[1]);
        const int g = lane >> 4, sub = lane & 15;
        int i = g;
        uint32_t w[4];
        int tc = 0;
        if (i < n) {
            tc = s_list[1 * 512 + i];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i) * 16 + sub);
            }
        }
        for (; i < n; ) {
            const int i_next = i + 2;
            const uint32_t wc0 = w[0], wc1 = w[1], wc2 = w[2], wc3 = w[3];
            const int tc_cur = tc;
            if (i_next < n) {
                tc = s_list[1 * 512 + i_next];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    w[r] = (MODE == 2) ? 0x55555555u : __ldcs(w32 + ((ir0 + r) * n + i_next) * 16 + sub);
                }
            }
            const uint32_t wcur[4] = { wc0, wc1, wc2, wc3 };
            const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                          : *(const float4 *)(x_vec + tc_cur * BMO_TILE_DIM + 4 * sub);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int sh = 15 - 8 * k;
                const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    au8[r] = fmaf(v4_uq(wcur[r], sh, 0x007F8000u), xk, au8[r]);
                }
            }
            if (sub == 0) a_x8 += s_xsum[tc_cur];
            i = i_next;
        }
    }

    // ---- FP16 (short; not pipelined) ----
    {
        const int n = s_cnt4[0];
        const half2 * f16 = (const half2 *)((const char *)header + band_tab[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const float2 xv = *(const float2 *)(x_vec + tc * BMO_TILE_DIM + 2 * lane);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float2 wf = __half22float2(f16[((ir0 + r) * n + i) * 32 + lane]);
                acc[r] = fmaf(wf.x, xv.x, acc[r]);
                acc[r] = fmaf(wf.y, xv.y, acc[r]);
            }
        }
    }

    const int32_t * row_starts = fuse_outliers
        ? (const int32_t *)((const char *)header + header->dequantized_cpu_ptr) : nullptr;
    const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
    const half * ov = (const half *)((const char *)header + header->outlier_values_offset);

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = r0 + r;
        if (row >= rows) break;
        float tot = acc[r]
            + (2.0f   * s2) * au2[r] - (s2 * (4.0f   + z2)) * a_x2
            + (8.0f   * s4) * au4[r] - (s4 * (16.0f  + z4)) * a_x4
            + (128.0f * s8) * au8[r] - (s8 * (256.0f + z8)) * a_x8;

        if (fuse_outliers) {
            const int ir = ir0 + r;
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + lane; k < e1; k += 32) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int p = s_pos[tc];
                const int cin = col & 63;
                const char * tb = (const char *)header + band_tab[tier];
                float base_w;
                if (tier == 3) {
                    const int nn = s_cnt4[3];
                    const uint8_t b = ((const uint8_t *)tb)[(ir * nn + p) * 16 + (cin >> 2)];
                    base_w = ((float)((b >> ((cin & 3) * 2)) & 3) - z2) * s2;
                } else if (tier == 2) {
                    const int nn = s_cnt4[2];
                    const uint8_t b = ((const uint8_t *)tb)[(ir * nn + p) * 32 + (cin >> 1)];
                    base_w = ((float)((cin & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
                } else if (tier == 1) {
                    const int nn = s_cnt4[1];
                    base_w = ((float)((const uint8_t *)tb)[(ir * nn + p) * 64 + cin] - z8) * s8;
                } else {
                    const int nn = s_cnt4[0];
                    base_w = __half2float(((const half *)tb)[(ir * nn + p) * 64 + cin]);
                }
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xFFFFFFFF, tot, o);
        if (lane == 0) y_out[row] = tot;
    }
}

static void launch_v7(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v7_gemv_kernel<0><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, true);
}

// ============================================================================
// V8 — V6 + x vector staged in dynamic SHARED memory.
//
// The uniform-x diagnostic showed the scattered x float4 global loads cost
// ~0.48 ms of memory-system time on their own (x is only cols*4 bytes, but
// the 8-tile-scattered 64B access pattern defeats the L1). Shared memory
// reads bypass L1/L2 entirely. Naive LDS layout would be 8-way bank
// conflicted (bank ignores the tile-group bits); rotating each lane group's
// float4 read order by its group index (jj_eff = (jj + g) & 3) spreads the
// accesses to <=2-way. Summation order changes; rel_l2 gate still applies.
// ============================================================================

static __global__ void __launch_bounds__(V5_THREADS, 2) v8_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
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
    extern __shared__ float s_x[];      // cols floats, dynamic

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->padding) + band * 4;

    for (int t = threadIdx.x; t < cols; t += blockDim.x) {
        s_x[t] = x_vec[t];
    }
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
        for (int tc = warp - 1; tc < n_tiles_col; tc += V5_WARPS - 1) {
            float s = s_x[tc * BMO_TILE_DIM + lane] + s_x[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    const int r0 = row_base + warp * 4;
    if (r0 >= rows) return;
    const int ir0 = warp * 4;

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float acc[4]  = {0, 0, 0, 0};
    float au2[4]  = {0, 0, 0, 0};
    float au4[4]  = {0, 0, 0, 0};
    float au8[4]  = {0, 0, 0, 0};
    float a_x2 = 0.0f, a_x4 = 0.0f, a_x8 = 0.0f;

    // ---- INT2: 8 tiles/step, 4 lanes/tile, rotated float4 reads ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[3]);
        const int g = lane >> 2, sub = lane & 3;
        for (int i = g; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) w[r] = __ldcs(w32 + ((ir0 + r) * n + i) * 4 + sub);
            const float4 * xp = (const float4 *)(s_x + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int je = (jj + g) & 3;    // rotation: <=2-way bank conflicts
                const float4 xv = xp[je];
                // byte je of w holds the 4 quants of this float4 chunk; PRMT
                // extraction keeps the per-element shifts compile-time
                uint32_t b[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) b[r] = __byte_perm(w[r], 0, je);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au2[r] = fmaf(v4_uq(b[r], 21 - 2 * k, 0x00600000u), xk, au2[r]);
                    }
                }
            }
            if (sub == 0) a_x2 += s_xsum[tc];
        }
    }

    // ---- INT4: 4 tiles/step, 8 lanes/tile, rotated float4 reads ----
    {
        const int n = s_cnt4[2];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[2]);
        const int g = lane >> 3, sub = lane & 7;
        for (int i = g; i < n; i += 4) {
            const int tc = s_list[2 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) w[r] = __ldcs(w32 + ((ir0 + r) * n + i) * 8 + sub);
            const float4 * xp = (const float4 *)(s_x + tc * BMO_TILE_DIM + 8 * sub);
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int je = (jj + g) & 1;
                const float4 xv = xp[je];
                // bytes 2*je..2*je+1 hold this chunk's 8 quants
                const uint32_t sel = 0x4410u + 0x22u * (uint32_t)je;
                uint32_t b[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) b[r] = __byte_perm(w[r], 0, sel);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        au4[r] = fmaf(v4_uq(b[r], 19 - 4 * k, 0x00780000u), xk, au4[r]);
                    }
                }
            }
            if (sub == 0) a_x4 += s_xsum[tc];
        }
    }

    // ---- INT8: 2 tiles/step, 16 lanes/tile ----
    {
        const int n = s_cnt4[1];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[1]);
        const int g = lane >> 4, sub = lane & 15;
        for (int i = g; i < n; i += 2) {
            const int tc = s_list[1 * 512 + i];
            uint32_t w[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) w[r] = __ldcs(w32 + ((ir0 + r) * n + i) * 16 + sub);
            const float4 xv = *(const float4 *)(s_x + tc * BMO_TILE_DIM + 4 * sub);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int sh = 15 - 8 * k;
                const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    au8[r] = fmaf(v4_uq(w[r], sh, 0x007F8000u), xk, au8[r]);
                }
            }
            if (sub == 0) a_x8 += s_xsum[tc];
        }
    }

    // ---- FP16: 1 tile/step, 2 elems/lane ----
    {
        const int n = s_cnt4[0];
        const half2 * f16 = (const half2 *)((const char *)header + band_tab[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const float2 xv = *(const float2 *)(s_x + tc * BMO_TILE_DIM + 2 * lane);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float2 wf = __half22float2(f16[((ir0 + r) * n + i) * 32 + lane]);
                acc[r] = fmaf(wf.x, xv.x, acc[r]);
                acc[r] = fmaf(wf.y, xv.y, acc[r]);
            }
        }
    }

    const int32_t * row_starts = fuse_outliers
        ? (const int32_t *)((const char *)header + header->dequantized_cpu_ptr) : nullptr;
    const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
    const half * ov = (const half *)((const char *)header + header->outlier_values_offset);

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = r0 + r;
        if (row >= rows) break;
        float tot = acc[r]
            + (2.0f   * s2) * au2[r] - (s2 * (4.0f   + z2)) * a_x2
            + (8.0f   * s4) * au4[r] - (s4 * (16.0f  + z4)) * a_x4
            + (128.0f * s8) * au8[r] - (s8 * (256.0f + z8)) * a_x8;

        if (fuse_outliers) {
            const int ir = ir0 + r;
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + lane; k < e1; k += 32) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int p = s_pos[tc];
                const int cin = col & 63;
                const char * tb = (const char *)header + band_tab[tier];
                float base_w;
                if (tier == 3) {
                    const int nn = s_cnt4[3];
                    const uint8_t b = ((const uint8_t *)tb)[(ir * nn + p) * 16 + (cin >> 2)];
                    base_w = ((float)((b >> ((cin & 3) * 2)) & 3) - z2) * s2;
                } else if (tier == 2) {
                    const int nn = s_cnt4[2];
                    const uint8_t b = ((const uint8_t *)tb)[(ir * nn + p) * 32 + (cin >> 1)];
                    base_w = ((float)((cin & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
                } else if (tier == 1) {
                    const int nn = s_cnt4[1];
                    base_w = ((float)((const uint8_t *)tb)[(ir * nn + p) * 64 + cin] - z8) * s8;
                } else {
                    const int nn = s_cnt4[0];
                    base_w = __half2float(((const half *)tb)[(ir * nn + p) * 64 + cin]);
                }
                tot += (__half2float(ov[k]) - base_w) * s_x[col];
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xFFFFFFFF, tot, o);
        if (lane == 0) y_out[row] = tot;
    }
}

static void launch_v8(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    const size_t smem = (size_t)cols * sizeof(float);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(v8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024);
        attr_set = true;
    }
    v8_gemv_kernel<<<n_blocks, V5_THREADS, smem, stream>>>(vx, x, y, cols, true);
}

// ============================================================================
// V9 — wavefront-optimal restructure on the tile-major payload.
//
// V8 taught us the binding resource is LSU wavefronts per load instruction
// (global-L1 and shared LDS share it; the uniform-x diagnostic collapsed the
// cost). V9 makes every load contiguous or broadcast:
//  * one tile per warp step, 8 rows per warp (256-thread block = 8 warps x
//    8 rows = the whole band); lane = (row_in_group, chunk) = (lane>>2,
//    lane&3); each lane owns 16 consecutive elements of its row.
//  * tile-major payload: the warp's weight load is one LDG spanning 8
//    consecutive row slices = contiguous 128B (int2) .. 1024B (fp16) = 1-8
//    wavefronts for 512 element-instances.
//  * x loads: the 4 chunks x 16B are contiguous 64B, broadcast across the 8
//    row-groups = 1 wavefront per float4 load.
//  * accumulators are per-lane (the lane's row), so registers stay low and
//    occupancy high; final reduction is a 2-step intra-quad shuffle.
// ============================================================================
#define V9_THREADS 256

template <int MODE>
static __global__ void __launch_bounds__(V9_THREADS, 6) v9_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
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

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;   // 8 warps; warp handles rows 8*warp..8*warp+7
    const int lane = threadIdx.x & 31;
    const int rg   = lane >> 2;          // row within the warp's 8-row group
    const int sub  = lane & 3;           // 16-element chunk within the row slice

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->padding) + band * 4;

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
        for (int tc = warp - 1; tc < n_tiles_col; tc += 7) {
            float s = x_vec[tc * BMO_TILE_DIM + lane] + x_vec[tc * BMO_TILE_DIM + 32 + lane];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
            if (lane == 0) s_xsum[tc] = s;
        }
    }
    __syncthreads();

    const int ir = warp * 8 + rg;        // this lane's in-band row
    const int row = row_base + ir;
    const bool live = row < rows;

    const float s8 = header->scale_int8, z8 = header->zp_int8;
    const float s4 = header->scale_int4, z4 = header->zp_int4;
    const float s2 = header->scale_low,  z2 = header->zp_low;

    float accf = 0.0f;                   // fp16 tiles, direct
    float au2a = 0.0f, au2b = 0.0f;      // split accumulators: 2 FMA chains/tier for ILP
    float au4a = 0.0f, au4b = 0.0f;
    float au8a = 0.0f, au8b = 0.0f;
    float a_x = 0.0f;                    // per-tier folds merged via constants below
    float a_x4s = 0.0f, a_x8s = 0.0f;

    // ---- INT2: one tile per step; lane's uint32 = 16 elems of its row ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[3]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[3 * 512 + i];
            const uint32_t w = (MODE == 2) ? 0x55555555u
                             : __ldcs(w32 + (i * BMO_TILE_DIM + ir) * 4 + sub);
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                              : xp[(MODE == 4) ? 0 : jj];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 21 - 2 * (4 * jj + k);
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    if (jj & 1) au2b = fmaf(v4_uq(w, sh, 0x00600000u), xk, au2b);
                    else        au2a = fmaf(v4_uq(w, sh, 0x00600000u), xk, au2a);
                }
            }
            if (sub == 0 && rg == 0) a_x += s_xsum[tc]; // folded once per row below
        }
    }

    // ---- INT4: lane's uint2 = 16 elems ----
    {
        const int n = s_cnt4[2];
        const uint2 * w64 = (const uint2 *)((const char *)header + band_tab[2]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[2 * 512 + i];
            const uint2 w2 = (MODE == 2) ? make_uint2(0x55555555u, 0x55555555u)
                           : __ldcs(w64 + (i * BMO_TILE_DIM + ir) * 4 + sub);
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
                const uint32_t w = (half_i == 0) ? w2.x : w2.y;
                #pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                                  : xp[(MODE == 4) ? 0 : (half_i * 2 + jj)];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int sh = 19 - 4 * (4 * jj + k);
                        const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                        if (half_i) au4b = fmaf(v4_uq(w, sh, 0x00780000u), xk, au4b);
                        else        au4a = fmaf(v4_uq(w, sh, 0x00780000u), xk, au4a);
                    }
                }
            }
            if (sub == 0 && rg == 0) a_x4s += s_xsum[tc];
        }
    }

    // ---- INT8: lane's uint4 = 16 elems ----
    {
        const int n = s_cnt4[1];
        const uint4 * w128 = (const uint4 *)((const char *)header + band_tab[1]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[1 * 512 + i];
            const uint4 w4 = (MODE == 2) ? make_uint4(0x55555555u, 0x55555555u, 0x55555555u, 0x55555555u)
                           : __ldcs(w128 + (i * BMO_TILE_DIM + ir) * 4 + sub);
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t w = (q == 0) ? w4.x : (q == 1) ? w4.y : (q == 2) ? w4.z : w4.w;
                const float4 xv = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f)
                                              : xp[(MODE == 4) ? 0 : q];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int sh = 15 - 8 * k;
                    const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                    if (q & 1) au8b = fmaf(v4_uq(w, sh, 0x007F8000u), xk, au8b);
                    else       au8a = fmaf(v4_uq(w, sh, 0x007F8000u), xk, au8a);
                }
            }
            if (sub == 0 && rg == 0) a_x8s += s_xsum[tc];
        }
    }

    // ---- FP16: lane's 16 halves = 2 x uint4 ----
    {
        const int n = s_cnt4[0];
        const uint4 * w128 = (const uint4 *)((const char *)header + band_tab[0]);
        for (int i = 0; i < n; ++i) {
            const int tc = s_list[0 * 512 + i];
            const uint4 wa = __ldcs(w128 + (i * BMO_TILE_DIM + ir) * 8 + sub * 2);
            const uint4 wb = __ldcs(w128 + (i * BMO_TILE_DIM + ir) * 8 + sub * 2 + 1);
            const float4 * xp = (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            const uint32_t ws[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float2 wf = __half22float2(*(const half2 *)&ws[q]);
                const float2 xv = *(const float2 *)((const float *)xp + 2 * q);
                accf = fmaf(wf.x, xv.x, accf);
                accf = fmaf(wf.y, xv.y, accf);
            }
        }
    }

    // combine: per-lane partials -> per-row total via intra-quad reduction.
    // a_x* were accumulated only on (sub==0, rg==0) lanes = once per TILE,
    // but the fold applies once per ROW: broadcast via shuffle from lane 0.
    float ax2_row = __shfl_sync(0xFFFFFFFF, a_x,   0);
    float ax4_row = __shfl_sync(0xFFFFFFFF, a_x4s, 0);
    float ax8_row = __shfl_sync(0xFFFFFFFF, a_x8s, 0);

    float tot = accf
        + (2.0f   * s2) * (au2a + au2b) - (s2 * (4.0f   + z2)) * (sub == 0 ? ax2_row : 0.0f)
        + (8.0f   * s4) * (au4a + au4b) - (s4 * (16.0f  + z4)) * (sub == 0 ? ax4_row : 0.0f)
        + (128.0f * s8) * (au8a + au8b) - (s8 * (256.0f + z8)) * (sub == 0 ? ax8_row : 0.0f);

    if (fuse_outliers && live) {
        const int32_t * row_starts = (const int32_t *)((const char *)header + header->dequantized_cpu_ptr);
        const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
        const half * ov = (const half *)((const char *)header + header->outlier_values_offset);
        const int e0 = row_starts[row], e1 = row_starts[row + 1];
        for (int k = e0 + sub; k < e1; k += 4) {
            const int col = oi[k] - row * cols;
            const int tc = col >> 6;
            const int tier = s_tiers[tc];
            const int p = s_pos[tc];
            const int cin = col & 63;
            const char * tb = (const char *)header + band_tab[tier];
            float base_w;
            if (tier == 3) {
                const uint8_t b = ((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 16 + (cin >> 2)];
                base_w = ((float)((b >> ((cin & 3) * 2)) & 3) - z2) * s2;
            } else if (tier == 2) {
                const uint8_t b = ((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 32 + (cin >> 1)];
                base_w = ((float)((cin & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
            } else if (tier == 1) {
                base_w = ((float)((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 64 + cin] - z8) * s8;
            } else {
                base_w = __half2float(((const half *)tb)[(p * BMO_TILE_DIM + ir) * 64 + cin]);
            }
            tot += (__half2float(ov[k]) - base_w) * x_vec[col];
        }
    }

    // intra-quad reduction: lanes (rg, sub 0..3) -> row total
    tot += __shfl_down_sync(0xFFFFFFFF, tot, 1);
    tot += __shfl_down_sync(0xFFFFFFFF, tot, 2);
    if (sub == 0 && live) y_out[row] = tot;
}

static void launch_v9(const void * vx, const float * x, float * y,
                      int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v9_gemv_kernel<0><<<n_blocks, V9_THREADS, 0, stream>>>(vx, x, y, cols, true);
}

// ============================================================================
// V10 — L1TEX-minimal: warps partition TILES; each warp computes all 64 rows
// of its tiles into 8 row-octet register accumulators.
//
// ncu on v9 showed the L1/TEX pipe at 80.6% (SM 47%, DRAM 46%): on Ampere's
// unified L1, global AND shared loads share that pipe, and v9's 8 warps each
// re-read all of x (8x duplicated L1TEX work). Here x is read 4 float4s per
// tile per BLOCK, weights stream as contiguous 128-512B per instruction, and
// per-warp per-row partials are combined once via a 2KB shared reduction.
// Same tile-major payload as V9. Register accumulators (the only storage not
// behind L1TEX) carry all reuse.
// ============================================================================

template <int MODE>
static __global__ void __launch_bounds__(V9_THREADS, 4) v10_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
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
    __shared__ float    s_part[8 * BMO_TILE_DIM];  // per-warp, per-row partials

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;   // 8 warps; warp w owns tile-list slots w, w+8, ...
    const int lane = threadIdx.x & 31;
    const int rg   = lane >> 2;          // row within each row octet
    const int sub  = lane & 3;           // 16-element chunk within the row slice

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->padding) + band * 4;

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

    // acc[m]: partial for row 8*m + rg of this band, over this warp's tiles,
    // with per-tier scale applied per tile (k1_t * sum(u*x)).
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float a_x2 = 0.0f, a_x4 = 0.0f, a_x8 = 0.0f; // this warp's tier xsum partials

    // ---- INT2: warp's tiles are list slots warp, warp+8, ... ----
    {
        const int n = s_cnt4[3];
        const uint32_t * w32 = (const uint32_t *)((const char *)header + band_tab[3]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[3 * 512 + i];
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            float4 xv[4];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                xv[jj] = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : jj];
            }
            uint32_t wv[8];
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                wv[m] = (MODE == 2) ? 0x55555555u
                      : __ldcs(w32 + (i * BMO_TILE_DIM + 8 * m + rg) * 4 + sub);
            }
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                const uint32_t w = wv[m];
                float part = 0.0f;
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int sh = 21 - 2 * (4 * jj + k);
                        const float xk = (k == 0) ? xv[jj].x : (k == 1) ? xv[jj].y : (k == 2) ? xv[jj].z : xv[jj].w;
                        part = fmaf(v4_uq(w, sh, 0x00600000u), xk, part);
                    }
                }
                acc[m] = fmaf(2.0f * s2, part, acc[m]);
            }
            if (lane == 0) a_x2 += s_xsum[tc];
        }
    }

    // ---- INT4 ----
    {
        const int n = s_cnt4[2];
        const uint2 * w64 = (const uint2 *)((const char *)header + band_tab[2]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[2 * 512 + i];
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            float4 xv[4];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                xv[jj] = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : jj];
            }
            uint2 w2v[4];
            #pragma unroll
            for (int mh = 0; mh < 2; ++mh) {
            #pragma unroll
            for (int mm = 0; mm < 4; ++mm) {
                w2v[mm] = (MODE == 2) ? make_uint2(0x55555555u, 0x55555555u)
                        : __ldcs(w64 + (i * BMO_TILE_DIM + 8 * (4 * mh + mm) + rg) * 4 + sub);
            }
            #pragma unroll
            for (int mm = 0; mm < 4; ++mm) {
                const int m = 4 * mh + mm;
                const uint2 w2 = w2v[mm];
                float part = 0.0f;
                #pragma unroll
                for (int half_i = 0; half_i < 2; ++half_i) {
                    const uint32_t w = (half_i == 0) ? w2.x : w2.y;
                    #pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int sh = 19 - 4 * (4 * jj + k);
                            const float4 & xq = xv[half_i * 2 + jj];
                            const float xk = (k == 0) ? xq.x : (k == 1) ? xq.y : (k == 2) ? xq.z : xq.w;
                            part = fmaf(v4_uq(w, sh, 0x00780000u), xk, part);
                        }
                    }
                }
                acc[m] = fmaf(8.0f * s4, part, acc[m]);
            }
            }
            if (lane == 0) a_x4 += s_xsum[tc];
        }
    }

    // ---- INT8 ----
    {
        const int n = s_cnt4[1];
        const uint4 * w128 = (const uint4 *)((const char *)header + band_tab[1]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[1 * 512 + i];
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            float4 xv[4];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                xv[jj] = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : jj];
            }
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                const uint4 w4 = (MODE == 2) ? make_uint4(0x55555555u, 0x55555555u, 0x55555555u, 0x55555555u)
                               : __ldcs(w128 + (i * BMO_TILE_DIM + 8 * m + rg) * 4 + sub);
                float part = 0.0f;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t w = (q == 0) ? w4.x : (q == 1) ? w4.y : (q == 2) ? w4.z : w4.w;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int sh = 15 - 8 * k;
                        const float xk = (k == 0) ? xv[q].x : (k == 1) ? xv[q].y : (k == 2) ? xv[q].z : xv[q].w;
                        part = fmaf(v4_uq(w, sh, 0x007F8000u), xk, part);
                    }
                }
                acc[m] = fmaf(128.0f * s8, part, acc[m]);
            }
            if (lane == 0) a_x8 += s_xsum[tc];
        }
    }

    // ---- FP16 ----
    {
        const int n = s_cnt4[0];
        const uint4 * w128 = (const uint4 *)((const char *)header + band_tab[0]);
        for (int i = warp; i < n; i += 8) {
            const int tc = s_list[0 * 512 + i];
            const float * xb = x_vec + tc * BMO_TILE_DIM + 16 * sub;
            float xr[16];
            #pragma unroll
            for (int q = 0; q < 16; ++q) xr[q] = xb[q];
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                const uint4 wa = __ldcs(w128 + (i * BMO_TILE_DIM + 8 * m + rg) * 8 + sub * 2);
                const uint4 wb = __ldcs(w128 + (i * BMO_TILE_DIM + 8 * m + rg) * 8 + sub * 2 + 1);
                const uint32_t ws[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
                float part = 0.0f;
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const float2 wf = __half22float2(*(const half2 *)&ws[q]);
                    part = fmaf(wf.x, xr[2 * q],     part);
                    part = fmaf(wf.y, xr[2 * q + 1], part);
                }
                acc[m] += part;
            }
        }
    }

    // fold the row-independent -k2*sum(x) terms once per row partial
    const float fold = - (s2 * (4.0f   + z2)) * a_x2
                       - (s4 * (16.0f  + z4)) * a_x4
                       - (s8 * (256.0f + z8)) * a_x8;
    const float fold_l0 = __shfl_sync(0xFFFFFFFF, fold, 0); // accumulated on lane 0 only

    // quad-reduce each acc[m] and stash this warp's per-row partials
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        float v = acc[m];
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        if (sub == 0) s_part[warp * BMO_TILE_DIM + 8 * m + rg] = v + fold_l0;
    }
    __syncthreads();

    // final: warp w owns output rows 8w..8w+7 (as in V9); sum the 8 warps'
    // partials, apply outlier corrections, write.
    {
        const int ir = warp * 8 + rg;
        const int row = row_base + ir;
        const bool live = row < rows;

        float tot = 0.0f;
        // lane quad splits the 8 warp-partials: sub reads j = sub and sub+4
        tot += s_part[sub * BMO_TILE_DIM + ir];
        tot += s_part[(sub + 4) * BMO_TILE_DIM + ir];
        // NOTE: fold_l0 was added by EVERY warp's partial; it must count once.
        // Remove the 7 extra copies here (fold values differ per warp — each
        // warp folded only ITS tiles' xsum, so the SUM of the 8 partials
        // carries exactly the full fold once. No correction needed.)

        if (fuse_outliers && live) {
            const int32_t * row_starts = (const int32_t *)((const char *)header + header->dequantized_cpu_ptr);
            const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
            const half * ov = (const half *)((const char *)header + header->outlier_values_offset);
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + sub; k < e1; k += 4) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int p = s_pos[tc];
                const int cin = col & 63;
                const char * tb = (const char *)header + band_tab[tier];
                float base_w;
                if (tier == 3) {
                    const uint8_t b = ((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 16 + (cin >> 2)];
                    base_w = ((float)((b >> ((cin & 3) * 2)) & 3) - z2) * s2;
                } else if (tier == 2) {
                    const uint8_t b = ((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 32 + (cin >> 1)];
                    base_w = ((float)((cin & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
                } else if (tier == 1) {
                    base_w = ((float)((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 64 + cin] - z8) * s8;
                } else {
                    base_w = __half2float(((const half *)tb)[(p * BMO_TILE_DIM + ir) * 64 + cin]);
                }
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        // intra-quad reduction of the (partial-sum + outlier) contributions
        tot += __shfl_down_sync(0xFFFFFFFF, tot, 1);
        tot += __shfl_down_sync(0xFFFFFFFF, tot, 2);
        if (sub == 0 && live) y_out[row] = tot;
    }
}

static void launch_v10(const void * vx, const float * x, float * y,
                       int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v10_gemv_kernel<0><<<n_blocks, V9_THREADS, 0, stream>>>(vx, x, y, cols, true);
}

static void launch_v10_dbg_nox(const void * vx, const float * x, float * y,
                               int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v10_gemv_kernel<1><<<n_blocks, V9_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

static void launch_v10_dbg_now(const void * vx, const float * x, float * y,
                               int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v10_gemv_kernel<2><<<n_blocks, V9_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

// ============================================================================
// V11 — V10 with a register diet to reach 6 blocks/SM (48 warps) WITHOUT
// spills. ncu on v10: register limit (64 regs) caps occupancy at 4 blocks/32
// warps and the kernel is latency-bound (7.1 of 13.5 cycles L1TEX scoreboard,
// 1.05 eligible warps/scheduler); forcing 6 blocks by launch bounds alone
// spilled 244B and regressed. Here the x float4 working set is halved (two
// half-tile passes: 8 x-registers instead of 16) and the fp16 tail loads x
// per row-octet, trading a few extra cheap broadcast loads for ~22 fewer
// live registers in the hot loop.
// ============================================================================

template <int MODE>
static __global__ void __launch_bounds__(V9_THREADS, 5) v11_gemv_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols,
    const bool fuse_outliers)
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

    if (n_tiles_col > 512) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768).\n", n_tiles_col);
        }
        __syncthreads();
        __trap();
    }

    const int band = blockIdx.x;
    const int row_base = band * BMO_TILE_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rg   = lane >> 2;
    const int sub  = lane & 3;

    const uint8_t * tile_tiers = (const uint8_t *)((const char *)header + header->tile_tiers_offset);
    const int32_t * band_tab = (const int32_t *)((const char *)header + header->padding) + band * 4;

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
                wv[m] = (MODE == 2) ? 0x55555555u
                      : __ldcs(w32 + (i * BMO_TILE_DIM + 8 * m + rg) * 4 + sub);
            }
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int jjh = 0; jjh < 2; ++jjh) {
                const float4 xq0 = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : 2 * jjh];
                const float4 xq1 = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : 2 * jjh + 1];
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
                            part = fmaf(v4_uq(wv[m], sh, 0x00600000u), xk, part);
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
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int mh = 0; mh < 2; ++mh) {
                uint2 w2v[4];
                #pragma unroll
                for (int mm = 0; mm < 4; ++mm) {
                    w2v[mm] = (MODE == 2) ? make_uint2(0x55555555u, 0x55555555u)
                            : __ldcs(w64 + (i * BMO_TILE_DIM + 8 * (4 * mh + mm) + rg) * 4 + sub);
                }
                #pragma unroll
                for (int half_i = 0; half_i < 2; ++half_i) {
                    const float4 xq0 = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : 2 * half_i];
                    const float4 xq1 = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : 2 * half_i + 1];
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
                                part = fmaf(v4_uq(w, sh, 0x00780000u), xk, part);
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
            const float4 * xp = (MODE == 4) ? (const float4 *)x_vec
                                            : (const float4 *)(x_vec + tc * BMO_TILE_DIM + 16 * sub);
            #pragma unroll
            for (int qh = 0; qh < 2; ++qh) {
                const float4 xq0 = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : 2 * qh];
                const float4 xq1 = (MODE == 1) ? make_float4(1.f, 1.f, 1.f, 1.f) : xp[(MODE == 4) ? 0 : 2 * qh + 1];
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const uint2 w2 = (MODE == 2) ? make_uint2(0x55555555u, 0x55555555u)
                                   : __ldcs(w64 + (i * BMO_TILE_DIM + 8 * m + rg) * 8 + 2 * sub + qh);
                    float part = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t w = q ? w2.y : w2.x;
                        const float4 & xq = q ? xq1 : xq0;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int sh = 15 - 8 * k;
                            const float xk = (k == 0) ? xq.x : (k == 1) ? xq.y : (k == 2) ? xq.z : xq.w;
                            part = fmaf(v4_uq(w, sh, 0x007F8000u), xk, part);
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

    {
        const int ir = warp * 8 + rg;
        const int row = row_base + ir;
        const bool live = row < rows;

        float tot = 0.0f;
        tot += s_part[sub * BMO_TILE_DIM + ir];
        tot += s_part[(sub + 4) * BMO_TILE_DIM + ir];
        // each warp folded only ITS tiles' xsum into its partial, so the sum
        // of the 8 partials carries the full fold exactly once.

        if (fuse_outliers && live) {
            const int32_t * row_starts = (const int32_t *)((const char *)header + header->dequantized_cpu_ptr);
            const int32_t * oi = (const int32_t *)((const char *)header + header->outlier_indices_offset);
            const half * ov = (const half *)((const char *)header + header->outlier_values_offset);
            const int e0 = row_starts[row], e1 = row_starts[row + 1];
            for (int k = e0 + sub; k < e1; k += 4) {
                const int col = oi[k] - row * cols;
                const int tc = col >> 6;
                const int tier = s_tiers[tc];
                const int p = s_pos[tc];
                const int cin = col & 63;
                const char * tb = (const char *)header + band_tab[tier];
                float base_w;
                if (tier == 3) {
                    const uint8_t b = ((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 16 + (cin >> 2)];
                    base_w = ((float)((b >> ((cin & 3) * 2)) & 3) - z2) * s2;
                } else if (tier == 2) {
                    const uint8_t b = ((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 32 + (cin >> 1)];
                    base_w = ((float)((cin & 1) ? (b >> 4) : (b & 0x0F)) - z4) * s4;
                } else if (tier == 1) {
                    base_w = ((float)((const uint8_t *)tb)[(p * BMO_TILE_DIM + ir) * 64 + cin] - z8) * s8;
                } else {
                    base_w = __half2float(((const half *)tb)[(p * BMO_TILE_DIM + ir) * 64 + cin]);
                }
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        tot += __shfl_down_sync(0xFFFFFFFF, tot, 1);
        tot += __shfl_down_sync(0xFFFFFFFF, tot, 2);
        if (sub == 0 && live) y_out[row] = tot;
    }
}

static void launch_v11(const void * vx, const float * x, float * y,
                       int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v11_gemv_kernel<0><<<n_blocks, V9_THREADS, 0, stream>>>(vx, x, y, cols, true);
}

static void launch_v9_dbg_nox(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v9_gemv_kernel<1><<<n_blocks, V9_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

static void launch_v9_dbg_now(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v9_gemv_kernel<2><<<n_blocks, V9_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

static void launch_v7_dbg_nox(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v7_gemv_kernel<1><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

static void launch_v7_dbg_now(const void * vx, const float * x, float * y,
                              int rows, int cols, int n_outliers, cudaStream_t stream) {
    const int n_blocks = (rows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    v7_gemv_kernel<2><<<n_blocks, V5_THREADS, 0, stream>>>(vx, x, y, cols, false);
}

// ============================================================================
// Harness
// ============================================================================
typedef void (*launch_fn)(const void *, const float *, float *, int, int, int, cudaStream_t);

struct VariantSpec {
    const char * name;
    launch_fn fn;
    int payload_kind; // 0 = shipped, 1 = v3 (sorted outliers + CSR), 2 = v6 (band-major)
};

static double median_of(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n == 0 ? 0.0 : (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [tensor_base_name]\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    std::string base_name = argc > 2 ? argv[2] : "transformer_layers_0_gating_linear_in_weight";

    GgufReader reader;
    if (!reader.open(gguf_path)) {
        fprintf(stderr, "error: failed to open %s\n", gguf_path);
        return 1;
    }
    BmoTensor t;
    if (!load_bmo_tensor(reader, base_name, t)) return 1;
    const block_bmo_tier & h = t.header;

    printf("tensor: %s\n", base_name.c_str());
    printf("  rows=%d cols=%d n_tiles_col=%d n_outliers=%d\n", h.rows, h.cols, h.cols / 64, h.n_outliers);
    printf("  n_tiles: fp16=%d int8=%d int4=%d int2=%d (total %d)\n",
           h.n_tiles[0], h.n_tiles[1], h.n_tiles[2], h.n_tiles[3],
           h.n_tiles[0] + h.n_tiles[1] + h.n_tiles[2] + h.n_tiles[3]);
    printf("  tier_offsets: %d %d %d %d %d\n", h.tier_offsets[0], h.tier_offsets[1], h.tier_offsets[2], h.tier_offsets[3], h.tier_offsets[4]);
    printf("  scales/zps: i8=(%g,%g) i4=(%g,%g) low=(%g,%g)\n", h.scale_int8, h.zp_int8, h.scale_int4, h.zp_int4, h.scale_low, h.zp_low);
    printf("  shipped payload bytes: %zu\n", t.shipped_bytes);
    for (int i = 0; i < 4; ++i) {
        if (h.tier_offsets[i] % 16 != 0) {
            printf("  WARNING: tier_offsets[%d]=%d not 16-byte aligned — vectorized variants unsafe\n", i, h.tier_offsets[i]);
        }
    }

    // fixed-seed random input vector
    std::mt19937 rng(1783708826u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(h.cols);
    for (auto & v : x) v = dist(rng);

    // CPU reference
    printf("building CPU reference (dequant + double-accum matvec)...\n");
    fflush(stdout);
    std::vector<float> w = dequant_reference(t);
    std::vector<double> y_ref(h.rows, 0.0);
    for (int r = 0; r < h.rows; ++r) {
        double acc = 0.0;
        const float * wr = w.data() + (int64_t)r * h.cols;
        for (int c = 0; c < h.cols; ++c) acc += (double)wr[c] * (double)x[c];
        y_ref[r] = acc;
    }
    double ref_l2 = 0.0;
    for (int r = 0; r < h.rows; ++r) ref_l2 += y_ref[r] * y_ref[r];
    ref_l2 = sqrt(ref_l2);

    // device buffers
    build_repacked_payload(t, false, t.payload_v6);
    build_repacked_payload(t, true,  t.payload_v9);
    void * d_payload = nullptr, * d_payload_v3 = nullptr, * d_payload_v6 = nullptr, * d_payload_v9 = nullptr;
    float * d_x = nullptr, * d_y = nullptr;
    CU_CHECK(cudaMalloc(&d_payload, t.payload.size()));
    CU_CHECK(cudaMalloc(&d_payload_v3, t.payload_v3.size()));
    CU_CHECK(cudaMalloc(&d_payload_v6, t.payload_v6.size()));
    CU_CHECK(cudaMalloc(&d_payload_v9, t.payload_v9.size()));
    CU_CHECK(cudaMalloc(&d_x, h.cols * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_y, h.rows * sizeof(float)));
    CU_CHECK(cudaMemcpy(d_payload, t.payload.data(), t.payload.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_payload_v3, t.payload_v3.data(), t.payload_v3.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_payload_v6, t.payload_v6.data(), t.payload_v6.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_payload_v9, t.payload_v9.data(), t.payload_v9.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_x, x.data(), h.cols * sizeof(float), cudaMemcpyHostToDevice));

    // ---- bandwidth ceiling probe ----
    {
        size_t n_words = t.payload.size() / 16;
        uint4 * d_sink = nullptr;
        CU_CHECK(cudaMalloc(&d_sink, sizeof(uint4)));
        const int probe_blocks = 512, probe_threads = 256;
        for (int i = 0; i < 20; ++i) bw_probe_kernel<<<probe_blocks, probe_threads>>>((const uint4 *)d_payload, n_words, d_sink);
        CU_CHECK(cudaDeviceSynchronize());
        cudaEvent_t ea, eb;
        CU_CHECK(cudaEventCreate(&ea)); CU_CHECK(cudaEventCreate(&eb));
        std::vector<float> ms(100);
        for (int i = 0; i < 100; ++i) {
            CU_CHECK(cudaEventRecord(ea));
            bw_probe_kernel<<<probe_blocks, probe_threads>>>((const uint4 *)d_payload, n_words, d_sink);
            CU_CHECK(cudaEventRecord(eb));
            CU_CHECK(cudaEventSynchronize(eb));
            CU_CHECK(cudaEventElapsedTime(&ms[i], ea, eb));
        }
        double med = median_of(ms);
        printf("\nbw_probe (uint4 grid-stride read of payload): %.4f ms  ->  %.2f GB/s ceiling\n",
               med, (double)t.payload.size() / (med * 1e6));
        cudaEventDestroy(ea); cudaEventDestroy(eb);
        cudaFree(d_sink);
    }

    // ---- variants ----
    std::vector<VariantSpec> variants = {
        { "v0_current",       launch_v0, 0 },
        { "v1_warprow",       launch_v1, 0 },
        { "v2_vectorized",    launch_v2, 0 },
        { "v3_fused_outlier", launch_v3, 1 },
        { "v4_2row_1shift",   launch_v4, 1 },
        { "v5_tileband",      launch_v5, 1 },
        { "v6_bandmajor",     launch_v6, 2 },
        { "v7_prefetch",      launch_v7, 2 },
        { "v8_shared_x",      launch_v8, 2 },
        { "v9_tilewarp",      launch_v9, 3 },
        { "v10_tilepar",      launch_v10, 3 },
        { "v11_regdiet",      launch_v11, 3 },
    };
    if (getenv("BMO_BENCH_DEBUG")) {
        // diagnostics: intentionally wrong output, timing-only
        variants.push_back({ "dbg_v5_no_xload", launch_v5_dbg_nox, 1 });
        variants.push_back({ "dbg_v5_no_wload", launch_v5_dbg_now, 1 });
        variants.push_back({ "dbg_v6_no_xload", launch_v6_dbg_nox, 2 });
        variants.push_back({ "dbg_v6_no_wload", launch_v6_dbg_now, 2 });
        variants.push_back({ "dbg_v7_no_xload", launch_v7_dbg_nox, 2 });
        variants.push_back({ "dbg_v7_no_wload", launch_v7_dbg_now, 2 });
        variants.push_back({ "dbg_v6_uniform_x", launch_v6_dbg_unix, 2 });
        variants.push_back({ "dbg_v9_no_xload", launch_v9_dbg_nox, 3 });
        variants.push_back({ "dbg_v9_no_wload", launch_v9_dbg_now, 3 });
        variants.push_back({ "dbg_v10_no_xload", launch_v10_dbg_nox, 3 });
        variants.push_back({ "dbg_v10_no_wload", launch_v10_dbg_now, 3 });
    }

    printf("\n%-22s %10s %10s %14s %12s  %s\n", "variant", "ms/call", "GB/s", "max_abs_diff", "rel_l2", "gate(rel_l2<1e-5)");
    std::vector<float> y_host(h.rows);

    const char * only = getenv("BMO_BENCH_ONLY"); // substring filter on variant names
    for (const auto & v : variants) {
        if (only && !strstr(only, v.name)) continue;
        const void * payload = v.payload_kind == 3 ? d_payload_v9
                             : v.payload_kind == 2 ? d_payload_v6
                             : v.payload_kind == 1 ? d_payload_v3 : d_payload;

        // correctness
        CU_CHECK(cudaMemset(d_y, 0, h.rows * sizeof(float)));
        v.fn(payload, d_x, d_y, h.rows, h.cols, h.n_outliers, 0);
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            printf("%-22s LAUNCH ERROR: %s\n", v.name, cudaGetErrorString(kerr));
            continue;
        }
        kerr = cudaDeviceSynchronize();
        if (kerr != cudaSuccess) {
            printf("%-22s EXEC ERROR: %s\n", v.name, cudaGetErrorString(kerr));
            continue;
        }
        CU_CHECK(cudaMemcpy(y_host.data(), d_y, h.rows * sizeof(float), cudaMemcpyDeviceToHost));
        double max_abs = 0.0, err_l2 = 0.0;
        for (int r = 0; r < h.rows; ++r) {
            double d = (double)y_host[r] - y_ref[r];
            max_abs = std::max(max_abs, fabs(d));
            err_l2 += d * d;
        }
        double rel_l2 = sqrt(err_l2) / ref_l2;

        // timing: median of 100 individually cudaEvent-timed calls
        for (int i = 0; i < 20; ++i) v.fn(payload, d_x, d_y, h.rows, h.cols, h.n_outliers, 0);
        CU_CHECK(cudaDeviceSynchronize());
        cudaEvent_t ea, eb;
        CU_CHECK(cudaEventCreate(&ea)); CU_CHECK(cudaEventCreate(&eb));
        std::vector<float> ms(100);
        for (int i = 0; i < 100; ++i) {
            CU_CHECK(cudaEventRecord(ea));
            v.fn(payload, d_x, d_y, h.rows, h.cols, h.n_outliers, 0);
            CU_CHECK(cudaEventRecord(eb));
            CU_CHECK(cudaEventSynchronize(eb));
            CU_CHECK(cudaEventElapsedTime(&ms[i], ea, eb));
        }
        cudaEventDestroy(ea); cudaEventDestroy(eb);
        double med = median_of(ms);
        double gbps = (double)t.shipped_bytes / (med * 1e6);
        printf("%-22s %10.4f %10.2f %14.6e %12.6e  %s\n",
               v.name, med, gbps, max_abs, rel_l2, rel_l2 < 1e-5 ? "PASS" : "FAIL");
    }

    cudaFree(d_payload); cudaFree(d_payload_v3); cudaFree(d_payload_v6); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
