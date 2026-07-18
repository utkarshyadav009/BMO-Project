// Standalone isolated per-call kernel-equivalence diff tool (Step 2 of the
// corrected kernel sign-off methodology — see moshi_oracle/HANDOFF.md and
// the task brief that produced this file).
//
// PURPOSE: decouple per-call kernel arithmetic differences (OLD production
// mul_mat_vec_bmo_tier_cuda_kernel vs NEW mul_mat_vec_bmo_tier_tilemajor_/
// _rowminor_kernel) from the CASCADE AMPLIFICATION that commit 7332756's
// full-forward-pass residual diff conflated them with. For each of the 31
// BMO_TIER gating layers (0-30) x 2 matmuls (linear_in, linear_out) =
// 62 calls, this tool:
//   1. Loads that tensor's REAL payload from the shipped GGUF, built via
//      each build's OWN real host-side repack logic (ported verbatim below
//      from each checkout's loader.h build_custom_ffn_tensor() — NOT
//      reimplemented/approximated; every offset/repack line was read from
//      the source and transcribed, only top-level symbol names were
//      suffixed _OLD/_NEW to let both coexist in one translation unit).
//   2. Loads the REAL captured input vector (x) that OLD's production
//      binary actually saw at that layer on its first real forward pass
//      (outputs/step2_gating_captures_OLD_8dfd1ba/cpp_gating_{nx,gated}_
//      layer_N.bin) — not synthetic/random data.
//   3. Runs OLD's real kernel and NEW's real kernel back-to-back on the
//      SAME input, in the SAME process, on the SAME GPU — single GEMV call
//      each, no cascading, no re-deriving the input.
//   4. Computes rel_l2 = norm(old_out - new_out) / (norm(old_out) + 1e-12)
//      and max_abs_diff directly between the two kernels' outputs (NOT vs
//      a CPU reference — this is an old-vs-new EQUIVALENCE check, not a
//      correctness-vs-reference check).
//
// Kernel source provenance (verbatim ports, mechanical renaming only):
//   OLD: BMO-Project-old/moshi_oracle/ggml/src/ggml-cuda/convert.cu
//        lines ~129-354 (mul_mat_vec_bmo_tier_cuda_kernel + outlier kernel
//        + host wrapper), checkout HEAD 8dfd1ba.
//   NEW: THIS repo's moshi_oracle/ggml/src/ggml-cuda/convert.cu lines
//        ~156-699 (tilemajor + rowminor kernels + host dispatch wrapper),
//        HEAD 5058f3d.
// Payload-builder source provenance (verbatim host-side ports):
//   OLD: BMO-Project-old/moshi_oracle/moshi.cpp/src/loader.h
//        build_custom_ffn_tensor(), lines ~578-769 ("shipped" format, no
//        band-major repack — OLD predates the band-major rewrite, commit
//        53c61ec, entirely).
//   NEW: THIS repo's moshi_oracle/moshi.cpp/src/loader.h
//        build_custom_ffn_tensor(), lines ~804-1129 (band-major repack).
// GGUF raw-tensor-byte reading pattern (GgufReader) ported from
// moshi_oracle/moshi.cpp/tools/bmo_kernel_bench.cu's GgufReader (same file
// access pattern as loader.h read_raw_bytes_from_gguf_file() — direct
// fread from data_offset+tensor_offset, cross-verified line-by-line against
// loader.h's own read_scalar_i32/read_bytes/read_raw_bytes_from_gguf_file
// before reuse, not blindly trusted).
//
// NOT reused: bmo_kernel_bench.cu's v0_gemv_kernel / launch_v0 and its
// build_repacked_payload() V6/V9 bench-only repack variants — those are
// this project's OWN internal optimization-variant scratch space (a
// different, bench-local block_bmo_tier struct lacking band_table_offset/
// band_layout/outlier_row_starts_offset entirely), not verified identical
// to either build's real production payload format, and explicitly flagged
// as untrusted for this production sign-off decision by the task brief.
//
// Gate: rel_l2 <= 2e-5 per call. 62 calls total.

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
#include <algorithm>

#define CU_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// ============================================================================
// GGUF raw reading — same pattern as bmo_kernel_bench.cu's GgufReader /
// loader.h's read_raw_bytes_from_gguf_file (direct fread from the file's
// tensor data region; scalar GGUF tensors like ".rows"/".cols" are stored
// as 1-element tensors, read the same way).
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
// OLD payload format ("shipped", pre-band-major-rewrite) — struct and
// build function transcribed verbatim from BMO-Project-old/moshi_oracle/
// moshi.cpp/src/loader.h build_custom_ffn_tensor() lines ~578-769 and
// BMO-Project-old/moshi_oracle/ggml/src/ggml-cuda/convert.cu lines 8-30.
// ============================================================================
struct block_bmo_tier_old {
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

    int64_t packed_weights_offset;
    int64_t tile_tiers_offset;
    int64_t outlier_indices_offset;
    int64_t outlier_values_offset;
    int64_t tile_stream_indices_offset;
};

static void build_payload_old(GgufReader & r, const std::string & base_name,
                               std::vector<uint8_t> & payload_out,
                               int32_t & rows_out, int32_t & cols_out, int32_t & n_outliers_out) {
    int32_t rows = r.scalar_i32(base_name + ".rows");
    if (rows <= 0) rows = r.scalar_i32(base_name + ".out_features");
    int32_t cols = r.scalar_i32(base_name + ".cols");
    if (cols <= 0) cols = r.scalar_i32(base_name + ".in_features");
    int32_t n_outliers = r.scalar_i32(base_name + ".n_outliers");

    float scale_int8 = r.scalar_f32(base_name + ".scale_int8", 1.0f);
    float zp_int8    = r.scalar_f32(base_name + ".zp_int8", 0.0f);
    float scale_int4 = r.scalar_f32(base_name + ".scale_int4", 1.0f);
    float zp_int4    = r.scalar_f32(base_name + ".zp_int4", 0.0f);
    float scale_low  = r.scalar_f32(base_name + ".scale_low", 1.0f);
    float zp_low     = r.scalar_f32(base_name + ".zp_low", 0.0f);

    std::vector<uint8_t> pw, tt, oi, ov, n_tiles_buf, tier_offsets_buf;
    r.tensor_bytes(base_name + ".packed_weights", pw);
    r.tensor_bytes(base_name + ".tile_tiers", tt);
    r.tensor_bytes(base_name + ".n_tiles", n_tiles_buf);
    r.tensor_bytes(base_name + ".tier_offsets", tier_offsets_buf);
    r.tensor_bytes(base_name + ".outlier_indices", oi);
    r.tensor_bytes(base_name + ".outlier_values", ov);

    block_bmo_tier_old header;
    memset(&header, 0, sizeof(header));
    header.rows = rows;
    header.cols = cols;
    header.scale_int8 = scale_int8;
    header.zp_int8    = zp_int8;
    header.scale_int4 = scale_int4;
    header.zp_int4    = zp_int4;
    header.scale_low  = scale_low;
    header.zp_low     = zp_low;
    header.n_outliers = n_outliers;
    header.padding = 0;

    if (n_tiles_buf.size() >= 4 * sizeof(int32_t)) memcpy(header.n_tiles, n_tiles_buf.data(), 4 * sizeof(int32_t));
    else memset(header.n_tiles, 0, 4 * sizeof(int32_t));
    if (tier_offsets_buf.size() >= 5 * sizeof(int32_t)) memcpy(header.tier_offsets, tier_offsets_buf.data(), 5 * sizeof(int32_t));
    else memset(header.tier_offsets, 0, 5 * sizeof(int32_t));

    // Compute tile stream indices (global per-tier stream index)
    std::vector<uint16_t> tile_stream_indices(tt.size());
    {
        int32_t ptrs[4] = {0, 0, 0, 0};
        for (size_t t_idx = 0; t_idx < tt.size(); ++t_idx) {
            uint8_t tier = tt[t_idx];
            tile_stream_indices[t_idx] = (uint16_t)ptrs[tier]++;
        }
    }

    size_t write_offset = sizeof(block_bmo_tier_old);
    header.packed_weights_offset = write_offset;
    write_offset += pw.size();
    write_offset = (write_offset + 3) & ~(size_t)3;

    header.tile_tiers_offset = write_offset;
    write_offset += tt.size();
    write_offset = (write_offset + 3) & ~(size_t)3;

    header.outlier_indices_offset = write_offset;
    write_offset += oi.size();
    write_offset = (write_offset + 3) & ~(size_t)3;

    header.outlier_values_offset = write_offset;
    write_offset += ov.size();
    write_offset = (write_offset + 15) & ~(size_t)15;

    header.tile_stream_indices_offset = write_offset;
    write_offset += tile_stream_indices.size() * sizeof(uint16_t);
    write_offset = (write_offset + 15) & ~(size_t)15;

    payload_out.assign(write_offset, 0);
    memcpy(payload_out.data(), &header, sizeof(block_bmo_tier_old));
    memcpy(payload_out.data() + header.packed_weights_offset, pw.data(), pw.size());
    memcpy(payload_out.data() + header.tile_tiers_offset, tt.data(), tt.size());
    if (!oi.empty()) memcpy(payload_out.data() + header.outlier_indices_offset, oi.data(), oi.size());
    if (!ov.empty()) memcpy(payload_out.data() + header.outlier_values_offset, ov.data(), ov.size());
    memcpy(payload_out.data() + header.tile_stream_indices_offset, tile_stream_indices.data(), tile_stream_indices.size() * sizeof(uint16_t));
    // dequantized_cpu_ptr intentionally left 0: real loader only populates it
    // on the CPU (non-CUDA-backend) path; the GPU kernel path (production,
    // what we're testing) never reads it. Verified by reading OLD's
    // convert.cu kernel bodies — no reference to header->dequantized_cpu_ptr.

    rows_out = rows; cols_out = cols; n_outliers_out = n_outliers;
}

// ============================================================================
// NEW payload format (band-major, post kernel-rewrite commit 53c61ec) —
// struct and build function transcribed verbatim from THIS repo's
// moshi_oracle/moshi.cpp/src/loader.h build_custom_ffn_tensor() lines
// ~804-1129 and moshi_oracle/ggml/src/ggml-cuda/convert.cu lines 8-46.
// ============================================================================
struct block_bmo_tier_new {
    int32_t rows;
    int32_t cols;
    int32_t n_tiles[4];
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

    int64_t packed_weights_offset;
    int64_t tile_tiers_offset;
    int64_t outlier_indices_offset;
    int64_t outlier_values_offset;
    int64_t tile_stream_indices_offset;

    int64_t outlier_row_starts_offset;

    int64_t band_table_offset;
    int32_t band_layout;
    int32_t reserved2;
};

static void build_payload_new(GgufReader & r, const std::string & base_name,
                               std::vector<uint8_t> & payload_out,
                               int32_t & rows_out, int32_t & cols_out, int32_t & n_outliers_out) {
    int32_t rows = r.scalar_i32(base_name + ".rows");
    if (rows <= 0) rows = r.scalar_i32(base_name + ".out_features");
    int32_t cols = r.scalar_i32(base_name + ".cols");
    if (cols <= 0) cols = r.scalar_i32(base_name + ".in_features");
    int32_t n_outliers = r.scalar_i32(base_name + ".n_outliers");

    float scale_int8 = r.scalar_f32(base_name + ".scale_int8", 1.0f);
    float zp_int8    = r.scalar_f32(base_name + ".zp_int8", 0.0f);
    float scale_int4 = r.scalar_f32(base_name + ".scale_int4", 1.0f);
    float zp_int4    = r.scalar_f32(base_name + ".zp_int4", 0.0f);
    float scale_low  = r.scalar_f32(base_name + ".scale_low", 1.0f);
    float zp_low     = r.scalar_f32(base_name + ".zp_low", 0.0f);

    std::vector<uint8_t> pw, tt, oi, ov, n_tiles_buf, tier_offsets_buf;
    r.tensor_bytes(base_name + ".packed_weights", pw);
    r.tensor_bytes(base_name + ".tile_tiers", tt);
    r.tensor_bytes(base_name + ".n_tiles", n_tiles_buf);
    r.tensor_bytes(base_name + ".tier_offsets", tier_offsets_buf);
    r.tensor_bytes(base_name + ".outlier_indices", oi);
    r.tensor_bytes(base_name + ".outlier_values", ov);

    block_bmo_tier_new header;
    memset(&header, 0, sizeof(header));
    header.rows = rows;
    header.cols = cols;
    header.scale_int8 = scale_int8;
    header.zp_int8    = zp_int8;
    header.scale_int4 = scale_int4;
    header.zp_int4    = zp_int4;
    header.scale_low  = scale_low;
    header.zp_low     = zp_low;
    header.n_outliers = n_outliers;
    header.padding = 0;

    if (n_tiles_buf.size() >= 4 * sizeof(int32_t)) memcpy(header.n_tiles, n_tiles_buf.data(), 4 * sizeof(int32_t));
    else memset(header.n_tiles, 0, 4 * sizeof(int32_t));
    if (tier_offsets_buf.size() >= 5 * sizeof(int32_t)) memcpy(header.tier_offsets, tier_offsets_buf.data(), 5 * sizeof(int32_t));
    else memset(header.tier_offsets, 0, 5 * sizeof(int32_t));

    if (rows % 64 != 0 || cols % 64 != 0) {
        fprintf(stderr, "FATAL: BMO tensor %s dims (%d x %d) not multiples of 64 — band repack cannot proceed\n",
                base_name.c_str(), rows, cols);
        exit(1);
    }
    const int32_t n_bands = rows / 64;
    const int32_t n_tiles_col_rp = cols / 64;
    std::vector<uint16_t> tile_stream_indices(tt.size());  // within-band position
    std::vector<int32_t> tile_global_stream(tt.size());    // global per-tier stream index
    {
        int32_t gptrs[4] = {0, 0, 0, 0};
        for (size_t t_idx = 0; t_idx < tt.size(); ++t_idx) {
            tile_global_stream[t_idx] = gptrs[tt[t_idx]]++;
        }
        for (int32_t b = 0; b < n_bands; ++b) {
            int32_t bptrs[4] = {0, 0, 0, 0};
            for (int32_t tc2 = 0; tc2 < n_tiles_col_rp; ++tc2) {
                const size_t t_idx = (size_t)b * n_tiles_col_rp + tc2;
                tile_stream_indices[t_idx] = (uint16_t)bptrs[tt[t_idx]]++;
            }
        }
    }

    // MUST match the dispatch rule in ggml/src/ggml-cuda/convert.cu
    // mul_mat_vec_bmo_tier_cuda(): tile-major (2) when cols <= 8192.
    const int32_t band_layout = (cols <= 8192) ? 2 : 1;
    const int slice_bytes_by_tier[4] = { 128, 64, 32, 16 };

    std::vector<int32_t> outlier_row_starts((size_t)rows + 1, 0);
    if (n_outliers > 0 && !oi.empty() && !ov.empty()) {
        int32_t * oi32 = (int32_t *) oi.data();
        uint16_t * ov16 = (uint16_t *) ov.data();
        std::vector<int32_t> order(n_outliers);
        for (int32_t i = 0; i < n_outliers; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return oi32[a] < oi32[b]; });
        std::vector<int32_t> oi_sorted(n_outliers);
        std::vector<uint16_t> ov_sorted(n_outliers);
        for (int32_t i = 0; i < n_outliers; ++i) { oi_sorted[i] = oi32[order[i]]; ov_sorted[i] = ov16[order[i]]; }
        memcpy(oi32, oi_sorted.data(), (size_t)n_outliers * sizeof(int32_t));
        memcpy(ov16, ov_sorted.data(), (size_t)n_outliers * sizeof(uint16_t));
        for (int32_t i = 0; i < n_outliers; ++i) outlier_row_starts[oi_sorted[i] / cols + 1]++;
        for (int32_t r2 = 0; r2 < rows; ++r2) outlier_row_starts[r2 + 1] += outlier_row_starts[r2];
    }

    size_t write_offset = (sizeof(block_bmo_tier_new) + 15) & ~(size_t)15;
    header.packed_weights_offset = write_offset;
    write_offset += pw.size();
    write_offset = (write_offset + 3) & ~(size_t)3;

    header.tile_tiers_offset = write_offset;
    write_offset += tt.size();
    write_offset = (write_offset + 3) & ~(size_t)3;

    header.outlier_indices_offset = write_offset;
    write_offset += oi.size();
    write_offset = (write_offset + 3) & ~(size_t)3;

    header.outlier_values_offset = write_offset;
    write_offset += ov.size();
    write_offset = (write_offset + 15) & ~(size_t)15;

    header.tile_stream_indices_offset = write_offset;
    write_offset += tile_stream_indices.size() * sizeof(uint16_t);
    write_offset = (write_offset + 15) & ~(size_t)15;

    header.outlier_row_starts_offset = write_offset;
    write_offset += (size_t)(rows + 1) * sizeof(int32_t);
    write_offset = (write_offset + 15) & ~(size_t)15;

    header.band_table_offset = write_offset;
    write_offset += ((size_t)n_bands * 4 + 1) * sizeof(int32_t);
    write_offset = (write_offset + 15) & ~(size_t)15;
    header.band_layout = band_layout;
    header.reserved2 = 0;

    std::vector<int32_t> band_table((size_t)n_bands * 4 + 1);
    {
        size_t cur = (size_t)header.packed_weights_offset;
        for (int32_t b = 0; b < n_bands; ++b) {
            for (int t2 = 0; t2 < 4; ++t2) {
                band_table[(size_t)b * 4 + t2] = (int32_t)cur;
                int32_t n_bt = 0;
                for (int32_t tc2 = 0; tc2 < n_tiles_col_rp; ++tc2) {
                    if (tt[(size_t)b * n_tiles_col_rp + tc2] == t2) n_bt++;
                }
                cur += (size_t)n_bt * 64 * slice_bytes_by_tier[t2];
            }
        }
        band_table[(size_t)n_bands * 4] = (int32_t)cur;
        if (cur > (size_t)header.packed_weights_offset + pw.size()) {
            fprintf(stderr, "FATAL: BMO band repack for %s needs %zu bytes but packed region is %zu\n",
                    base_name.c_str(), cur - (size_t)header.packed_weights_offset, pw.size());
            exit(1);
        }
    }

    payload_out.assign(write_offset, 0);
    memcpy(payload_out.data(), &header, sizeof(block_bmo_tier_new));
    for (int32_t b = 0; b < n_bands; ++b) {
        int32_t n_bt[4] = {0, 0, 0, 0};
        for (int32_t tc2 = 0; tc2 < n_tiles_col_rp; ++tc2) n_bt[tt[(size_t)b * n_tiles_col_rp + tc2]]++;
        for (int32_t tc2 = 0; tc2 < n_tiles_col_rp; ++tc2) {
            const size_t t_idx = (size_t)b * n_tiles_col_rp + tc2;
            const int tier = tt[t_idx];
            const int sb = slice_bytes_by_tier[tier];
            const int32_t pos = tile_stream_indices[t_idx];
            const uint8_t * src_tile = pw.data() + header.tier_offsets[tier]
                                     + (size_t)tile_global_stream[t_idx] * 64 * sb;
            uint8_t * dst_base = payload_out.data() + band_table[(size_t)b * 4 + tier];
            for (int ir = 0; ir < 64; ++ir) {
                const size_t dst_slice = (band_layout == 2)
                    ? ((size_t)pos * 64 + ir)
                    : ((size_t)ir * n_bt[tier] + pos);
                memcpy(dst_base + dst_slice * sb, src_tile + (size_t)ir * sb, sb);
            }
        }
    }
    memcpy(payload_out.data() + header.tile_tiers_offset, tt.data(), tt.size());
    if (!oi.empty()) memcpy(payload_out.data() + header.outlier_indices_offset, oi.data(), oi.size());
    if (!ov.empty()) memcpy(payload_out.data() + header.outlier_values_offset, ov.data(), ov.size());
    memcpy(payload_out.data() + header.tile_stream_indices_offset, tile_stream_indices.data(), tile_stream_indices.size() * sizeof(uint16_t));
    memcpy(payload_out.data() + header.outlier_row_starts_offset, outlier_row_starts.data(), ((size_t)rows + 1) * sizeof(int32_t));
    memcpy(payload_out.data() + header.band_table_offset, band_table.data(), band_table.size() * sizeof(int32_t));

    rows_out = rows; cols_out = cols; n_outliers_out = n_outliers;
}

// ============================================================================
// OLD kernel — verbatim port of BMO-Project-old/moshi_oracle/ggml/src/
// ggml-cuda/convert.cu lines 129-354 (checkout HEAD 8dfd1ba). ONLY change:
// top-level names suffixed _OLD and struct type renamed to
// block_bmo_tier_old (to coexist with NEW's kernel in one translation
// unit) — internal logic is byte-for-byte identical to the source.
// ============================================================================
#define BMO_GEMV_BLOCK_SIZE 256
#define BMO_TILE_DIM 64
#define BMO_TILE_ELEMS 4096

static __device__ __forceinline__ float bmo_dequant_element_fast_OLD(
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

static __global__ void mul_mat_vec_bmo_tier_cuda_kernel_OLD(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const int row = blockIdx.x;
    const block_bmo_tier_old * header = (const block_bmo_tier_old *) vx;
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
            printf("FATAL: BMO fused GEMV kernel n_tiles_col=%d exceeds shared memory bound of 512 (max cols=32768). "
                   "Increase s_tiers/s_stream_indices arrays and recompile.\n", n_tiles_col);
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

        float w = bmo_dequant_element_fast_OLD(pw, s_tier_offsets, s_scales_and_zps, tier, stream_idx, in_tile_idx);
        thread_sum += w * x_vec[col];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    __shared__ float warp_sums[BMO_GEMV_BLOCK_SIZE / 32];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

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

static __global__ void apply_outliers_gemv_bmo_tier_kernel_OLD(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const block_bmo_tier_old * header = (const block_bmo_tier_old *) vx;
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

static void mul_mat_vec_bmo_tier_cuda_OLD(
    const void * vx, const float * x_vec, float * y_out,
    const int32_t nrows, const int32_t ncols, const int32_t n_outliers, cudaStream_t stream)
{
    mul_mat_vec_bmo_tier_cuda_kernel_OLD<<<nrows, BMO_GEMV_BLOCK_SIZE, 0, stream>>>(
        vx, x_vec, y_out, ncols);
    CU_CHECK(cudaGetLastError());

    if (n_outliers > 0) {
        const int outlier_block_size = 256;
        const int outlier_grid_size = (n_outliers + outlier_block_size - 1) / outlier_block_size;
        apply_outliers_gemv_bmo_tier_kernel_OLD<<<outlier_grid_size, outlier_block_size, 0, stream>>>(
            vx, x_vec, y_out, ncols);
        CU_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// NEW kernels — verbatim port of THIS repo's moshi_oracle/ggml/src/
// ggml-cuda/convert.cu lines 156-699 (HEAD 5058f3d). ONLY change: top-level
// names suffixed _NEW and struct type renamed to block_bmo_tier_new — the
// device math (bmo_uq exact-float trick, warp partitioning, affine-sum
// fold, outlier CSR correction) is byte-for-byte identical to the source.
// ============================================================================
#define BMO_RM_THREADS 512
#define BMO_RM_WARPS   (BMO_RM_THREADS / 32)

static __device__ __forceinline__ int bmo_slice_bytes_NEW(int tier) {
    return (tier == 0) ? 128 : (tier == 1) ? 64 : (tier == 2) ? 32 : 16;
}

static __device__ __forceinline__ float bmo_uq_NEW(uint32_t w, int sh_left, uint32_t mask) {
    const uint32_t m = (sh_left >= 0) ? (w << sh_left) : (w >> (-sh_left));
    return __int_as_float((m & mask) | 0x40000000u);
}

static __device__ __forceinline__ float bmo_dequant_one_NEW(
    const block_bmo_tier_new * header, const char * tb, int tier, int slice, int cin)
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

static __global__ void __launch_bounds__(BMO_GEMV_BLOCK_SIZE, 5) mul_mat_vec_bmo_tier_tilemajor_kernel_NEW(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const block_bmo_tier_new * header = (const block_bmo_tier_new *) vx;
    const int32_t rows = header->rows;
    const int32_t cols = header->cols;
    const int32_t n_tiles_col = cols / BMO_TILE_DIM;

    __shared__ uint8_t  s_tiers[512];
    __shared__ uint16_t s_pos[512];
    __shared__ float    s_xsum[512];
    __shared__ uint16_t s_list[4 * 512];
    __shared__ int      s_cnt4[4];
    __shared__ float    s_part[8 * BMO_TILE_DIM];

    if (n_tiles_col > 512 || header->band_layout != 2) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL: BMO tile-major GEMV kernel: n_tiles_col=%d (max 512) band_layout=%d (need 2)\n",
                   n_tiles_col, header->band_layout);
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
                            part = fmaf(bmo_uq_NEW(wv[m], sh, 0x00600000u), xk, part);
                        }
                    }
                    acc[m] = fmaf(2.0f * s2, part, acc[m]);
                }
            }
            if (lane == 0) a_x2 += s_xsum[tc];
        }
    }

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
                                part = fmaf(bmo_uq_NEW(w, sh, 0x00780000u), xk, part);
                            }
                        }
                        acc[m] = fmaf(8.0f * s4, part, acc[m]);
                    }
                }
            }
            if (lane == 0) a_x4 += s_xsum[tc];
        }
    }

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
                            part = fmaf(bmo_uq_NEW(w, sh, 0x007F8000u), xk, part);
                        }
                    }
                    acc[m] = fmaf(128.0f * s8, part, acc[m]);
                }
            }
            if (lane == 0) a_x8 += s_xsum[tc];
        }
    }

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
                const float base_w = bmo_dequant_one_NEW(header, tb, tier, slice, col & 63);
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        tot += __shfl_down_sync(0xFFFFFFFF, tot, 1);
        tot += __shfl_down_sync(0xFFFFFFFF, tot, 2);
        if (sub == 0 && live) y_out[row] = tot;
    }
    GGML_UNUSED(ncols);
}

static __global__ void __launch_bounds__(BMO_RM_THREADS, 2) mul_mat_vec_bmo_tier_rowminor_kernel_NEW(
    const void * __restrict__ vx,
    const float * __restrict__ x_vec,
    float * __restrict__ y_out,
    const int32_t ncols)
{
    const block_bmo_tier_new * header = (const block_bmo_tier_new *) vx;
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
        __trap();
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
                        au2[r] = fmaf(bmo_uq_NEW(w[r], sh, 0x00600000u), xk, au2[r]);
                    }
                }
            }
            if (sub == 0) a_x2 += s_xsum[tc];
        }
    }

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
                        au4[r] = fmaf(bmo_uq_NEW(w[r], sh, 0x00780000u), xk, au4[r]);
                    }
                }
            }
            if (sub == 0) a_x4 += s_xsum[tc];
        }
    }

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
                    au8[r] = fmaf(bmo_uq_NEW(w[r], sh, 0x007F8000u), xk, au8[r]);
                }
            }
            if (sub == 0) a_x8 += s_xsum[tc];
        }
    }

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
                const int nn = (band_tab[tier + 1] - band_tab[tier]) / (BMO_TILE_DIM * bmo_slice_bytes_NEW(tier));
                const int slice = ir * nn + (int)s_pos[tc];
                const char * tb = (const char *)header + band_tab[tier];
                const float base_w = bmo_dequant_one_NEW(header, tb, tier, slice, col & 63);
                tot += (__half2float(ov[k]) - base_w) * x_vec[col];
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xFFFFFFFF, tot, o);
        if (lane == 0) y_out[row] = tot;
    }
    GGML_UNUSED(ncols);
}

static void mul_mat_vec_bmo_tier_cuda_NEW(
    const void * vx, const float * x_vec, float * y_out,
    const int32_t nrows, const int32_t ncols, const int32_t n_outliers, cudaStream_t stream)
{
    const int n_blocks = (nrows + BMO_TILE_DIM - 1) / BMO_TILE_DIM;
    if (ncols <= 8192) {
        mul_mat_vec_bmo_tier_tilemajor_kernel_NEW<<<n_blocks, BMO_GEMV_BLOCK_SIZE, 0, stream>>>(
            vx, x_vec, y_out, ncols);
    } else {
        mul_mat_vec_bmo_tier_rowminor_kernel_NEW<<<n_blocks, BMO_RM_THREADS, 0, stream>>>(
            vx, x_vec, y_out, ncols);
    }
    CU_CHECK(cudaGetLastError());
    GGML_UNUSED(n_outliers); // outlier correction is fused in-kernel (CSR ranges) for NEW
}

// ============================================================================
// main: iterate 31 layers x 2 matmuls = 62 calls
// ============================================================================
int main(int argc, char ** argv) {
    const char * gguf_path = (argc > 1) ? argv[1] :
        "moshi_oracle/models_h100_actual/qat_heavy_int2_dir/qat_heavy_int2.gguf";
    const char * x_dir = (argc > 2) ? argv[2] :
        "outputs/step2_gating_captures_OLD_8dfd1ba";
    const char * csv_path = (argc > 3) ? argv[3] :
        "outputs/step2_isolated_diff_results.csv";

    GgufReader r;
    if (!r.open(gguf_path)) {
        fprintf(stderr, "error: failed to open %s\n", gguf_path);
        return 1;
    }

    FILE * csv = fopen(csv_path, "w");
    if (!csv) { fprintf(stderr, "error: cannot open %s for write\n", csv_path); return 1; }
    fprintf(csv, "layer,tensor,dispatch,rows,cols,n_outliers,rel_l2,max_abs_diff,gate,status\n");

    printf("%-6s %-11s %-10s %-7s %-7s %-9s %-14s %-14s %-6s\n",
           "layer", "tensor", "dispatch", "rows", "cols", "n_outl", "rel_l2", "max_abs_diff", "status");

    int n_fail = 0;
    int n_calls = 0;

    for (int li = 0; li <= 30; ++li) {
        for (int kind = 0; kind < 2; ++kind) {
            const char * tname = (kind == 0) ? "linear_in" : "linear_out";
            const char * xkind = (kind == 0) ? "nx" : "gated";
            std::string base_name = "transformer_layers_" + std::to_string(li) + "_gating_" + tname + "_weight";
            std::string xfile = std::string(x_dir) + "/cpp_gating_" + xkind + "_layer_" + std::to_string(li) + ".bin";

            FILE * xf = fopen(xfile.c_str(), "rb");
            if (!xf) { fprintf(stderr, "error: missing x file %s\n", xfile.c_str()); return 1; }
            fseek(xf, 0, SEEK_END);
            long xbytes = ftell(xf);
            fseek(xf, 0, SEEK_SET);
            std::vector<float> x_host((size_t)xbytes / sizeof(float));
            size_t nread = fread(x_host.data(), sizeof(float), x_host.size(), xf);
            fclose(xf);
            if (nread != x_host.size()) { fprintf(stderr, "error: short read on %s\n", xfile.c_str()); return 1; }

            std::vector<uint8_t> payload_old, payload_new;
            int32_t rows_o = 0, cols_o = 0, no_o = 0;
            int32_t rows_n = 0, cols_n = 0, no_n = 0;
            build_payload_old(r, base_name, payload_old, rows_o, cols_o, no_o);
            build_payload_new(r, base_name, payload_new, rows_n, cols_n, no_n);

            if (rows_o != rows_n || cols_o != cols_n || no_o != no_n) {
                fprintf(stderr, "WARNING: header mismatch for %s: OLD(rows=%d,cols=%d,n_outl=%d) NEW(rows=%d,cols=%d,n_outl=%d)\n",
                        base_name.c_str(), rows_o, cols_o, no_o, rows_n, cols_n, no_n);
            }
            if ((int64_t)x_host.size() != cols_o) {
                fprintf(stderr, "FATAL: x vector size %zu != tensor cols %d for %s\n",
                        x_host.size(), cols_o, base_name.c_str());
                return 1;
            }

            void * d_payload_old = nullptr; void * d_payload_new = nullptr;
            float * d_x = nullptr; float * d_y_old = nullptr; float * d_y_new = nullptr;
            CU_CHECK(cudaMalloc(&d_payload_old, payload_old.size()));
            CU_CHECK(cudaMalloc(&d_payload_new, payload_new.size()));
            CU_CHECK(cudaMalloc(&d_x, x_host.size() * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_y_old, (size_t)rows_o * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_y_new, (size_t)rows_n * sizeof(float)));
            CU_CHECK(cudaMemcpy(d_payload_old, payload_old.data(), payload_old.size(), cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_payload_new, payload_new.data(), payload_new.size(), cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_x, x_host.data(), x_host.size() * sizeof(float), cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemset(d_y_old, 0, (size_t)rows_o * sizeof(float)));
            CU_CHECK(cudaMemset(d_y_new, 0, (size_t)rows_n * sizeof(float)));

            mul_mat_vec_bmo_tier_cuda_OLD(d_payload_old, d_x, d_y_old, rows_o, cols_o, no_o, 0);
            mul_mat_vec_bmo_tier_cuda_NEW(d_payload_new, d_x, d_y_new, rows_n, cols_n, no_n, 0);
            CU_CHECK(cudaDeviceSynchronize());

            std::vector<float> y_old(rows_o), y_new(rows_n);
            CU_CHECK(cudaMemcpy(y_old.data(), d_y_old, (size_t)rows_o * sizeof(float), cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(y_new.data(), d_y_new, (size_t)rows_n * sizeof(float), cudaMemcpyDeviceToHost));

            double diff_sq = 0.0, norm_sq = 0.0, max_abs = 0.0;
            for (int i = 0; i < rows_o; ++i) {
                double d = (double)y_old[i] - (double)y_new[i];
                diff_sq += d * d;
                norm_sq += (double)y_old[i] * (double)y_old[i];
                if (fabs(d) > max_abs) max_abs = fabs(d);
            }
            double rel_l2 = sqrt(diff_sq) / (sqrt(norm_sq) + 1e-12);
            bool pass = rel_l2 <= 2e-5;
            n_calls++;
            if (!pass) n_fail++;
            const char * dispatch = (cols_o <= 8192) ? "tilemajor" : "rowminor";

            printf("%-6d %-11s %-10s %-7d %-7d %-9d %-14.6e %-14.6e %-6s\n",
                   li, tname, dispatch, rows_o, cols_o, no_o, rel_l2, max_abs, pass ? "PASS" : "FAIL");
            fprintf(csv, "%d,%s,%s,%d,%d,%d,%.6e,%.6e,2e-05,%s\n",
                    li, tname, dispatch, rows_o, cols_o, no_o, rel_l2, max_abs, pass ? "PASS" : "FAIL");
            fflush(stdout); fflush(csv);

            cudaFree(d_payload_old); cudaFree(d_payload_new);
            cudaFree(d_x); cudaFree(d_y_old); cudaFree(d_y_new);
        }
    }

    fclose(csv);
    printf("\n%d / %d calls FAILED (gate: rel_l2 <= 2e-05)\n", n_fail, n_calls);
    printf("STEP2_VERDICT: %s\n", (n_fail == 0) ? "PASS" : "FAIL");
    return (n_fail > 0) ? 1 : 0;
}
