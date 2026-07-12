#pragma once

#include <functional>
#include <gguf.h>
#include "crc-bbf.h"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cctype>  // isdigit (nvmap client-size parsing)
#ifndef _WIN32
#include <fcntl.h>   // posix_fadvise, POSIX_FADV_DONTNEED (Fix A)
#include <unistd.h>  // fileno's off_t and related POSIX types
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
#include <malloc.h>  // malloc_trim (Fix B, per-layer)
#endif

// MEMLEDGER instrumentation — measurement-only diagnostic for Stage 2 OOM investigation.
// Not gated behind a build flag: this session's ask is to log unconditionally so the
// exact same binary that OOMs also produces the ledger.
static size_t memledger_cuda_free_mib( ggml_backend * backend ) {
    if ( ! backend ) return 0;
    ggml_backend_dev_t dev = ggml_backend_get_device( backend );
    if ( ! dev ) return 0;
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props( dev, &props );
    return props.memory_free / 1024 / 1024;
}

static size_t memledger_rss_mib() {
    FILE * f = fopen( "/proc/self/status", "r" );
    if ( ! f ) return 0;
    size_t rss_kb = 0;
    char line[256];
    while ( fgets( line, sizeof(line), f ) ) {
        if ( strncmp( line, "VmRSS:", 6 ) == 0 ) {
            sscanf( line + 6, "%zu", &rss_kb );
            break;
        }
    }
    fclose( f );
    return rss_kb / 1024;
}

// --- Trustworthy-instrument additions (replace cuda_free as primary signal) ---
// The probe (tools/nvmap_alloc_probe.cu) showed cudaMemGetInfo/the UMA
// free-memory estimate is not precise at the granularity prior sessions
// relied on. These read ground-truth kernel accounting instead.

// /sys/kernel/debug/nvmap/iovmm/clients "total ... <N>K" line — this
// process's real NvMap allocation. Root-only; if unreadable, returns 0 and
// the caller distinguishes via nvmap_readable().
static bool memledger_nvmap_readable() {
    FILE * f = fopen( "/sys/kernel/debug/nvmap/iovmm/clients", "r" );
    if ( ! f ) return false;
    fclose( f );
    return true;
}

static size_t memledger_nvmap_client_kib() {
    FILE * f = fopen( "/sys/kernel/debug/nvmap/iovmm/clients", "r" );
    if ( ! f ) return 0;
    char line[256];
    size_t total_kib = 0;
    while ( fgets( line, sizeof(line), f ) ) {
        // Line format: "  total                                               66820K"
        char * p = line;
        while ( *p == ' ' ) p++;
        if ( strncmp( p, "total", 5 ) == 0 ) {
            // Find the trailing "<digits>K"
            char * end = line + strlen(line);
            while ( end > line && (*(end-1) == '\n' || *(end-1) == '\r') ) end--;
            char * digits_end = end;
            if ( digits_end > line && *(digits_end - 1) == 'K' ) {
                char * digits_start = digits_end - 1;
                while ( digits_start > line && isdigit((unsigned char)*(digits_start - 1)) ) digits_start--;
                total_kib = (size_t) atoll( digits_start );
            }
            break;
        }
    }
    fclose( f );
    return total_kib;
}

// /proc/self/smaps_rollup has Rss and Anonymous (NOT RssFile/RssShmem —
// those live in /proc/self/status instead; the assigning task named
// smaps_rollup for all four fields, which is factually off for two of
// them. Reading each field from where it actually exists rather than
// force-fitting the wrong file.
static size_t memledger_smaps_rollup_field_kib( const char * field ) {
    FILE * f = fopen( "/proc/self/smaps_rollup", "r" );
    if ( ! f ) return 0;
    char line[256];
    size_t field_len = strlen(field);
    size_t val_kb = 0;
    while ( fgets( line, sizeof(line), f ) ) {
        if ( strncmp( line, field, field_len ) == 0 && line[field_len] == ':' ) {
            sscanf( line + field_len + 1, "%zu", &val_kb );
            break;
        }
    }
    fclose( f );
    return val_kb;
}

static size_t memledger_status_field_kib( const char * field ) {
    FILE * f = fopen( "/proc/self/status", "r" );
    if ( ! f ) return 0;
    char line[256];
    size_t field_len = strlen(field);
    size_t val_kb = 0;
    while ( fgets( line, sizeof(line), f ) ) {
        if ( strncmp( line, field, field_len ) == 0 && line[field_len] == ':' ) {
            sscanf( line + field_len + 1, "%zu", &val_kb );
            break;
        }
    }
    fclose( f );
    return val_kb;
}

static size_t memledger_meminfo_field_kib( const char * field ) {
    FILE * f = fopen( "/proc/meminfo", "r" );
    if ( ! f ) return 0;
    char line[256];
    size_t field_len = strlen(field);
    size_t val_kb = 0;
    while ( fgets( line, sizeof(line), f ) ) {
        if ( strncmp( line, field, field_len ) == 0 && line[field_len] == ':' ) {
            sscanf( line + field_len + 1, "%zu", &val_kb );
            break;
        }
    }
    fclose( f );
    return val_kb;
}

static size_t memledger_meminfo_field_mib( const char * field ) {
    return memledger_meminfo_field_kib( field ) / 1024;
}

static void memledger_log( const char * event, const char * name, const char * type,
                            size_t payload_B, size_t alloc_B, size_t nbytes_B,
                            ggml_backend * backend ) {
    size_t nvmap_kib   = memledger_nvmap_client_kib();
    bool   nvmap_ok    = memledger_nvmap_readable();
    size_t smaps_rss   = memledger_smaps_rollup_field_kib( "Rss" );
    size_t smaps_anon  = memledger_smaps_rollup_field_kib( "Anonymous" );
    size_t status_file = memledger_status_field_kib( "RssFile" );
    size_t status_shm  = memledger_status_field_kib( "RssShmem" );
    size_t mem_avail   = memledger_meminfo_field_kib( "MemAvailable" );
    size_t mem_cached  = memledger_meminfo_field_kib( "Cached" );
    fprintf( stderr,
        "MEMLEDGER event=%-20s name=%-60s type=%-10s payload_B=%10zu alloc_B=%10zu nbytes_B=%10zu "
        "UNRELIABLE_REF_cuda_free_MiB=%7zu rss_MiB=%7zu "
        "nvmap_client_KiB=%s%9zu smaps_Rss_KiB=%9zu smaps_Anonymous_KiB=%9zu status_RssFile_KiB=%9zu status_RssShmem_KiB=%9zu "
        "meminfo_MemAvailable_KiB=%9zu meminfo_Cached_KiB=%9zu\n",
        event, name, type, payload_B, alloc_B, nbytes_B,
        memledger_cuda_free_mib(backend), memledger_rss_mib(),
        nvmap_ok ? " " : "U", nvmap_kib, smaps_rss, smaps_anon, status_file, status_shm,
        mem_avail, mem_cached );
    fflush( stderr );
}

// Attribution instrumentation: per-class breakdown at each bmo_layer_done,
// to distinguish page-cache (Cached), anon-heap (AnonPages), and other
// memory classes from /proc/meminfo — none of this changes any allocation
// behavior, measurement only.
static void memledger_log_meminfo( const char * layer_label ) {
    size_t mem_free  = memledger_meminfo_field_mib( "MemFree" );
    size_t cached    = memledger_meminfo_field_mib( "Cached" );
    size_t buffers   = memledger_meminfo_field_mib( "Buffers" );
    size_t anon      = memledger_meminfo_field_mib( "AnonPages" );
    size_t mapped    = memledger_meminfo_field_mib( "Mapped" );

    // /sys/kernel/debug/nvmap/iovmm/clients requires root; log "unavailable"
    // explicitly rather than silently omitting the column when unreadable.
    FILE * nvf = fopen( "/sys/kernel/debug/nvmap/iovmm/clients", "r" );
    char nvmap_str[32];
    if ( nvf ) {
        // Content format is client-listing text, not a single number — report
        // that it WAS readable (root) rather than parse the whole table here;
        // the actual root-cause question this instrumentation answers is
        // page-cache vs anon-heap, both already covered by the fields above.
        snprintf( nvmap_str, sizeof(nvmap_str), "readable" );
        fclose( nvf );
    } else {
        snprintf( nvmap_str, sizeof(nvmap_str), "unavailable" );
    }

    fprintf( stderr,
        "MEMLEDGER_MEMINFO layer=%-8s MemFree_MiB=%7zu Cached_MiB=%7zu Buffers_MiB=%7zu AnonPages_MiB=%7zu Mapped_MiB=%7zu nvmap_iovmm=%s\n",
        layer_label, mem_free, cached, buffers, anon, mapped, nvmap_str );
    fflush( stderr );
}

// Falsification probe: BMO_DRY_ALLOC=1 runs the full per-layer loop (file
// reads, payload assembly, MEMLEDGER logging) but SKIPS the actual device
// allocation (ggml_backend_alloc_ctx_tensors) and upload
// (ggml_backend_tensor_set) for BMO tensors. If cuda_free/d_free per layer
// still moves with this set, the gap is not from BMO's own device
// allocations at all. Measurement only — not for production use, tensors
// have no valid buffer afterward.
static bool bmo_dry_alloc_enabled() {
    static int cached = -1;
    if ( cached < 0 ) cached = ( getenv("BMO_DRY_ALLOC") != NULL ) ? 1 : 0;
    return cached == 1;
}

// BMO double-storage fix: identifies the 4 LARGE raw sub-component tensors of
// BMO_TIER gating weights (layers 0-30 only — layer 31's gating is a single
// plain tensor with no dot-suffix, so it never matches this and is untouched,
// per the "layer 31 stays F16, do not optimize" constraint). The small scalar
// siblings (.rows/.cols/.n_outliers/.scale_*/.zp_*/.n_tiles/.tier_offsets/
// .packing_version, all 4-20 bytes) are deliberately left on the normal
// device-buffer path — they're negligible and touching them adds risk for
// no measurable gain.
static bool is_bmo_big_subcomponent( const std::string & name ) {
    if ( name.find( "_gating_linear_in_weight." ) == std::string::npos &&
         name.find( "_gating_linear_out_weight." ) == std::string::npos )
        return false;
    static const char * big_suffixes[] = {
        ".packed_weights", ".tile_tiers", ".outlier_indices", ".outlier_values"
    };
    for ( auto suf : big_suffixes ) {
        size_t L = strlen(suf);
        if ( name.size() >= L && name.compare( name.size() - L, L, suf ) == 0 )
            return true;
    }
    return false;
}

inline void quantize_wht_nf2(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    struct block_q2_K {
        uint8_t scales[16];
        uint8_t qs[64];
        ggml_fp16_t d;
        ggml_fp16_t dmin;
    };

    block_q2_K * out = (block_q2_K *) dst;
    int64_t num_blocks = n_per_row / 256;

    for (int64_t r = 0; r < nrows; r++) {
        const float * row = src + r * n_per_row;
        block_q2_K * row_out = out + r * num_blocks;

        for (int64_t b = 0; b < num_blocks; b++) {
            const float * block_data = row + b * 256;
            block_q2_K & block = row_out[b];

            // Compute RMS scales
            float sum0 = 0.0f;
            for (int i = 0; i < 128; i++) {
                sum0 += block_data[i] * block_data[i];
            }
            float scale0 = sqrtf(sum0 / 128.0f);
            if (scale0 < 1e-8f) scale0 = 1e-8f;

            float sum1 = 0.0f;
            for (int i = 128; i < 256; i++) {
                sum1 += block_data[i] * block_data[i];
            }
            float scale1 = sqrtf(sum1 / 128.0f);
            if (scale1 < 1e-8f) scale1 = 1e-8f;

            // Store scales in d and dmin
            block.d = ggml_fp32_to_fp16(scale0);
            block.dmin = ggml_fp32_to_fp16(scale1);

            // Initialize scales array to 0
            memset(block.scales, 0, sizeof(block.scales));

            // Quantize and pack codes (4 codes per byte, little-endian)
            for (int n = 0; n < 2; n++) {
                float scale = (n == 0) ? scale0 : scale1;
                const float * group_data = block_data + 128 * n;

                for (int l = 0; l < 32; l++) {
                    uint8_t q = 0;
                    int offsets[4] = {0, 32, 64, 96};
                    for (int o = 0; o < 4; o++) {
                        float val = group_data[l + offsets[o]];
                        float normalized = val / scale;

                        // NF2 Lloyd-Max Quantization
                        uint8_t code = 0;
                        if (normalized < -0.9816f) {
                            code = 0;
                        } else if (normalized < 0.0f) {
                            code = 1;
                        } else if (normalized < 0.9816f) {
                            code = 2;
                        } else {
                            code = 3;
                        }

                        q |= (code << (2 * o));
                    }
                    block.qs[32 * n + l] = q;
                }
            }
        }
    }
}

// TODO: remove prefix
std::tuple<std::string, std::string> split_first( const std::string& input, char c ) {
    size_t pos = input.find(c);
    if (pos == std::string::npos)
        return {input, ""};
    return {input.substr(0, pos), input.substr(pos + 1)};
}

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

    // Outliers are stored sorted by flat index (row-major), and a CSR-style
    // per-row range table (int32 x (rows+1)) lives at this offset, so the
    // fused GEMV kernel can apply each row's outliers in-kernel instead of
    // a second atomicAdd kernel launch.
    // MUST stay in sync with ggml/src/ggml-quants.h and
    // ggml/src/ggml-cuda/convert.cu.
    int64_t outlier_row_starts_offset;

    // Band-major packed stream layout: per 64-row tile band, per tier,
    // tiles in tile-column order. band_table_offset points at int32
    // absolute byte offsets [n_bands*4 + 1] (last entry = end sentinel,
    // so (tab[i+1]-tab[i])/(64*slice_bytes) is the band+tier tile count).
    // band_layout: 1 = row-minor [ir][pos][slice] (large cols),
    //              2 = tile-major [pos][ir][slice] (cols <= 8192).
    // tile_stream_indices hold each tile's POSITION within its band+tier
    // list (no longer the global per-tier stream index).
    int64_t band_table_offset;
    int32_t band_layout;
    int32_t reserved2;
};

class WeightLoader {
    public:

    struct alloc_request_t {
        ggml_tensor ** result;
        int n_dims;
        NE ne;
        ggml_type type;
        std::string name;
    };

    std::string filename;

    unref_ptr<SafeTensorFile> stf;
    gguf_context * gguf;

    ScratchContext * scratch;
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_context * custom_ctx;
    std::vector<float*> cpu_dequantized_ptrs;
    ggml_backend_buffer_t buffer;
    std::vector<alloc_request_t> alloc_requests;
    std::vector<std::function<void(WeightLoader*)>> init_requests;
    bool quantize;
    ggml_type qtype;

    bool is_gguf;
    std::map<std::string,ggml_tensor*> tensors;

    // V2 fix (host heap retention during BMO repack): the 4 large BMO
    // sub-components (packed_weights, tile_tiers, outlier_indices,
    // outlier_values) must coexist within one tensor's processing, so they
    // need 4 separate buffers — but each is REUSED across all 31 layers
    // (grown once to its max size, then resize()'d without reallocating)
    // instead of a fresh std::vector allocated and freed per call. Measured
    // root cause: ~92 MiB/layer RSS growth that glibc wasn't returning to
    // the OS between the 62 rapid alloc/free cycles (see commit message).
    std::vector<uint8_t> reuse_buf_packed_weights;
    std::vector<uint8_t> reuse_buf_tile_tiers;
    std::vector<uint8_t> reuse_buf_outlier_indices;
    std::vector<uint8_t> reuse_buf_outlier_values;

    // Fix B (anon heap): the repacked `payload` buffer in build_custom_ffn_tensor
    // was the dominant per-layer retention source (~56 MiB/layer, matching the
    // tracked device allocation almost exactly) — reused the same way as the 4
    // read-staging buffers above. Reserved to ~60 MiB (largest single-layer
    // payload observed) on first use in build_custom_ffn_tensor.
    std::vector<uint8_t> reuse_buf_payload;

    // MEMLEDGER: tracks how many of a BMO gating layer's 2 tensors (in/out) have
    // been repacked, so bmo_layer_done can be logged once per layer (not per tensor).
    std::map<int,int> memledger_bmo_layer_tensor_count;

private:
    WeightLoader(const char * filename, SafeTensorFile * stf, ScratchContext * scratch, ggml_backend * backend = NULL) {
        this->filename = filename;
        this->stf = stf;
        this->gguf = NULL;
        this->scratch = scratch;
        this->backend = backend;
        this->ctx = NULL;
        this->custom_ctx = NULL;
        buffer = NULL;
        quantize = false;
        qtype = GGML_TYPE_Q4_0;
        is_gguf = false;
    }
    WeightLoader(const char * filename, gguf_context * gguf, ggml_context * ctx, ScratchContext * scratch, ggml_backend * backend = NULL) {
        this->filename = filename;
        this->stf = NULL;
        this->gguf = gguf;
        this->scratch = scratch;
        this->backend = backend;
        this->ctx = ctx;
        this->custom_ctx = NULL;
        buffer = NULL;
        quantize = false;
        qtype = GGML_TYPE_Q4_0;
        is_gguf = true;
    }
public:
    ~WeightLoader() {
        if (buffer)
            ggml_backend_buffer_free( buffer );
        if (ctx)
            ggml_free( ctx );
        if (custom_ctx)
            ggml_free( custom_ctx );
        if (gguf)
            gguf_free( gguf );
        for (float * ptr : cpu_dequantized_ptrs) {
            free(ptr);
        }
    }

    static WeightLoader * from_safetensor( const char * filename, ScratchContext * scratch, ggml_backend * backend = NULL ) {
        auto stf = SafeTensorFile::from_file( filename );
        if ( ! stf )
            return NULL;
        return new WeightLoader( filename, stf, scratch, backend );
    }

    static WeightLoader * from_gguf( const char * filename, ScratchContext * scratch, ggml_backend * backend = NULL ) {
        ggml_context * ctx;
        gguf_init_params params;
        params.no_alloc = true;
        params.ctx = &ctx;

        auto gguf = gguf_init_from_file( filename, params );
        if ( ! gguf ) {
            return NULL;
        }
        assert( ctx );
        auto loader = new WeightLoader( filename, gguf, ctx, scratch, backend );

        return loader;
    }

    safetensor_t * find( std::string name ) {
        // TODO: remove the the prefix
        auto [_, _name] = split_first(name, '.');
        auto res = stf->find( _name );
        printf("DEBUG: WeightLoader::find: name=%s, _name=%s, found=%d\n", name.c_str(), _name.c_str(), res != NULL); fflush(stdout);
        return res;
    }

    void init( safetensor_t * safetensor, ggml_tensor * tensor ) {
        stf->init( safetensor, tensor, backend );
    }

    void add_alloc( ggml_tensor ** result, int n_dims, NE ne, ggml_type type, std::string name ) {
        assert( ctx == NULL );
        alloc_requests.push_back({ result, n_dims, {ne[0], ne[1], ne[2], ne[3]}, type, name });
    }

    void add_init( std::function<void(WeightLoader*)> on_init ) {
        init_requests.push_back( on_init );
    }

    std::string tensor_name( std::string & name ) {
        auto name_size = name.size();
        if ( name_size < GGML_MAX_NAME )
            return name;
        crc_t crc;
        crc = crc_init();
        crc = crc_update(crc, (unsigned char *)name.c_str(), name_size );
        crc = crc_finalize(crc);
        std::string crc_name;
        crc_name.resize(8);
        static const char * hex = "0123456789abcdef";
        for ( int i = 0; i < 8; i++ ) {
            crc_name[i] = hex[ (crc >> 4) & 0xf ];
            crc_name[i] = hex[ crc & 0xf ];
            crc >>= 8;
        }
        return crc_name;
    }

    int32_t read_scalar_i32(std::string name, int32_t def_val = 0) {
        auto t = tensors.find(name);
        if (t == tensors.end()) return def_val;
        int32_t val = def_val;
        ggml_backend_tensor_get(t->second, &val, 0, sizeof(int32_t));
        return val;
    }

    float read_scalar_f32(std::string name, float def_val = 0.0f) {
        auto t = tensors.find(name);
        if (t == tensors.end()) return def_val;
        float val = def_val;
        ggml_backend_tensor_get(t->second, &val, 0, sizeof(float));
        return val;
    }

    std::vector<uint8_t> read_bytes(std::string name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return {};
        size_t nbytes = ggml_nbytes(it->second);
        std::vector<uint8_t> buf(nbytes);
        ggml_backend_tensor_get(it->second, buf.data(), 0, nbytes);
        return buf;
    }

    // BMO double-storage fix: reads a large sub-component's bytes directly from
    // the GGUF file, bypassing the device buffer entirely. Used only for the
    // 4 large BMO sub-components (see is_bmo_big_subcomponent) — those are
    // never allocated in the shared device buffer (load_gguf marks them with
    // a sentinel data pointer so ggml_backend_alloc_ctx_tensors skips them),
    // so read_bytes()/ggml_backend_tensor_get() cannot be used for them.
    //
    // V2 fix: `out` is one of the 4 reuse_buf_* members, not a fresh local —
    // resize() only reallocates if the new size exceeds current capacity, so
    // after the first (largest) tensor of each component type, subsequent
    // calls reuse the same backing storage instead of allocating and freeing
    // a new one 62 times. Measured: this was the source of ~92 MiB/layer of
    // RSS growth that glibc wasn't returning to the OS between calls.
    void read_raw_bytes_from_gguf_file( const std::string & name, std::vector<uint8_t> & out ) {
        int64_t tid = gguf_find_tensor( gguf, name.c_str() );
        if ( tid < 0 ) { out.resize(0); return; }
        size_t data_offset   = gguf_get_data_offset( gguf );
        size_t tensor_offset = gguf_get_tensor_offset( gguf, tid );
        size_t nbytes        = gguf_get_tensor_size( gguf, tid );
        out.resize( nbytes );
        if ( nbytes == 0 ) return;
        FILE * f = fopen( filename.c_str(), "rb" );
        if ( ! f ) {
            fprintf( stderr, "error: failed to reopen \"%s\" for BMO host-side read of %s\n",
                      filename.c_str(), name.c_str() );
            exit(1);
        }
#ifdef _WIN32
        _fseeki64( f, data_offset + tensor_offset, SEEK_SET );
#else
        fseek( f, data_offset + tensor_offset, SEEK_SET );
#endif
        size_t r = fread( out.data(), nbytes, 1, f );
        // Fix A (page cache): this range won't be re-read (each tensor's raw
        // bytes are read exactly once), so tell the kernel to drop it from
        // the page cache immediately rather than let it accumulate across
        // all 62 tensors. Advisory only — doesn't touch `out`'s contents.
#ifndef _WIN32
        posix_fadvise( fileno(f), (off_t)(data_offset + tensor_offset), (off_t)nbytes, POSIX_FADV_DONTNEED );
#endif
        fclose( f );
        if ( r != 1 ) {
            fprintf( stderr, "error: failed to read tensor %s from file (BMO host-side read)\n", name.c_str() );
            exit(1);
        }
    }

    // Fix A (page cache): drop the whole GGUF file from page cache once BMO
    // loading is done, in case any cache pages accumulated outside the
    // per-tensor ranges above (e.g. from load_gguf()'s own earlier read of
    // the non-BMO tensors, which goes through a separate code path).
    void fadvise_dontneed_whole_file() {
#ifndef _WIN32
        FILE * f = fopen( filename.c_str(), "rb" );
        if ( ! f ) return;
        posix_fadvise( fileno(f), 0, 0, POSIX_FADV_DONTNEED ); // len=0 means "to EOF"
        fclose( f );
#endif
    }

    std::vector<float> dequantize_attn_to_f32(std::string & base_name, int32_t & rows, int32_t & cols) {
        rows = read_scalar_i32(base_name + ".rows");
        if (rows <= 0) rows = read_scalar_i32(base_name + ".out_features");
        cols = read_scalar_i32(base_name + ".cols");
        if (cols <= 0) cols = read_scalar_i32(base_name + ".in_features");
        int32_t group_size = read_scalar_i32(base_name + ".group_size", 128);
        int32_t n_groups = read_scalar_i32(base_name + ".n_groups", cols / group_size);

        auto pw = read_bytes(base_name + ".packed_weights");
        auto scales = read_bytes(base_name + ".scales");
        auto zeros = read_bytes(base_name + ".zeros");

        const float * p_scales = (const float *) scales.data();
        const float * p_zeros = (const float *) zeros.data();

        std::vector<float> w_f32(rows * cols);

        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < n_groups; ++g) {
                float scale = p_scales[r * n_groups + g];
                float zero = p_zeros[r * n_groups + g];
                for (int i = 0; i < group_size; ++i) {
                    int c = g * group_size + i;
                    int flat_idx = r * cols + c;
                    int byte_idx = flat_idx / 2;
                    uint8_t b = pw[byte_idx];
                    uint8_t q = (flat_idx % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                    w_f32[flat_idx] = (float)q * scale + zero;
                }
            }
        }
        return w_f32;
    }

    ggml_tensor * build_quantized_attn_tensor(std::string & name) {
        std::string base_name = name;
        std::replace(base_name.begin(), base_name.end(), '.', '_');

        std::string name_pw = base_name + ".packed_weights";
        if (tensors.find(name_pw) == tensors.end()) {
            return NULL;
        }

        int32_t rows = 0;
        int32_t cols = 0;
        std::vector<float> w_f32 = dequantize_attn_to_f32(base_name, rows, cols);

        if (!custom_ctx) {
            ggml_init_params params;
            params.mem_size = 10 * 1024 * 1024;
            params.mem_buffer = NULL;
            params.no_alloc = true;
            custom_ctx = ggml_init(params);
        }

        ggml_type target_qtype = GGML_TYPE_Q4_K;
        ggml_tensor * custom_tensor = ggml_new_tensor_2d(custom_ctx, target_qtype, cols, rows);
        ggml_set_name(custom_tensor, base_name.c_str());

        if (backend) {
            ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
        } else {
            custom_tensor->data = malloc(ggml_row_size(target_qtype, cols) * rows);
        }

        size_t qsize = ggml_row_size(target_qtype, cols) * rows;
        std::vector<uint8_t> qdata(qsize);
        ggml_quantize_chunk(target_qtype, w_f32.data(), qdata.data(), 0, rows, cols, nullptr);

        if (backend) {
            ggml_backend_tensor_set(custom_tensor, qdata.data(), 0, qsize);
        } else {
            memcpy(custom_tensor->data, qdata.data(), qsize);
        }

        tensors[base_name] = custom_tensor;
        return custom_tensor;
    }

    float * dequantize_ffn_cpu(const block_bmo_tier & header,
                               const uint8_t * pw,
                               const uint8_t * tile_tiers,
                               const int32_t * outlier_indices,
                               const ggml_fp16_t * outlier_values) {
        int32_t rows = header.rows;
        int32_t cols = header.cols;
        int64_t total = (int64_t) rows * (int64_t) cols;
        float * w_f32 = (float *) malloc(total * sizeof(float));
        memset(w_f32, 0, total * sizeof(float));

        int32_t n_fp16 = header.n_tiles[0];
        int32_t n_int8 = header.n_tiles[1];
        int32_t n_int4 = header.n_tiles[2];
        int32_t n_int2 = header.n_tiles[3];

        int32_t off0 = header.tier_offsets[0];
        int32_t off1 = header.tier_offsets[1];
        int32_t off2 = header.tier_offsets[2];
        int32_t off3 = header.tier_offsets[3];
        int32_t off4 = header.tier_offsets[4];

        int32_t tile_rows = 64;
        int32_t tile_cols = 64;
        int32_t tile_size = tile_rows * tile_cols;

        std::vector<float> fp16_tiles_deq(n_fp16 * tile_size);
        if (n_fp16 > 0) {
            const ggml_fp16_t * raw_fp16 = (const ggml_fp16_t *)(pw + off0);
            for (int64_t i = 0; i < (int64_t)n_fp16 * tile_size; ++i) {
                fp16_tiles_deq[i] = ggml_fp16_to_fp32(raw_fp16[i]);
            }
        }

        std::vector<float> int8_tiles_deq(n_int8 * tile_size);
        if (n_int8 > 0) {
            const uint8_t * raw_int8 = pw + off1;
            for (int64_t i = 0; i < (int64_t)n_int8 * tile_size; ++i) {
                int8_tiles_deq[i] = ((float)raw_int8[i] - header.zp_int8) * header.scale_int8;
            }
        }

        std::vector<float> int4_tiles_deq(n_int4 * tile_size);
        if (n_int4 > 0) {
            const uint8_t * raw_int4 = pw + off2;
            for (int64_t t = 0; t < n_int4; ++t) {
                for (int32_t i = 0; i < tile_size; ++i) {
                    int32_t flat_idx = t * tile_size + i;
                    int32_t byte_idx = flat_idx / 2;
                    uint8_t b = raw_int4[byte_idx];
                    uint8_t q = (flat_idx % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                    int4_tiles_deq[flat_idx] = ((float)q - header.zp_int4) * header.scale_int4;
                }
            }
        }

        std::vector<float> int2_tiles_deq(n_int2 * tile_size);
        if (n_int2 > 0) {
            const uint8_t * raw_int2 = pw + off3;
            for (int64_t t = 0; t < n_int2; ++t) {
                for (int32_t i = 0; i < tile_size; ++i) {
                    int32_t flat_idx = t * tile_size + i;
                    int32_t byte_idx = flat_idx / 4;
                    uint8_t b = raw_int2[byte_idx];
                    uint8_t q = (flat_idx % 4 == 0) ? (b & 0x03) :
                                ((flat_idx % 4 == 1) ? ((b >> 2) & 0x03) :
                                 ((flat_idx % 4 == 2) ? ((b >> 4) & 0x03) : ((b >> 6) & 0x03)));
                    int2_tiles_deq[flat_idx] = ((float)q - header.zp_low) * header.scale_low;
                }
            }
        }

        int32_t n_tiles_col = cols / 64;
        int32_t fp16_ptr = 0;
        int32_t int8_ptr = 0;
        int32_t int4_ptr = 0;
        int32_t int2_ptr = 0;

        int32_t n_tiles_total = rows * cols / tile_size;
        for (int32_t t_idx = 0; t_idx < n_tiles_total; ++t_idx) {
            uint8_t tier = tile_tiers[t_idx];
            int32_t tile_r = t_idx / n_tiles_col;
            int32_t tile_c = t_idx % n_tiles_col;
            int32_t row_start = tile_r * tile_rows;
            int32_t col_start = tile_c * tile_cols;

            const float * src_tile = NULL;
            if (tier == 0) {
                src_tile = fp16_tiles_deq.data() + fp16_ptr * tile_size;
                fp16_ptr++;
            } else if (tier == 1) {
                src_tile = int8_tiles_deq.data() + int8_ptr * tile_size;
                int8_ptr++;
            } else if (tier == 2) {
                src_tile = int4_tiles_deq.data() + int4_ptr * tile_size;
                int4_ptr++;
            } else if (tier == 3) {
                src_tile = int2_tiles_deq.data() + int2_ptr * tile_size;
                int2_ptr++;
            }

            for (int32_t tr = 0; tr < tile_rows; ++tr) {
                for (int32_t tc = 0; tc < tile_cols; ++tc) {
                    w_f32[(row_start + tr) * cols + (col_start + tc)] = src_tile[tr * tile_cols + tc];
                }
            }
        }

        if (header.n_outliers > 0 && outlier_indices && outlier_values) {
            for (int32_t i = 0; i < header.n_outliers; ++i) {
                int32_t idx = outlier_indices[i];
                w_f32[idx] = ggml_fp16_to_fp32(outlier_values[i]);
            }
        }

        return w_f32;
    }

    ggml_tensor * build_custom_ffn_tensor(std::string & name) {
        std::string base_name = name;
        std::replace(base_name.begin(), base_name.end(), '.', '_');

        std::string name_pw = base_name + ".packed_weights";
        // BMO double-storage fix: .packed_weights is deliberately excluded from
        // `tensors[]` (see load_gguf()), so existence must be checked against the
        // GGUF file's own tensor index instead of the device-tensor map.
        if (gguf_find_tensor(gguf, name_pw.c_str()) < 0) {
            return NULL;
        }

        int32_t rows = read_scalar_i32(base_name + ".rows");
        if (rows <= 0) rows = read_scalar_i32(base_name + ".out_features");
        int32_t cols = read_scalar_i32(base_name + ".cols");
        if (cols <= 0) cols = read_scalar_i32(base_name + ".in_features");
        int32_t n_outliers = read_scalar_i32(base_name + ".n_outliers");

        float scale_int8 = read_scalar_f32(base_name + ".scale_int8", 1.0f);
        float zp_int8 = read_scalar_f32(base_name + ".zp_int8", 0.0f);
        float scale_int4 = read_scalar_f32(base_name + ".scale_int4", 1.0f);
        float zp_int4 = read_scalar_f32(base_name + ".zp_int4", 0.0f);
        float scale_low = read_scalar_f32(base_name + ".scale_low", 1.0f);
        float zp_low = read_scalar_f32(base_name + ".zp_low", 0.0f);

        // BMO double-storage fix: these 4 are the large fields, excluded from the
        // device buffer in load_gguf() — read directly from the file instead. The
        // small scalar fields below (n_tiles, tier_offsets) were left on the normal
        // device-buffer path (negligible size, not worth the extra risk to touch).
        // V2 fix: pw/tt/oi/ov are now REFERENCES to the 4 reusable member buffers,
        // not fresh per-call std::vectors — see read_raw_bytes_from_gguf_file.
        read_raw_bytes_from_gguf_file(base_name + ".packed_weights", reuse_buf_packed_weights);
        read_raw_bytes_from_gguf_file(base_name + ".tile_tiers", reuse_buf_tile_tiers);
        auto & pw = reuse_buf_packed_weights;
        auto & tt = reuse_buf_tile_tiers;
        auto n_tiles_buf = read_bytes(base_name + ".n_tiles");
        auto tier_offsets_buf = read_bytes(base_name + ".tier_offsets");
        read_raw_bytes_from_gguf_file(base_name + ".outlier_indices", reuse_buf_outlier_indices);
        read_raw_bytes_from_gguf_file(base_name + ".outlier_values", reuse_buf_outlier_values);
        auto & oi = reuse_buf_outlier_indices;
        auto & ov = reuse_buf_outlier_values;

        // Pre-compute payload offset to optimize memory allocation size.
        // The packed region start is rounded up to 16 so the band-major
        // sub-streams (whose sizes are all multiples of 1024) stay 16-byte
        // aligned for the GEMV kernels' uint4/float4 loads — sizeof(header)
        // itself is not a multiple of 16.
        size_t offset = (sizeof(block_bmo_tier) + 15) & ~(size_t)15;
        offset += pw.size();
        offset = (offset + 3) & ~3;
        offset += tt.size();
        offset = (offset + 3) & ~3;
        offset += oi.size();
        offset = (offset + 3) & ~3;
        offset += ov.size();
        offset = (offset + 15) & ~15;
        offset += tt.size() * sizeof(uint16_t);
        offset = (offset + 15) & ~15;
        offset += (size_t)(rows + 1) * sizeof(int32_t); // outlier CSR row starts
        offset = (offset + 15) & ~15;
        offset += ((size_t)(rows / 64) * 4 + 1) * sizeof(int32_t); // band table (+ end sentinel)
        offset = (offset + 15) & ~15;

        if (!custom_ctx) {
            ggml_init_params params;
            params.mem_size = 10 * 1024 * 1024;
            params.mem_buffer = NULL;
            params.no_alloc = true;
            custom_ctx = ggml_init(params);
        }

        ggml_tensor * custom_tensor = ggml_new_tensor_2d(custom_ctx, GGML_TYPE_BMO_TIER, cols, rows);
        ggml_set_name(custom_tensor, base_name.c_str());

        // Stash outlier count in op_params[0] for CUDA backend to use.
        // This is safe because BMO_TIER tensors are weight leaves, not op outputs — op_params is unused.
        custom_tensor->op_params[0] = n_outliers;

        // Diagnostic: print actual dims for every BMO tensor so n_tiles_col can be verified
        // against the s_tiers[] shared memory bound (512 max = 32768 cols max).
        // Payload bytes (physical) vs logical nbytes (cols*rows) are intentionally different — see comment below.
        {
            int n_tiles_col_diag = (cols > 0) ? cols / 64 : 0;
            fprintf(stderr, "BMO_TENSOR_LOAD: %-60s  rows=%5d cols=%5d n_tiles_col=%3d n_outliers=%5d  payload_bytes=%7zu  logical_nbytes=%7zu\n",
                    base_name.c_str(), rows, cols, n_tiles_col_diag, n_outliers,
                    offset, (size_t)cols * rows);
            fflush(stderr);
        }

        size_t memledger_bmo_alloc_B = 0;
        if (backend) {
            // EXPERIMENT BRANCH ONLY — DO NOT SHIP WITHOUT AUDIT.
            // We temporarily lie to the ggml allocator about tensor shape so it allocates
            // exactly `offset` bytes (the packed BMO_TIER payload) instead of cols*rows bytes.
            // After allocation we restore the logical dims so kernel dispatch sees the correct shape.
            //
            // KNOWN LANDMINE: After restore, ggml_nbytes(tensor) returns cols*rows while the
            // backing buffer is only `offset` bytes (~2x smaller). Any call to
            // ggml_backend_tensor_get/copy or bad_padding_clear that uses ggml_nbytes() as the
            // copy length will read/write past the allocation into the next tensor's memory.
            // Safe here only because BMO_TIER tensors go through the custom fused GEMV path
            // exclusively and are never copied back to host during inference.
            // If this ever changes, replace with a proper custom allocator that tracks
            // physical_size separately from logical ne[].
            custom_tensor->ne[0] = offset;
            custom_tensor->ne[1] = 1;
            custom_tensor->nb[1] = offset;
            custom_tensor->nb[2] = offset;
            custom_tensor->nb[3] = offset;
            if ( ! bmo_dry_alloc_enabled() ) {
                // MEMLEDGER: capture the buffer this call actually allocates (previously discarded).
                // Each call to ggml_backend_alloc_ctx_tensors on the shared custom_ctx allocates a
                // NEW separate backend buffer for whatever tensor(s) in custom_ctx are still
                // unallocated — i.e. one dedicated buffer per BMO tensor, not one shared pool.
                ggml_backend_buffer_t memledger_buf = ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
                if (memledger_buf) memledger_bmo_alloc_B = ggml_backend_buffer_get_size(memledger_buf);
            }
            // BMO_DRY_ALLOC: custom_tensor->data/buffer stay NULL — fine, this
            // mode only observes the loading loop, never reaches compute.
            // Restore logical shape so ggml graph dispatch sees correct rows/cols.
            // Strides nb[2]/nb[3] are set to cols*rows to keep ggml_is_contiguous() == true.
            custom_tensor->ne[0] = cols;
            custom_tensor->ne[1] = rows;
            custom_tensor->nb[1] = cols;
            custom_tensor->nb[2] = (size_t)cols * rows;
            custom_tensor->nb[3] = (size_t)cols * rows;
        } else {
            custom_tensor->data = malloc(offset); // Allocates exactly offset bytes on CPU instead of cols * rows
        }

        block_bmo_tier header;
        header.rows = rows;
        header.cols = cols;
        header.scale_int8 = scale_int8;
        header.zp_int8 = zp_int8;
        header.scale_int4 = scale_int4;
        header.zp_int4 = zp_int4;
        header.scale_low = scale_low;
        header.zp_low = zp_low;
        header.n_outliers = n_outliers;
        header.padding = 0;

        if (n_tiles_buf.size() >= 4 * sizeof(int32_t)) {
            memcpy(header.n_tiles, n_tiles_buf.data(), 4 * sizeof(int32_t));
        } else {
            memset(header.n_tiles, 0, 4 * sizeof(int32_t));
        }

        if (tier_offsets_buf.size() >= 5 * sizeof(int32_t)) {
            memcpy(header.tier_offsets, tier_offsets_buf.data(), 5 * sizeof(int32_t));
        } else {
            memset(header.tier_offsets, 0, 5 * sizeof(int32_t));
        }

        // Band-major repack prep. tile_stream_indices now hold each tile's
        // POSITION within its band+tier list (the fused GEMV and dequant
        // kernels address the band-major streams by position); the GLOBAL
        // per-tier stream index (original semantics) is still needed below
        // to locate each tile's source slice inside the raw packed bytes.
        if (rows % 64 != 0 || cols % 64 != 0) {
            fprintf(stderr, "FATAL: BMO tensor %s dims (%d x %d) not multiples of 64 — band repack cannot proceed\n",
                    base_name.c_str(), rows, cols);
            exit(1);
        }
        const int32_t n_bands = rows / 64;
        const int32_t n_tiles_col_rp = cols / 64;
        std::vector<uint16_t> tile_stream_indices(tt.size()); // within-band positions
        std::vector<int32_t> tile_global_stream(tt.size());
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

        // Band-major layout selection — MUST match the dispatch rule in
        // ggml/src/ggml-cuda/convert.cu mul_mat_vec_bmo_tier_cuda():
        // tile-major wins when the x vector (cols*4B) is L1-resident.
        const int32_t band_layout = (cols <= 8192) ? 2 : 1;
        const int slice_bytes_by_tier[4] = { 128, 64, 32, 16 }; // fp16, int8, int4, int2

        // Fused-outlier GEMV prep: sort the outlier (index, value) pairs by
        // flat index (== by row, then column; done host-side, once, at load)
        // and build CSR-style per-row ranges. Order of outliers is
        // irrelevant to every existing consumer (the CPU dequant and the
        // atomicAdd correction kernel are both order-independent), so this
        // is safe on its own.
        std::vector<int32_t> outlier_row_starts((size_t)rows + 1, 0);
        if (n_outliers > 0 && !oi.empty() && !ov.empty()) {
            int32_t * oi32 = (int32_t *) oi.data();
            uint16_t * ov16 = (uint16_t *) ov.data();
            std::vector<int32_t> order(n_outliers);
            for (int32_t i = 0; i < n_outliers; ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(),
                             [&](int32_t a, int32_t b) { return oi32[a] < oi32[b]; });
            std::vector<int32_t> oi_sorted(n_outliers);
            std::vector<uint16_t> ov_sorted(n_outliers);
            for (int32_t i = 0; i < n_outliers; ++i) {
                oi_sorted[i] = oi32[order[i]];
                ov_sorted[i] = ov16[order[i]];
            }
            memcpy(oi32, oi_sorted.data(), (size_t)n_outliers * sizeof(int32_t));
            memcpy(ov16, ov_sorted.data(), (size_t)n_outliers * sizeof(uint16_t));
            for (int32_t i = 0; i < n_outliers; ++i) {
                outlier_row_starts[oi_sorted[i] / cols + 1]++;
            }
            for (int32_t r2 = 0; r2 < rows; ++r2) {
                outlier_row_starts[r2 + 1] += outlier_row_starts[r2];
            }
        }

        size_t write_offset = (sizeof(block_bmo_tier) + 15) & ~(size_t)15; // 16-align (see precompute)
        header.packed_weights_offset = write_offset;
        write_offset += pw.size();
        write_offset = (write_offset + 3) & ~3;

        header.tile_tiers_offset = write_offset;
        write_offset += tt.size();
        write_offset = (write_offset + 3) & ~3;

        header.outlier_indices_offset = write_offset;
        write_offset += oi.size();
        write_offset = (write_offset + 3) & ~3;

        header.outlier_values_offset = write_offset;
        write_offset += ov.size();
        write_offset = (write_offset + 15) & ~15;

        header.tile_stream_indices_offset = write_offset;
        write_offset += tile_stream_indices.size() * sizeof(uint16_t);
        write_offset = (write_offset + 15) & ~15;

        header.outlier_row_starts_offset = write_offset;
        write_offset += (size_t)(rows + 1) * sizeof(int32_t);
        write_offset = (write_offset + 15) & ~15;

        header.band_table_offset = write_offset;
        write_offset += ((size_t)n_bands * 4 + 1) * sizeof(int32_t);
        write_offset = (write_offset + 15) & ~15;
        header.band_layout = band_layout;
        header.reserved2 = 0;

        // Band table: absolute byte offsets (from struct start) of each
        // band+tier sub-stream inside the packed region, plus end sentinel.
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

        // Fix B: reuse the member buffer instead of a fresh per-call vector.
        // resize() alone would leave stale bytes from the PREVIOUS tensor's
        // payload in any padding gaps between fields — explicitly zero the
        // needed range every time so the assembled bytes are identical to
        // the original fresh-zero-initialized-vector behavior.
        auto & payload = reuse_buf_payload;
        payload.resize(offset);
        std::fill(payload.begin(), payload.end(), 0);
        memcpy(payload.data(), &header, sizeof(block_bmo_tier));
        // Band-major repack into the packed region: destination slice order
        // is [band][tier][pos][ir] (tile-major, band_layout==2) or
        // [band][tier][ir][pos] (row-minor, ==1); source slices come from
        // the original per-tier global streams in the raw packed bytes.
        for (int32_t b = 0; b < n_bands; ++b) {
            int32_t n_bt[4] = {0, 0, 0, 0};
            for (int32_t tc2 = 0; tc2 < n_tiles_col_rp; ++tc2) {
                n_bt[tt[(size_t)b * n_tiles_col_rp + tc2]]++;
            }
            for (int32_t tc2 = 0; tc2 < n_tiles_col_rp; ++tc2) {
                const size_t t_idx = (size_t)b * n_tiles_col_rp + tc2;
                const int tier = tt[t_idx];
                const int sb = slice_bytes_by_tier[tier];
                const int32_t pos = tile_stream_indices[t_idx];
                const uint8_t * src_tile = pw.data() + header.tier_offsets[tier]
                                         + (size_t)tile_global_stream[t_idx] * 64 * sb;
                uint8_t * dst_base = payload.data() + band_table[(size_t)b * 4 + tier];
                for (int ir = 0; ir < 64; ++ir) {
                    const size_t dst_slice = (band_layout == 2)
                        ? ((size_t)pos * 64 + ir)
                        : ((size_t)ir * n_bt[tier] + pos);
                    memcpy(dst_base + dst_slice * sb, src_tile + (size_t)ir * sb, sb);
                }
            }
        }
        memcpy(payload.data() + header.tile_tiers_offset, tt.data(), tt.size());
        if (!oi.empty()) memcpy(payload.data() + header.outlier_indices_offset, oi.data(), oi.size());
        if (!ov.empty()) memcpy(payload.data() + header.outlier_values_offset, ov.data(), ov.size());
        memcpy(payload.data() + header.tile_stream_indices_offset, tile_stream_indices.data(), tile_stream_indices.size() * sizeof(uint16_t));
        memcpy(payload.data() + header.outlier_row_starts_offset, outlier_row_starts.data(), ((size_t)rows + 1) * sizeof(int32_t));
        memcpy(payload.data() + header.band_table_offset, band_table.data(), band_table.size() * sizeof(int32_t));

        float * deq_cpu = NULL;
        if (!backend) {
            // Perform CPU dequantization once to store CPU floats only if CUDA is not used.
            // NOTE: uses the RAW packed bytes (original per-tier global-stream
            // layout, which dequantize_ffn_cpu expects) — the payload's packed
            // region is band-major after the repack above.
            const uint8_t * p_pw = pw.data();
            const uint8_t * p_tt = payload.data() + header.tile_tiers_offset;
            const int32_t * p_oi = oi.empty() ? NULL : (const int32_t *)(payload.data() + header.outlier_indices_offset);
            const ggml_fp16_t * p_ov = ov.empty() ? NULL : (const ggml_fp16_t *)(payload.data() + header.outlier_values_offset);
            deq_cpu = dequantize_ffn_cpu(header, p_pw, p_tt, p_oi, p_ov);
            cpu_dequantized_ptrs.push_back(deq_cpu);
        }
        
        block_bmo_tier * header_in_payload = (block_bmo_tier *) payload.data();
        header_in_payload->dequantized_cpu_ptr = (int64_t) deq_cpu;

        if (backend) {
            if ( ! bmo_dry_alloc_enabled() ) {
                ggml_backend_tensor_set(custom_tensor, payload.data(), 0, offset);
            }
        } else {
            memcpy(custom_tensor->data, payload.data(), offset);
        }

        // MEMLEDGER: nbytes_B uses ggml_nbytes() as literally specified — note this reads
        // cols*rows (the logical/restored shape), NOT the offset bytes actually resident,
        // per the KNOWN LANDMINE comment above. payload_B is the true physical byte count.
        memledger_log("per_tensor_bmo", base_name.c_str(), ggml_type_name(custom_tensor->type),
                       offset, memledger_bmo_alloc_B, ggml_nbytes(custom_tensor), backend);

        // MEMLEDGER: log bmo_layer_done once both of a layer's gating tensors
        // (in + out) have been repacked — requested as one line per LAYER, not
        // per tensor. Parses "transformer_layers_<N>_gating_linear_..." for N;
        // any name not matching this exact pattern (there shouldn't be any,
        // since is_bmo_big_subcomponent already restricts to this family) is
        // silently skipped rather than crashing.
        {
            const std::string prefix = "transformer_layers_";
            if ( base_name.compare(0, prefix.size(), prefix) == 0 ) {
                size_t digits_start = prefix.size();
                size_t digits_end = base_name.find('_', digits_start);
                if ( digits_end != std::string::npos ) {
                    int layer = atoi( base_name.substr(digits_start, digits_end - digits_start).c_str() );
                    int count = ++memledger_bmo_layer_tensor_count[layer];
                    if ( count >= 2 ) {
                        // Fix B: trim once per layer (inside the loop), not only
                        // once at the very end of moshi_lm_load() — retention
                        // accumulates layer-by-layer, so a single end-of-load
                        // trim cannot prevent the peak from building up during
                        // the loop itself.
#if !defined(_WIN32) && !defined(__APPLE__)
                        malloc_trim(0);
#endif
                        char layer_name[32];
                        snprintf(layer_name, sizeof(layer_name), "layer=%d", layer);
                        memledger_log("bmo_layer_done", layer_name, "-", 0, 0, 0, backend);
                        char layer_label[16];
                        snprintf(layer_label, sizeof(layer_label), "%d", layer);
                        memledger_log_meminfo(layer_label);
                    }
                }
            }
        }

        tensors[base_name] = custom_tensor;
        return custom_tensor;
    }

    ggml_tensor * get_tensor( std::string & name ) {
        if ( ! gguf )
            return NULL;

        std::string orig_name = name;

        // 1. Try exact match of original requested name
        auto it_orig = tensors.find( orig_name );
        if ( it_orig != tensors.end() ) {
            return it_orig->second;
        }

        // 2. Try hash match of original requested name
        std::string hashed_orig = tensor_name( orig_name );
        auto it_hashed_orig = tensors.find( hashed_orig );
        if ( it_hashed_orig != tensors.end() ) {
            name = hashed_orig;
            return it_hashed_orig->second;
        }

        size_t pos_dot;
        while ((pos_dot = name.find(".in_projs.0.weight")) != std::string::npos) {
            name.replace(pos_dot, 18, ".in_proj_weight");
        }
        while ((pos_dot = name.find(".out_projs.0.weight")) != std::string::npos) {
            name.replace(pos_dot, 19, ".out_proj.weight");
        }

        auto it_exact = tensors.find( name );
        if ( it_exact != tensors.end() ) {
            return it_exact->second;
        }

        std::string hashed_name = tensor_name( name );
        auto it_hashed = tensors.find( hashed_name );
        if ( it_hashed != tensors.end() ) {
            name = hashed_name;
            return it_hashed->second;
        }

        std::string lookup_name = name;
        auto [prefix, rest] = split_first(lookup_name, '.');
        if (!rest.empty()) {
            auto it_rest = tensors.find( rest );
            if ( it_rest != tensors.end() ) {
                name = rest;
                return it_rest->second;
            }
        }

        if (!rest.empty() && (prefix == "lm" || prefix == "mimi")) {
            lookup_name = rest;
        }
        
        std::replace(lookup_name.begin(), lookup_name.end(), '.', '_');
        
        // Map self-attention projections from plural to singular
        size_t pos;
        while ((pos = lookup_name.find("_in_projs_0_")) != std::string::npos) {
            lookup_name.replace(pos, 12, "_in_proj_");
        }
        while ((pos = lookup_name.find("_out_projs_0_")) != std::string::npos) {
            lookup_name.replace(pos, 13, "_out_proj_");
        }
        
        auto it = tensors.find( lookup_name );
        if ( it != tensors.end() ) {
            name = lookup_name;
            return it->second;
        }

        // Try replacing _alpha with _weight
        if (lookup_name.size() > 6 && lookup_name.substr(lookup_name.size() - 6) == "_alpha") {
            std::string weight_name = lookup_name.substr(0, lookup_name.size() - 6) + "_weight";
            it = tensors.find( weight_name );
            if ( it != tensors.end() ) {
                name = weight_name;
                return it->second;
            }
        }
        
        std::string name_pw = lookup_name + ".packed_weights";
        // BMO double-storage fix: BMO gating tensors' .packed_weights is deliberately
        // excluded from `tensors[]` (see load_gguf()), so this existence check must
        // use the GGUF file's own tensor index instead — tensors.find() would always
        // return false for BMO tensors otherwise, silently skipping build_custom_ffn_tensor()
        // entirely. gguf_find_tensor() also correctly finds attention's (packing_version==10)
        // .packed_weights, which is untouched by the exclusion — so this single check is
        // correct for both branches below.
        if ( gguf_find_tensor( gguf, name_pw.c_str() ) >= 0 ) {
            int32_t packing_version = read_scalar_i32( lookup_name + ".packing_version", 0 );
            if ( packing_version == 10 ) {
                auto built = build_quantized_attn_tensor( lookup_name );
                if ( built ) {
                    name = lookup_name;
                    return built;
                }
            } else if ( packing_version == 6 ) {
                auto built = build_custom_ffn_tensor( lookup_name );
                if ( built ) {
                    name = lookup_name;
                    return built;
                }
            }
        }
        
        std::string search_name = rest.empty() ? name : rest;
        search_name = tensor_name( search_name );
        it = tensors.find( search_name );
        if ( it != tensors.end() ) {
            name = search_name;
            return it->second;
        }
            
        printf("DEBUG: get_tensor(%s) -> NULL (original requested name: %s)\n", lookup_name.c_str(), name.c_str()); fflush(stdout);
        return NULL;
    }

    bool fetch( ggml_tensor ** result, std::string name, ggml_type dst_type, int offset = 0 ) {
        if ( gguf ) {
            *result = get_tensor( name );
            return *result? true : false;
        }
        safetensor_t * safetensor =  find( name );
        *result = NULL;
        if (!safetensor)
            return false;
        // get source info
        ggml_type src_type = safetensor_get_type( safetensor->dtype );
        NE ne;
        int n_dims = safetensor_get_shape(safetensor, ne, offset);
        if ( dst_type == GGML_TYPE_Q2_K && ne[0] % 256 ) {
            dst_type = GGML_TYPE_Q4_0;
        }
        if ( dst_type == GGML_TYPE_Q4_K && ne[0] % 256 ) {
            dst_type = GGML_TYPE_Q4_0;
        }
        if ( dst_type == GGML_TYPE_Q4_0 && ne[0] % 32 ) {
            dst_type = src_type;
        }
        if ( dst_type == GGML_TYPE_Q8_K && ne[0] % 256 ) {
            dst_type = GGML_TYPE_Q8_0;
        }
        if ( dst_type == GGML_TYPE_Q8_0 && ne[0] % 32 ) {
            dst_type = src_type;
        }
        add_alloc( result, n_dims, ne, dst_type, name );
        if (dst_type == src_type) {
            add_init([ safetensor, result ]( WeightLoader * loader ) {
                loader->init( safetensor, *result );
            } );
        } else {
            add_init([ safetensor, result ]( WeightLoader * loader ) {
                if ((*result)->type == GGML_TYPE_Q2_K) {
                    // Create a local CPU ggml_context
                    ggml_init_params params;
                    params.mem_size = (size_t)1024 * 1024 * 1024;
                    params.mem_buffer = NULL;
                    params.no_alloc = false;
                    ggml_context * cpu_ctx = ggml_init(params);
                    assert(cpu_ctx);

                    NE ne;
                    int n_dims = safetensor_get_shape(safetensor, ne, 0);
                    ggml_tensor * original_cpu = ggml_new_tensor(cpu_ctx, GGML_TYPE_F32, n_dims, ne);
                    assert(original_cpu);

                    int64_t size = safetensor->data_offsets[1] - safetensor->data_offsets[0];
                    std::vector<char> raw_data(size);
                    int64_t offset = safetensor->data_offsets[0] + loader->stf->header_length;
#ifdef _WIN32
                    _fseeki64(loader->stf->f, offset, SEEK_SET);
#else
                    fseek(loader->stf->f, offset, SEEK_SET);
#endif
                    fread(raw_data.data(), size, 1, loader->stf->f);

                    ggml_type src_type = safetensor_get_type(safetensor->dtype);
                    if (src_type == GGML_TYPE_BF16) {
                        int64_t nelements = ggml_nelements(original_cpu);
                        const uint16_t * src_bf16 = (const uint16_t *) raw_data.data();
                        float * dst_f32 = (float *) original_cpu->data;
                        for (int64_t k = 0; k < nelements; k++) {
                            uint32_t val = ((uint32_t)src_bf16[k]) << 16;
                            dst_f32[k] = *(float*)&val;
                        }
                    } else if (src_type == GGML_TYPE_F16) {
                        int64_t nelements = ggml_nelements(original_cpu);
                        const uint16_t * src_f16 = (const uint16_t *) raw_data.data();
                        float * dst_f32 = (float *) original_cpu->data;
                        for (int64_t k = 0; k < nelements; k++) {
                            dst_f32[k] = ggml_fp16_to_fp32(src_f16[k]);
                        }
                    } else if (src_type == GGML_TYPE_F32) {
                        memcpy(original_cpu->data, raw_data.data(), size);
                    } else {
                        printf("Unsupported safetensor weight type %s\n", ggml_type_name(src_type));
                        exit(-1);
                    }

                    // Apply WHT pre-rotation (Walsh-Hadamard Transform) on F32 CPU tensor
                    int64_t row_size = original_cpu->ne[0];
                    int64_t num_rows = ggml_nrows(original_cpu);
                    if (row_size % 128 == 0) {
                        float * data = (float *) original_cpu->data;
                        for (int64_t r = 0; r < num_rows; r++) {
                            float * row = data + r * row_size;
                            for (int64_t c = 0; c < row_size; c += 128) {
                                int h = 1;
                                while (h < 128) {
                                    for (int i = 0; i < 128; i += 2 * h) {
                                        for (int j = i; j < i + h; j++) {
                                            float a = row[c + j];
                                            float b = row[c + j + h];
                                            row[c + j] = a + b;
                                            row[c + j + h] = a - b;
                                        }
                                    }
                                    h *= 2;
                                }
                                for (int i = 0; i < 128; i++) {
                                    row[c + i] *= 0.0883883476f;
                                }
                            }
                        }
                    }

                    // Quantize the rotated F32 tensor to Q2_K on the CPU
                    int64_t quantized_size = ggml_row_size(GGML_TYPE_Q2_K, row_size) * num_rows;
                    std::vector<char> quantized_data(quantized_size);
                    
                    quantize_wht_nf2((const float *) original_cpu->data, quantized_data.data(), num_rows, row_size);

                    // Copy the quantized data directly to the GPU tensor *result
                    ggml_backend_tensor_set(*result, quantized_data.data(), 0, quantized_size);

                    ggml_free(cpu_ctx);
                } else {
                    auto & scratch_ctx = *loader->scratch;
                    auto original = scratch_ctx.load( loader->stf, safetensor );
                    auto cast = ggml_cast( scratch_ctx, original, (*result)->type );
                    scratch_ctx.build_forward_expand( cast, *result );
                    scratch_ctx.compute();
                }
            } );
        }
        return true;
    }

    bool fetch( ggml_tensor ** result, std::string name, void *func = NULL, int offset = 0 ) {
        if ( gguf ) {
            *result = get_tensor( name );
            return *result? true : false;
        }
        safetensor_t * safetensor =  find( name );
        *result = NULL;
        if (!safetensor)
            return false;
        // get source info
        ggml_type src_type = safetensor_get_type( safetensor->dtype );
        NE ne;
        int n_dims = safetensor_get_shape(safetensor, ne, offset);
        // get destination type
        ggml_type dst_type = src_type;
        if (func == ggml_mul) dst_type = GGML_TYPE_F32;
        else if (func == ggml_add) dst_type = GGML_TYPE_F32;
        else if (func == ggml_rms_norm) dst_type = GGML_TYPE_F32;
        else if (func == ggml_conv_1d) dst_type = GGML_TYPE_F16;
        add_alloc( result, n_dims, ne, dst_type, name );
        if (dst_type == src_type) {
            add_init([ safetensor, result ]( WeightLoader * loader ) {
                loader->init( safetensor, *result );
            } );
        } else {
            add_init([ safetensor, result ]( WeightLoader * loader ) {
                if ((*result)->type == GGML_TYPE_Q2_K) {
                    // Create a local CPU ggml_context
                    ggml_init_params params;
                    params.mem_size = (size_t)1024 * 1024 * 1024;
                    params.mem_buffer = NULL;
                    params.no_alloc = false;
                    ggml_context * cpu_ctx = ggml_init(params);
                    assert(cpu_ctx);

                    NE ne;
                    int n_dims = safetensor_get_shape(safetensor, ne, 0);
                    ggml_tensor * original_cpu = ggml_new_tensor(cpu_ctx, GGML_TYPE_F32, n_dims, ne);
                    assert(original_cpu);

                    int64_t size = safetensor->data_offsets[1] - safetensor->data_offsets[0];
                    std::vector<char> raw_data(size);
                    int64_t offset = safetensor->data_offsets[0] + loader->stf->header_length;
#ifdef _WIN32
                    _fseeki64(loader->stf->f, offset, SEEK_SET);
#else
                    fseek(loader->stf->f, offset, SEEK_SET);
#endif
                    fread(raw_data.data(), size, 1, loader->stf->f);

                    ggml_type src_type = safetensor_get_type(safetensor->dtype);
                    if (src_type == GGML_TYPE_BF16) {
                        int64_t nelements = ggml_nelements(original_cpu);
                        const uint16_t * src_bf16 = (const uint16_t *) raw_data.data();
                        float * dst_f32 = (float *) original_cpu->data;
                        for (int64_t k = 0; k < nelements; k++) {
                            uint32_t val = ((uint32_t)src_bf16[k]) << 16;
                            dst_f32[k] = *(float*)&val;
                        }
                    } else if (src_type == GGML_TYPE_F16) {
                        int64_t nelements = ggml_nelements(original_cpu);
                        const uint16_t * src_f16 = (const uint16_t *) raw_data.data();
                        float * dst_f32 = (float *) original_cpu->data;
                        for (int64_t k = 0; k < nelements; k++) {
                            dst_f32[k] = ggml_fp16_to_fp32(src_f16[k]);
                        }
                    } else if (src_type == GGML_TYPE_F32) {
                        memcpy(original_cpu->data, raw_data.data(), size);
                    } else {
                        printf("Unsupported safetensor weight type %s\n", ggml_type_name(src_type));
                        exit(-1);
                    }

                    // Apply WHT pre-rotation (Walsh-Hadamard Transform) on F32 CPU tensor
                    int64_t row_size = original_cpu->ne[0];
                    int64_t num_rows = ggml_nrows(original_cpu);
                    if (row_size % 128 == 0) {
                        float * data = (float *) original_cpu->data;
                        for (int64_t r = 0; r < num_rows; r++) {
                            float * row = data + r * row_size;
                            for (int64_t c = 0; c < row_size; c += 128) {
                                int h = 1;
                                while (h < 128) {
                                    for (int i = 0; i < 128; i += 2 * h) {
                                        for (int j = i; j < i + h; j++) {
                                            float a = row[c + j];
                                            float b = row[c + j + h];
                                            row[c + j] = a + b;
                                            row[c + j + h] = a - b;
                                        }
                                    }
                                    h *= 2;
                                }
                                for (int i = 0; i < 128; i++) {
                                    row[c + i] *= 0.0883883476f;
                                }
                            }
                        }
                    }

                    // Quantize the rotated F32 tensor to Q2_K on the CPU
                    int64_t quantized_size = ggml_row_size(GGML_TYPE_Q2_K, row_size) * num_rows;
                    std::vector<char> quantized_data(quantized_size);
                    
                    quantize_wht_nf2((const float *) original_cpu->data, quantized_data.data(), num_rows, row_size);

                    // Copy the quantized data directly to the GPU tensor *result
                    ggml_backend_tensor_set(*result, quantized_data.data(), 0, quantized_size);

                    ggml_free(cpu_ctx);
                } else {
                    auto & scratch_ctx = *loader->scratch;
                    auto original = scratch_ctx.load( loader->stf, safetensor );
                    auto cast = ggml_cast( scratch_ctx, original, (*result)->type );
                    scratch_ctx.build_forward_expand( cast, *result );
                    scratch_ctx.compute();
                }
            } );
        }
        return true;
    }

    void save_gguf( const char * filename ) {
        auto gguf = gguf_init_empty();
        for ( auto tensor = ggml_get_first_tensor( ctx ); tensor;
                   tensor = ggml_get_next_tensor( ctx, tensor ) )
            gguf_add_tensor( gguf, tensor );
        gguf_write_to_file( gguf, filename, false );
    }

    bool load_gguf() {

        assert( backend );

        // BMO double-storage fix: pre-mark the 4 large BMO sub-component tensors
        // (per layer, layers 0-30 only) with a non-NULL sentinel `data` pointer.
        // ggml_backend_alloc_ctx_tensors's internal allocator (ggml-alloc.c) skips
        // any tensor where `t->data != NULL`, so this excludes them from the shared
        // buffer's size and backing memory entirely — their bytes are never uploaded
        // to the device at all; build_custom_ffn_tensor() reads them straight from
        // the file instead (read_raw_bytes_from_gguf_file). The sentinel value is
        // never dereferenced — nothing below touches these tensors' ->data again.
        int n_tensors_prescan = (int) gguf_get_n_tensors( gguf );
        int n_bmo_excluded = 0;
        for (int i = 0; i < n_tensors_prescan; i++) {
            std::string name = gguf_get_tensor_name( gguf, i );
            if ( is_bmo_big_subcomponent( name ) ) {
                auto tensor = ggml_get_tensor( ctx, name.c_str() );
                assert( tensor );
                tensor->data = (void*)(intptr_t)1;
                n_bmo_excluded++;
            }
        }
        memledger_log("bmo_subcomponents_excluded", "(prescan_done)", "-", 0, (size_t)n_bmo_excluded, 0, backend);

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // MEMLEDGER: this is ONE shared buffer sized for every non-BMO tensor in ctx
        // (now correctly excluding the large BMO sub-components marked above) —
        // contrast with build_custom_ffn_tensor(), which allocates a separate buffer
        // PER BMO tensor. alloc_B here is the total shared-buffer size, not per-tensor.
        memledger_log("after_gguf_buffer_alloc", "(shared_ctx_buffer)", "-",
                       0, buffer ? ggml_backend_buffer_get_size(buffer) : 0, 0, backend);

        auto f = fopen( filename.c_str(), "rb" );

        std::vector<char> data;
        auto data_offset = gguf_get_data_offset( gguf );
        int n_tensors = (int) gguf_get_n_tensors( gguf );
        for (int i = 0; i < n_tensors; i++) {
            std::string name = gguf_get_tensor_name( gguf, i );

            if ( is_bmo_big_subcomponent( name ) ) {
                // Excluded above — no device tensor to upload to, and deliberately
                // not added to `tensors[]` so any accidental lookup elsewhere fails
                // loudly (empty/NULL) instead of silently reading the sentinel pointer.
                memledger_log("bmo_subcomponent_skipped", name.c_str(), "-",
                               (size_t)gguf_get_tensor_size(gguf, i), 0, 0, backend);
                continue;
            }

            auto tensor      = ggml_get_tensor( ctx, name.c_str() );
            auto offset      = data_offset + gguf_get_tensor_offset( gguf, i );
            auto nbytes      = gguf_get_tensor_size( gguf, i );

            if ( data.size() < nbytes ) data.resize( nbytes );
#ifdef _WIN32
            auto e = _fseeki64(f, offset, SEEK_SET);
#else
            auto e = fseek(f, offset, SEEK_SET);
#endif
            assert( e == 0 );
            int64_t r = fread(data.data(), nbytes, 1, f);
            if (r != 1) {
                printf("failed to read tensor %s\n", name.c_str());
                exit(-1);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);

            tensors[name] = tensor;

            // MEMLEDGER: alloc_B=0 here is deliberate, not a bug — this tensor lives inside
            // the ONE shared buffer already accounted for in the "after_gguf_buffer_alloc"
            // milestone line; there is no separate per-tensor buffer to query for this path.
            memledger_log("per_tensor_regular", name.c_str(), ggml_type_name(tensor->type),
                           (size_t)nbytes, 0, ggml_nbytes(tensor), backend);
        }

        fclose( f );

        memledger_log("after_load_gguf_raw", "(all_raw_tensors_done)", "-", 0, 0, 0, backend);

        return true;
    }

    void alloc() {
        assert( ctx == NULL );
        size_t nbytes = ggml_tensor_overhead() * alloc_requests.size();
        if (backend) {
            ctx = ggml_init({ nbytes, NULL, true });
        } else {
            for (auto req : alloc_requests) {
                int64_t ne = req.ne[0] * req.ne[1] * req.ne[2] * req.ne[3];
                nbytes += ggml_row_size(req.type, ne);
            }
            ctx = ggml_init({ nbytes, NULL, false });
        }
        for (auto req : alloc_requests) {
            *req.result = ggml_new_tensor( ctx, req.type, req.n_dims, req.ne );
            auto name_size = req.name.size();
            assert( name_size );
            if ( name_size ) {
                ggml_set_name( *req.result, tensor_name( req.name ).c_str() );
            }
        }
        alloc_requests.clear();
        if (backend)
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    void init() {
        assert( ctx );
        assert( backend == NULL || buffer != NULL);
        for (auto req : init_requests) {
            req( this );
        }
        init_requests.clear();
    }

    void load() {
        alloc();
        init();
    }
};


