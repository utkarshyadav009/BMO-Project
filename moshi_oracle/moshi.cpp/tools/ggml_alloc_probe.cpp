// STEP 1 certification probe for the "ggml_backend_alloc_ctx_tensors
// per-call overhead" hypothesis raised by RUN 1's nvmap_client_KiB
// attribution (see RESUME_NOTES.md): the standalone raw-cudaMalloc probe
// (nvmap_alloc_probe.cu) showed ~0 per-call overhead, but the real loader
// never calls raw cudaMalloc — it calls ggml_backend_alloc_ctx_tensors()
// once per BMO tensor (62x) on an incrementally growing ggml_context. This
// probe replicates THAT exact mechanism (not raw cudaMalloc) to test
// whether the overhead lives in ggml's own allocator wrapper.
//
// PATTERN A: one ggml_context (no_alloc=true). Loop 62x: create one 1D
//   GGML_TYPE_I8 tensor at the real alternating BMO sizes, then call
//   ggml_backend_alloc_ctx_tensors(ctx, backend) immediately — mirrors
//   loader.h's build_custom_ffn_tensor() calling it once per tensor on the
//   same growing custom_ctx.
// PATTERN B: fresh context, all 62 tensors created first, ONE
//   ggml_backend_alloc_ctx_tensors call — mirrors StateContext::alloc()'s
//   bulk pattern (the after_kv_cache case that showed zero overhead).
//
// CERTIFIED = A overhead >= 900 MiB AND B overhead < 100 MiB.
//
// Root required (reads /sys/kernel/debug/nvmap/iovmm/clients). Run twice
// for reproducibility, per task spec.

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <vector>

static size_t nvmap_client_kib() {
    FILE * f = fopen( "/sys/kernel/debug/nvmap/iovmm/clients", "r" );
    if ( ! f ) return 0;
    char line[256];
    size_t total_kib = 0;
    while ( fgets( line, sizeof(line), f ) ) {
        char * p = line;
        while ( *p == ' ' ) p++;
        if ( strncmp( p, "total", 5 ) == 0 ) {
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

static bool nvmap_readable() {
    FILE * f = fopen( "/sys/kernel/debug/nvmap/iovmm/clients", "r" );
    if ( ! f ) return false;
    fclose( f );
    return true;
}

// Real BMO tensor sizes (alternating), from loader.h's build_custom_ffn_tensor
// payload byte counts observed in prior sessions: gating_linear_in / _out.
static const size_t SIZE_A = 39279104;
static const size_t SIZE_B = 19639296;
static const int N_TENSORS = 62;

static size_t total_requested() {
    size_t total = 0;
    for ( int i = 0; i < N_TENSORS; i++ )
        total += (i % 2 == 0) ? SIZE_A : SIZE_B;
    return total;
}

// PATTERN A: mirrors loader.h exactly — one growing context, one
// ggml_backend_alloc_ctx_tensors call PER tensor.
static size_t run_pattern_a( ggml_backend * backend, size_t nvmap_before_kib ) {
    ggml_init_params params;
    params.mem_size = 16 * 1024 * 1024; // headroom for 62 tensor structs' metadata
    params.mem_buffer = NULL;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);

    size_t last_kib = nvmap_before_kib;
    for ( int i = 0; i < N_TENSORS; i++ ) {
        size_t nbytes = (i % 2 == 0) ? SIZE_A : SIZE_B;
        ggml_tensor * t = ggml_new_tensor_1d( ctx, GGML_TYPE_I8, (int64_t) nbytes );
        char name[64];
        snprintf( name, sizeof(name), "probe_a_%02d", i );
        ggml_set_name( t, name );

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors( ctx, backend );
        size_t alloc_B = buf ? ggml_backend_buffer_get_size(buf) : 0;

        size_t kib = nvmap_client_kib();
        fprintf( stdout, "PATTERN_A call=%2d requested_B=%10zu backend_alloc_B=%10zu nvmap_client_KiB=%9zu d_nvmap_since_prev_KiB=%9zd\n",
                 i, nbytes, alloc_B, kib, (ssize_t)kib - (ssize_t)last_kib );
        fflush(stdout);
        last_kib = kib;
    }

    size_t nvmap_after_kib = nvmap_client_kib();
    ggml_free( ctx ); // frees backend buffers too (ggml_context owns them via ggml_backend_alloc_ctx_tensors bookkeeping)
    return nvmap_after_kib;
}

// PATTERN B: mirrors StateContext::alloc()'s bulk pattern — all tensors
// created first in one context, ONE ggml_backend_alloc_ctx_tensors call.
static size_t run_pattern_b( ggml_backend * backend, size_t nvmap_before_kib ) {
    ggml_init_params params;
    params.mem_size = 16 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);

    for ( int i = 0; i < N_TENSORS; i++ ) {
        size_t nbytes = (i % 2 == 0) ? SIZE_A : SIZE_B;
        ggml_tensor * t = ggml_new_tensor_1d( ctx, GGML_TYPE_I8, (int64_t) nbytes );
        char name[64];
        snprintf( name, sizeof(name), "probe_b_%02d", i );
        ggml_set_name( t, name );
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors( ctx, backend );
    size_t alloc_B = buf ? ggml_backend_buffer_get_size(buf) : 0;
    size_t nvmap_after_kib = nvmap_client_kib();
    fprintf( stdout, "PATTERN_B call=single requested_B=%10zu backend_alloc_B=%10zu nvmap_client_KiB=%9zu d_nvmap_since_prev_KiB=%9zd\n",
             total_requested(), alloc_B, nvmap_after_kib, (ssize_t)nvmap_after_kib - (ssize_t)nvmap_before_kib );
    fflush(stdout);

    ggml_free( ctx );
    return nvmap_after_kib;
}

int main() {
    if ( ! nvmap_readable() ) {
        fprintf( stderr, "error: /sys/kernel/debug/nvmap/iovmm/clients not readable — run as root.\n" );
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend * backend = ggml_backend_init_best();
    if ( ! backend ) {
        fprintf( stderr, "error: failed to init backend.\n" );
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device( backend );
    fprintf( stdout, "using device: %s\n", ggml_backend_dev_name(dev) );

    size_t requested_total_B = total_requested();
    fprintf( stdout, "N_TENSORS=%d requested_total_B=%zu requested_total_MiB=%.2f\n",
             N_TENSORS, requested_total_B, requested_total_B / 1024.0 / 1024.0 );

    // --- PATTERN A ---
    size_t nvmap_start_a = nvmap_client_kib();
    fprintf( stdout, "\n=== PATTERN A (62 separate ggml_backend_alloc_ctx_tensors calls) ===\n" );
    fprintf( stdout, "nvmap_client_KiB_before=%zu\n", nvmap_start_a );
    size_t nvmap_end_a = run_pattern_a( backend, nvmap_start_a );
    double delta_a_mib = ( (double)nvmap_end_a - (double)nvmap_start_a ) / 1024.0;
    double overhead_a_mib = delta_a_mib - ( requested_total_B / 1024.0 / 1024.0 );
    fprintf( stdout, "nvmap_client_KiB_after=%zu  d_nvmap_MiB=%.2f  requested_MiB=%.2f  OVERHEAD_A_MiB=%.2f\n",
             nvmap_end_a, delta_a_mib, requested_total_B / 1024.0 / 1024.0, overhead_a_mib );

    // --- PATTERN B ---
    size_t nvmap_start_b = nvmap_client_kib();
    fprintf( stdout, "\n=== PATTERN B (1 bulk ggml_backend_alloc_ctx_tensors call) ===\n" );
    fprintf( stdout, "nvmap_client_KiB_before=%zu\n", nvmap_start_b );
    size_t nvmap_end_b = run_pattern_b( backend, nvmap_start_b );
    double delta_b_mib = ( (double)nvmap_end_b - (double)nvmap_start_b ) / 1024.0;
    double overhead_b_mib = delta_b_mib - ( requested_total_B / 1024.0 / 1024.0 );
    fprintf( stdout, "nvmap_client_KiB_after=%zu  d_nvmap_MiB=%.2f  requested_MiB=%.2f  OVERHEAD_B_MiB=%.2f\n",
             nvmap_end_b, delta_b_mib, requested_total_B / 1024.0 / 1024.0, overhead_b_mib );

    fprintf( stdout, "\n=== VERDICT ===\n" );
    bool certified = ( overhead_a_mib >= 900.0 ) && ( overhead_b_mib < 100.0 );
    fprintf( stdout, "OVERHEAD_A_MiB=%.2f (need >= 900)  OVERHEAD_B_MiB=%.2f (need < 100)  CERTIFIED=%s\n",
             overhead_a_mib, overhead_b_mib, certified ? "YES" : "NO" );

    return 0;
}
