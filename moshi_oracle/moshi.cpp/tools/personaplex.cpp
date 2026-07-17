#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <ctype.h> // isdigit (nvmap client-size parsing, MEMLEDGER)
#include <inttypes.h> // PRIx64 (token-hash correctness gate)
#include <chrono>  // STEP 1 frame-phase timing instrumentation
#include <vector>
#include <algorithm>
// PIPELINE (mimi off the critical path): worker threads + bounded queues
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <functional>

#include "common_ggml.h"
#include <moshi/moshi.h>
#include "common_av.h"
#include "common_sdl.h"
#include "common_utils.h"

static void print_usage(const char * program) {
    fprintf( stderr, R"(usage: %s [option(s)]

uses sdl to listen and respond to audio i/o.

options:
  -h,       --help             shows this help message

  -l,       --list-devices     list hardware and exits.
  -d NAME,  --device NAME      use named hardware.
            --threads N        number of CPU threads.

  -r PATH,  --model-root PATH  path to where all models are stored and replaces
                               MODEL_CACHE environment variable. the models at
                               root are in subdirectories of 'publisher/model'
  -m PATH,  --model PATH       path to where model is, can be relative to the
                               MODEL_CACHE environment variable, or program
                               directory, or working directory. by default is
                               'Codes4Fun/personaplex-7b-v1-q4_k-GGUF'
  -q QUANT, --quantize QUANT   convert weights to: q8_0, q4_0, q4_k
  -g,       --gguf-caching     loads gguf if exists, saves gguf if it does not.
  -k TYPE,  --kv-type TYPE     quantization type of the KV cache (bf16, f16, q8_0, q4_0, q2_k)
                               model is saved alongside the original
                               safetensors file.

  -c N,     --context N        default: auto adjusted to device memory with a
                               max of 3000. lowering values reduces vram usage
                               but reduces effective conversation time.
  -s N,     --seed N           seed value.
  -t N,     --temperature N    consistency vs creativity, default 0.8
  -b        --bench            benchmark mode that disables sdl io and ends
                               after a few seconds.
  -i FNAME                     talk to moshi from an audio file.
            --delay            delay the audio file in frames (12.5 fps)

personaplex options:
  -p PROMPT --prompt PROMPT    system prompt
  -v NAME,  --voice NAME       either a filepath to an audio file, a saved
                               embedding as safetensors, gguf, or of defaults:
                                    NATF0
                                    NATF1
                                    NATF2
                                    NATF3
                                    NATM0
                                    NATM1
                                    NATM2
                                    NATM3
                                    VARF0
                                    VARF1
                                    VARF2
                                    VARF3
                                    VARF4
                                    VARM0
                                    VARM1
                                    VARM2
                                    VARM3
                                    VARM4

)", program);
    exit(1);
}

bool shutdown = false;
int64_t lm_delta_time = 0;
int64_t lm_frames = 0;

void log_metrics() {
    printf( "\n\nrun frames: %d\n", (int)lm_frames );
    printf( "run time: %.3f s\n", lm_delta_time / 1000000.f );
    printf( "\nframe rate:  %f frames/s\n", lm_frames * 1000000.f / lm_delta_time );
}

#include <signal.h>
void signal_handler(int dummy) {
    shutdown = true;
}

// MEMLEDGER instrumentation — measurement-only diagnostic for Stage 2 OOM investigation.
static size_t memledger_status_field_kib( const char * field ) {
    FILE * f = fopen( "/proc/self/status", "r" );
    if ( ! f ) return 0;
    size_t field_len = strlen(field);
    size_t val_kb = 0;
    char line[256];
    while ( fgets( line, sizeof(line), f ) ) {
        if ( strncmp( line, field, field_len ) == 0 && line[field_len] == ':' ) {
            sscanf( line + field_len + 1, "%zu", &val_kb );
            break;
        }
    }
    fclose( f );
    return val_kb;
}
static size_t memledger_rss_mib() {
    return memledger_status_field_kib("VmRSS") / 1024;
}
static size_t memledger_smaps_rollup_field_kib( const char * field ) {
    FILE * f = fopen( "/proc/self/smaps_rollup", "r" );
    if ( ! f ) return 0;
    size_t field_len = strlen(field);
    size_t val_kb = 0;
    char line[256];
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
    size_t field_len = strlen(field);
    size_t val_kb = 0;
    char line[256];
    while ( fgets( line, sizeof(line), f ) ) {
        if ( strncmp( line, field, field_len ) == 0 && line[field_len] == ':' ) {
            sscanf( line + field_len + 1, "%zu", &val_kb );
            break;
        }
    }
    fclose( f );
    return val_kb;
}
// /sys/kernel/debug/nvmap/iovmm/clients "total ... <N>K" — root-only.
static bool memledger_nvmap_readable() {
    FILE * f = fopen( "/sys/kernel/debug/nvmap/iovmm/clients", "r" );
    if ( ! f ) return false;
    fclose( f );
    return true;
}
static size_t memledger_nvmap_client_kib() {
    FILE * f = fopen( "/sys/kernel/debug/nvmap/iovmm/clients", "r" );
    if ( ! f ) return 0;
    size_t total_kib = 0;
    char line[256];
    while ( fgets( line, sizeof(line), f ) ) {
        char * p = line;
        while ( *p == ' ' ) p++;
        if ( strncmp( p, "total", 5 ) == 0 ) {
            char * end = line + strlen(line);
            while ( end > line && (*(end-1) == '\n' || *(end-1) == '\r') ) end--;
            if ( end > line && *(end - 1) == 'K' ) {
                char * digits_start = end - 1;
                while ( digits_start > line && isdigit((unsigned char)*(digits_start - 1)) ) digits_start--;
                total_kib = (size_t) atoll( digits_start );
            }
            break;
        }
    }
    fclose( f );
    return total_kib;
}
static void memledger_log_simple( const char * event, size_t cuda_free_mib ) {
    size_t nvmap_kib   = memledger_nvmap_client_kib();
    bool   nvmap_ok    = memledger_nvmap_readable();
    size_t smaps_rss   = memledger_smaps_rollup_field_kib( "Rss" );
    size_t smaps_anon  = memledger_smaps_rollup_field_kib( "Anonymous" );
    size_t status_file = memledger_status_field_kib( "RssFile" );
    size_t status_shm  = memledger_status_field_kib( "RssShmem" );
    size_t mem_avail   = memledger_meminfo_field_kib( "MemAvailable" );
    size_t mem_cached  = memledger_meminfo_field_kib( "Cached" );
    fprintf( stderr,
        "MEMLEDGER event=%-20s name=%-60s type=%-10s payload_B=%10d alloc_B=%10d nbytes_B=%10d "
        "UNRELIABLE_REF_cuda_free_MiB=%7zu rss_MiB=%7zu "
        "nvmap_client_KiB=%s%9zu smaps_Rss_KiB=%9zu smaps_Anonymous_KiB=%9zu status_RssFile_KiB=%9zu status_RssShmem_KiB=%9zu "
        "meminfo_MemAvailable_KiB=%9zu meminfo_Cached_KiB=%9zu\n",
        event, "-", "-", 0, 0, 0, cuda_free_mib, memledger_rss_mib(),
        nvmap_ok ? " " : "U", nvmap_kib, smaps_rss, smaps_anon, status_file, status_shm,
        mem_avail, mem_cached );
    fflush( stderr );
}

// STEP 1 (frame-phase timing task): /proc/self/stat field 12 = majflt
// (major page faults), cumulative since process start. comm (field 2) can
// contain spaces/parens, so parse from the last ')' rather than splitting
// the whole line — fields after ')' start at absolute field 3 (state).
static long read_majflt() {
    FILE * f = fopen( "/proc/self/stat", "r" );
    if ( ! f ) return -1;
    char buf[4096];
    size_t n = fread( buf, 1, sizeof(buf) - 1, f );
    fclose( f );
    if ( n == 0 ) return -1;
    buf[n] = 0;
    char * p = strrchr( buf, ')' );
    if ( ! p ) return -1;
    p++;
    int field = 0; // relative to p; absolute field 12 (majflt) = relative field 10
    long majflt = -1;
    char * tok = strtok( p, " " );
    while ( tok ) {
        field++;
        if ( field == 10 ) { majflt = atol(tok); break; }
        tok = strtok( NULL, " " );
    }
    return majflt;
}

static double median_ms( std::vector<double> & v ) {
    if ( v.empty() ) return 0.0;
    std::sort( v.begin(), v.end() );
    size_t n = v.size();
    return (n % 2) ? v[n/2] : (v[n/2 - 1] + v[n/2]) / 2.0;
}

// PIPELINE: bounded FIFO between the frame loop and the codec workers.
// push() blocks when full, pop() blocks when empty; both record how often
// and for how long they waited (the frame loop is the only pusher of dec_in
// and the only popper of enc_out, so those counters are exactly the
// critical-path stalls). close() wakes everyone; pop() keeps returning
// queued items after close (drain) and returns false only when closed+empty.
template<typename T>
struct BoundedQueue {
    std::mutex m;
    std::condition_variable cv_push, cv_pop;
    std::deque<T> q;
    size_t cap;
    bool closed = false;
    long push_waits = 0, pop_waits = 0;
    double push_wait_ms = 0.0, pop_wait_ms = 0.0;
    explicit BoundedQueue( size_t cap ) : cap(cap) {}
    bool push( T && v ) {
        std::unique_lock<std::mutex> lk(m);
        if ( q.size() >= cap && ! closed ) {
            auto t0 = std::chrono::steady_clock::now();
            cv_push.wait( lk, [&]{ return q.size() < cap || closed; } );
            push_waits++;
            push_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0 ).count();
        }
        if ( closed ) return false;
        q.push_back( std::move(v) );
        cv_pop.notify_one();
        return true;
    }
    bool pop( T & out ) {
        std::unique_lock<std::mutex> lk(m);
        if ( q.empty() && ! closed ) {
            auto t0 = std::chrono::steady_clock::now();
            cv_pop.wait( lk, [&]{ return ! q.empty() || closed; } );
            pop_waits++;
            pop_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0 ).count();
        }
        if ( q.empty() ) return false; // closed and fully drained
        out = std::move( q.front() );
        q.pop_front();
        cv_push.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lk(m);
        closed = true;
        cv_push.notify_all();
        cv_pop.notify_all();
    }
};

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    memledger_log_simple("process_start", 0); // no backend yet — cuda_free_MiB is N/A (0)

    const char * device = NULL;
    int n_threads = 0;

    const char * model_cache = getenv("MODEL_CACHE");
    std::string model_root = model_cache? model_cache : "";
    std::string model_path = "Codes4Fun/personaplex-7b-v1-q4_k-GGUF/";
    bool model_path_set = false;
    const char * quant = NULL;
    bool gguf_caching = false;

    int context = -1;
    int seed = (int)time(NULL);
    float depth_temperature = 0.8f;
    float text_temperature = 0.7f;
    bool bench = false;

    const char * input = NULL;
    int input_delay = 0;

    std::string personaplex_voice_filepath = "";
    std::string personaplex_system_prompt = "";
    std::string kv_type_str = "bf16";

    //////////////////////
    // MARK: Parse Args
    //////////////////////

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        }
        if (arg == "-l" || arg == "--list-devices") {
            list_devices();
        }
        if (arg == "-d" || arg == "--device") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires name of device\n", argv[i] );
                exit(1);
            }
            device = argv[++i];
            continue;
        }
        if (arg == "--threads") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            n_threads = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-r" || arg == "--model-root") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires path to models\n", argv[i] );
                exit(1);
            }
            model_root = argv[++i];
            continue;
        }
        if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to model\n", argv[i] );
                exit(1);
            }
            model_path = argv[++i];
            model_path_set = true;
            continue;
        }
        if (arg == "-q" || arg == "--quantize") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires type\n", argv[i] );
                exit(1);
            }
            quant = argv[++i];
            continue;
        }
        if (arg == "-g" || arg == "--gguf-caching" ) {
            gguf_caching = true;
            continue;
        }
        if (arg == "-k" || arg == "--kv-type") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires type (bf16, f16, q8_0, q4_0)\n", argv[i] );
                exit(1);
            }
            kv_type_str = argv[++i];
            continue;
        }
        if (arg == "-c" || arg == "--context") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            context = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-s" || arg == "--seed") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            seed = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-t" || arg == "--temperature") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            text_temperature = (float) std::stod(argv[++i]);
            depth_temperature = text_temperature;
            continue;
        }
        if (arg == "-i") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to audio file\n", argv[i] );
                exit(1);
            }
            input = argv[++i];
            continue;
        }
        if (arg == "--delay") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            input_delay = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-b" || arg == "--bench")  {
            bench = true;
            continue;
        }
        if (arg == "-v" || arg == "--voice") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" must be followed by filepath or voice name\n", argv[i] );
                exit(1);
            }
            personaplex_voice_filepath = argv[++i];
            continue;
        }
        if (arg == "-p" || arg == "--prompt") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires quoted text\n", argv[i] );
                exit(1);
            }
            personaplex_system_prompt = argv[++i];
            continue;
        }
        if (arg[0] == '-') {
            fprintf( stderr, "error: unrecognized option \"%s\"\n", argv[i] );
            exit(1);
        }
        fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
        exit(1);
    }

    /////////////////////////
    // MARK: Initialize
    /////////////////////////

    std::string program_path = get_program_path(argv[0]);
    ensure_path( program_path );
    ensure_path( model_root );
    ensure_path( model_path );

    common_ggml_t ggml;
    init_ggml( ggml, device, n_threads );
    printf("DEBUG: init_ggml returned successfully, free memory: %d MiB\n", ggml.memory_free_mb ); fflush(stdout);
    memledger_log_simple("after_ggml_init", device_memory_free(ggml.dev) / 1024 / 1024); // CUDA context floor, cf. Stage 1
    const int memory_base = 4990;
    const int memory_ctx = 758;
    const int memory_min = 5368;
    const int memory_spec = 7264;
    printf("DEBUG: memory limits defined\n"); fflush(stdout);
    if ( ggml.memory_free_mb < memory_min ) {
        fprintf(stderr, "warning: might fail due to low memory!\n");
    } else if ( context == -1 && ggml.memory_free_mb - 100 < memory_spec ) {
        fprintf( stderr, "auto adjusting context to fit in memory, use context option to override.\n" );
        context = (ggml.memory_free_mb - 100 - memory_base) * 1000 / memory_ctx;
        fprintf( stderr, "context: %d\n", context );
    }
    printf("DEBUG: memory check done, context=%d\n", context); fflush(stdout);

    if ( input && ! file_exists( input ) ) {
        fprintf( stderr, "error: input file does not exist: %s\n", input );
        exit(1);
    }
    printf("DEBUG: input file check done\n"); fflush(stdout);

    // find model path
    bool found_file, found_dir;
    if ( is_abs_or_rel( model_path ) ) {
        printf("DEBUG: model_path is abs or rel: %s\n", model_path.c_str()); fflush(stdout);
        check_arg_path( model_path, found_file, found_dir );
        if ( ! found_dir ) {
            if ( found_file ) {
                fprintf( stderr, "error: expected directory but found file: %s\n",
                     model_path.c_str() );
                exit(1);
            } else {
                fprintf( stderr, "error: could not find directory: %s\n",
                    model_path.c_str() );
                exit(1);
            }
        }
    } else if ( ! model_path_set ) {
        printf("DEBUG: model_path is NOT set, using defaults\n"); fflush(stdout);
        // check defaults
        std::vector<std::string> paths;
        paths.push_back( model_root + "Codes4Fun/personaplex-7b-v1-q4_k-GGUF/" );
        paths.push_back( program_path + "Codes4Fun/personaplex-7b-v1-q4_k-GGUF/" );
        paths.push_back( "Codes4Fun/personaplex-7b-v1-q4_k-GGUF/" );
        paths.push_back( model_root + "nvidia/personaplex-7b-v1/" );
        paths.push_back( program_path + "nvidia/personaplex-7b-v1/" );
        paths.push_back( "nvidia/personaplex-7b-v1/" );
        for ( auto & path : paths ) {
            check_arg_path( path, found_file, found_dir );
            if ( found_dir ) {
                model_path = path;
                break;
            }
        }
        if ( ! found_dir ) {
            fprintf( stderr, "error: could not find a default model directory\n" );
            exit(1);
        }
    } else {
        printf("DEBUG: model_path is set: %s\n", model_path.c_str()); fflush(stdout);
        std::string full_path = model_root + model_path;
        check_arg_path( full_path, found_file, found_dir );
        printf("DEBUG: check_arg_path full_path done, found_file=%d, found_dir=%d\n", found_file, found_dir); fflush(stdout);
        if ( found_dir ) {
            model_path = full_path;
        } else {
            full_path = program_path + model_path;
            check_arg_path( full_path, found_file, found_dir );
            printf("DEBUG: check_arg_path program_path done, found_file=%d, found_dir=%d\n", found_file, found_dir); fflush(stdout);
            if ( found_dir ) {
                model_path = full_path;
            } else {
                check_arg_path( model_path, found_file, found_dir );
                printf("DEBUG: check_arg_path model_path done, found_file=%d, found_dir=%d\n", found_file, found_dir); fflush(stdout);
                if ( ! found_dir ) {
                    fprintf( stderr, "error: could not find directory: %s\n",
                        model_path.c_str() );
                    exit(1);
                }
            }
        }
    }
    printf( "found model path: %s\n", model_path.c_str() ); fflush(stdout);

    // default config
    moshi_config_t config;
    std::string config_filepath;
    config_filepath = model_path + "personaplex-config.json";
    printf("DEBUG: config_filepath = %s\n", config_filepath.c_str()); fflush(stdout);
    if ( ! file_exists( config_filepath.c_str() ) ) {
        config_filepath = program_path + "personaplex-config.json";
        printf("DEBUG: trying program_path config_filepath = %s\n", config_filepath.c_str()); fflush(stdout);
        if ( ! file_exists( config_filepath.c_str() ) ) {
            fprintf( stderr, "error: failed to find a config.json\n" );
            exit(1);
        }
    }

    printf("DEBUG: reading config from %s\n", config_filepath.c_str()); fflush(stdout);
    if ( moshi_get_config( &config, config_filepath.c_str() ) != 0 ) {
        fprintf( stderr, "error: reading config\n");
        exit(1);
    }
    printf("DEBUG: config loaded successfully\n"); fflush(stdout);

    if ( context > 0 ) {
        config.context = context;
    }
    // Not a memory metric — logged separately from MEMLEDGER's cuda_free/rss format
    // to avoid mislabeling a token count as a byte quantity.
    fprintf(stderr, "MEMLEDGER_CONTEXT final_context_size=%d\n", config.context); fflush(stderr);

    if ( kv_type_str == "q8_0" ) {
        config.kv_cache_type = GGML_TYPE_Q8_0;
    } else if ( kv_type_str == "q4_0" ) {
        config.kv_cache_type = GGML_TYPE_Q4_0;
    } else if ( kv_type_str == "q2_k" ) {
        config.kv_cache_type = GGML_TYPE_Q2_K;
    } else if ( kv_type_str == "f16" ) {
        config.kv_cache_type = GGML_TYPE_F16;
    } else if ( kv_type_str == "bf16" ) {
        config.kv_cache_type = GGML_TYPE_BF16;
    } else {
        fprintf( stderr, "error: unknown KV cache type \"%s\", must be bf16, f16, q8_0, q4_0, q2_k\n", kv_type_str.c_str() );
        exit(1);
    }
    std::string model_filepath = model_path + config.moshi_name;
    std::string mimi_filepath = model_path + config.mimi_name;
    std::string tokenizer_filepath = model_path + config.tokenizer_name;

    printf("DEBUG: checking moshi file: %s\n", model_filepath.c_str()); fflush(stdout);
    if ( ! file_exists( model_filepath.c_str() ) ) {
        fprintf( stderr, "error: missing moshi file \"%s\"\n", model_filepath.c_str() );
        exit(1);
    }
    printf("DEBUG: moshi file exists\n"); fflush(stdout);

    printf("DEBUG: checking mimi file: %s\n", mimi_filepath.c_str()); fflush(stdout);
    if ( ! file_exists( mimi_filepath.c_str() ) ) {
        printf("DEBUG: mimi file does not exist at path, trying defaults\n"); fflush(stdout);
        // files can be deleted or not downloaded to save memory
        bool found = false;
        std::vector<std::string> paths = {
            "nvidia/personaplex-7b-v1/tokenizer-e351c8d8-checkpoint125.safetensors",
            "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors",
            "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors",
            "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors",
            "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors",
            "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors",
        };
        if ( model_root.size() ) {
            paths.push_back( model_root + "nvidia/personaplex-7b-v1/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( model_root + "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( model_root + "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors" );
        }
        if ( program_path.size() ) {
            paths.push_back( program_path + "nvidia/personaplex-7b-v1/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( program_path + "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( program_path + "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( program_path + "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( program_path + "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( program_path + "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors" );
        }
        for ( auto & path : paths ) {
            printf("DEBUG: checking path: %s, file_exists=%d\n", path.c_str(), file_exists( path.c_str() )); fflush(stdout);
            if ( file_exists( path.c_str() ) ) {
                mimi_filepath = path;
                found = true;
                break;
            }
        }

        if ( ! found ) {
            fprintf( stderr, "error: missing mimi file \"%s\"\n", mimi_filepath.c_str() );
            exit(1);
        }
    }
    printf("DEBUG: mimi file path verified: %s\n", mimi_filepath.c_str()); fflush(stdout);

    if ( ! file_exists( tokenizer_filepath.c_str() ) ) {
        // files can be deleted or not downloaded to save memory
        bool found = false;
        if ( config.tokenizer_name == "tokenizer_spm_32k_3.model" ) {
            // the file is the same for several models
            std::vector<std::string> paths = {
                "nvidia/personaplex-7b-v1/tokenizer_spm_32k_3.model",
                "kyutai/moshika-pytorch-bf16/tokenizer_spm_32k_3.model",
                "kyutai/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model"
            };
            if ( model_root.size() ) {
                paths.push_back( model_root + "nvidia/personaplex-7b-v1/tokenizer_spm_32k_3.model" );
                paths.push_back( model_root + "kyutai/moshika-pytorch-bf16/tokenizer_spm_32k_3.model" );
                paths.push_back( model_root + "kyutai/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model" );
            }
            if ( program_path.size() ) {
                paths.push_back( program_path + "nvidia/personaplex-7b-v1/tokenizer_spm_32k_3.model" );
                paths.push_back( program_path + "kyutai/moshika-pytorch-bf16/tokenizer_spm_32k_3.model" );
                paths.push_back( program_path + "kyutai/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model" );
            }
            for ( auto & path : paths ) {
                if ( file_exists( path.c_str() ) ) {
                    tokenizer_filepath = path;
                    found = true;
                    break;
                }
            }
        }
        if ( ! found ) {
            fprintf( stderr, "error: missing tokenizer file \"%s\"\n", tokenizer_filepath.c_str() );
            exit(1);
        }
    }
    printf("DEBUG: tokenizer file path verified: %s\n", tokenizer_filepath.c_str()); fflush(stdout);

    bool personaplex_voice_embedding = false;
    bool personaplex_voice_mimi = false;
    if ( personaplex_voice_filepath.size() ) {
        if ( ! file_exists( personaplex_voice_filepath.c_str() ) ) {
            std::vector<std::string> paths;
            if ( personaplex_voice_filepath.size() == 5 ) {
                std::string expanded_filepath = model_path + "voices/" + personaplex_voice_filepath;
                paths.push_back( expanded_filepath + ".gguf" );
                paths.push_back( expanded_filepath + ".safetensors" );
            }
            paths.push_back( model_path + personaplex_voice_filepath );
            if ( model_root.size() ) {
                paths.push_back( model_root + personaplex_voice_filepath );
            }
            if ( program_path.size() ) {
                paths.push_back( program_path + personaplex_voice_filepath );
            }

            bool found = false;
            for ( auto & path : paths ) {
                if ( file_exists( path.c_str() ) ) {
                    personaplex_voice_filepath = path;
                    found = true;
                    break;
                }
            }

            if ( ! found ) {
                fprintf( stderr, "error: failed to find voice file \"%s\"\n", personaplex_voice_filepath.c_str() );
                exit(1);
            }
        }
        const char * ext = get_ext( personaplex_voice_filepath.c_str() );
        if ( strcmp( ext, ".gguf" ) == 0 || strcmp( ext, ".safetensors" ) == 0 )
            personaplex_voice_embedding = true;
        else if ( strcmp( ext, ".mimi" ) == 0 )
            personaplex_voice_mimi = true;
    }

    // MARK: Loading

    srand( seed );
    printf( "DEBUG: seed set, seed=%d\n", seed ); fflush(stdout);

    // context
    printf("DEBUG: calling moshi_alloc\n"); fflush(stdout);
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( ggml.backend, ggml.backend_cpu );
    printf("DEBUG: moshi_alloc returned=%p\n", (void*)moshi.ptr); fflush(stdout);

    printf( "loading...\n" ); fflush(stdout);
    auto load_start = ggml_time_ms();

    if ( quant ) {
        uint32_t uquant = *(uint32_t*)quant;
        switch (uquant) {
        case 0x305f3471: // "q4_0"
            break;
        case 0x6b5f3471: // "q4_k"
            break;
        case 0x6b5f3271: // "q2_k"
            break;
        case 0x305f3871: // "q8_0"
            break;
        default:
            fprintf( stderr, "error: invalid quant %s\n", quant );
            exit(-1);
        }
    }

    std::string model_gguf = "";
    if ( gguf_caching ) {
        if ( quant ) {
            model_gguf = model_filepath + "." + quant + ".gguf";
            if ( file_exists( model_gguf.c_str() ) ) {
                model_filepath = model_gguf;
                model_gguf = "";
                quant = NULL;
            }
        } else {
            model_gguf = model_filepath + ".gguf";
            if ( file_exists( model_gguf.c_str() ) ) {
                model_filepath = model_gguf;
                model_gguf = "";
            }
        }
    }

    // model
    unref_ptr<moshi_lm_t> lm = moshi_lm_from_files( moshi, &config,
        model_filepath.c_str() );
    if ( quant ) {
        if ( ! moshi_lm_quantize( lm, quant ) ) {
            fprintf( stderr, "error: unknown quant %s\n", quant );
            exit(-1);
        }
    }

    // generator
    printf("DEBUG: calling moshi_lm_generator\n"); fflush(stdout);
    unref_ptr<moshi_lm_gen_t> gen = moshi_lm_generator( lm );
    printf("DEBUG: moshi_lm_generator returned successfully\n"); fflush(stdout);

    // tokenizer
    printf("DEBUG: calling tokenizer_alloc\n"); fflush(stdout);
    unref_ptr<tokenizer_t> tok = tokenizer_alloc(
        tokenizer_filepath.c_str(),
        config.cross_attention );
    printf("DEBUG: tokenizer_alloc returned successfully\n"); fflush(stdout);

    // codec
    int num_audio_codebooks = 8; // personaplex hard codes this
    printf("DEBUG: calling mimi_alloc\n"); fflush(stdout);
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi,
        mimi_filepath.c_str(),
        num_audio_codebooks );
    printf("DEBUG: mimi_alloc returned successfully\n"); fflush(stdout);
    float frame_rate = mimi_frame_rate( codec );
    int frame_size = mimi_frame_size( codec );
    printf("DEBUG: frame_rate=%f, frame_size=%d\n", frame_rate, frame_size); fflush(stdout);

    // model
    printf("DEBUG: calling moshi_lm_load\n"); fflush(stdout);
    moshi_lm_load( lm );
    printf("DEBUG: moshi_lm_load returned successfully\n"); fflush(stdout);
    if ( model_gguf.size() ) {
        moshi_lm_save_gguf( lm, model_gguf.c_str() );
    }

    // encoder
    printf("DEBUG: calling mimi_encode_alloc_context\n"); fflush(stdout);
    unref_ptr<mimi_encode_context_t> encoder;
    encoder = mimi_encode_alloc_context( codec );
    printf("DEBUG: mimi_encode_alloc_context returned successfully\n"); fflush(stdout);
    memledger_log_simple("after_mimi_encode_alloc", device_memory_free(ggml.dev) / 1024 / 1024);

    // decoder
    printf("DEBUG: calling mimi_decode_alloc_context\n"); fflush(stdout);
    unref_ptr<mimi_decode_context_t> decoder;
    decoder = mimi_decode_alloc_context( codec );
    printf("DEBUG: mimi_decode_alloc_context returned successfully\n"); fflush(stdout);
    memledger_log_simple("after_mimi_decode_alloc", device_memory_free(ggml.dev) / 1024 / 1024);

    auto load_end = ggml_time_ms();
    printf("done loading. %f\n", (load_end - load_start) / 1000.f); fflush(stdout);

    // --- VRAM MEASUREMENT PROTOCOL (three independent sources, labelled) ---
    // Measurement 1: driver-level free memory delta since process start.
    // Captures everything the CUDA driver allocated: model weights + scratch buffers + KV cache.
    size_t post_load_driver_free = device_memory_free(ggml.dev);
    size_t vram_M1_driver_delta_bytes = ggml.memory_free - post_load_driver_free;
    printf("VRAM_M1_driver_delta_MiB: %zu  (pre_load_free=%zu MiB minus post_load_free=%zu MiB)\n",
           vram_M1_driver_delta_bytes / 1024 / 1024,
           ggml.memory_free / 1024 / 1024,
           post_load_driver_free / 1024 / 1024);
    fflush(stdout);

    // Measurement 2: sum of ggml_backend_buffer_get_size() across all known context buffers.
    // This is the PHYSICAL bytes the ggml allocator requested from the CUDA driver.
    // For BMO_TIER tensors this reflects the ACTUAL allocation (offset bytes, not cols*rows).
    size_t vram_M2_phys_alloc_bytes = moshi_get_allocated_memory(moshi, lm, codec, gen, encoder, decoder);
    printf("VRAM_M2_phys_alloc_MiB:  %zu  (ggml_backend_buffer_get_size sum)\n",
           vram_M2_phys_alloc_bytes / 1024 / 1024);
    fflush(stdout);

    // Note: M1 > M2 is expected \u2014 M1 includes CUDA runtime overhead, driver reserved pages,
    // and any buffers allocated outside ggml (e.g. cuBLAS workspace, KV scratch).
    // M2 < 5021 MiB means the 1D-alloc trick successfully reduced physical allocation.
    // M2 == 5021 MiB means the trick had no effect (e.g. allocator padded to logical size).
    // --- END VRAM MEASUREMENT PROTOCOL ---


    Decoder input_decoder;
    Resampler resampler;
    if ( input ) {
        input_decoder.init( input );
        AVChannelLayout mono;
        av_channel_layout_default( &mono, 1 );
        resampler.set_input( input_decoder.codec_ctx );
        resampler.set_output( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
        resampler.init();
    }

    if ( personaplex_voice_filepath.size() ) {
        if ( personaplex_voice_embedding ) {
            moshi_lm_personaplex_load_voice( moshi, gen,
                personaplex_voice_filepath.c_str() );
            printf("using voice embedding: %s\n", personaplex_voice_filepath.c_str() );
        } else if ( personaplex_voice_mimi ) {
            auto f = fopen( personaplex_voice_filepath.c_str(), "rb" );
            if ( ! f ) {
                fprintf( stderr, "error: failed to open \"%s\"\n", "temp.mimi" );
                exit(1);
            }
            int i32;
            auto n = fread( &i32, 4, 1, f );
            assert( n == 1 );
            assert( i32 == 0x494d494d );
            n = fread( &i32, 4, 1, f );
            assert( n == 1 );
            assert( i32 == num_audio_codebooks );
            std::deque<std::vector<int16_t>> audio_prompt;
            while ( true ) {
                audio_prompt.push_back({});
                auto & audio_codes = audio_prompt.back();
                audio_codes.resize( num_audio_codebooks );
                n = fread( audio_codes.data(), num_audio_codebooks * 2, 1, f );
                if ( n != 1 ) {
                    audio_prompt.pop_back();
                    break;
                }
            }
            fclose( f );
            moshi_lm_personaplex_audio_prompt( gen, audio_prompt );
            mimi_encode_reset( encoder );
            printf("using audio prompt: %s\n", personaplex_voice_filepath.c_str() );
        } else {
            // mimi encode audio file
            Decoder voice_decoder;
            voice_decoder.init( personaplex_voice_filepath.c_str() );
            AVChannelLayout mono;
            av_channel_layout_default( &mono, 1 );
            Resampler voice_resampler;
            voice_resampler.set_input( voice_decoder.codec_ctx );
            voice_resampler.set_output( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
            voice_resampler.init();
            std::deque<std::vector<int16_t>> audio_prompt;
            AVFrame * dec_frame;
            while ( ( dec_frame = voice_decoder.frame() ) ) {
                auto frame = voice_resampler.frame( dec_frame );
                while ( frame ) {
                    audio_prompt.push_back({});
                    auto & audio_codes = audio_prompt.back();
                    audio_codes.resize( num_audio_codebooks );
                    mimi_encode_send( encoder, (float*)frame->data[0] );
                    mimi_encode_receive( encoder, audio_codes.data() );
                    frame = voice_resampler.frame();
                }
            }
            auto frame = voice_resampler.flush( true ); // inject silence
            if ( frame ) {
                audio_prompt.push_back({});
                auto & audio_codes = audio_prompt.back();
                audio_codes.resize( num_audio_codebooks );
                mimi_encode_send( encoder, (float*)frame->data[0] );
                mimi_encode_receive( encoder, audio_codes.data() );
            }
            moshi_lm_personaplex_audio_prompt( gen, audio_prompt );
            mimi_encode_reset( encoder );
            printf("using audio prompt: %s\n", personaplex_voice_filepath.c_str() );
        }
    }
    if ( personaplex_system_prompt.size() ) {
        moshi_lm_personaplex_system_prompt( moshi, gen, tok,
            personaplex_system_prompt.c_str() );
    }

    /////////////////////////
    // MARK: SDL
    /////////////////////////

    AudioState input_state, output_state;
    SDL_AudioDeviceID cap_dev, dev;

    if ( ! bench ) {
        if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) != 0) {
            fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
            return 1;
        }

        sdl_init_frames( input_state, 3, frame_size*4 );

        SDL_AudioSpec want, have;

        want.freq = 24000; // Sample rate
        want.format = AUDIO_F32; // Audio format
        want.channels = 1; // Mono audio
        want.samples = frame_size;
        want.callback = sdl_capture_callback;
        want.userdata = &input_state;

        cap_dev = SDL_OpenAudioDevice(NULL, 1, &want, &have, 0);
        if (cap_dev <= 0) {
            fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
            return 1;
        }
        assert( want.freq == have.freq );
        assert( want.format == have.format );
        assert( want.channels == have.channels );
        assert( want.samples == have.samples );

        sdl_init_frames( output_state, 3, frame_size*4 );

        want.callback = sdl_audio_callback;
        want.userdata = &output_state;
        dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if (dev <= 0) {
            fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
            return 1;
        }
        assert( want.freq == have.freq );
        assert( want.format == have.format );
        assert( want.channels == have.channels );
        assert( want.samples == have.samples );
    }

    /////////////////////////
    // MARK: Loop
    /////////////////////////

    // model
    printf("DEBUG: calling moshi_lm_start\n"); fflush(stdout);
    moshi_lm_start( moshi, gen, depth_temperature, text_temperature );
    printf("DEBUG: moshi_lm_start returned successfully\n"); fflush(stdout);

    // PIPELINE: enable scratch buffer reuse now that prompt prefill is done —
    // steady decode then performs zero cudaMalloc/cudaFree on the LM thread
    // (cudaFree blocks the caller against in-flight kernels on other streams;
    // measured 100.7 ms vs a 100 ms busy stream on Orin). Enabled in BOTH
    // serial and pipelined modes so the two differ only in scheduling.
    moshi_lm_scratch_reuse( gen, 1 );

    // DEFAULTS TO SERIAL. The pipelined path failed its own correctness gate
    // (TOKEN_HASH mismatch vs serial at identical seed/frame-count, this-boot
    // smoke test — see HANDOFF.md §4) and root cause is NOT YET FOUND (ruled
    // out: shared/global rand() touched from a worker thread — grep confirms
    // no ctx.exponential()/rand() call anywhere in the mimi encode/decode
    // path). Per explicit instruction not to spend further session budget
    // debugging concurrency, this reverts the DEFAULT to the known-correct
    // serial loop; the pipelined implementation and its correctness-gate
    // instrumentation are left in place (BMO_PIPELINE=1 opts in) for whoever
    // picks up the root-cause investigation next.
    bool pipeline_on = false;
    {
        const char * e = getenv( "BMO_PIPELINE" );
        if ( e && strcmp( e, "1" ) == 0 ) pipeline_on = true;
    }
    printf( "PIPELINE: %s\n", pipeline_on
        ? "ON (mimi encode/decode on worker threads, own CUDA streams) -- KNOWN BROKEN, opt-in via BMO_PIPELINE=1 for debugging only, see HANDOFF.md"
        : "OFF (serial loop)" );
    fflush( stdout );

    // Correctness gate: FNV-1a 64 over the generated token sequence
    // (text token + 8 audio codes per emitted frame, frame order).
    // Identical placement in both loops; serial vs pipelined must match.
    uint64_t token_hash = 1469598103934665603ULL;
    long token_hash_frames = 0;
    auto token_hash_mix = [&]( uint64_t v ) {
        token_hash ^= v;
        token_hash *= 1099511628211ULL;
    };

    std::vector<int16_t> tokens(num_audio_codebooks);
    int text_token;

    std::vector<float> blank(frame_size);

    // BENCH_DUMP_PCM=<path>: dump each decoded frame (raw f32, 24 kHz mono)
    // during serial bench runs, for waveform-parity gates between builds.
    // Env-gated and bench/serial-only: unset (the official measurement
    // configuration) it is a single NULL check per frame. Not wired into the
    // pipelined decode worker — waveform gates run in the serial mode of
    // record.
    FILE * bench_pcm_f = NULL;
    if ( bench ) {
        const char * e = getenv( "BENCH_DUMP_PCM" );
        if ( e && e[0] ) {
            bench_pcm_f = fopen( e, "wb" );
            if ( ! bench_pcm_f )
                fprintf( stderr, "BENCH_DUMP_PCM: cannot open %s\n", e );
        }
    }

    // STEP 1 (frame-phase timing task): per-frame phase timing windows,
    // flushed to a median-per-25-frames report. Measurement only — does not
    // affect the decode path. t_temporal/t_depformer/t_sample_sync come from
    // g_moshi_phase_timing (accumulated inside lm.h, reset here each frame);
    // t_mimi_enc/t_mimi_dec are timed directly around the mimi calls below.
    std::vector<double> prof_t_mimi_enc, prof_t_mimi_dec, prof_t_temporal,
        prof_t_depformer, prof_t_dep_substep_mean, prof_t_sample_sync,
        prof_t_other, prof_t_frame_total;
    std::vector<long> prof_majflt_delta;
    long prof_last_majflt = read_majflt();

    if ( ! bench ) {
        SDL_PauseAudioDevice(cap_dev, 0);
        SDL_PauseAudioDevice(dev, 0);
    }

    AVFrame * dec_frame = input? dec_frame = input_decoder.frame() : NULL;
    AVFrame * res_frame = NULL;

    printf("ready\n");

    uint64_t lm_start = ggml_time_us();

    // ---- PIPELINE state (threads started only when pipeline_on) ----
    BoundedQueue<std::vector<int16_t>> pipe_enc_out( 2 ); // encode worker -> frame loop
    BoundedQueue<std::vector<int16_t>> pipe_dec_in( 2 );  // frame loop -> decode worker
    std::atomic<bool> pipe_enc_stop( false );
    std::thread pipe_enc_thread, pipe_dec_thread;
    std::atomic<long> pipe_enc_frames( 0 ), pipe_dec_frames( 0 );
    std::mutex pipe_stat_m;
    std::vector<double> pipe_enc_ms, pipe_dec_ms; // worker-side compute times
    bool first_pop_pending = true;
    long enc_first_pop_waits = 0; // frame-1 encode wait is structural, reported separately

    if ( pipeline_on ) {
        // Input acquisition, one frame ahead of the LM, mirroring the serial
        // loop's mode branches verbatim (file -> live handover included).
        // Runs exclusively on the encode worker after this point.
        std::function<bool(std::vector<float>&)> input_source =
            [&]( std::vector<float> & audio ) -> bool {
            if ( input ) {
                if ( input_delay > 0 ) {
                    input_delay--;
                    memset( audio.data(), 0, audio.size() * sizeof(float) );
                    if ( input_delay == 0 ) {
                        printf(" | ");
                        fflush( stdout );
                    }
                    return true;
                }
                if ( res_frame ) {
                    // drain resampler
                    res_frame = resampler.frame();
                }
                while ( ! res_frame ) { // fill resampler if needed
                    dec_frame = input_decoder.frame();
                    if ( ! dec_frame ) { // we are done
                        break;
                    } else {
                        res_frame = resampler.frame( dec_frame );
                    }
                }
                if ( res_frame ) {
                    memcpy( audio.data(), res_frame->data[0], audio.size() * sizeof(float) );
                    res_frame = resampler.frame();
                } else {
                    // no more decoder frames
                    input = NULL;
                    memset( audio.data(), 0, audio.size() * sizeof(float) );
                    printf(" | ");
                    fflush( stdout );
                }
                return true;
            } else if ( bench ) {
                memset( audio.data(), 0, audio.size() * sizeof(float) );
                return true;
            } else {
                sdl_frame_t * input_frame = sdl_receive_frame( input_state, true );
                memcpy( audio.data(), input_frame->data, audio.size() * sizeof(float) );
                sdl_free_frame( input_state, input_frame );
                return true;
            }
        };

        // ---- encode worker: own backend/stream, feeds the frame loop ----
        pipe_enc_thread = std::thread( [&]() {
            std::vector<float> audio( frame_size );
            std::vector<int16_t> codes( num_audio_codebooks );
            while ( ! pipe_enc_stop.load() && ! shutdown ) {
                if ( ! input_source( audio ) ) break;
                auto t0 = std::chrono::steady_clock::now();
                mimi_encode_send( encoder, audio.data() );
                mimi_encode_receive( encoder, codes.data() );
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0 ).count();
                {
                    std::lock_guard<std::mutex> lk( pipe_stat_m );
                    pipe_enc_ms.push_back( ms );
                }
                pipe_enc_frames++;
                std::vector<int16_t> out( codes );
                if ( ! pipe_enc_out.push( std::move( out ) ) ) break; // closed
            }
        } );

        // ---- decode worker: own backend/stream, consumes generated codes ----
        pipe_dec_thread = std::thread( [&]() {
            std::vector<int16_t> codes;
            std::vector<float> audio_out( frame_size );
            while ( pipe_dec_in.pop( codes ) ) { // drains queued frames after close
                auto t0 = std::chrono::steady_clock::now();
                mimi_decode_send( decoder, codes.data() );
                if ( bench ) {
                    mimi_decode_receive( decoder, audio_out.data() );
                } else {
                    // sdl_get_frame/sdl_send_frame pace against the audio device
                    sdl_frame_t * frame = sdl_get_frame( output_state );
                    mimi_decode_receive( decoder, (float*)frame->data );
                    sdl_send_frame( output_state, frame );
                }
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0 ).count();
                {
                    std::lock_guard<std::mutex> lk( pipe_stat_m );
                    pipe_dec_ms.push_back( ms );
                }
                pipe_dec_frames++;
            }
        } );

        // ---- pipelined frame loop: temporal/depformer/sampling only ----
        while ( ! shutdown ) {
            if ( bench ) lm_start = ggml_time_us();
            auto _prof_frame_t0 = std::chrono::steady_clock::now();
            moshi_phase_timing_reset();

            // this frame's input codes (worker runs one frame ahead);
            // t_mimi_enc now measures the critical-path WAIT, not encode itself
            auto _prof_menc_t0 = std::chrono::steady_clock::now();
            std::vector<int16_t> in_codes;
            bool got = pipe_enc_out.pop( in_codes );
            auto _prof_menc_t1 = std::chrono::steady_clock::now();
            double _prof_t_mimi_enc = std::chrono::duration<double, std::milli>( _prof_menc_t1 - _prof_menc_t0 ).count();
            if ( first_pop_pending ) {
                first_pop_pending = false;
                enc_first_pop_waits = pipe_enc_out.pop_waits;
            }
            if ( ! got ) break;
            for ( int i = 0; i < num_audio_codebooks; i++ )
                tokens[i] = in_codes[i];
            if ( ! bench ) lm_start = ggml_time_us(); // live mode: mic pacing excluded

            printf("DEBUG: calling moshi_lm_send2\n"); fflush(stdout);
            moshi_lm_send2( gen, tokens );
            printf("DEBUG: after moshi_lm_send2\n"); fflush(stdout);

            printf("DEBUG: calling moshi_lm_receive\n"); fflush(stdout);
            if ( moshi_lm_receive( gen, text_token, tokens ) ) {
                printf("DEBUG: after moshi_lm_receive\n"); fflush(stdout);

                token_hash_mix( (uint64_t)(uint32_t)text_token );
                for ( int i = 0; i < num_audio_codebooks; i++ )
                    token_hash_mix( (uint64_t)(uint16_t)tokens[i] );
                token_hash_frames++;

                // audio out -> decode worker; t_mimi_dec now measures the
                // critical-path WAIT (queue-full = decode not keeping up)
                auto _prof_mdec_t0 = std::chrono::steady_clock::now();
                std::vector<int16_t> out_codes( tokens.begin(), tokens.end() );
                pipe_dec_in.push( std::move( out_codes ) );
                auto _prof_mdec_t1 = std::chrono::steady_clock::now();
                double _prof_t_mimi_dec = std::chrono::duration<double, std::milli>( _prof_mdec_t1 - _prof_mdec_t0 ).count();

                lm_delta_time += ggml_time_us() - lm_start;
                lm_frames++;

                // STEP 1 phase timing: identical window/report format to the
                // serial loop; t_mimi_enc/t_mimi_dec are queue waits here.
                if ( bench ) {
                    auto _prof_frame_t1 = std::chrono::steady_clock::now();
                    double _prof_frame_total = std::chrono::duration<double, std::milli>( _prof_frame_t1 - _prof_frame_t0 ).count();
                    double _prof_t_temporal = g_moshi_phase_timing.t_temporal_ms;
                    double _prof_t_depformer = g_moshi_phase_timing.t_depformer_ms;
                    double _prof_t_sample_sync = g_moshi_phase_timing.t_sample_sync_ms;
                    int _prof_dep_substeps = g_moshi_phase_timing.depformer_substeps;
                    double _prof_t_other = _prof_frame_total - ( _prof_t_mimi_enc + _prof_t_mimi_dec +
                        _prof_t_temporal + _prof_t_depformer + _prof_t_sample_sync );
                    long _prof_majflt_now = read_majflt();
                    long _prof_majflt_delta = ( prof_last_majflt >= 0 && _prof_majflt_now >= 0 )
                        ? ( _prof_majflt_now - prof_last_majflt ) : -1;
                    prof_last_majflt = _prof_majflt_now;

                    prof_t_mimi_enc.push_back( _prof_t_mimi_enc );
                    prof_t_mimi_dec.push_back( _prof_t_mimi_dec );
                    prof_t_temporal.push_back( _prof_t_temporal );
                    prof_t_depformer.push_back( _prof_t_depformer );
                    prof_t_dep_substep_mean.push_back( _prof_dep_substeps > 0 ? _prof_t_depformer / _prof_dep_substeps : 0.0 );
                    prof_t_sample_sync.push_back( _prof_t_sample_sync );
                    prof_t_other.push_back( _prof_t_other );
                    prof_t_frame_total.push_back( _prof_frame_total );
                    prof_majflt_delta.push_back( _prof_majflt_delta );

                    if ( lm_frames % 25 == 0 ) {
                        long majflt_sum = 0;
                        for ( long d : prof_majflt_delta ) if ( d > 0 ) majflt_sum += d;
                        printf(
                            "PHASE_TIMING frame=%4d window=%zu median_ms: "
                            "t_frame_total=%7.3f t_temporal=%7.3f t_depformer=%7.3f t_depformer_substep_mean(n=%d)=%7.3f "
                            "t_sample_sync=%7.3f t_mimi_enc=%7.3f t_mimi_dec=%7.3f t_other=%7.3f  majflt_sum_window=%ld majflt_per_frame=%.2f\n",
                            (int)lm_frames, prof_t_frame_total.size(),
                            median_ms(prof_t_frame_total), median_ms(prof_t_temporal), median_ms(prof_t_depformer),
                            _prof_dep_substeps, median_ms(prof_t_dep_substep_mean),
                            median_ms(prof_t_sample_sync), median_ms(prof_t_mimi_enc), median_ms(prof_t_mimi_dec),
                            median_ms(prof_t_other), majflt_sum, (double)majflt_sum / prof_t_frame_total.size() );
                        fflush(stdout);
                        prof_t_mimi_enc.clear(); prof_t_mimi_dec.clear(); prof_t_temporal.clear();
                        prof_t_depformer.clear(); prof_t_dep_substep_mean.clear(); prof_t_sample_sync.clear();
                        prof_t_other.clear(); prof_t_frame_total.clear(); prof_majflt_delta.clear();
                    }
                }

                if ( lm_frames == 1 ) {
                    memledger_log_simple("after_first_decode_frame", device_memory_free(ggml.dev) / 1024 / 1024);
                }

                if ( bench && (lm_frames % 25 == 0) ) {
                    size_t cur_driver_free = device_memory_free(ggml.dev);
                    size_t m1 = ggml.memory_free - cur_driver_free;
                    size_t m2 = moshi_get_allocated_memory(moshi, lm, codec, gen, encoder, decoder);
                    printf("VRAM_FRAME: frame=%4d  M1_driver_delta_MiB=%5zu  M2_phys_alloc_MiB=%5zu  outside_ggml_MiB=%5zd\n",
                           lm_frames,
                           m1 / 1024 / 1024,
                           m2 / 1024 / 1024,
                           ((long long)m1 - (long long)m2) / 1024 / 1024);
                    fflush(stdout);
                }

                if ( bench && lm_frames >= 1250 ) {
                    break;
                }

                // text out
                if ( text_token != 0 && text_token != 3 /*&& text_token > 0*/ ) {
                    auto piece = tokenizer_id_to_piece( tok, text_token );
                    std::string _text;
                    for ( size_t ci = 0; ci < piece.size(); ci++ ) {
                        if ( piece.c_str()[ci] == -30 ) {
                            _text += ' ';
                            ci += 2;
                            continue;
                        }
                        _text += piece[ci];
                    }
                    fprintf( stdout, "%s", _text.c_str() );
                    fflush( stdout );
                }
            }
        }
    }

    if ( ! pipeline_on )
    while ( ! shutdown ) {
        if ( input ) {
            if ( input_delay > 0 ) {
                input_delay--;
                memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
                lm_start = ggml_time_us();
                mimi_encode_send( encoder, blank.data() );
                if ( input_delay == 0 ) {
                    printf(" | ");
                    fflush( stdout );
                }
            } else {
                if ( res_frame ) {
                    // drain resampler
                    res_frame = resampler.frame();
                }
                while ( ! res_frame ) { // fill resampler if needed
                    dec_frame = input_decoder.frame();
                    if ( ! dec_frame ) { // we are done
                        break;
                    } else {
                        res_frame = resampler.frame( dec_frame );
                    }
                }
                if ( res_frame ) {
                    lm_start = ggml_time_us();
                    mimi_encode_send( encoder, (float*)res_frame->data[0] );
                    res_frame = resampler.frame();
                } else {
                    // no more decoder frames
                    input = NULL;
                    memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
                    lm_start = ggml_time_us();
                    mimi_encode_send( encoder, blank.data() );
                    printf(" | ");
                    fflush( stdout );
                }
            }
        } else if ( bench ) {
            memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
            lm_start = ggml_time_us();
            printf("DEBUG: calling mimi_encode_send\n"); fflush(stdout);
            mimi_encode_send( encoder, blank.data() );
            printf("DEBUG: after mimi_encode_send\n"); fflush(stdout);
        } else {
            // sdl_receive_frame can block, don't include in frame rate
            sdl_frame_t * input_frame = sdl_receive_frame( input_state, true );

            lm_start = ggml_time_us();
            mimi_encode_send( encoder, (float*)input_frame->data );
            lm_delta_time += ggml_time_us() - lm_start;

            sdl_free_frame( input_state, input_frame );
            lm_start = ggml_time_us();
        }

        // STEP 1 phase timing: frame boundary + reset the LM-side accumulator
        // (filled inside moshi_lm_receive -> moshi_lmgen_step, see lm.h).
        auto _prof_frame_t0 = std::chrono::steady_clock::now();
        moshi_phase_timing_reset();

        printf("DEBUG: calling mimi_encode_receive\n"); fflush(stdout);
        auto _prof_menc_t0 = std::chrono::steady_clock::now();
        mimi_encode_receive( encoder, tokens.data() );
        auto _prof_menc_t1 = std::chrono::steady_clock::now();
        double _prof_t_mimi_enc = std::chrono::duration<double, std::milli>( _prof_menc_t1 - _prof_menc_t0 ).count();
        printf("DEBUG: after mimi_encode_receive\n"); fflush(stdout);

        printf("DEBUG: calling moshi_lm_send2\n"); fflush(stdout);
        moshi_lm_send2( gen, tokens );
        printf("DEBUG: after moshi_lm_send2\n"); fflush(stdout);

        printf("DEBUG: calling moshi_lm_receive\n"); fflush(stdout);
        if ( moshi_lm_receive( gen, text_token, tokens ) ) {
            printf("DEBUG: after moshi_lm_receive\n"); fflush(stdout);

            token_hash_mix( (uint64_t)(uint32_t)text_token );
            for ( int i = 0; i < num_audio_codebooks; i++ )
                token_hash_mix( (uint64_t)(uint16_t)tokens[i] );
            token_hash_frames++;

            // audio out
            mimi_decode_send( decoder, tokens.data() );

            double _prof_t_mimi_dec = 0;
            if ( bench ) {
                auto _prof_mdec_t0 = std::chrono::steady_clock::now();
                mimi_decode_receive( decoder, blank.data() );
                auto _prof_mdec_t1 = std::chrono::steady_clock::now();
                _prof_t_mimi_dec = std::chrono::duration<double, std::milli>( _prof_mdec_t1 - _prof_mdec_t0 ).count();
                if ( bench_pcm_f )
                    fwrite( blank.data(), sizeof( float ), blank.size(), bench_pcm_f );
            } else {
                // sdl_get_frame can block, don't include in frame rate
                lm_delta_time += ggml_time_us() - lm_start;
                sdl_frame_t * frame = sdl_get_frame( output_state );
                lm_start = ggml_time_us();

                auto _prof_mdec_t0 = std::chrono::steady_clock::now();
                mimi_decode_receive( decoder, (float*)frame->data );
                auto _prof_mdec_t1 = std::chrono::steady_clock::now();
                _prof_t_mimi_dec = std::chrono::duration<double, std::milli>( _prof_mdec_t1 - _prof_mdec_t0 ).count();
                sdl_send_frame( output_state, frame ); // this can block
            }
            lm_delta_time += ggml_time_us() - lm_start;
            lm_frames++;

            // STEP 1 phase timing: record this frame's window sample (bench only —
            // matches the existing VRAM_FRAME probe's cadence/scope).
            if ( bench ) {
                auto _prof_frame_t1 = std::chrono::steady_clock::now();
                double _prof_frame_total = std::chrono::duration<double, std::milli>( _prof_frame_t1 - _prof_frame_t0 ).count();
                double _prof_t_temporal = g_moshi_phase_timing.t_temporal_ms;
                double _prof_t_depformer = g_moshi_phase_timing.t_depformer_ms;
                double _prof_t_sample_sync = g_moshi_phase_timing.t_sample_sync_ms;
                int _prof_dep_substeps = g_moshi_phase_timing.depformer_substeps;
                double _prof_t_other = _prof_frame_total - ( _prof_t_mimi_enc + _prof_t_mimi_dec +
                    _prof_t_temporal + _prof_t_depformer + _prof_t_sample_sync );
                long _prof_majflt_now = read_majflt();
                long _prof_majflt_delta = ( prof_last_majflt >= 0 && _prof_majflt_now >= 0 )
                    ? ( _prof_majflt_now - prof_last_majflt ) : -1;
                prof_last_majflt = _prof_majflt_now;

                prof_t_mimi_enc.push_back( _prof_t_mimi_enc );
                prof_t_mimi_dec.push_back( _prof_t_mimi_dec );
                prof_t_temporal.push_back( _prof_t_temporal );
                prof_t_depformer.push_back( _prof_t_depformer );
                prof_t_dep_substep_mean.push_back( _prof_dep_substeps > 0 ? _prof_t_depformer / _prof_dep_substeps : 0.0 );
                prof_t_sample_sync.push_back( _prof_t_sample_sync );
                prof_t_other.push_back( _prof_t_other );
                prof_t_frame_total.push_back( _prof_frame_total );
                prof_majflt_delta.push_back( _prof_majflt_delta );

                if ( lm_frames % 25 == 0 ) {
                    long majflt_sum = 0;
                    for ( long d : prof_majflt_delta ) if ( d > 0 ) majflt_sum += d;
                    printf(
                        "PHASE_TIMING frame=%4d window=%zu median_ms: "
                        "t_frame_total=%7.3f t_temporal=%7.3f t_depformer=%7.3f t_depformer_substep_mean(n=%d)=%7.3f "
                        "t_sample_sync=%7.3f t_mimi_enc=%7.3f t_mimi_dec=%7.3f t_other=%7.3f  majflt_sum_window=%ld majflt_per_frame=%.2f\n",
                        (int)lm_frames, prof_t_frame_total.size(),
                        median_ms(prof_t_frame_total), median_ms(prof_t_temporal), median_ms(prof_t_depformer),
                        _prof_dep_substeps, median_ms(prof_t_dep_substep_mean),
                        median_ms(prof_t_sample_sync), median_ms(prof_t_mimi_enc), median_ms(prof_t_mimi_dec),
                        median_ms(prof_t_other), majflt_sum, (double)majflt_sum / prof_t_frame_total.size() );
                    fflush(stdout);
                    prof_t_mimi_enc.clear(); prof_t_mimi_dec.clear(); prof_t_temporal.clear();
                    prof_t_depformer.clear(); prof_t_dep_substep_mean.clear(); prof_t_sample_sync.clear();
                    prof_t_other.clear(); prof_t_frame_total.clear(); prof_majflt_delta.clear();
                }
            }

            if ( lm_frames == 1 ) {
                memledger_log_simple("after_first_decode_frame", device_memory_free(ggml.dev) / 1024 / 1024);
            }

            // Frame-indexed VRAM probe every 25 frames.
            // Prints both M1 (driver free delta) and M2 (ggml physical alloc sum) so
            // growth curve can distinguish pool growth (M1 rises, M2 flat) from
            // ggml-internal growth (both rise together).
            if ( bench && (lm_frames % 25 == 0) ) {
                size_t cur_driver_free = device_memory_free(ggml.dev);
                size_t m1 = ggml.memory_free - cur_driver_free;
                size_t m2 = moshi_get_allocated_memory(moshi, lm, codec, gen, encoder, decoder);
                printf("VRAM_FRAME: frame=%4d  M1_driver_delta_MiB=%5zu  M2_phys_alloc_MiB=%5zu  outside_ggml_MiB=%5zd\n",
                       lm_frames,
                       m1 / 1024 / 1024,
                       m2 / 1024 / 1024,
                       ((long long)m1 - (long long)m2) / 1024 / 1024);
                fflush(stdout);
            }

            if ( bench && lm_frames >= 1250 ) {
                break;
            }

            // text out
            if ( text_token != 0 && text_token != 3 /*&& text_token > 0*/ ) {
                auto piece = tokenizer_id_to_piece( tok, text_token );
                std::string _text;
                for ( size_t ci = 0; ci < piece.size(); ci++ ) {
                    if ( piece.c_str()[ci] == -30 ) {
                        _text += ' ';
                        ci += 2;
                        continue;
                    }
                    _text += piece[ci];
                }
                fprintf( stdout, "%s", _text.c_str() );
                fflush( stdout );
            }
        }
    }

    if ( bench_pcm_f )
        fclose( bench_pcm_f );

    // PIPELINE: shut down workers cleanly. Encode worker exits its own loop
    // once told to stop (shutdown flag / pipe_enc_stop) or once input runs
    // out; closing pipe_enc_out also wakes it if blocked mid-push. Decode
    // worker keeps draining pipe_dec_in after close() (finishes whatever the
    // frame loop already queued) and exits only once empty+closed.
    if ( pipeline_on ) {
        pipe_enc_stop.store( true );
        pipe_enc_out.close();
        if ( pipe_enc_thread.joinable() ) pipe_enc_thread.join();

        pipe_dec_in.close();
        if ( pipe_dec_thread.joinable() ) pipe_dec_thread.join();

        // Critical-path stall audit. enc_out.pop() is called by the frame
        // loop (consumer) and dec_in.push() is called by the frame loop
        // (producer) — those are the only two queue operations the
        // temporal/depformer/sampling chain performs, so waits there are the
        // only ones that count as the chain blocking on codec work. Frame 1's
        // enc_out pop is a structural startup fill (nothing to pop_wait on
        // dec_in.push() at that point ever a startup case, since the queue
        // starts empty with capacity 2). dec_in.pop_waits (decode worker idle
        // between frames) and enc_out.push_waits (encode worker throttled 2
        // frames ahead) are expected/benign and reported for completeness only.
        long enc_out_pop_waits_steady = pipe_enc_out.pop_waits - enc_first_pop_waits;
        long dec_in_push_waits = pipe_dec_in.push_waits;
        long underrun_count = enc_out_pop_waits_steady + dec_in_push_waits;

        double enc_ms_sum = 0, dec_ms_sum = 0;
        {
            std::lock_guard<std::mutex> lk( pipe_stat_m );
            for ( double v : pipe_enc_ms ) enc_ms_sum += v;
            for ( double v : pipe_dec_ms ) dec_ms_sum += v;
        }
        printf( "\nPIPELINE_STATS enc_frames=%ld dec_frames=%ld "
                "enc_out_pop_waits_total=%ld enc_out_pop_waits_startup=%ld enc_out_pop_waits_steady=%ld "
                "enc_out_push_waits=%ld (benign) dec_in_pop_waits=%ld (benign, decode-idle) "
                "dec_in_push_waits=%ld enc_mean_ms=%.3f dec_mean_ms=%.3f\n",
                (long)pipe_enc_frames.load(), (long)pipe_dec_frames.load(),
                pipe_enc_out.pop_waits, enc_first_pop_waits, enc_out_pop_waits_steady,
                pipe_enc_out.push_waits, pipe_dec_in.pop_waits,
                dec_in_push_waits,
                pipe_enc_frames.load() ? enc_ms_sum / pipe_enc_frames.load() : 0.0,
                pipe_dec_frames.load() ? dec_ms_sum / pipe_dec_frames.load() : 0.0 );
        printf( "QUEUE_UNDERRUNS: %ld (%s)\n", underrun_count,
                underrun_count == 0 ? "PASS — critical path never blocked on codec work beyond startup fill"
                                    : "FAIL — critical path stalled on codec queue" );
        fflush( stdout );
    }

    printf( "TOKEN_HASH: 0x%016" PRIx64 " frames=%ld mode=%s\n",
            token_hash, token_hash_frames, pipeline_on ? "pipelined" : "serial" );
    fflush( stdout );

    auto memory_delta = ggml.memory_free - device_memory_free( ggml.dev );
    printf("\ndevice memory delta: %d MiB", (int)( memory_delta / 1024 / 1024 ) );
    size_t allocated_bytes = moshi_get_allocated_memory( moshi, lm, codec, gen, encoder, decoder );
    printf("\nexact tensor VRAM allocation: %d MiB\n", (int)( allocated_bytes / 1024 / 1024 ) );

    log_metrics();

    return 0;
}


