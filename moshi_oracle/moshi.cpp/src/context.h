#pragma once

#include <inttypes.h>
#include <algorithm>

#include <stdexcept>

typedef int64_t NE[GGML_MAX_DIMS]; // number of elements per dimension
class GGML_NE {
public:
    int64_t ne[GGML_MAX_DIMS];
    GGML_NE( int64_t ne0=1, int64_t ne1=1, int64_t ne2=1, int64_t ne3=1 ) {
        ne[0] = ne0;
        ne[1] = ne1;
        ne[2] = ne2;
        ne[3] = ne3;
    }

    GGML_NE( std::vector<int> _ne, bool reverse = false ) {
        assert( _ne.size() <= GGML_MAX_DIMS );
        int i = 0;
        if ( ! reverse ) {
            for ( ; i < GGML_MAX_DIMS && i < _ne.size(); i++ )
                ne[i] = _ne[i];
        } else {
            for ( ; i < GGML_MAX_DIMS && i < _ne.size(); i++ )
                ne[i] = _ne[_ne.size()-1 - i];
        }
        for ( ; i < GGML_MAX_DIMS; i++ )
            ne[i] = 1;
    }

    operator int64_t* () {
        return ne;
    }
    
    int64_t nelements() {
        return ne[0] * ne[1] * ne[2] * ne[3];
    }
};

ggml_type safetensor_get_type(std::string dtype) {
    if (dtype == "F32")
        return GGML_TYPE_F32;
    if (dtype == "F16")
        return GGML_TYPE_F16;
    if (dtype == "BF16")
        return GGML_TYPE_BF16;
    if (dtype == "I32")
        return GGML_TYPE_I32;
    assert(false);
    return (ggml_type)-1;
}

int safetensor_get_shape(safetensor_t * safetensor, NE &ne, int offset = 0) {
    // dimensions are inverted
    int last_index = (int) safetensor->shape.size() - 1;
    assert( last_index + offset < 4 );
    for (int i = 0; i < offset; i++)
        ne[i] = 1;
    for (int i = 0; i <= last_index; i++)
        ne[i + offset] = safetensor->shape[last_index-i];
    for (int i = offset + last_index + 1; i < 4; i++) {
        ne[i] = 1;
    }
    return (int) safetensor->shape.size() + offset;
}

ggml_tensor * safetensor_alloc( ggml_context * ctx, safetensor_t * safetensor) {
    auto type = safetensor_get_type(safetensor->dtype);
    // dimensions are inverted
    int last_index = (int) safetensor->shape.size() - 1;
    NE ne = {1, 1, 1, 1};
    for (int i = 0; i <= last_index; i++)
        ne[i] = safetensor->shape[last_index-i];
    return ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
}

class SafeTensorFile {
    public:
    FILE * f;
    int64_t header_length;
    safetensors_t tensors;
    SafeTensorFile() {}
    ~SafeTensorFile() {fclose(f);}

    int ref_count = 1;
    void ref() {
        ref_count++;
    }
    void unref() {
        --ref_count;
        if (ref_count < 1)
            delete this;
    }

    static SafeTensorFile * from_file(const char * filename) {
        FILE * f = fopen(filename, "rb");
        if (!f)
            return NULL;
        int64_t length;
        size_t r;
        r = fread(&length, sizeof(length), 1, f);
        if (r != 1 || length == 0) {
            fclose(f);
            return NULL;
        }
        std::vector<char> data(length+1);
        r = fread(data.data(), length, 1, f);
        if (r != 1) {
            fclose(f);
            return NULL;
        }
        data[length] = 0;

        const_str_t json = {data.data(), (int)length};

        safetensors_t tensors;
        if (!safetensor_parse(json, tensors)) {
            fclose(f);
            return NULL;
        }

        auto stf = new SafeTensorFile();
        stf->f = f;
        stf->header_length = length + 8;
        stf->tensors.swap(tensors);
        return stf;
    }

    safetensor_t * find(std::string name) {
        auto it = tensors.find(name);
        if (it == tensors.end())
            return NULL;
        return & it->second;
    }

    void init( safetensor_t * safetensor, ggml_tensor * tensor, ggml_backend * backend = NULL ) {
        int64_t nbytes = ggml_nbytes(tensor);
        int64_t offset = safetensor->data_offsets[0] + header_length;
        int64_t size = safetensor->data_offsets[1] - safetensor->data_offsets[0];
        if (nbytes > size) {
            printf("data is smaller than expected, got %" PRId64 " needed %" PRId64 "\n", size, nbytes);
            exit(-1);
        }
#ifdef _WIN32
        //fseeko64(f, offset, SEEK_SET);
        auto e = _fseeki64(f, offset, SEEK_SET);
#else
        auto e = fseek(f, offset, SEEK_SET);
#endif
        assert( e == 0 );
        if (backend) {
            std::vector<char*> data(nbytes);
            int64_t r = fread(data.data(), nbytes, 1, f);
            if (r != 1) {
                printf("failed to read tensor %s\n", safetensor->key.c_str());
                exit(-1);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);
        } else {
            int64_t r = fread(tensor->data, nbytes, 1, f);
            if (r != 1) {
                printf("failed to read tensor %s\n", safetensor->key.c_str());
                exit(-1);
            }
        }
    }
};

SafeTensorFile * ref( SafeTensorFile * stf ) {
    stf->ref();
    return stf;
}

void unref( SafeTensorFile * stf ) {
    stf->unref();
}

// single tensor context
class own_ctx_tensor {
public:
    ggml_context * ctx;
    ggml_tensor * tensor;
    ggml_backend_buffer * buffer;
    own_ctx_tensor() {
        ctx = NULL;
        tensor = NULL;
        buffer = NULL;
    }
    void reset() {
        if ( buffer )
            ggml_backend_buffer_free( buffer );
        if ( ctx )
            ggml_free( ctx );
        buffer = NULL;
        ctx = NULL;
    }
    ~own_ctx_tensor() {
        reset();
    }
    void new_tensor( NE ne, ggml_type type, ggml_backend * backend ) {
        assert( backend ); // TODO: support non-backend options
        reset();
        if ( backend ) {
            ctx = ggml_init({
                /*.mem_size   =*/ ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            });
            assert( ctx );
            tensor = ggml_new_tensor( ctx, type, 4, ne );
            assert( tensor );
            buffer = ggml_backend_alloc_ctx_tensors( ctx, backend );
            assert( buffer );
        }
    }
    operator ggml_tensor* () {
        return tensor;
    }
    ggml_tensor * operator->() {
        return tensor;
    }
    own_ctx_tensor( const own_ctx_tensor& ) = delete; 
    own_ctx_tensor & operator=( own_ctx_tensor& ) = delete;
};

class GraphContext {
    public:
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_backend_buffer * buffer;

    // to load tensors
    struct load_t {
        SafeTensorFile * src;
        safetensor_t * safetensor;
        ggml_tensor * tensor;
    };
    std::vector<load_t> loaders;
    // to load 32 constant
    struct constant_32_t {
        ggml_tensor * tensor;
        int32_t value;
    };
    std::vector<constant_32_t> constants32;
    // to load a vector constant
    struct constant_t {
        ggml_tensor * tensor;
        std::vector<uint8_t> data;
    };
    std::vector<constant_t> constants;
    // exponential (random)
    std::vector<uint8_t> scratch_data;
    struct exponential_t {
        ggml_tensor * tensor;
        float lambd;
    };
    std::vector<exponential_t> exponentials;
    // input convert
    struct input_convert_t {
        ggml_tensor * dst;
        ggml_tensor * src;
    };
    std::vector<input_convert_t> input_converts;
    //
    struct backend_tensor_t {
        ggml_tensor * src;
        ggml_tensor * dst;
    };
    std::vector<backend_tensor_t> backend_copies;
    //
    struct copy_t {
        ggml_tensor * src;
        void * dst;
    };
    std::vector<copy_t> copies;

    // PIPELINE (mimi off the critical path): cudaFree blocks the calling
    // thread until every in-flight kernel on the device finishes (measured
    // 100.7 ms against a 100 ms busy stream on Orin, cudafree_stall_test).
    // The default per-compute() alloc/free of scratch buffers is therefore a
    // cross-thread stall once mimi decode runs concurrently on its own
    // stream. With reuse_buffer enabled the backing buffer is kept high-water
    // style: steady state performs zero cudaMalloc/cudaFree. Growth (warmup
    // only, if a later graph is bigger) still frees, loudly.
    bool reuse_buffer = false;
    ggml_backend_buffer_t reuse_buf = NULL;
    size_t reuse_buf_size = 0;

    GraphContext( size_t mb, ggml_backend * backend = NULL ) {
        this->backend = backend;
        ctx = ggml_init({
            /*.mem_size   =*/ mb * 1024 * 1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ backend? true : false, // NOTE: this should be false when using the legacy API
        });
        gf = NULL;
        buffer = NULL;
    }

    ~GraphContext() {
        if ( buffer && buffer != reuse_buf )
            ggml_backend_buffer_free( buffer );
        if ( reuse_buf )
            ggml_backend_buffer_free( reuse_buf );
        ggml_free(ctx);
    }

    operator ggml_context * () {
        return ctx;
    }

    ggml_tensor * new_tensor( ggml_type type, NE ne ) {
        auto tensor = ggml_new_tensor( ctx, type, 4, ne );
        return tensor;
    }

    virtual void tensor_set( ggml_tensor * tensor, int32_t value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == 1 );
        ggml_backend_tensor_set( tensor, &value, 0, 4 );
    }

    virtual void tensor_set( ggml_tensor * tensor, float value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == 1 );
        ggml_backend_tensor_set( tensor, &value, 0, 4 );
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<int32_t> & value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == value.size() );
        ggml_backend_tensor_set( tensor, value.data(), 0, 4 * value.size() );
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<float> & value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == value.size() );
        ggml_backend_tensor_set( tensor, value.data(), 0, 4 * value.size() );
    }

    ggml_tensor * constant( int32_t i32 ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, 1 );
            constants32.push_back({ tensor, i32 });
            return tensor;
        }
        assert(false);
        //return ggml_new_i32( ctx, i32 );
        return NULL;
    }

    ggml_tensor * constant( float f32 ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, 1 );
            constants32.push_back({ tensor, *(int32_t*)&f32 });
            return tensor;
        }
        assert(false);
        //return ggml_new_f32( ctx, f32 );
        return NULL;
    }

    ggml_tensor * input( NE ne, std::vector<int> & i32 ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_I32, 4, ne );
        size_t nelements = ggml_nelements( tensor );
        assert( nelements == i32.size() );
        int * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (int*)constant.data.data();
        } else {
            data = (int*)tensor->data;
        }
        memcpy( data, i32.data(), ggml_nbytes( tensor ) );
        return tensor;
    }

    ggml_tensor * input( NE ne, std::vector<float> & f32 ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_F32, 4, ne );
        size_t nelements = ggml_nelements( tensor );
        assert( nelements == f32.size() );
        float * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (float*)constant.data.data();
        } else {
            data = (float*)tensor->data;
        }
        memcpy( data, f32.data(), ggml_nbytes( tensor ) );
        return tensor;
    }

    ggml_tensor * fill( int count, float value ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, count );
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            float * data = (float*)constant.data.data();
            for (int64_t i = 0; i < count; i++) {
                data[i] = value;
            }
            return tensor;
        }
        assert(false);
    }

    ggml_tensor * fill( NE ne, float value ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_F32, 4, ne );
        auto nelements = ggml_nelements( tensor );
        float * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (float*)constant.data.data();
        } else {
            assert( tensor->data );
            data = (float*)tensor->data;
        }
        for (int64_t i = 0; i < nelements; i++) {
            data[i] = value;
        }
        return tensor;
    }

    ggml_tensor * fill( NE ne, int32_t value ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_I32, 4, ne );
        auto nelements = ggml_nelements( tensor );
        int32_t * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (int32_t*)constant.data.data();
        } else {
            assert( tensor->data );
            data = (int32_t*)tensor->data;
        }
        for (int64_t i = 0; i < nelements; i++) {
            data[i] = value;
        }
        return tensor;
    }

    // arange was not supported by all backends
    ggml_tensor * arange( float start, float stop, float step ) {
        if (backend) {
            const int64_t steps = (int64_t) ceilf((stop - start) / step);
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, steps );
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            float * data = (float*)constant.data.data();
            for (int64_t i = 0; i < steps; i++) {
                data[i] = start + step * i;
            }
            return tensor;
        }
        return ggml_arange( ctx, start, stop, step );
    }

    // probability density function
    ggml_tensor * exponential( NE ne, float lambd = 1.f ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_F32, 4, ne );
        if (backend) {
            exponentials.push_back({tensor, lambd});
        }
        return tensor;
    }

    void _exponential_compute() {
        for ( auto & exp : exponentials ) {
            auto nbytes = ggml_nbytes( exp.tensor );
            if ( scratch_data.size() < nbytes )
                scratch_data.resize( nbytes );
            auto data = (float*)scratch_data.data();
            int64_t n = ggml_nelements( exp.tensor );
#ifdef DISABLE_RAND
            for (int64_t i = 0; i < n; i++)
                data[i] = -logf(0.5) / exp.lambd;
#else
            for (int64_t i = 0; i < n; i++)
                data[i] = -logf(rand() / (float)RAND_MAX) / exp.lambd;
#endif
            ggml_backend_tensor_set( exp.tensor, data, 0, nbytes );
        }
    }

    std::string name;
    void set_name(std::string name) {
        this->name = name;
    }

    ggml_cgraph * get_graph() {
        if (!gf)
            gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 4, false );
        return gf;
    }

    void build_forward_expand( ggml_tensor * tensor ) {
        ggml_build_forward_expand( get_graph(), tensor );
    }

    bool debug_enable = false;
    struct debug_sum_t {
        const char * label;
        ggml_tensor * src;
    };
    std::vector<debug_sum_t> debug_sums;
    void debug( const char * label, ggml_tensor * src ) {
        if (!debug_enable)
            return;
        if (src->type != GGML_TYPE_F32)
            src = ggml_cast( ctx, src, GGML_TYPE_F32 );
        auto sum = ggml_sum( ctx, src );
        ggml_build_forward_expand( get_graph(), sum );
        debug_sums.push_back({label, sum});
    }
    void _debug_compute() {
        for (auto sum : debug_sums) {
            float fsum;
            ggml_backend_tensor_get( sum.src, &fsum, 0, 4 );
            printf( "%s %f\n", sum.label, fsum );
        }
    }

    // MEMLEDGER (scratch-graph breakdown, measurement-only): self-contained since
    // loader.h (which has the full memledger_log helpers) is included AFTER this
    // file in moshi.cpp — can't call those here, so this duplicates the minimal
    // pieces needed, same pattern used in personaplex.cpp for the same reason.
    static void memledger_graph_alloc_prescan( ggml_context * ctx, const std::string & label ) {
        struct entry_t { std::string name; std::string type; int64_t ne[4]; size_t bytes; };
        std::vector<entry_t> entries;
        size_t total = 0;
        for ( ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t) ) {
            if ( t->data != NULL || t->view_src != NULL ) continue; // already allocated / a view, not a fresh allocation
            size_t bytes = ggml_nbytes(t);
            total += bytes;
            entries.push_back({ ggml_get_name(t), ggml_type_name(t->type), {t->ne[0],t->ne[1],t->ne[2],t->ne[3]}, bytes });
        }
        std::sort( entries.begin(), entries.end(), []( const entry_t & a, const entry_t & b ) { return a.bytes > b.bytes; } );
        fprintf(stderr, "MEMLEDGER_GRAPH label=%s total_bytes=%zu total_MiB=%.2f n_tensors=%zu\n",
                label.c_str(), total, total / 1024.0 / 1024.0, entries.size());

        // Log ALL tensors (not just top-10) so bucket categorization is complete.
        for ( size_t i = 0; i < entries.size(); i++ ) {
            auto & e = entries[i];
            fprintf(stderr, "MEMLEDGER_GRAPH_ALL label=%s rank=%zu name=%-40s type=%-8s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] bytes=%10zu MiB=%8.4f\n",
                    label.c_str(), i, e.name.c_str(), e.type.c_str(), e.ne[0], e.ne[1], e.ne[2], e.ne[3], e.bytes, e.bytes / 1024.0 / 1024.0);
        }

        // Four-bucket category aggregation (task-specified):
        // (a) Attention KV copies: large tensors with ne[1] > 1 where shape is [head_dim, ctx, n_heads, 1]
        //     These are Q/K/V copies where ne[1] == context_length (dominant mass in large contexts).
        // (b) Attention score tensors: shape [ctx, T, n_heads, 1] where T is 1 or ctx (softmax input/output)
        // (c) Logits / vocab-sized: ne[0] is 32000 (text vocab) or 2048/1024 (audio codebook)
        // (d) Other: everything not in (a)-(c)
        size_t cat_a = 0, cat_b = 0, cat_c = 0, cat_d = 0;
        size_t cnt_a = 0, cnt_b = 0, cnt_c = 0, cnt_d = 0;

        // Find the dominant context length from largest tensors with ne[1] > 1 && ne[2] > 1
        int64_t ctx_len = 0;
        for ( auto & e : entries ) {
            if ( e.ne[1] > 1 && e.ne[2] > 1 && e.ne[3] == 1 && e.bytes > 1024*1024 ) {
                if ( e.ne[1] > ctx_len ) ctx_len = e.ne[1];
            }
        }

        for ( auto & e : entries ) {
            bool in_c = ( e.ne[0] == 32000 || e.ne[0] == 2048 || e.ne[0] == 1024 );
            // KV copy: [head_dim, ctx_len, n_heads, 1] — ne[1] == ctx_len, ne[2] > 1, f32/f16
            bool in_a = ( !in_c && ctx_len > 0 && e.ne[1] == ctx_len && e.ne[2] > 1 && e.ne[3] <= 1 );
            // Attn score: [ctx_len, small, n_heads, 1] — ne[0] == ctx_len, n_heads > 1
            bool in_b = ( !in_c && !in_a && ctx_len > 0 && e.ne[0] == ctx_len && e.ne[2] > 1 );
            if ( in_c ) { cat_c += e.bytes; cnt_c++; }
            else if ( in_a ) { cat_a += e.bytes; cnt_a++; }
            else if ( in_b ) { cat_b += e.bytes; cnt_b++; }
            else            { cat_d += e.bytes; cnt_d++; }
        }
        fprintf(stderr, "MEMLEDGER_GRAPH_CAT label=%s ctx_len_detected=%" PRId64 " "
                "catA_attn_kv_MiB=%.2f(n=%zu) catB_attn_score_MiB=%.2f(n=%zu) "
                "catC_logits_MiB=%.2f(n=%zu) catD_other_MiB=%.2f(n=%zu) "
                "check_total_MiB=%.2f\n",
                label.c_str(), ctx_len,
                cat_a/1024.0/1024.0, cnt_a,
                cat_b/1024.0/1024.0, cnt_b,
                cat_c/1024.0/1024.0, cnt_c,
                cat_d/1024.0/1024.0, cnt_d,
                (cat_a+cat_b+cat_c+cat_d)/1024.0/1024.0);
        fflush(stderr);
    }


    // Reuse-mode replacement for ggml_backend_alloc_ctx_tensors: identical
    // size formula and placement loop as upstream alloc_tensor_range /
    // ggml_backend_alloc_ctx_tensors_from_buft (ggml-alloc.c), but the
    // backing buffer is retained across clear() and only grows.
    ggml_backend_buffer_t alloc_ctx_tensors_reused() {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type( backend );
        size_t alignment = ggml_backend_buft_get_alignment( buft );
        size_t needed = 0;
        for ( ggml_tensor * t = ggml_get_first_tensor( ctx ); t != NULL; t = ggml_get_next_tensor( ctx, t ) ) {
            if ( t->data == NULL && t->view_src == NULL )
                needed += GGML_PAD( ggml_backend_buft_get_alloc_size( buft, t ), alignment );
        }
        if ( needed == 0 )
            needed = alignment; // degenerate graph: keep a valid buffer anyway
        if ( needed > reuse_buf_size ) {
            if ( reuse_buf ) {
                fprintf( stderr, "SCRATCH_REUSE grow name=%s old_B=%zu new_B=%zu\n",
                    name.size() ? name.c_str() : "(unnamed)", reuse_buf_size, needed );
                fflush( stderr );
                ggml_backend_buffer_free( reuse_buf ); // warmup-only cudaFree
                reuse_buf = NULL;
                reuse_buf_size = 0;
            }
            reuse_buf = ggml_backend_buft_alloc_buffer( buft, needed );
            if ( ! reuse_buf ) {
                fprintf( stderr, "FATAL: SCRATCH_REUSE failed to allocate %zu bytes\n", needed );
                fflush( stderr );
                exit( 1 );
            }
            reuse_buf_size = needed;
        }
        ggml_backend_buffer_reset( reuse_buf );
        ggml_tallocr tallocr = ggml_tallocr_new( reuse_buf );
        for ( ggml_tensor * t = ggml_get_first_tensor( ctx ); t != NULL; t = ggml_get_next_tensor( ctx, t ) ) {
            enum ggml_status status = GGML_STATUS_SUCCESS;
            if ( t->data == NULL ) {
                if ( t->view_src == NULL ) {
                    status = ggml_tallocr_alloc( &tallocr, t );
                } else if ( t->buffer == NULL ) {
                    status = ggml_backend_view_init( t );
                }
            } else if ( t->view_src != NULL && t->buffer == NULL ) {
                status = ggml_backend_view_init( t );
            }
            if ( status != GGML_STATUS_SUCCESS ) {
                fprintf( stderr, "FATAL: SCRATCH_REUSE failed to place tensor %s\n", ggml_get_name( t ) );
                fflush( stderr );
                exit( 1 );
            }
        }
        return reuse_buf;
    }

    void alloc() {
        assert( ! buffer );
        memledger_graph_alloc_prescan( ctx, name.size() ? name : "(unnamed)" );
        if ( reuse_buffer && backend ) {
            buffer = alloc_ctx_tensors_reused();
        } else {
            buffer = ggml_backend_alloc_ctx_tensors( ctx, backend );
        }
        assert( buffer );
        for (auto load : loaders) {
            load.src->init( load.safetensor, load.tensor, backend );
        }
        for (auto i32 : constants32) {
            ggml_backend_tensor_set(i32.tensor, &i32.value, 0, ggml_nbytes(i32.tensor));
        }
        for (auto constant : constants) {
            ggml_backend_tensor_set(constant.tensor, constant.data.data(), 0, ggml_nbytes(constant.tensor));
        }
        for (auto convert : input_converts) {
            ggml_backend_tensor_set(convert.dst, convert.src->data, 0, ggml_nbytes(convert.dst));
        }
    }

    void compute() {
        _exponential_compute();
        assert( backend );
        if (name.size()) {CAPTURE(name, gf);}
        ggml_backend_graph_compute( backend, gf );
        _debug_compute();
    }
};

class ScratchContext : public GraphContext {
    public:
    ScratchContext( size_t mb, ggml_backend * backend = NULL )
        : GraphContext( mb, backend ) {}

    ggml_tensor * load( SafeTensorFile * src, safetensor_t * safetensor ) {
        auto tensor = safetensor_alloc( ctx, safetensor );
        if (backend) {
            loaders.push_back({ src, safetensor, tensor });
            return tensor;
        }
        src->init(safetensor, tensor);
        return tensor;
    }

    virtual void tensor_set( ggml_tensor * tensor, int32_t value ) {
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == 1 );
        constants32.push_back({ tensor, value });
    }

    virtual void tensor_set( ggml_tensor * tensor, float value ) {
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == 1 );
            constants32.push_back({ tensor, *(int32_t*)&value });
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<int32_t> & value ) {
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == value.size() );
        constants.push_back({tensor});
        auto & constant = constants.back();
        constant.data.resize( ggml_nbytes( tensor ) );
        memcpy( constant.data.data(), value.data(), ggml_nbytes( tensor ) );
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<float> & value ) {
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == value.size() );
        constants.push_back({tensor});
        auto & constant = constants.back();
        constant.data.resize( ggml_nbytes( tensor ) );
        memcpy( constant.data.data(), value.data(), ggml_nbytes( tensor ) );
    }

    void build_forward_expand( ggml_tensor * tensor ) {
        assert( tensor->op == GGML_OP_CPY ); // scratch context will not store data
        ggml_build_forward_expand( get_graph(), tensor );
    }

    // this should only be used for cpu to gpu copies, otherwise use ggml_cpy
    void build_forward_expand( ggml_tensor * tensor, ggml_tensor * copy_tensor ) {
        assert( copy_tensor->buffer ); // copy to a backend
        assert( ggml_nbytes(tensor) == ggml_nbytes(copy_tensor) );
        ggml_build_forward_expand( get_graph(), tensor );
        backend_copies.push_back({ tensor, copy_tensor });
    }

    void build_forward_expand( ggml_tensor * tensor, int32_t * dst ) {
        ggml_build_forward_expand( get_graph(), tensor );
        copies.push_back({ tensor, dst });
    }

    void build_forward_expand( ggml_tensor * tensor, float * dst ) {
        ggml_build_forward_expand( get_graph(), tensor );
        copies.push_back({ tensor, dst });
    }

    void clear() {
        debug_sums.clear();
        backend_copies.clear();
        copies.clear();
        //tensor_copies.clear();
        input_converts.clear();
        exponentials.clear();
        constants.clear();
        constants32.clear();
        loaders.clear();
        if ( buffer != reuse_buf ) // reuse mode keeps the backing buffer (no cudaFree)
            ggml_backend_buffer_free( buffer );
        buffer = NULL;
        ggml_reset(ctx);
        gf = NULL;
        name = "";
    }

    void compute() {
        assert( backend );
        alloc();
        _exponential_compute();

        // compute
        if (name.size()) {CAPTURE(name, gf);}
        ggml_backend_graph_compute( backend, gf );

        // debug
        _debug_compute();
        debug_enable = false;
        // copy results
        for (auto copy : copies) {
            size_t nbytes = ggml_nbytes(copy.src);
            ggml_backend_tensor_get(copy.src, copy.dst, 0, nbytes);
        }
        for (auto copy : backend_copies) {
            int64_t nbytes = ggml_nbytes( copy.dst );
            std::vector<uint8_t> buf( nbytes );
            ggml_backend_tensor_get( copy.src, buf.data(), 0, nbytes );
            ggml_backend_tensor_set( copy.dst, buf.data(), 0, nbytes );
        }
        // cleanup
        clear();
    }
};

class StateContext {
    public:
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct state_tensor_t {
        ggml_tensor ** ptensor;
        ggml_type type;
        NE ne;
        std::vector<uint8_t> data;
    };
    std::vector<state_tensor_t> states;

    ggml_type kv_cache_type = GGML_TYPE_BF16;
    ggml_tensor * hadamard64 = NULL;
    ggml_tensor * hadamard128 = NULL;

    void add_hadamard( int head_dim ) {
        if (head_dim == 64) {
            if (hadamard64 != NULL) return;
            NE ne = { 64, 64, 1, 1 };
            std::vector<float> hadamard_data(64 * 64);
            float scale = 1.0f / sqrtf(64.0f);
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 64; j++) {
                    int val = i & j;
                    int count = 0;
                    while (val) {
                        count ^= (val & 1);
                        val >>= 1;
                    }
                    hadamard_data[i * 64 + j] = (count == 0 ? 1.0f : -1.0f) * scale;
                }
            }
            new_tensor( ne, hadamard_data, &hadamard64 );
        } else if (head_dim == 128) {
            if (hadamard128 != NULL) return;
            NE ne = { 128, 128, 1, 1 };
            std::vector<float> hadamard_data(128 * 128);
            float scale = 1.0f / sqrtf(128.0f);
            for (int i = 0; i < 128; i++) {
                for (int j = 0; j < 128; j++) {
                    int val = i & j;
                    int count = 0;
                    while (val) {
                        count ^= (val & 1);
                        val >>= 1;
                    }
                    hadamard_data[i * 128 + j] = (count == 0 ? 1.0f : -1.0f) * scale;
                }
            }
            new_tensor( ne, hadamard_data, &hadamard128 );
        }
    }

    StateContext( ggml_backend * backend = NULL ) {
        ctx = NULL;
        buffer = NULL;
        this->backend = backend;
    }

    ~StateContext() {
        if (buffer)
            ggml_backend_buffer_free( buffer );
        if (ctx)
            ggml_free( ctx );
    }

    void new_tensor( NE ne, ggml_type type, ggml_tensor ** ptensor ) {
        // will be initialized later
        states.push_back({ ptensor, type });
        auto & state = states.back();
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
        }
    }
    
    void new_tensor( NE ne, std::vector<float> & src, ggml_tensor ** ptensor ) {
        states.push_back({ ptensor, GGML_TYPE_F32 });
        auto & state = states.back();
        int64_t nelements = 1;
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
            nelements *= ne[i];
        }
        assert( nelements == (int64_t)src.size() );
        state.data.resize( nelements * 4 );
        float * dst = (float*)state.data.data();
        for ( int i = 0; i < nelements; i++)
            dst[i] = src[i];
        *ptensor = NULL;
    }

    void fill16( NE ne, ggml_type type, int16_t value, ggml_tensor ** ptensor ) {
        assert( type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 );
        states.push_back({ ptensor, type });
        auto & state = states.back();
        int64_t nelements = 1;
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
            nelements *= ne[i];
        }
        state.data.resize( nelements * 2 );
        int16_t * data = (int16_t*)state.data.data();
        for ( int i = 0; i < nelements; i++)
            data[i] = value;
        *ptensor = NULL;
    }

    void fill_quant( NE ne, ggml_type type, ggml_tensor ** ptensor ) {
        states.push_back({ ptensor, type });
        auto & state = states.back();
        int64_t nelements = 1;
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
            nelements *= ne[i];
        }
        size_t nbytes = ggml_row_size( type, nelements );
        state.data.resize( nbytes, 0 );
        *ptensor = NULL;
    }

    void fill32( NE ne, ggml_type type, int32_t value, ggml_tensor ** ptensor ) {
        assert( type == GGML_TYPE_F32 || type == GGML_TYPE_I32 );
        states.push_back({ ptensor, type });
        auto & state = states.back();
        int64_t nelements = 1;
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
            nelements *= ne[i];
        }
        state.data.resize( nelements * 4 );
        int32_t * data = (int32_t*)state.data.data();
        for ( int i = 0; i < nelements; i++)
            data[i] = value;
        *ptensor = NULL;
    }

    void fill( NE ne, float value, ggml_tensor ** ptensor ) {
        fill32( ne, GGML_TYPE_F32, *(int32_t*)&value, ptensor );
    }

    void fill( NE ne, int32_t value, ggml_tensor ** ptensor ) {
        fill32( ne, GGML_TYPE_I32, value, ptensor );
    }

    void alloc() {
        assert( ctx == NULL ); // can only alloc once!
        size_t nbytes = ggml_tensor_overhead() * states.size();
        if (backend) {
            ctx = ggml_init({ nbytes, NULL, true });
        } else {
            for ( auto state : states )
                nbytes += state.data.size();
            ctx = ggml_init({ nbytes, NULL, false });
        }
        for ( auto state : states )
            *state.ptensor = ggml_new_tensor( ctx, state.type, 4, state.ne );
        if (backend)
            buffer = ggml_backend_alloc_ctx_tensors( ctx, backend );
    }

    void init() {
        if (backend) {
            for ( auto state : states ) {
                if ( ! state.data.size() )
                    continue;
                ggml_backend_tensor_set( *state.ptensor, state.data.data(), 0,
                    state.data.size() );
            }
        } else {
            for ( auto state : states ) {
                if ( ! state.data.size() )
                    continue;
                memcpy( (*state.ptensor)->data, state.data.data(), state.data.size() );
            }
        }
    }
};

