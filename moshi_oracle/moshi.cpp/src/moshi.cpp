
#include <assert.h>
#include <math.h>
#if !defined(_WIN32) && !defined(__APPLE__)
#include <malloc.h> // malloc_trim (V2 fix, moshi_lm_load)
#endif

#include <iostream>
#include <chrono> // STEP 1 frame-phase timing instrumentation

#define MOSHI_BUILD
#include <moshi/moshi.h>

// for src/context.h
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef ENABLE_REPLAY
#include "replay.h"
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#else
#ifdef ENABLE_CAPTURE
#include "src/ggml_cap.h"
#else
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#endif
#endif
#define ONCE(code) {static bool once=false; if (!once) {{code;}; once=true;}}
#define ON_NTH(nth, code) {static int count=0; if (count++ == (nth)) {code;}}

#include "../src/config.h"
#include "../src/context.h"
#include "../src/wav.h"
#include "../src/loader.h"
#include "../src/torch.h"
#include "../src/moshi/modules/transformer.h"
#include "../src/moshi/utils/sampling.h"
#include "../src/moshi/models/lm_utils.h"
#include "../src/moshi/models/lm.h"
#include "../src/moshi/quantization/core_vq.h"
#include "../src/moshi/quantization/vq.h"
#include "../src/moshi/modules/conv.h"
#include "../src/moshi/modules/seanet.h"
#include "../src/moshi/models/compression.h"
#include "../src/moshi/models/lm_default.h"
#include "../src/moshi/models/tts.h"

moshi_phase_timing_t g_moshi_phase_timing = {};
void moshi_phase_timing_reset() { g_moshi_phase_timing = moshi_phase_timing_t{}; }

struct moshi_context_t {
    ggml_backend * backend;
    ggml_backend * backend_cpu;
    own_ptr<ScratchContext> scratch_cpu;
    own_ptr<ScratchContext> scratch;
};

struct mimi_codec_t {
    int n_q;
    moshi_context_t * moshi;
    own_ptr<moshi_mimi_t> mimi;
    own_ptr<WeightLoader> mimi_weights;
};

struct mimi_encode_context_t {
    mimi_codec_t * codec;
    own_ptr<StateContext> state_ctx;
    own_ptr<moshi_mimi_state_t> states;
    std::vector<int> tokens;
    std::vector<float> frame;
};

struct mimi_decode_context_t {
    mimi_codec_t * codec;
    own_ptr<StateContext> state_ctx;
    own_ptr<moshi_mimi_state_t> states;
    std::vector<int> tokens;
    std::vector<float> frame;
};

struct tokenizer_t {
    sentencepiece::SentencePieceProcessor sp;
    int padding_between = 1;
    bool insert_bos = true;

    std::string tail;

    enum {
        FIND_START,
        FIND_END,
        CHECK_WORD,
        TOKENIZE,
    } state = FIND_START;
    int offset = 0, start_offset = 0, end_offset = 0;

    int found_break = 0;
    float time;
    std::deque<std::string> words;
};

// MARK: Moshi Context

moshi_context_t * moshi_alloc( ggml_backend * backend, ggml_backend * backend_cpu ) {
    printf("DEBUG: moshi_alloc starting\n"); fflush(stdout);
    assert( backend );
    assert( backend_cpu );
    auto dev_cpu = ggml_backend_get_device( backend_cpu );
    printf("DEBUG: dev_cpu=%p\n", (void*)dev_cpu); fflush(stdout);
    assert ( ggml_backend_dev_type( dev_cpu ) == GGML_BACKEND_DEVICE_TYPE_CPU );
    printf("DEBUG: dev_cpu type check passed\n"); fflush(stdout);

    auto moshi = new moshi_context_t;
    moshi->backend_cpu = backend_cpu;
    moshi->backend = backend;
    printf("DEBUG: allocating scratch_cpu\n"); fflush(stdout);
    moshi->scratch_cpu = new ScratchContext( 256, backend_cpu );
    printf("DEBUG: allocating scratch\n"); fflush(stdout);
    moshi->scratch = new ScratchContext( 256, backend );
    printf("DEBUG: moshi_alloc ending\n"); fflush(stdout);
    return moshi;
}

void unref( moshi_context_t * moshi ) {
    delete moshi;
}

// MARK: Mimi Codec

static void mimi_alloc( mimi_codec_t * codec, moshi_context_t * moshi,
        const char * filename, int n_q ) {
    auto mimi = moshi_mimi_alloc_default( n_q );

    std::string filepath = filename;
    WeightLoader * mimi_weights;
    if ( filepath.ends_with( ".safetensors" ) ) {
        mimi_weights = WeightLoader::from_safetensor( filename,
            moshi->scratch_cpu, moshi->backend );
        if ( ! mimi_weights ) {
            fprintf(stderr, "error: mimi weights not found\n" );
            exit(1);
        }
    } else {
        mimi_weights = WeightLoader::from_gguf( filename,
            moshi->scratch_cpu, moshi->backend );
        if ( ! mimi_weights ) {
            fprintf(stderr, "error: mimi weights not found\n" );
            exit(1);
        }
        mimi_weights->load_gguf();
    }

    get_weights( mimi_weights, "mimi.quantizer.", mimi->quantizer );
    get_weights( mimi_weights, "mimi.upsample.convtr.", mimi->upsample );
    get_weights( mimi_weights, "mimi.decoder_transformer.transformer.", mimi->decoder_transformer );
    get_weights( mimi_weights, "mimi.decoder.", mimi->decoder );
    if ( mimi->encoder ) {
        get_weights( mimi_weights, "mimi.downsample.conv.", mimi->downsample );
        get_weights( mimi_weights, "mimi.encoder_transformer.transformer.", mimi->encoder_transformer );
        get_weights( mimi_weights, "mimi.encoder.", mimi->encoder );
    }
    if ( ! mimi_weights->is_gguf )
        mimi_weights->load();

    codec->n_q = n_q;
    codec->moshi = moshi;
    codec->mimi = mimi;
    codec->mimi_weights = mimi_weights;
}

mimi_codec_t * mimi_alloc( moshi_context_t * moshi, const char * filename, int n_q ) {
    auto codec = new mimi_codec_t;
    mimi_alloc( codec, moshi, filename, n_q );
    return codec;
}

void unref( mimi_codec_t * codec ) {
    delete codec;
}

float mimi_frame_rate( mimi_codec_t * codec ) {
    return codec->mimi->frame_rate;
}

int mimi_frame_size( mimi_codec_t * codec ) {
    return 1920;
}

void mimi_save_gguf( mimi_codec_t * codec, const char * filepath ) {
    codec->mimi_weights->save_gguf( filepath );
}

// MARK: Mimi Encode

static void mimi_encode_alloc_context( mimi_encode_context_t * context, mimi_codec_t * codec ) {
    auto state_ctx = new StateContext( codec->moshi->backend );
    auto mimi_states = moshi_mimi_encoder_states( state_ctx, codec->mimi );
    int frame_size = mimi_frame_size( codec );

    state_ctx->alloc();
    state_ctx->init();
    init( codec->moshi->scratch, mimi_states, codec->mimi );

    context->codec = codec;
    context->state_ctx = state_ctx;
    context->states = mimi_states;
    context->frame.resize( frame_size );
}

mimi_encode_context_t * mimi_encode_alloc_context( mimi_codec_t * codec ) {
    auto context = new mimi_encode_context_t;
    mimi_encode_alloc_context( context, codec );
    return context;
}

void unref( mimi_encode_context_t * context ) {
    delete context;
}

void mimi_encode_reset( mimi_encode_context_t * context ) {
    auto scratch = context->codec->moshi->scratch.ptr;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    init( scratch, states, mimi );
}

void mimi_encode_send( mimi_encode_context_t * context, float * frame ) {
    memcpy( context->frame.data(), frame, context->frame.size() * 4 );
}

void mimi_encode_receive( mimi_encode_context_t * context, int16_t * tokens ) {
    auto & ctx = *context->codec->moshi->scratch;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;

    mimi_encode(
        ctx,
        mimi,
        states,
        context->frame,
        context->tokens
    );

    for ( int i = 0; i < context->tokens.size(); i++ )
        tokens[i] = context->tokens[i];
}

// MARK: Mimi Decode

static void mimi_decode_alloc_context( mimi_decode_context_t * context, mimi_codec_t * codec ) {
    auto state_ctx = new StateContext( codec->moshi->backend );
    NE upsample_ne = {1, 512, 1, 1};
    NE decoder_ne = {2, 512, 1, 1};
    auto mimi_states = moshi_mimi_states( state_ctx, codec->mimi, upsample_ne, decoder_ne );
    int frame_size = mimi_frame_size( codec );

    state_ctx->alloc();
    state_ctx->init();
    init( codec->moshi->scratch, mimi_states, codec->mimi );

    context->codec = codec;
    context->state_ctx = state_ctx;
    context->states = mimi_states;
    context->tokens.resize( codec->n_q );
    context->frame.resize( frame_size );
}

mimi_decode_context_t * mimi_decode_alloc_context( mimi_codec_t * codec ) {
    auto context = new mimi_decode_context_t;
    mimi_decode_alloc_context( context, codec );
    return context;
}

void unref( mimi_decode_context_t * context ) {
    delete context;
}

void mimi_decode_reset( mimi_decode_context_t * context ) {
    auto scratch = context->codec->moshi->scratch.ptr;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    init( scratch, states, mimi );
}

void mimi_decode_send( mimi_decode_context_t * context, int16_t * tokens ) {
    auto n_q = context->codec->n_q;
    for ( int i = 0; i < n_q; i++ )
        context->tokens[i] = tokens[i];
}

void mimi_decode_receive( mimi_decode_context_t * context, float * frame ) {
    auto & ctx = *context->codec->moshi->scratch;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    mimi_decode(
        ctx,
        mimi,
        states,
        context->tokens,
        context->frame
    );
    if ( frame )
        memcpy( frame, context->frame.data(), context->frame.size() * 4 );
}

// MARK: Voice Condition

static void voice_condition( voice_t * voice,
        conditioners_t * cond,
        ggml_tensor * speaker_wavs,
        moshi_context_t * moshi
) {
    ScratchContext & b_voice_ctx = *moshi->scratch.ptr;
    auto backend = moshi->backend;

    // cfg {'1.0': 0, '1.5': 1, '2.0': 2, '2.5': 3, '3.0': 4, '3.5': 5, '4.0': 6}
    auto cfg_val = b_voice_ctx.constant( 2 );
    auto cfg_emb = ggml_get_rows(b_voice_ctx, cond->cfg_embed_weight, cfg_val);
    auto cfg_cond = ggml_mul_mat(b_voice_ctx, cond->cfg_output_proj_weight, cfg_emb);

    // control {'ok': 0}
    auto control_val = b_voice_ctx.constant( 0 );
    auto control_emb = ggml_get_rows(b_voice_ctx, cond->control_embed_weight, control_val);
    auto control_cond = ggml_mul_mat(b_voice_ctx, cond->control_output_proj_weight, control_emb);

    auto condition_sum = ggml_add(b_voice_ctx, cfg_cond, control_cond);

    // speaker_wavs
    auto speaker_wavs_a = ggml_cont(b_voice_ctx, ggml_transpose(b_voice_ctx, speaker_wavs));
    auto speaker_wavs_b = ggml_mul_mat(b_voice_ctx, cond->speaker_wavs_output_proj_weight,
        speaker_wavs_a);
    //
    //auto speaker_wavs_cond = ggml_new_tensor_2d(b_voice_ctx, GGML_TYPE_F32,
    //    speaker_wavs_b->ne[0], speaker_wavs_b->ne[1] * 5);
    // fill with learnt padding
    //speaker_wavs_cond = ggml_scale_inplace(b_voice_ctx, speaker_wavs_cond, 0);
    //speaker_wavs_cond = ggml_add_inplace(b_voice_ctx, speaker_wavs_cond,
    //    speaker_wavs_learnt_padding);
    auto speaker_wavs_cond = ggml_repeat_4d( b_voice_ctx,
        cond->speaker_wavs_learnt_padding,
        speaker_wavs_b->ne[0], speaker_wavs_b->ne[1] * 5, 1, 1 );
    // set first speaker
    auto speaker_0 = ggml_view_2d(b_voice_ctx, speaker_wavs_cond,
        speaker_wavs_b->ne[0], speaker_wavs_b->ne[1],
        speaker_wavs_cond->nb[1], 0);
    speaker_0 = ggml_cpy(b_voice_ctx, speaker_wavs_b, speaker_0);
    speaker_wavs_cond = ggml_view_2d(b_voice_ctx, speaker_0,
        speaker_wavs_cond->ne[0], speaker_wavs_cond->ne[1],
        speaker_wavs_cond->nb[1], 0);

    float cross_attention_pos_emb_scale = 1;
    //auto positions = ggml_arange(b_voice_ctx, 0, speaker_wavs_cond->ne[1], 1);
    auto positions = b_voice_ctx.arange(0, (float) speaker_wavs_cond->ne[1], 1);
    auto pos_emb = ggml_timestep_embedding(b_voice_ctx, positions, 2048, 10000);
    auto condition_cross = ggml_add(b_voice_ctx, speaker_wavs_cond, ggml_scale(b_voice_ctx, pos_emb, cross_attention_pos_emb_scale));

    size_t mem_size = 2 * ggml_tensor_overhead();
    bool no_alloc = true;
    if (! backend ) {
        mem_size += ggml_nbytes( condition_sum );
        mem_size += ggml_nbytes( condition_cross );
        no_alloc = false;
    }
    voice->ctx = ggml_init({
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ no_alloc,
    });
    voice->sum = ggml_dup_tensor( voice->ctx, condition_sum );
    voice->cross = ggml_dup_tensor( voice->ctx, condition_cross );
    ggml_set_name( voice->sum, "sum" );
    ggml_set_name( voice->cross, "cross" );
    voice->buffer = ggml_backend_alloc_ctx_tensors( voice->ctx, backend );
    b_voice_ctx.build_forward_expand( condition_sum, voice->sum );
    b_voice_ctx.build_forward_expand( condition_cross, voice->cross );
    ONCE( b_voice_ctx.set_name("voice") );
    b_voice_ctx.compute();
}

// MARK: Tokenizer

tokenizer_t * tokenizer_alloc( const char * filepath, bool insert_bos ) {
    auto tok = new tokenizer_t;
    tok->sp.Load( filepath );
    tok->insert_bos = insert_bos;
    return tok;
}

void unref( tokenizer_t * tok ) {
    delete tok;
}

bool tokenizer_empty( tokenizer_t * tok ) {
    return ! tok->tail.size();
}

static int tokenizer_tokenize( tokenizer_t * tok, std::string & word, Entry * entry ) {
    const int text_bos_token = 1;

    std::vector<int> tokens;
    tok->sp.Encode(word, &tokens);
    if ( tok->insert_bos ) {
        tok->insert_bos = false;
        std::vector<int> new_tokens(1 + tokens.size());
        new_tokens[0] = text_bos_token;
        for (size_t i = 0; i < tokens.size(); i++)
            new_tokens[i+1] = tokens[i];
        tokens.swap( new_tokens );
    }
    int padding = 0;
    if ( tok->padding_between > 0 ) {
        padding = tok->padding_between + (int) tokens.size() - 1;
        if ( padding < 0 )
            padding = 0;
    }
    entry->tokens.swap( tokens );
    entry->text = word;
    entry->padding = padding;
    entry->time = ggml_time_ms();
    //queue_push( tok->outlet );
    //entries.push_back( Entry( tokens, word, padding ) );
    return 0;
}

static int tokenizer_break( tokenizer_t * tok, Entry * entry ) {
    const int tok_pad_id = 3;
    const float frame_rate = 12.5f;

    float time = tok->time;
    if ( time <= 0.f)
        return 0;

    if ( time > 10.f )
        time = 10.f;
    int npad = (int)( time * frame_rate );
    if ( npad == 0 )
        npad = 1;

    entry->text = "";
    while ( tok->words.size() ) {
        entry->text += tok->words.front() + " ";
        tok->words.pop_front();
    }
    entry->time = ggml_time_ms();
#ifdef OLD_WAY
    // tts.py: script_to_entries
    entry->tokens.reset();
    entry->padding = npad;
#else
    // tts_preprocess.rs: Tokenizer::  preprocess
    std::vector<int> tokens={ tok_pad_id, npad };
    entry->tokens.swap( tokens );
    entry->padding = 0;
#endif
    return 1;
}

static int tokenizer_break_time( tokenizer_t * tok ) {
    std::string & word = tok->words.back();
    if ( ! word.starts_with( "time=\"" ) )
        return 0;

    const_str_t s = { word.c_str(), (int)word.size() };
    int offset = 6;

    //int end_offset = str_find_not_of( s, offset, "0123456789." );
	char *end;
	const char *start = s.s + offset;
	tok->time = (float)strtod(start, &end);
	offset = (int) (end - start) + offset;

    int remaining = s.length - offset;
    if ( remaining < 2 || s.s[offset] != 's' || s.s[offset+1] != '"' )
        return 0;
    if ( remaining == 3 )
        return 0; // unexpected extra character

    if ( remaining >= 4 ) {
        if ( s.s[offset+2] != '/' || s.s[offset+3] != '>' )
            return 0; // expected "/>"
        return 3; // completed break
    }
    return 2; // incomplete break
}

int tokenizer_send( tokenizer_t * tok, std::string text ) {
    if ( text.size() == 0 ) {
        if ( tok->words.size() ) {
            tok->state = tokenizer_t::TOKENIZE;
        }
        // flush tail
        if ( tok->tail.size() )
            tok->tail += " ";
        return 0;
    }

    tok->tail += text;
    return 0;
}

int tokenizer_receive( tokenizer_t * tok, Entry * entry ) {
    const_str_t s = { tok->tail.c_str(), (int)tok->tail.size() };

    int found = 0;
    int & offset = tok->offset;
    int & start_offset = tok->start_offset;
    int & end_offset = tok->end_offset;

    while( true ) {
        switch( tok->state ) {
        case tokenizer_t::FIND_START:
            offset = str_skip_whitespaces( s, end_offset );
            if ( offset == s.length ) { // tail was all white space chars
                tok->tail = "";
                offset = 0;
                end_offset = 0;
                tok->state = tokenizer_t::FIND_START;
                return found;
            }
            start_offset = offset;
        case tokenizer_t::FIND_END:
            end_offset = str_find_whitespaces( s, offset );
            if ( end_offset == s.length ) { // maybe partial word
                offset = end_offset - start_offset;
                if ( start_offset != 0 ) {
                    tok->tail = tok->tail.substr( start_offset );
                    start_offset = 0;
                }
                tok->state = tokenizer_t::FIND_END;
                return found;
            }
            tok->words.push_back({});
            tok->words.back().assign( s.s + start_offset, end_offset - start_offset );
            if ( found ) {
                tok->state = tokenizer_t::CHECK_WORD;
                return found + 1;
            }
        case tokenizer_t::CHECK_WORD:
            if ( tok->found_break == 1 ) { // found break, check for time
                tok->found_break = tokenizer_break_time( tok );
                if ( tok->found_break == 0 ) { // no time property
                    tok->state = tokenizer_t::TOKENIZE;
                    break;
                }
                if ( tok->found_break == 2 ) { // need another word
                    tok->state = tokenizer_t::FIND_START;
                    break;
                }
                tok->state = tokenizer_t::CHECK_WORD;
                break;
            } else if ( tok->found_break == 2 ) {
                if ( tok->words.back().starts_with( "/>" ) ) {
                    tok->found_break = 3;
                    tok->state = tokenizer_t::CHECK_WORD;
                    break;
                } else {
                    tok->found_break = 0;
                    tok->state = tokenizer_t::TOKENIZE;
                    break;
                }
            } else if ( tok->found_break == 3 ) {
                int front_size = (int) tok->words.front().size();
                if ( front_size > 6 ) {
                    // must have token before break
                    auto word = tok->words.front().substr( 0, front_size - 6 );
                    tok->words.front() = tok->words.front().substr( front_size - 6 );
                    tok->words.push_front(word);
                    tok->state = tokenizer_t::TOKENIZE;
                    break;
                }
                int back_size = (int) tok->words.back().size();
                int tail = (int) tok->words.back().find( "/>" ) + 2;
                std::string word;
                if ( back_size != tail ) {
                    word = tok->words.back().substr( tail );
                    tok->words.back() = tok->words.back().substr( 0, tail );
                }
                tokenizer_break( tok, entry ); // consumes words
                tok->found_break = 0;
                if ( word.size() ) {
                    tok->words.push_back( word );
                    tok->state = tokenizer_t::CHECK_WORD;
                    return 2;
                }
                found = 1;
                tok->state = tokenizer_t::FIND_START;
                break;
            } else if ( tok->words.front().ends_with( "<break" ) ) {
                tok->found_break = 1;
                tok->state = tokenizer_t::FIND_START;
                break;
            }
        case tokenizer_t::TOKENIZE:
            tokenizer_tokenize( tok, tok->words.front(), entry );
            tok->words.pop_front();
            if ( tok->words.size() ) {
                tok->state = tokenizer_t::CHECK_WORD;
                return 2;
            }
            found = 1;
            tok->state = tokenizer_t::FIND_START;
        }
    }

    return 0;
}

std::string tokenizer_id_to_piece( tokenizer_t * tok, int token ) {
    return tok->sp.IdToPiece( token );
}

// MARK: LM

struct moshi_lm_t {
    std::string filepath;
    bool uses_cross;
    moshi_lmmodel_t * model;
    own_ptr<WeightLoader> weights;
    own_ptr<conditioners_t> cond;
    bool second_stream_ahead;
    ggml_type kv_cache_type = GGML_TYPE_BF16;
};

moshi_lm_t * moshi_lm_from_files(
    moshi_context_t * moshi,
    moshi_config_t * config,
    const char * filepath
) {
    printf("DEBUG: moshi_lm_from_files starting, filepath=%s\n", filepath); fflush(stdout);
    auto lm = new moshi_lm_t;

    lm->filepath = filepath;

    if ( lm->filepath.ends_with( ".safetensors" ) ) {
        printf("DEBUG: loading safetensors weights\n"); fflush(stdout);
        lm->weights = WeightLoader::from_safetensor( filepath, moshi->scratch_cpu, moshi->backend );
        if ( ! lm->weights ) {
            printf("DEBUG: safetensors weights load failed\n"); fflush(stdout);
            return NULL;
        }
    } else {
        printf("DEBUG: loading gguf weights\n"); fflush(stdout);
        lm->weights = WeightLoader::from_gguf( filepath, moshi->scratch_cpu, moshi->backend );
        if ( ! lm->weights ) {
            printf("DEBUG: gguf weights load failed\n"); fflush(stdout);
            return NULL;
        }
        // MEMLEDGER: file is open and header/tensor-info parsed (no_alloc=true in from_gguf),
        // but zero tensor data has been uploaded yet — this is the pre-upload baseline.
        memledger_log("after_gguf_parse", "(file_opened_no_alloc)", "-", 0, 0, 0, moshi->backend);
    }
    printf("DEBUG: weights loaded successfully\n"); fflush(stdout);

    printf("DEBUG: calling moshi_lmmodel_alloc_default\n"); fflush(stdout);
    lm->model = moshi_lmmodel_alloc_default( config );
    printf("DEBUG: moshi_lmmodel_alloc_default returned successfully\n"); fflush(stdout);

    lm->uses_cross = config->cross_attention;
    lm->second_stream_ahead = config->tts_config.second_stream_ahead;
    lm->kv_cache_type = (ggml_type)config->kv_cache_type;

    printf("DEBUG: moshi_lm_from_files ending\n"); fflush(stdout);
    return lm;
}

void unref( moshi_lm_t * lm ) {
    delete lm;
}

void moshi_lm_set_delay_steps( moshi_lm_t * lm, int delay_steps ) {
    lm->model->delay_steps = delay_steps;
}

int moshi_lm_get_max_delay( moshi_lm_t * lm ) {
    return lm->model->max_delay;
}

int moshi_lm_get_delay_steps( moshi_lm_t * lm ) {
    return lm->model->delay_steps;
}

bool moshi_lm_quantize( moshi_lm_t * lm, const char * quant ) {
    uint32_t uquant = *(uint32_t*)quant;
    ggml_type qtype = (ggml_type)0;
    switch (uquant) {
        case 0x305f3471: // "q4_0"
            qtype = GGML_TYPE_Q4_0;
            break;
        case 0x6b5f3471: // "q4_k"
            qtype = GGML_TYPE_Q4_K;
            break;
        case 0x6b5f3271: // "q2_k"
            qtype = GGML_TYPE_Q2_K;
            break;
        case 0x305f3871: // "q8_0"
            qtype = GGML_TYPE_Q8_0;
            break;
        default:
            return false;
    }
    lm->weights->quantize = true;
    lm->weights->qtype = qtype;
    return true;
}

// RUN 2 (reserve-then-release): STEP 1's ggml_alloc_probe certification came
// back NOT CERTIFIED (per-call ggml_backend_alloc_ctx_tensors overhead is
// ~1 MiB/call in isolation, not the ~36 MiB/layer seen in the real loader —
// see tools/ggml_alloc_probe.cpp), so consolidation is not evidenced. This
// is the scoped fallback instead: hold one large device buffer across the
// entire BMO load + KV cache alloc (keeping NvMap's carveout from draining/
// fragmenting during that phase), then free it immediately before the text
// graph's GraphContext::alloc() call so that allocation gets first crack at
// whatever contiguity the reserve was protecting. Measurement/experiment
// only — env-gated, disabled (0 MiB) unless GRAPH_RESERVE_MIB is set.
static ggml_backend_buffer_t g_graph_reserve_buf = NULL;

static size_t graph_reserve_mib_env() {
    const char * v = getenv("GRAPH_RESERVE_MIB");
    if ( ! v ) return 0;
    return (size_t) atoll(v);
}

static void graph_reserve_acquire( ggml_backend * backend ) {
    size_t mib = graph_reserve_mib_env();
    if ( mib == 0 || g_graph_reserve_buf || ! backend ) return;
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type( backend );
    g_graph_reserve_buf = ggml_backend_buft_alloc_buffer( buft, mib * 1024 * 1024 );
    memledger_log( "graph_reserve_acquire", "(reserve)", "-", 0,
                    g_graph_reserve_buf ? ggml_backend_buffer_get_size(g_graph_reserve_buf) : 0, 0, backend );
    if ( ! g_graph_reserve_buf ) {
        fprintf( stderr, "GRAPH_RESERVE: failed to acquire %zu MiB reserve buffer\n", mib );
        fflush( stderr );
    }
}

static void graph_reserve_release( ggml_backend * backend ) {
    if ( ! g_graph_reserve_buf ) return;
    ggml_backend_buffer_free( g_graph_reserve_buf );
    g_graph_reserve_buf = NULL;
    memledger_log( "graph_reserve_release", "(reserve)", "-", 0, 0, 0, backend );
}

int moshi_lm_load( moshi_lm_t * lm ) {
    graph_reserve_acquire( lm->weights->backend );
    if ( lm->weights->is_gguf ) {
        lm->weights->load_gguf( );
    }

    get_weights( lm->weights, "lm.", lm->model );
    // V2 fix: return whatever glibc is still holding from the BMO repack's
    // per-tensor read/payload churn back to the OS. The 4 large read-staging
    // buffers are now reused across all 62 tensors (see loader.h), but the
    // per-tensor `payload` vector in build_custom_ffn_tensor is not — this
    // catches that and any other residual heap retention from the loop.
    // glibc-specific (not available on Windows/macOS malloc).
#if !defined(_WIN32) && !defined(__APPLE__)
    malloc_trim(0);
#endif
    // Fix A (page cache): drop any GGUF page-cache pages that accumulated
    // outside the per-tensor ranges already advised in
    // read_raw_bytes_from_gguf_file (e.g. from load_gguf()'s own read of
    // the non-BMO tensors).
    lm->weights->fadvise_dontneed_whole_file();
    if ( lm->uses_cross ) {
        lm->cond = new conditioners_t;
        get_weights( lm->weights, lm->cond );
    }

    if ( ! lm->weights->is_gguf ) {
        lm->weights->load();
    }

    return 0;
}

void moshi_lm_save_gguf( moshi_lm_t * lm, const char * filepath ) {
    lm->weights->save_gguf( filepath );
}


// MARK: Generator

struct moshi_lm_gen_t {
    moshi_lm_t * lm;

    own_ptr<voice_t> voice;
    own_ptr<WeightLoader> voice_weights;

    std::vector<int> text_prompt_tokens;

    moshi_lmgen_t lmgen;
    StateMachine * machine = NULL;
    State * machine_state = NULL;
    StateContext * state_ctx = NULL;
    ScratchContext * ctx = NULL;
    std::vector<int> audio_tokens;

    moshi_lmmodel_states_t * lm_states = NULL;
    moshi_lmgen_state_t * lmgen_state = NULL;
};

moshi_lm_gen_t * moshi_lm_generator( moshi_lm_t * lm ) {
    auto gen = new moshi_lm_gen_t;
    gen->lm = lm;
    return gen;
}

void unref( moshi_lm_gen_t * gen ) {
    delete gen;
}

int moshi_lm_set_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filepath ) {
    if ( ! gen->lm->uses_cross )
        return -1;

    gen->voice_weights = WeightLoader::from_safetensor( filepath, moshi->scratch_cpu, moshi->backend );
    if ( ! gen->voice_weights )
        return -1;

    return 0;
}

int moshi_lm_load_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen ) {
    if ( ! gen->lm->uses_cross )
        return -1;

    if ( ! gen->lm->cond )
        return -2;

    gen->voice = new voice_t;
    gen->voice->ctx = NULL;
    gen->voice->buffer = NULL;
    gen->voice->sum = NULL;
    gen->voice->cross = NULL;

    ggml_tensor * speaker_wavs;
    gen->voice_weights->fetch( &speaker_wavs, "voice.speaker_wavs" );
    gen->voice_weights->load();

    voice_condition( gen->voice, gen->lm->cond, speaker_wavs, moshi );

    return 0;
}

int moshi_lm_voice_prefix( moshi_lm_gen_t * gen, std::deque<int> & text_prefix, std::deque<std::vector<int>> & audio_prefix ) {
    auto voice = new voice_t;
    voice->ctx = NULL;
    voice->buffer = NULL;
    voice->sum = NULL;
    voice->cross = NULL;
    voice->text_prefixes.swap( text_prefix );
    voice->audio_prefixes.swap( audio_prefix );
    voice->prompt_embeddings = NULL;
    voice->prompt_cache = NULL;
    gen->voice = voice;
    return 0;
}

int moshi_lm_personaplex_audio_prompt( moshi_lm_gen_t * gen, std::deque<std::vector<int16_t>> & audio_prompt ) {
    auto voice = new voice_t;
    voice->ctx = NULL;
    voice->buffer = NULL;
    voice->sum = NULL;
    voice->cross = NULL;
    voice->prompt_audio.swap ( audio_prompt );
    voice->prompt_embeddings = NULL;
    voice->prompt_cache = NULL;
    gen->voice = voice;
    return 0;
}

int moshi_lm_personaplex_load_voice( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filepath ) {
    std::string filename = filepath;
    auto ext_index = filename.find_last_of('.');
    if ( ext_index == std::string::npos ) {
        return -1;
    }
    std::string ext = filename.substr(ext_index);
    if ( ext == ".safetensors" ) {
        gen->voice_weights = WeightLoader::from_safetensor( filepath,
            moshi->scratch_cpu, moshi->backend );
        if ( ! gen->voice_weights )
            return -1;
    } else if ( ext == ".gguf" ) {
        gen->voice_weights = WeightLoader::from_gguf( filepath,
            moshi->scratch_cpu, moshi->backend );
        if ( ! gen->voice_weights )
            return -1;

        gen->voice_weights->load_gguf();
    } else {
        return -1;
    }

    int n;
    ggml_tensor * voice_prompt_embeddings, * voice_prompt_cache;
    n = gen->voice_weights->fetch( &voice_prompt_embeddings, "voice.embeddings" );
    assert( n );
    n = gen->voice_weights->fetch( &voice_prompt_cache, "voice.cache" );
    assert( n );
    if ( ! gen->voice_weights->is_gguf ) {
        gen->voice_weights->load();
//#define SAVE_GGUF
#ifdef SAVE_GGUF
        gen->voice_weights->save_gguf( (filename.substr(0, ext_index) + ".gguf").c_str() );
#endif
    }

    auto voice = new voice_t;
    voice->ctx = NULL;
    voice->buffer = NULL;
    voice->sum = NULL;
    voice->cross = NULL;
    voice->prompt_embeddings = voice_prompt_embeddings;
    voice->prompt_cache = voice_prompt_cache;
    gen->voice = voice;

    return 0;
}

int moshi_lm_personaplex_system_prompt(
    moshi_context_t * moshi,
    moshi_lm_gen_t * gen,
    tokenizer_t * tok,
    const char * prompt
) {
    std::string system_prompt = "<system> ";
    system_prompt += prompt;
    system_prompt += " <system>";
    tok->sp.Encode(system_prompt, &gen->text_prompt_tokens);
    return 0;
}

void moshi_lm_start( moshi_context_t * moshi, moshi_lm_gen_t * gen, float depth_temperature, float text_temperature, bool logging ) {
    printf("DEBUG: moshi_lm_start starting\n"); fflush(stdout);
    const int max_padding = 8;
    const int initial_padding = 2;
    const int second_stream_ahead = gen->lm->second_stream_ahead;
    printf("DEBUG: creating StateContext, backend=%p\n", (void*)moshi->backend); fflush(stdout);
    gen->state_ctx = new StateContext( moshi->backend );
    printf("DEBUG: state_ctx created\n"); fflush(stdout);
    gen->state_ctx->kv_cache_type = gen->lm->kv_cache_type;
    printf("DEBUG: kv_cache_type set to %d\n", (int)gen->state_ctx->kv_cache_type); fflush(stdout);
    ggml_tensor * condition_cross = NULL;
    if ( gen->voice && ! gen->lm->model->personaplex ) {
        condition_cross = gen->voice->cross;
        gen->machine = new StateMachine(gen->lm->model->text_card + 1, second_stream_ahead, max_padding, initial_padding);
        gen->machine->logging = logging;
        gen->machine_state = gen->machine->new_state();
        gen->lmgen = moshi_lmgen_t{
            gen->lm->model,
            true, depth_temperature, text_temperature, 250, 25,
            gen->machine, gen->machine_state,
            gen->voice->sum,
            &gen->voice->text_prefixes, &gen->voice->audio_prefixes
        };
        printf("DEBUG: calling moshi_lmmodel_states (voice path)\n"); fflush(stdout);
        gen->lm_states = moshi_lmmodel_states( gen->state_ctx, gen->lm->model, gen->voice->cross );
    } else {
        gen->lmgen = moshi_lmgen_t{
            gen->lm->model,
            true, depth_temperature, text_temperature, 250, 25,
            NULL, NULL, // no state machine
            NULL, // no cross
            NULL, NULL, // empty prefixes
        };
        printf("DEBUG: calling moshi_lmmodel_states (default path)\n"); fflush(stdout);
        gen->lm_states = moshi_lmmodel_states( gen->state_ctx, gen->lm->model, NULL );
    }
    printf("DEBUG: moshi_lmmodel_states returned successfully\n"); fflush(stdout);
    gen->lmgen_state = moshi_lmgen_state( gen->lm->model );
    printf("DEBUG: calling state_ctx->alloc()\n"); fflush(stdout);
    gen->state_ctx->alloc();
    printf("DEBUG: calling state_ctx->init()\n"); fflush(stdout);
    gen->state_ctx->init();
    memledger_log("after_kv_cache", "(state_ctx_alloc_init_done)", "-",
                   0, gen->state_ctx->buffer ? ggml_backend_buffer_get_size(gen->state_ctx->buffer) : 0,
                   0, moshi->backend);
    printf("DEBUG: calling init (scratch)\n"); fflush(stdout);
    init( moshi->scratch, gen->lm_states, gen->lm->model, condition_cross );
    printf("DEBUG: creating gen->ctx ScratchContext\n"); fflush(stdout);

    gen->ctx = new ScratchContext( 256, moshi->backend );
    gen->audio_tokens.resize( gen->lm->model->num_audio_codebooks );
    printf("DEBUG: ScratchContext created\n"); fflush(stdout);

    // personaplex process system prompts
    if ( gen->lm->model->personaplex ) {
        graph_reserve_release( moshi->backend );
        printf("DEBUG: processing system prompts\n"); fflush(stdout);
        moshi_lmgen_step_system_prompts(
            *gen->ctx,
            &gen->lmgen, gen->lmgen_state,
            gen->lm_states,
            gen->voice,
            gen->text_prompt_tokens
        );
        printf("DEBUG: system prompts processed successfully\n"); fflush(stdout);
        // MEMLEDGER: this is the Hybrid System Prompt prefill (cuBLAS fall-through
        // path) — the task's stated highest-water-mark candidate for VRAM usage.
        memledger_log("after_first_prefill_step", "(system_prompts_processed)", "-", 0, 0, 0, moshi->backend);
    }
    memledger_log("after_lm_start", "(moshi_lm_start_returning)", "-", 0, 0, 0, moshi->backend);
    printf("DEBUG: moshi_lm_start ending\n"); fflush(stdout);
}

void moshi_lm_send( moshi_lm_gen_t * gen, Entry * entry ) {
    gen->machine_state->entries.push_back( *entry );
}

int moshi_lm_receive( moshi_lm_gen_t * gen, int & text_token, std::vector<int16_t> & audio_tokens ) {
    bool depformer_replace_tokens = (gen->lmgen_state->offset < gen->lm->model->delay_steps);
    bool has_audio_tokens = moshi_lmgen_step(
        *gen->ctx,
        &gen->lmgen, gen->lmgen_state,
        gen->lm_states,
        depformer_replace_tokens,
        text_token,
        gen->audio_tokens
    );
    audio_tokens.resize( gen->audio_tokens.size() );
    if ( has_audio_tokens ) {
        for ( int i = 0; i < gen->audio_tokens.size(); ++i )
            audio_tokens[i] = gen->audio_tokens[i];
    }
    return has_audio_tokens? 1 : 0;
}

void moshi_lm_send2( moshi_lm_gen_t * gen, std::vector<int16_t> & audio_tokens ) {
    gen->audio_tokens.resize( audio_tokens.size() );
    for ( int i = 0; i < audio_tokens.size(); ++i )
        gen->audio_tokens[i] = audio_tokens[i];
}

void moshi_lm_receive2( moshi_lm_gen_t * gen, int & text_token, float & vad ) {
    moshi_lmgen_step(
        *gen->ctx,
        &gen->lmgen, gen->lmgen_state,
        gen->lm_states,
        false, //depformer_replace_tokens
        text_token,
        gen->audio_tokens,
        &vad
    );
}

int moshi_lm_is_active( moshi_lm_gen_t * gen ) {
    const int final_padding = 4;
    int end_offset = gen->machine_state->end_step + gen->lm->model->delay_steps + final_padding;
    int offset = gen->lmgen_state->offset;
    return ( offset < end_offset || gen->machine_state->end_step == -1 );
}

int moshi_lm_is_empty( moshi_lm_gen_t * gen ) {
    return gen->machine_state->is_empty();
}

void moshi_lm_machine_reset( moshi_lm_gen_t * gen ) {
    gen->machine->reset_state( gen->machine_state );
}

size_t moshi_get_allocated_memory(
    moshi_context_t * moshi,
    moshi_lm_t * lm,
    mimi_codec_t * codec,
    moshi_lm_gen_t * gen,
    mimi_encode_context_t * encoder,
    mimi_decode_context_t * decoder
) {
    size_t total = 0;
    if (moshi) {
        if (moshi->scratch && moshi->scratch->buffer) total += ggml_backend_buffer_get_size(moshi->scratch->buffer);
        if (moshi->scratch_cpu && moshi->scratch_cpu->buffer) total += ggml_backend_buffer_get_size(moshi->scratch_cpu->buffer);
    }
    if (lm) {
        if (lm->weights && lm->weights->buffer) total += ggml_backend_buffer_get_size(lm->weights->buffer);
    }
    if (codec) {
        if (codec->mimi_weights && codec->mimi_weights->buffer) total += ggml_backend_buffer_get_size(codec->mimi_weights->buffer);
    }
    if (gen) {
        if (gen->state_ctx && gen->state_ctx->buffer) total += ggml_backend_buffer_get_size(gen->state_ctx->buffer);
        if (gen->voice_weights && gen->voice_weights->buffer) total += ggml_backend_buffer_get_size(gen->voice_weights->buffer);
    }
    if (encoder) {
        if (encoder->state_ctx && encoder->state_ctx->buffer) total += ggml_backend_buffer_get_size(encoder->state_ctx->buffer);
    }
    if (decoder) {
        if (decoder->state_ctx && decoder->state_ctx->buffer) total += ggml_backend_buffer_get_size(decoder->state_ctx->buffer);
    }
    return total;
}

// MARK: Misc

/*int moshi_lm_n_q( moshi_lmmodel_t * lm ) {
    return lm->n_q;
}

int moshi_lm_max_delay( moshi_lmmodel_t * lm ) {
    return lm->max_delay;
}

int moshi_lm_delay_steps( moshi_lmmodel_t * lm ) {
    return lm->delay_steps;
}*/






