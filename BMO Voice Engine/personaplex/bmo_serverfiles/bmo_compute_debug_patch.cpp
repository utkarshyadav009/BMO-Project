/*
 * INSTRUCTIONS: Add this debug code to bmo_compute.cpp in bmo_build_depth_graph()
 * 
 * Find this section around line 520:
 *   ggml_tensor * z_s = ggml_mul_mat(wctx, model.depformer_in[(size_t) codebook_step], temporal_out);
 *   if (!z_s) {
 *       throw std::runtime_error("bmo_build_depth_graph: failed to build depformer input projection");
 *   }
 *
 * And add debug dumps after z_s is created:
 */

// DEBUG: Dump z_s (depformer_in output)
if (codebook_step == 0) {
    FILE * dbg_z_s = fopen("cpp_debug_z_s.bin", "wb");
    if (dbg_z_s) {
        // Convert to float32 for comparison
        float z_s_f32[1024];
        const float * z_s_ptr = (const float *) z_s->data;
        if (z_s->type == GGML_TYPE_F32) {
            std::memcpy(z_s_f32, z_s_ptr, 1024 * sizeof(float));
        } else {
            // Handle other types if needed (would need ggml_get_data_f32 or similar)
            std::fill_n(z_s_f32, 1024, 0.0f);
        }
        fwrite(z_s_f32, sizeof(float), 1024, dbg_z_s);
        fclose(dbg_z_s);
        std::cout << "[DEBUG] Wrote cpp_debug_z_s.bin\n";
    }
}

// Then find the embedding selection around line 528:
/*
    ggml_tensor * last_tok = nullptr;
    if (codebook_step == 0) {
        last_tok = ggml_get_rows(wctx, model.text_emb, text_tokens);
    } else {
        last_tok = ggml_get_rows(wctx, model.audio_embs[(size_t) (codebook_step - 1)], audio_tokens);
    }
*/

// And add after last_tok is created:
if (codebook_step == 0 && last_tok) {
    FILE * dbg_emb = fopen("cpp_debug_text_emb.bin", "wb");
    if (dbg_emb) {
        float emb_f32[1024];
        const float * emb_ptr = (const float *) last_tok->data;
        if (last_tok->type == GGML_TYPE_F32) {
            std::memcpy(emb_f32, emb_ptr, 1024 * sizeof(float));
        } else {
            std::fill_n(emb_f32, 1024, 0.0f);
        }
        fwrite(emb_f32, sizeof(float), 1024, dbg_emb);
        fclose(dbg_emb);
        std::cout << "[DEBUG] Wrote cpp_debug_text_emb.bin\n";
    }
}

// Then find where x is computed (around line 537):
/*
    ggml_tensor * x = ggml_add(wctx, z_s, ggml_reshape_2d(wctx, last_tok, 1024, 1));
*/

// And add after x is created:
if (codebook_step == 0 && x) {
    FILE * dbg_x_init = fopen("cpp_debug_x_init.bin", "wb");
    if (dbg_x_init) {
        float x_init_f32[1024];
        const float * x_ptr = (const float *) x->data;
        if (x->type == GGML_TYPE_F32) {
            std::memcpy(x_init_f32, x_ptr, 1024 * sizeof(float));
        } else {
            std::fill_n(x_init_f32, 1024, 0.0f);
        }
        fwrite(x_init_f32, sizeof(float), 1024, dbg_x_init);
        fclose(dbg_x_init);
        std::cout << "[DEBUG] Wrote cpp_debug_x_init.bin\n";
    }
}

