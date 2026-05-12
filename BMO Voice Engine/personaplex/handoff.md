# Handoff — BMO PersonaPlex / `bmo_inference` stream parity

**Date:** 2026-05-12

## Summary

`bmo_inference.py stream` produced **garbage text/audio** while **`moshi.offline`** with **`BMO_USE_CPP=1`** was **good**. The fix was to **stop hand-rolling** the streaming token path and **reuse the same stack as offline**: **`LMGen`** + **`patch_lm_for_bmo`** + warmup + **`bmo_engine.reset()`** after warmup + **`step_system_prompts`** + per-frame **`lm_gen.step`** with Mimi-encoded user codes.

## Root cause (why standalone stream broke)

1. **`TokenDelayer` + direct `BMOEngine.forward_temporal` / `CppDepth`** could not stay bit-for-bit aligned with **`LMGen.prepare_step_input`** (delay ring, `provided` flags, cache positions) and **`LMGen.depformer_step`** teacher-forcing on **user** duplex slots.
2. Even after **`depformer_targets_from_stream`** / **`CppDepth`** fixes, subtle mismatches vs **`prepare_step_input`** remained a risk.
3. **`moshi.offline`** was already the **gold path**: same **`LMGen`** loop as the server.

## What changed (2026-05-12)

### `bmo_inference.py` — `mode_stream`

- **Requires `BMO_USE_CPP=1`**. Stream mode raises a clear error otherwise (GGUF + LMGen shell only).
- Flow mirrors **`moshi.offline.run_inference`** with C++ bridge:
  - `loaders.get_moshi_lm(None, ...)`, **`LMGen`**
  - **`patch_lm_for_bmo`**, **`disable_cuda_graphs_in_lmgen`**, **`warmup`**
  - **`bmo_engine.reset()`** after warmup (KV / position must not carry warmup garbage)
  - **`lm_gen.step_system_prompts(mimi)`** then **`mimi.reset_streaming()`**
  - Generation: **`lm_gen.step(step_in)`** with user codes from **`_encode_user_wav_codes_like_offline`**; after input is exhausted, feed **Mimi-encoded silence** (zero PCM chunk), not arbitrary numpy silence tokens.
  - Output WAV: **concatenate** per-frame PCM from **`decode_tokens_to_pcm`** (streaming decode parity with offline), not a one-shot **`mimi.decode`** on stacked codes only.
- **Removed from this codebase path:** `PERSONAPLEX_DELAYS`, **`depformer_targets_from_stream`**, **`TokenDelayer`**, and **`_encode_voice_prompt_codes`** (voice prompt goes through **`lm_gen.load_voice_prompt`**).
- **Added:** **`_resolve_voice_prompt_wav_for_lmgen`** for **`--voice-prompt-seconds`** (trim last window → temp WAV for **`load_voice_prompt`**). **`--device`**, **`--cpu-offload`**.
- **`--depth-mode`** is **ignored** in stream (depth always goes through patched **`forward_depformer`**). **`--force-text-pad`** is ignored (no LMGen hook yet).
- **`n_ctx`:** uses **`int(os.environ.get("BMO_N_CTX", str(args.n_ctx)))`** so **`BMO_N_CTX`** matches **`offline.py`** when set; otherwise **`--n-ctx`** applies.
- **Device:** Mimi + LM shell load on **`--device`** or cuda-if-available (previously stream forced Mimi on CPU).

### `moshi/moshi/offline.py` (context from prior work; still authoritative)

- **`BMO_USE_CPP=1`:** **`get_moshi_lm(None, ...)`** — no large safetensors LM; **`--moshi-weight`** ignored (logged).
- **`.pt` voice embeddings** rejected in GGUF-only mode (would need **`forward_embeddings`** on real PT weights).
- **`BMO_N_CTX`** from env (default **2048** in offline for GGUF init), **`BMO_GGUF`**, **`bmo_engine.reset()`** after **`warmup`**.

## Operational checklist

1. Set **`BMO_SO_PATH`** / **`--so-path`** before anything imports **`moshi.bmo_engine`** ( **`moshi.offline`** imports it at module load ).
2. **`BMO_USE_CPP=1`**, **`BMO_GGUF`** or **`--gguf`**, WAV **`--voice-prompt`** (not **`.pt`** in GGUF-only mode).
3. **`BMO_N_CTX`** large enough: `voice_frames + 2×silence + text_frames + n_frames` (approx); KV wrap destroys quality.

## Example command (stream)

```bash
PYTHONPATH=./moshi BMO_USE_CPP=1 BMO_SO_PATH=$PWD/build_jetson/libbmo.so BMO_N_CTX=2048 \
  python bmo_inference.py stream \
    --gguf $PWD/bmo_septq_v3.gguf \
    --mimi $PWD/tokenizer-e351c8d8-checkpoint125.safetensors \
    --tokenizer $PWD/tokenizer_spm_32k_3.model \
    --voice-prompt $PWD/bmo_621.wav \
    --text-prompt "Tell me a joke." \
    --input-wav $PWD/user.wav \
    --n-frames 125 \
    --output-wav /tmp/bmo_response.wav \
    --output-text /tmp/bmo_response.json
```

## Still separate: `mode_text` / `mode_smoke` / `mode_harness`

These still use **`BMOEngine`** directly and **`DepthStrategy`** (`CppDepth` etc.); only **`stream`** was switched to full **`LMGen`** parity.

## Open follow-ups (optional)

- Re-add **`--force-text-pad`** for stream if needed (would require a small LMGen hook or post-step override).
- If **`bmo_inference text`** should match **`LMGen`** text-only behavior, that would be a separate alignment pass.
