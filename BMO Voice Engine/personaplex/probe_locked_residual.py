#!/usr/bin/env python3
"""probe_locked_residual.py -- 3-way falsification test for the "locked top-5"
symptom seen in `python bmo_inference.py stream`.

Background (see Claude's analysis): the C++ engine produces near-identical
top-5 text logits across all sampling positions, even though pos_base /
RoPE / KV are supposed to make every frame different. There are 3 buckets
this can fall in:

  A. Position information is broken (RoPE layout/base, pos_base not
     advancing in the kernel, KV indexing wrong). Test: feed IDENTICAL
     17-token input at every call after prefill. Only thing that changes
     across calls is the position. If z[t] is approximately z[0] for all
     t, position info is broken.

  B. Audio embedding stream dominates the residual (one stream's
     embedding magnitude is ~10x the others), so changing the text token
     has no observable effect on z. Test: feed IDENTICAL audio (silence)
     but VARYING text tokens. If z[t] is approximately z[0] across t,
     the text stream isn't reaching the residual.

  C. The transformer math is fine, residual genuinely varies, but
     out_norm/text_linear collapse it back into the same top-5. Test:
     compute cos(z[t], z[0]) on the OUTPUT of forward_temporal in test A.
     If it's reasonably varying (e.g. < 0.99) but logits are still
     locked, the bug is downstream of the residual.

Usage (mirror your normal stream invocation, just swap the script name):

  PYTHONPATH=./moshi BMO_SO_PATH="$PWD/build_jetson/libbmo.so" \\
    python probe_locked_residual.py \\
      --gguf "$PWD/bmo_septq_v3.gguf" \\
      --mimi "$PWD/tokenizer-e351c8d8-checkpoint125.safetensors" \\
      --tokenizer "$PWD/tokenizer_spm_32k_3.model" \\
      --voice-prompt "$PWD/bmo_621.wav" \\
      --voice-prompt-seconds 3 \\
      --text-prompt "Tell me a joke." \\
      --n-ctx 1024 --n-probe 16

The probe loads the same model, runs the same prefill phases as
`mode_stream`, and then runs three controlled tests on top of the
populated KV cache. Output is a verdict line at the bottom that points
at one of buckets A / B / C.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Reuse all the heavy lifting from bmo_inference.py so we don't fork the
# loader logic. NOTE: `BMOEngine` is NOT a module-level symbol in
# bmo_inference.py -- it is deferred-imported inside `_load_engine_class()`
# so that BMO_SO_PATH can still be populated by argparse before the dlopen
# happens. We replicate that pattern here: call _load_engine_class() in
# main() to obtain the class and use it as a local.
from bmo_inference import (
    TokenDelayer,
    SILENCE_TOKENS,
    SINE_TOKENS,
    TEXT_PAD_ID,
    DEFAULT_N_MOSHI_CODEBOOKS,
    _load_engine_class,
    _load_mimi,
    _encode_voice_prompt_codes,
    wrap_with_system_tags,
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _run_prefill(engine, delayer, *, voice_codes, text_prompt_ids,
                 silence_frames, moshi_silence_tokens, user_sine_tokens,
                 n_moshi):
    """Run all 4 canonical prefill phases. Mirrors mode_stream() exactly."""

    def _prefill_one(text_tok, moshi_in, user_in):
        toks = delayer.step(int(text_tok), moshi_in, user_in)
        engine.forward_temporal(toks)

    # Phase 1: voice prompt
    if voice_codes is not None:
        for f in range(voice_codes.shape[0]):
            _prefill_one(TEXT_PAD_ID, voice_codes[f], user_sine_tokens)
        # Phase 2: silence spacer
        for _ in range(silence_frames):
            _prefill_one(TEXT_PAD_ID, moshi_silence_tokens, user_sine_tokens)

    # Phase 3: text prompt
    if text_prompt_ids:
        for tid in text_prompt_ids:
            _prefill_one(int(tid), moshi_silence_tokens, user_sine_tokens)
        # Phase 4: silence spacer
        for _ in range(silence_frames):
            _prefill_one(TEXT_PAD_ID, moshi_silence_tokens, user_sine_tokens)


def _make_input_with_delayer(delayer, text_token, moshi_now, user_now,
                             *, delay_moshi=False, delay_user=True):
    """Build a single 17-token input via the same delayer the stream loop uses.

    For the post-prefill probes we pin delay_moshi=False, delay_user=True
    so that whatever we feed as moshi_now/user_now is what reaches the
    model (otherwise prev_moshi from prefill would mask cb1-7).
    """
    return delayer.step(int(text_token), moshi_now, user_now,
                        delay_moshi=delay_moshi, delay_user=delay_user)


def _print_zstats(label, z, lt):
    print(f"  [{label}] |z|_2={np.linalg.norm(z):.3f} "
          f"z_mean={z.mean():+.4f} z_std={z.std():.4f} "
          f"|lt|_2={np.linalg.norm(lt):.3f} "
          f"argmax_lt={int(np.argmax(lt))} "
          f"max_lt={float(np.max(lt)):+.3f}")


def test_single_forward_baseline(engine, n_moshi, n_user,
                                 moshi_silence_tokens, user_silence_tokens):
    """Test 0a: reset existing engine and run ONE forward at position 0.

    No prefill. No prior context. KV cache cleared by reset(). This
    isolates the embedding + 32-layer cascade + out_norm path with zero
    streaming influence. If z_mean is already ~+0.9 here, the DC offset
    is being introduced by the embedding sum, the per-layer norms, or
    out_norm itself -- not by streaming KV. If z_mean is ~0 here but
    ~+0.9 after prefill, the streaming/KV path is what's accumulating DC.

    Uses the SAME engine instance the rest of the probe uses; we cannot
    create two engines because the Jetson Orin Nano runs out of pinned
    host memory for the second cudaHostRegister.
    """
    print(f"\n[probe 0a] single-forward baseline: reset(), "
          f"NO prefill, NO prior context, position 0")
    print(f"  fixed input: text=PAD, moshi=SILENCE, user=SILENCE")

    engine.reset()
    K = engine.n_codebooks
    delayer = TokenDelayer(n_codebooks=K, n_moshi=n_moshi)
    toks = delayer.step(TEXT_PAD_ID, moshi_silence_tokens, user_silence_tokens,
                        delay_moshi=False, delay_user=False)
    z, lt = engine.forward_temporal(toks)
    z = np.asarray(z, dtype=np.float32).copy()
    lt = np.asarray(lt, dtype=np.float32).copy()
    _print_zstats("frame 0 (no-prefill, reset engine)", z, lt)
    top5 = np.argsort(lt)[::-1][:5].tolist()
    top5_str = " ".join(f"{int(i)}({float(lt[i]):+.3f})" for i in top5)
    print(f"  top-5 (no-prefill): [{top5_str}]")
    print(f"  EPAD={float(lt[0]):+.3f} BOS={float(lt[1]):+.3f} "
          f"EOS={float(lt[2]):+.3f} PAD={float(lt[3]):+.3f}")
    return z, lt


def test_reset_every_frame(engine, n_probe, n_moshi, n_user,
                           moshi_silence_tokens, user_silence_tokens):
    """Test 0b: reset engine before EVERY forward. Each call is at
    position 0 with empty KV.

    Two outcomes are diagnostic:
      (i) z is bit-identical across all calls -> single-forward path is
          deterministic (good); z_mean tells us the no-streaming baseline.
      (ii) z varies despite reset -> there's stale state somewhere in the
           engine (KV cache, RoPE pos_base, work_ctx) that isn't being
           cleared by reset, which would also explain the streaming locked
           behaviour.
    """
    print(f"\n[probe 0b] reset-every-frame: same engine, reset() between "
          f"each call, position 0 every time")
    print(f"  same input as 0a; expect bit-identical z if reset() is "
          f"complete")

    K = engine.n_codebooks
    delayer = TokenDelayer(n_codebooks=K, n_moshi=n_moshi)
    zs = []
    lts = []
    for t in range(min(n_probe, 4)):
        engine.reset()
        toks = delayer.step(TEXT_PAD_ID, moshi_silence_tokens,
                            user_silence_tokens,
                            delay_moshi=False, delay_user=False)
        z, lt = engine.forward_temporal(toks)
        zs.append(np.asarray(z, dtype=np.float32).copy())
        lts.append(np.asarray(lt, dtype=np.float32).copy())
        _print_zstats(f"frame {t:3d} (after reset)", zs[-1], lts[-1])

    # Are they bit-identical?
    print("  pairwise cos(z[t], z[0]) (each frame uses a fresh reset):")
    for t in range(1, len(zs)):
        c = cosine(zs[0], zs[t])
        diff = np.abs(zs[0] - zs[t]).max()
        flag = " <-- bit-identical" if diff < 1e-6 else \
               (" <-- nearly identical" if c > 0.999999 else
                " <-- DIFFERENT (reset() does not clear all state!)")
        print(f"    t={t:3d}  cos={c:+.9f}  max|diff|={diff:.2e}{flag}")
    return zs, lts


def test_dc_accumulation_curve(engine, n_steps, n_moshi, n_user,
                               moshi_silence_tokens, user_silence_tokens,
                               varied=False, label=""):
    """Test 0c / 0d: how does z_mean evolve as we run N consecutive forwards?

    With `varied=False`: feed identical silence input every frame. The
    residual at position N is what the model produces after N forwards
    through the streaming KV path with no information ever being
    introduced. If z_mean rises at all, every call is corrupting state;
    if it stays at 0, only specific input content (e.g. voice prompt,
    text prompt) causes the collapse.

    With `varied=True`: feed varied content (random text token, random
    moshi cb0). Mirrors what real prefill provides. Tells us whether the
    DC accumulation is content-dependent or purely a function of N.

    The reported curve is z_mean / z_std / |z| at every position. We log
    a few key positions (0, 1, 2, 5, 10, 20, 30, 60).
    """
    print(f"\n[probe 0{'d' if varied else 'c'}] DC-accumulation curve: "
          f"reset(), then run {n_steps} consecutive forwards "
          f"({'varied' if varied else 'constant silence'} input)"
          f"{(' -- ' + label) if label else ''}")

    engine.reset()
    K = engine.n_codebooks
    delayer = TokenDelayer(n_codebooks=K, n_moshi=n_moshi)
    rng = np.random.default_rng(seed=99)

    z_means = np.zeros(n_steps, dtype=np.float64)
    z_stds = np.zeros(n_steps, dtype=np.float64)
    z_norms = np.zeros(n_steps, dtype=np.float64)
    argmaxes = np.zeros(n_steps, dtype=np.int32)
    epad_logits = np.zeros(n_steps, dtype=np.float64)
    pad_logits = np.zeros(n_steps, dtype=np.float64)

    moshi_in = moshi_silence_tokens.copy()
    user_in = user_silence_tokens.copy() if user_silence_tokens is not None \
        else np.zeros(0, dtype=np.int32)

    for t in range(n_steps):
        if varied:
            text_t = int(rng.integers(low=4, high=31000))
            moshi_in_t = moshi_silence_tokens.copy()
            moshi_in_t[0] = int(rng.integers(low=10, high=2040))
            user_in_t = user_in.copy()
            if n_user > 0:
                user_in_t[0] = int(rng.integers(low=10, high=2040))
        else:
            text_t = TEXT_PAD_ID
            moshi_in_t = moshi_in
            user_in_t = user_in

        toks = delayer.step(text_t, moshi_in_t, user_in_t,
                            delay_moshi=False, delay_user=False)
        z, lt = engine.forward_temporal(toks)
        z_means[t] = float(z.mean())
        z_stds[t] = float(z.std())
        z_norms[t] = float(np.linalg.norm(z))
        argmaxes[t] = int(np.argmax(lt))
        epad_logits[t] = float(lt[0])
        pad_logits[t] = float(lt[3])

    # Print key positions
    keys = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60]
    keys = [k for k in keys if k < n_steps]
    if (n_steps - 1) not in keys:
        keys.append(n_steps - 1)
    print(f"  pos | z_mean   z_std   |z|     argmax  EPAD     PAD     "
          f"unique_top_argmax_so_far")
    seen_argmax = set()
    for t in range(n_steps):
        seen_argmax.add(int(argmaxes[t]))
        if t in keys:
            print(f"  {t:3d} | {z_means[t]:+.4f}  {z_stds[t]:.4f}  "
                  f"{z_norms[t]:.3f}  {int(argmaxes[t]):6d}  "
                  f"{epad_logits[t]:+.3f}  {pad_logits[t]:+.3f}  "
                  f"{len(seen_argmax)}")

    # Detect the inflection: where does z_mean cross 0.1, 0.3, 0.5?
    crossings = {}
    for thresh in [0.05, 0.1, 0.3, 0.5, 0.8]:
        idx = np.argmax(z_means > thresh) if (z_means > thresh).any() else -1
        if idx > 0 or (idx == 0 and z_means[0] > thresh):
            crossings[thresh] = int(idx)
    if crossings:
        print(f"  z_mean threshold crossings: " +
              ", ".join(f"{th:.2f}@N={n}" for th, n in crossings.items()))
    else:
        print(f"  z_mean stays below 0.05 throughout {n_steps} steps "
              f"(no DC accumulation)")
    return z_means, z_stds, z_norms, argmaxes


def test_per_channel_isolation(engine, n_moshi, n_user,
                               moshi_silence_tokens, user_silence_tokens):
    """Test 0e: at position 0 with empty KV, isolate which input channel
    is responsible for the DC injection.

    Baseline (0a) showed: text=PAD, moshi=SILENCE, user=SILENCE -> z_mean=0.03.
    Test 0d showed:       text=RAND, moshi[0]=RAND, user[0]=RAND -> z_mean=0.34.

    Here we vary ONE channel at a time at position 0 and see which one
    moves z_mean by how much. The biggest mover is where the embedding
    bug lives.
    """
    print(f"\n[probe 0e] per-channel embedding isolation: at position 0, "
          f"flip ONE channel from silence/PAD to a non-silence value")
    print(f"  baseline (0a): text=PAD, moshi=SILENCE, user=SILENCE -> "
          f"z_mean ~ 0.03")

    K = engine.n_codebooks
    delayer = TokenDelayer(n_codebooks=K, n_moshi=n_moshi)
    rng = np.random.default_rng(seed=2025)

    cases = [
        ("baseline (PAD,silence,silence)",
         TEXT_PAD_ID, moshi_silence_tokens.copy(),
         user_silence_tokens.copy() if user_silence_tokens is not None
         else np.zeros(0, dtype=np.int32)),
    ]
    # text-only flip
    for tid in [0, 1, 2, 100, 1000, 10000, 21000, 31000]:
        cases.append((f"text only id={tid:5d}",
                      tid,
                      moshi_silence_tokens.copy(),
                      user_silence_tokens.copy() if user_silence_tokens is not None
                      else np.zeros(0, dtype=np.int32)))
    # moshi cb-k flip (one cb at a time)
    for k in range(n_moshi):
        m = moshi_silence_tokens.copy()
        m[k] = int(rng.integers(low=10, high=2040))
        cases.append((f"moshi cb{k}={int(m[k]):4d} (others=SIL)",
                      TEXT_PAD_ID, m,
                      user_silence_tokens.copy() if user_silence_tokens is not None
                      else np.zeros(0, dtype=np.int32)))
    # user cb-k flip
    if n_user > 0:
        for k in range(n_user):
            u = user_silence_tokens.copy()
            u[k] = int(rng.integers(low=10, high=2040))
            cases.append((f"user  cb{k}={int(u[k]):4d} (others=SIL)",
                          TEXT_PAD_ID, moshi_silence_tokens.copy(), u))

    print(f"  {'case':40s} | z_mean   z_std   |z|     argmax  EPAD     PAD")
    for (label, text_t, m_now, u_now) in cases:
        engine.reset()
        delayer.reset()
        toks = delayer.step(int(text_t), m_now, u_now,
                            delay_moshi=False, delay_user=False)
        z, lt = engine.forward_temporal(toks)
        z = np.asarray(z, dtype=np.float32)
        lt = np.asarray(lt, dtype=np.float32)
        print(f"  {label:40s} | {float(z.mean()):+.4f}  {float(z.std()):.4f}  "
              f"{float(np.linalg.norm(z)):.3f}  {int(np.argmax(lt)):6d}  "
              f"{float(lt[0]):+.3f}  {float(lt[3]):+.3f}")


def test_prefill_mean_curve(engine, voice_codes, text_prompt_ids,
                            silence_frames, n_moshi, n_user,
                            moshi_silence_tokens,
                            user_sine_tokens, user_silence_tokens):
    """Test 0f: run the full canonical prefill BUT log z_mean at every
    single frame, annotated with which phase we're in. Find the frame at
    which DC starts to ramp up.

    Phases (mirrors mode_stream + the probe's own _run_prefill):
       1) voice prompt    moshi=voice_codes,    user=SINE,  text=PAD
       2) silence spacer  moshi=SILENCE,        user=SINE,  text=PAD
       3) text prompt     moshi=SILENCE,        user=SINE,  text=tid_i
       4) silence spacer  moshi=SILENCE,        user=SINE,  text=PAD
    """
    print(f"\n[probe 0f] prefill z_mean curve: log z_mean at every frame "
          f"of the canonical prefill")
    engine.reset()
    K = engine.n_codebooks
    delayer = TokenDelayer(n_codebooks=K, n_moshi=n_moshi)

    rows = []  # (global_idx, phase, frame_in_phase, z_mean, z_std, argmax)

    def _step(text_tok, moshi_in, user_in, phase, fip):
        toks = delayer.step(int(text_tok), moshi_in, user_in)
        z, lt = engine.forward_temporal(toks)
        rows.append((len(rows), phase, fip,
                     float(z.mean()), float(z.std()),
                     float(np.linalg.norm(z)),
                     int(np.argmax(lt)),
                     float(lt[0]), float(lt[3])))

    if voice_codes is not None:
        for f in range(voice_codes.shape[0]):
            _step(TEXT_PAD_ID, voice_codes[f], user_sine_tokens, "voice", f)
        for f in range(silence_frames):
            _step(TEXT_PAD_ID, moshi_silence_tokens, user_sine_tokens,
                  "silence_after_voice", f)
    if text_prompt_ids:
        for i, tid in enumerate(text_prompt_ids):
            _step(int(tid), moshi_silence_tokens, user_sine_tokens,
                  "text_prompt", i)
        for f in range(silence_frames):
            _step(TEXT_PAD_ID, moshi_silence_tokens, user_sine_tokens,
                  "silence_after_text", f)

    print(f"  global  phase                fr  z_mean   z_std   |z|     argmax  EPAD     PAD")
    last_phase = None
    for (g, phase, fip, mean, std, norm, am, epad, pad) in rows:
        marker = ""
        if phase != last_phase:
            marker = " <- phase change"
            last_phase = phase
        print(f"  {g:5d}  {phase:20s} {fip:3d}  {mean:+.4f}  {std:.4f}  "
              f"{norm:.3f}  {am:6d}  {epad:+.3f}  {pad:+.3f}{marker}")

    return rows


def test_position_only(engine, delayer, n_probe, n_moshi, n_user,
                       moshi_silence_tokens, user_silence_tokens):
    """Test A: every frame has IDENTICAL input; only pos_base advances.

    If the model is well-formed, z[t] should differ across positions
    (because RoPE rotates Q/K differently and the KV cache keeps growing).
    cos(z[t], z[0]) should drift below 0.999 within a few frames. If it
    stays near 1.0, position information is being ignored.
    """
    print(f"\n[probe A] position-only variation: identical 17-token input "
          f"at every call, only pos_base advances")
    print(f"  fixed input: text=PAD, moshi=SILENCE, user=SILENCE "
          f"(no delay shift on either stream)")

    zs = []
    lts = []
    top5_ids = []
    for t in range(n_probe):
        toks = _make_input_with_delayer(
            delayer, TEXT_PAD_ID,
            moshi_silence_tokens, user_silence_tokens,
            delay_moshi=False, delay_user=False)
        z, lt = engine.forward_temporal(toks)
        zs.append(np.asarray(z, dtype=np.float32).copy())
        lts.append(np.asarray(lt, dtype=np.float32).copy())
        top5 = np.argsort(lt)[::-1][:5].tolist()
        top5_ids.append(top5)
        if t < 4 or t == n_probe - 1:
            _print_zstats(f"frame {t:3d}", zs[-1], lts[-1])

    print("  pairwise cos(z[t], z[0]) for t in 1..{}:".format(n_probe - 1))
    z0 = zs[0]
    for t in range(1, n_probe):
        c = cosine(z0, zs[t])
        flag = " <-- " if c > 0.9999 else ""
        print(f"    t={t:3d}  cos={c:+.6f}{flag}")

    # logit-side: are top-5 the same set across frames?
    sets = [tuple(sorted(s)) for s in top5_ids]
    n_unique = len(set(sets))
    print(f"  unique top-5 sets across {n_probe} frames: {n_unique}")
    if n_unique == 1:
        print(f"  -> SAME top-5 set every frame: "
              f"{sorted(top5_ids[0])}")

    return zs, lts, top5_ids


def test_text_only_varying(engine, delayer, n_probe, n_moshi, n_user,
                           moshi_silence_tokens, user_silence_tokens):
    """Test B: vary ONLY the text token across frames. If z barely
    moves, the text stream is being drowned by the audio stream in the
    embedding sum (or the text embedding tensor is broken)."""
    print(f"\n[probe B] text-only variation: text token varies, "
          f"moshi/user pinned to silence")

    rng = np.random.default_rng(seed=1234)
    # pick text tokens spanning the special-token region (0..3) and the
    # mid-vocab range, so we'd expect the residual to land in different
    # places of embedding space if the text stream actually reaches it.
    text_choices = [0, 1, 2, 3, 100, 1000, 5000, 12345, 21000, 31000]
    text_seq = [text_choices[i % len(text_choices)] for i in range(n_probe)]

    zs = []
    lts = []
    for t in range(n_probe):
        toks = _make_input_with_delayer(
            delayer, text_seq[t],
            moshi_silence_tokens, user_silence_tokens,
            delay_moshi=False, delay_user=False)
        z, lt = engine.forward_temporal(toks)
        zs.append(np.asarray(z, dtype=np.float32).copy())
        lts.append(np.asarray(lt, dtype=np.float32).copy())
        if t < 4 or t == n_probe - 1:
            _print_zstats(
                f"frame {t:3d} text={text_seq[t]:6d}", zs[-1], lts[-1])

    print("  pairwise cos(z[t], z[0]) (input text differs each frame):")
    z0 = zs[0]
    for t in range(1, n_probe):
        c = cosine(z0, zs[t])
        print(f"    t={t:3d} text={text_seq[t]:6d}  cos={c:+.6f}")

    # Same-position re-test: is changing the text token making any
    # difference at all? Run two forwards back-to-back with text=0 vs
    # text=21000 at the same KV state — but we can't rewind KV cheaply,
    # so we approximate by checking variance across the n_probe frames.
    z_stack = np.stack(zs, axis=0)
    per_dim_std = z_stack.std(axis=0)
    print(f"  z across-frames stats: per-dim std mean={per_dim_std.mean():.6f} "
          f"max={per_dim_std.max():.6f} min={per_dim_std.min():.6f}")

    return zs, lts, text_seq


def test_audio_only_varying(engine, delayer, n_probe, n_moshi, n_user):
    """Test C: vary ONLY the moshi audio (sweep cb0). Text and user pinned.
    Mirror image of test B; useful to confirm the audio stream IS reaching
    the residual (otherwise both streams could be silent contributors)."""
    print(f"\n[probe C] audio-only variation: moshi cb0 varies, "
          f"text/user pinned to PAD/silence")

    rng = np.random.default_rng(seed=4567)
    moshi_base = SILENCE_TOKENS[:n_moshi].astype(np.int32, copy=True)
    user_pinned = SILENCE_TOKENS[:n_user].astype(np.int32, copy=True) \
        if n_user > 0 else np.zeros(0, dtype=np.int32)

    zs = []
    lts = []
    for t in range(n_probe):
        moshi_now = moshi_base.copy()
        # Sweep cb0 across distinct codewords. Skip the silence value.
        moshi_now[0] = int(rng.integers(low=10, high=2040))
        toks = _make_input_with_delayer(
            delayer, TEXT_PAD_ID, moshi_now, user_pinned,
            delay_moshi=False, delay_user=False)
        z, lt = engine.forward_temporal(toks)
        zs.append(np.asarray(z, dtype=np.float32).copy())
        lts.append(np.asarray(lt, dtype=np.float32).copy())
        if t < 4 or t == n_probe - 1:
            _print_zstats(
                f"frame {t:3d} cb0={int(moshi_now[0]):4d}", zs[-1], lts[-1])

    print("  pairwise cos(z[t], z[0]) (input cb0 differs each frame):")
    z0 = zs[0]
    for t in range(1, n_probe):
        c = cosine(z0, zs[t])
        print(f"    t={t:3d}  cos={c:+.6f}")

    return zs, lts


def main(argv=None):
    p = argparse.ArgumentParser("probe_locked_residual")
    p.add_argument("--gguf", required=True)
    p.add_argument("--n-ctx", type=int, default=1024)
    p.add_argument("--mimi", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--voice-prompt", default=None)
    p.add_argument("--voice-prompt-seconds", type=float, default=3.0)
    p.add_argument("--silence-seconds", type=float, default=0.5)
    p.add_argument("--text-prompt", default=None)
    p.add_argument("--n-probe", type=int, default=16,
                   help="Number of frames per probe (A, B, C).")
    args = p.parse_args(argv)

    print(f"[probe] gguf={args.gguf} n_ctx={args.n_ctx}")
    Engine = _load_engine_class()
    mimi = _load_mimi(args, device="cpu")
    print(f"[probe] mimi loaded: codebooks={mimi.num_codebooks} "
          f"sr={mimi.sample_rate}Hz frame_rate={mimi.frame_rate}Hz")
    engine = Engine(args.gguf, n_ctx=args.n_ctx)
    print(f"[probe] engine: K={engine.n_codebooks} d_embd={engine.n_embd} "
          f"dep_q={engine.dep_q} text_vocab={engine.text_vocab} "
          f"audio_vocab={engine.audio_vocab}")

    K = engine.n_codebooks
    n_audio = max(0, K - 1)
    n_moshi = min(DEFAULT_N_MOSHI_CODEBOOKS, n_audio)
    n_user = n_audio - n_moshi
    print(f"[probe] token layout: text=1 moshi={n_moshi} user={n_user} K={K}")

    sp = None
    if args.tokenizer:
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(args.tokenizer)
        except ImportError:
            print("[probe] WARN: sentencepiece not installed; "
                  "skipping text-prompt prefill.")

    # ----- prefill setup (mirrors mode_stream) -----
    voice_codes = None
    if args.voice_prompt:
        voice_codes = _encode_voice_prompt_codes(
            mimi, args.voice_prompt, n_moshi)
        if args.voice_prompt_seconds and args.voice_prompt_seconds > 0:
            keep = max(1, int(round(args.voice_prompt_seconds *
                                    float(mimi.frame_rate))))
            if voice_codes.shape[0] > keep:
                voice_codes = voice_codes[-keep:]
        print(f"[probe] voice_codes shape={voice_codes.shape}")

    text_prompt_ids = []
    if sp is not None and args.text_prompt:
        wrapped = wrap_with_system_tags(args.text_prompt)
        text_prompt_ids = sp.EncodeAsIds(wrapped)
        print(f"[probe] text_prompt_ids={text_prompt_ids}")

    silence_frames = int(round(args.silence_seconds *
                               float(mimi.frame_rate)))
    voice_frames = voice_codes.shape[0] if voice_codes is not None else 0
    text_frames = len(text_prompt_ids)
    total_prefill = (voice_frames +
                     (silence_frames if voice_frames else 0) +
                     text_frames +
                     (silence_frames if text_frames else 0))
    print(f"[probe] prefill budget: voice={voice_frames} silence={silence_frames} "
          f"text={text_frames} total_prefill={total_prefill} "
          f"+ probe_frames=3*{args.n_probe} -> "
          f"total={total_prefill + 3 * args.n_probe} vs n_ctx={args.n_ctx}")

    moshi_silence_tokens = SILENCE_TOKENS[:n_moshi].astype(np.int32, copy=False)
    user_sine_tokens = SINE_TOKENS[:n_user].astype(np.int32, copy=False) \
        if n_user > 0 else np.zeros(0, dtype=np.int32)
    user_silence_tokens = SILENCE_TOKENS[:n_user].astype(np.int32, copy=False) \
        if n_user > 0 else np.zeros(0, dtype=np.int32)

    # ----- 0a: single-forward baseline (no prefill, no streaming).
    # Uses the SAME engine instance to avoid double-init OOM on Jetson.
    z_baseline, lt_baseline = test_single_forward_baseline(
        engine, n_moshi, n_user,
        moshi_silence_tokens, user_silence_tokens)

    # ----- 0b: reset every frame (does reset() clear KV/pos/work_ctx?)
    test_reset_every_frame(
        engine, args.n_probe, n_moshi, n_user,
        moshi_silence_tokens, user_silence_tokens)

    # ----- 0c: DC accumulation curve, identical silence input every step
    # 0a established that position 0 has z_mean ~= 0 (healthy).
    # The full prefill drives z_mean to ~0.9. This curve tells us where
    # in [1..total_prefill] the residual collapses.
    n_acc = max(64, total_prefill or 64)
    test_dc_accumulation_curve(
        engine, n_acc, n_moshi, n_user,
        moshi_silence_tokens, user_silence_tokens,
        varied=False, label="every frame is silence + PAD text")

    # ----- 0d: same curve but with varied input each step
    test_dc_accumulation_curve(
        engine, n_acc, n_moshi, n_user,
        moshi_silence_tokens, user_silence_tokens,
        varied=True, label="random text token + random moshi/user cb0")

    # ----- 0e: per-channel embedding isolation. At position 0 with empty
    # KV (after reset), flip ONE channel away from silence/PAD and see
    # how much z_mean moves. The biggest mover is where the DC injection
    # comes from.
    test_per_channel_isolation(
        engine, n_moshi, n_user,
        moshi_silence_tokens, user_silence_tokens)

    # ----- 0f: prefill curve. Run the full prefill but log z_mean at
    # every step, annotated with phase, so we can see which phase first
    # introduces DC.
    test_prefill_mean_curve(
        engine, voice_codes, text_prompt_ids, silence_frames,
        n_moshi, n_user,
        moshi_silence_tokens, user_sine_tokens, user_silence_tokens)

    # ----- one prefill, then 3 probes back-to-back. The probes ALL run
    # against the same KV cache, so cosines compare positions that share
    # everything except pos_base + KV history depth.
    engine.reset()
    delayer = TokenDelayer(n_codebooks=K, n_moshi=n_moshi)
    print(f"\n[probe] running prefill ({total_prefill} frames) on "
          f"main engine...")
    _run_prefill(
        engine, delayer,
        voice_codes=voice_codes,
        text_prompt_ids=text_prompt_ids,
        silence_frames=silence_frames,
        moshi_silence_tokens=moshi_silence_tokens,
        user_sine_tokens=user_sine_tokens,
        n_moshi=n_moshi,
    )
    print(f"[probe] prefill complete; entering probes.")

    # Test A: position-only variation (everything else identical)
    zs_a, lts_a, top5_a = test_position_only(
        engine, delayer, args.n_probe, n_moshi, n_user,
        moshi_silence_tokens, user_silence_tokens)

    # Test B: text-only variation
    zs_b, lts_b, _ = test_text_only_varying(
        engine, delayer, args.n_probe, n_moshi, n_user,
        moshi_silence_tokens, user_silence_tokens)

    # Test C: moshi-audio cb0 variation
    zs_c, lts_c = test_audio_only_varying(
        engine, delayer, args.n_probe, n_moshi, n_user)

    # ----- Verdict heuristics -----
    print("\n=================== VERDICT ===================")

    a_cosines = [cosine(zs_a[0], zs_a[t]) for t in range(1, len(zs_a))]
    b_cosines = [cosine(zs_b[0], zs_b[t]) for t in range(1, len(zs_b))]
    c_cosines = [cosine(zs_c[0], zs_c[t]) for t in range(1, len(zs_c))]

    a_min = min(a_cosines) if a_cosines else float("nan")
    b_min = min(b_cosines) if b_cosines else float("nan")
    c_min = min(c_cosines) if c_cosines else float("nan")

    print(f"A (position-only)   min cos(z[t],z[0]) = {a_min:+.6f}")
    print(f"B (text-only)       min cos(z[t],z[0]) = {b_min:+.6f}")
    print(f"C (moshi cb0-only)  min cos(z[t],z[0]) = {c_min:+.6f}")

    # A healthy 32-layer transformer with KV cache should show much more
    # angular movement than these thresholds across 15 positions / inputs.
    # cos > ~0.95 over 15 positions is already extremely suspicious.
    A_LOCKED = 0.985     # position-only: should drop noticeably with depth
    B_LOCKED = 0.985     # text token sweeping 0..31000 should move z meaningfully
    C_LOCKED = 0.990     # audio cb0 sweep is a smaller perturbation, slightly tighter

    a_locked = a_min >= A_LOCKED
    b_locked = b_min >= B_LOCKED
    c_locked = c_min >= C_LOCKED

    # ----- DC-attractor heuristic -----
    # In a healthy transformer the residual stream has |mean|/std << 1 and
    # |z| varies significantly when the input or position changes. If we
    # see (mean ~ std) AND |z| is locked to a constant magnitude across
    # frames AND mean is monotonically drifting toward an asymptote, the
    # transformer is in a degenerate attractor regardless of input.
    z_a = np.stack(zs_a, axis=0)
    a_norms = np.linalg.norm(z_a, axis=1)
    a_means = z_a.mean(axis=1)
    a_stds = z_a.std(axis=1)
    norm_range = float(a_norms.max() - a_norms.min())
    mean_range = float(a_means.max() - a_means.min())
    std_range = float(a_stds.max() - a_stds.min())
    mean_over_std_first = float(a_means[0] / max(a_stds[0], 1e-6))
    print(f"\nResidual statistics across Test A's {len(zs_a)} frames:")
    print(f"  |z|_2:  min={a_norms.min():.3f} max={a_norms.max():.3f} "
          f"range={norm_range:.4f}")
    print(f"  z_mean: min={a_means.min():+.4f} max={a_means.max():+.4f} "
          f"range={mean_range:.4f}  (frame0 mean/std = {mean_over_std_first:.3f})")
    print(f"  z_std:  min={a_stds.min():.4f} max={a_stds.max():.4f} "
          f"range={std_range:.4f}")
    z_norm_pinned = norm_range < 0.5  # < 0.5% relative change in 96.x scale
    z_dc_heavy = abs(mean_over_std_first) > 0.5
    z_drifting = (
        (a_means[-1] - a_means[0]) > 0.05 and
        (a_stds[-1] - a_stds[0]) < -0.02
    )
    if z_norm_pinned and z_dc_heavy and z_drifting:
        print("\n!! DC-attractor pattern detected: |z| is essentially "
              "constant (< 0.5% drift), residual mean is "
              f"comparable to std (mean/std={mean_over_std_first:.2f}), "
              "and across positions z is drifting toward a fixed-point "
              "(mean grows, std shrinks). This is consistent with "
              "attention/MLP outputs being too small to perturb the "
              "residual meaningfully -- the model is essentially passing "
              "through a near-constant vector.")

    if a_locked and b_locked and c_locked:
        print("\nDIAGNOSIS: residual is fully collapsed across position AND "
              "input variation. The transformer output is constant "
              "regardless of what's fed in. This is consistent with:")
        print("  - per-codebook embedding sum producing a near-constant "
              "vector (e.g. one stream with massive scale dominates and "
              "varies trivially), OR")
        print("  - a pre-LM-head layer (out_norm, final residual) being "
              "loaded as zeros / wrong tensor, OR")
        print("  - all attention being masked out (zero attention to "
              "anything).")
        print("Suggested next step: verify the OUTPUT of bmo_embed_input_tokens "
              "in C++ has non-trivial dimension-wise variance "
              "(BMO_LOG_EMBED diagnostic) and compare against PyTorch.")
    elif a_locked and not b_locked and not c_locked:
        print("\nDIAGNOSIS: input variation moves z (good), but pos-only "
              "doesn't (bad). Position information is broken.")
        print("Suggested next step: verify pos_base is advancing inside "
              "the kernel (BMO_LOG_ROPE), and verify RoPE Q/K outputs "
              "differ across calls. Check rope_freq_base value and the "
              "interleaved-vs-pair RoPE convention against training.")
    elif a_locked and b_locked and not c_locked:
        print("\nDIAGNOSIS: audio stream IS reaching the residual, but "
              "text stream and position info are ignored.")
        print("Suggested next step: verify text_emb tensor is loaded "
              "and its norm > 0; verify pos_base advancement.")
    elif a_locked and not b_locked and c_locked:
        print("\nDIAGNOSIS: text stream IS reaching the residual, but "
              "audio stream and position info are ignored.")
        print("Suggested next step: verify per-codebook audio embedding "
              "tensors are loaded; verify pos_base advancement.")
    elif not a_locked and b_locked and c_locked:
        print("\nDIAGNOSIS: position info is fine, but neither text nor "
              "audio input changes z. The embedding sum is being "
              "discarded somewhere upstream of the transformer (e.g. "
              "the layer-0 input is being overwritten by KV cache or "
              "norm path).")
        print("Suggested next step: dump bmo_embed_input_tokens output and "
              "the layer-0 attn input; they should be the same tensor.")
    elif not a_locked and not b_locked and not c_locked:
        print("\nDIAGNOSIS: residual moves correctly with position AND "
              "input variation. The bug is downstream of the transformer "
              "output (out_norm, text_linear, or sampling).")
        print("Suggested next step: dump out_norm and text_linear weight "
              "norms; if those are sane, run the same z through PyTorch's "
              "out_norm + text_linear and compare logits.")
    else:
        print("\nMixed signal; see per-cosine numbers above. Investigate "
              "the dimensions where cos > 0.9999 first.")

    print("===============================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
