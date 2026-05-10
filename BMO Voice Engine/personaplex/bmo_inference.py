"""bmo_inference.py - Hybrid Python / C++ inference driver for BMO on Jetson.

Phase 4.2/4.3/4.4 driver: instead of patching `moshi.offline` (which would
force a 17 GB PyTorch LMModel into 8 GB of unified memory), we orchestrate
inference from Python and keep the language model entirely inside libbmo.so:

    libbmo.so   (via BMOEngine ctypes wrapper)  - Temporal + depth transformers
    sentencepiece                               - Text tokenizer
    MimiModel  (PyTorch, ~200 MB)               - Audio codec   [stream mode]
    DepthStrategy                               - dummy | pytorch | cpp

Three modes, designed to be run *in order* so each stage isolates failures:

    smoke   forward_temporal stress test, all-zero tokens, no I/O.
            Validates: ctypes binding, libbmo.so load, KV alloc, kernel chain.

    text    Text-only autoregressive generation. Dummy moshi/user audio = 0.
            Validates: tokenizer wiring, KV cache across steps, text head.

    stream  Full audio-in/audio-out streaming via Mimi + a depth strategy.
            Validates: Mimi encode/decode, full duplex token layout (text +
            8 moshi + 8 user), real audio output. Requires --depth-mode cpp
            and a Mimi safetensors checkpoint via --mimi.

Examples (all from the personaplex root, with libbmo.so already built):

    # Pure forward-temporal stress, no audio.
    PYTHONPATH=./moshi BMO_SO_PATH=$PWD/build_jetson/libbmo.so \\
        python bmo_inference.py smoke --gguf $PWD/bmo_septq_v3.gguf

    # Text-only generation (no audio output, validates the temporal head).
    PYTHONPATH=./moshi BMO_SO_PATH=$PWD/build_jetson/libbmo.so \\
        python bmo_inference.py text \\
            --gguf $PWD/bmo_septq_v3.gguf \\
            --tokenizer $PWD/tokenizer_spm_32k_3.model \\
            --text-prompt "Hello, my name is" \\
            --n-generate 30

    # Canonical run: voice prompt + system text prompt + user audio + sampled
    # text/audio output. Mirrors `moshi.offline` argument-for-argument.
    PYTHONPATH=./moshi BMO_SO_PATH=$PWD/build_jetson/libbmo.so \\
        python bmo_inference.py stream \\
            --gguf $PWD/bmo_septq_v3.gguf \\
            --mimi $PWD/tokenizer-e351c8d8-checkpoint125.safetensors \\
            --tokenizer $PWD/tokenizer_spm_32k_3.model \\
            --voice-prompt $PWD/bmo_621.wav \\
            --text-prompt "Tell me a joke." \\
            --input-wav $PWD/tellmeajoke_padded.wav \\
            --depth-mode cpp \\
            --n-frames 125 \\
            --n-ctx 256 \\
            --output-wav /tmp/bmo_response.wav \\
            --output-text /tmp/bmo_response.json

    # Smoke-test with NO voice prompt (just to validate the decode path).
    PYTHONPATH=./moshi BMO_SO_PATH=$PWD/build_jetson/libbmo.so \\
        python bmo_inference.py stream \\
            --gguf $PWD/bmo_septq_v3.gguf \\
            --mimi $PWD/tokenizer-e351c8d8-checkpoint125.safetensors \\
            --depth-mode cpp \\
            --force-text-pad \\
            --n-frames 50 --n-ctx 128 \\
            --output-wav /tmp/bmo_smoke.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

# Make `from moshi.bmo_engine ...` resolve when run from the personaplex root
# without the user having to set PYTHONPATH manually.
_HERE = Path(__file__).resolve().parent
_MOSHI_PKG = _HERE / "moshi"
if _MOSHI_PKG.exists() and str(_MOSHI_PKG) not in sys.path:
    sys.path.insert(0, str(_MOSHI_PKG))

# NOTE: `bmo_engine` calls `ctypes.CDLL(os.environ['BMO_SO_PATH'])` at module
# import time, so we MUST defer the import until after argparse has had a
# chance to populate BMO_SO_PATH from the --so-path flag. We therefore expose
# a tiny lazy loader and only refer to BMOEngine for type-checking purposes
# in the rest of the module.
if TYPE_CHECKING:
    from moshi.bmo_engine import BMOEngine  # pragma: no cover


def _load_engine_class():
    so_path = os.environ.get("BMO_SO_PATH", "./build_jetson/libbmo.so")
    if not Path(so_path).exists():
        raise RuntimeError(
            f"BMO_SO_PATH={so_path!r} does not exist.\n"
            f"  Did you build libbmo.so?  cmake --build build_jetson --target bmo_shared -j 4\n"
            f"  Pass it explicitly with --so-path, or export BMO_SO_PATH=/path/to/libbmo.so."
        )
    try:
        from moshi.bmo_engine import BMOEngine as _BMOEngine
    except OSError as ex:
        raise RuntimeError(
            f"Failed to dlopen {so_path!r}: {ex}.\n"
            f"  Check `nm -D --defined-only {so_path} | grep ' T bmo_'` to confirm "
            f"the C-API symbols are exported."
        ) from ex
    return _BMOEngine


# ---------------------------------------------------------------------------
# Token-layout helpers
# ---------------------------------------------------------------------------
# Moshi/PersonaPlex convention (per moshi/models/lm.py):
#   tokens[0]                          : text channel
#   tokens[1 .. 1 + n_moshi]           : Moshi/agent audio codebooks
#   tokens[1 + n_moshi .. K-1]         : user audio codebooks
# where K = engine.n_codebooks (= n_q + 1). PersonaPlex has dep_q == n_q (the
# depformer predicts BOTH moshi and user codebooks for the full-duplex echo
# task), so we cannot infer the split from dep_q alone -- callers pass in
# n_moshi_codebooks (default 8 = one Mimi stream) so we know where the Moshi
# slots end and the user slots begin.

# Default count of Moshi-side audio codebooks. Matches Mimi's standard
# `set_num_codebooks(8)` and PersonaPlex's [moshi-8 | user-8] interleave.
DEFAULT_N_MOSHI_CODEBOOKS = 8

# PersonaPlex / Helium text padding token id (config: existing_text_padding_id).
# This is what the model emits on "no-text" frames; feeding 0 as text instead
# is OOD and is what produced earlier gibberish-only outputs.
TEXT_PAD_ID = 3

# Pre-encoded Mimi codes for "8 codebooks of pure silence" (used on BOTH the
# moshi/agent and the user audio channels during prefill) and "8 codebooks of
# a 440 Hz placeholder sine wave" (kept here as documented reference -- vanilla
# Moshi feeds this on the user channel during prompt phases, but PersonaPlex/
# BMO empirically requires SILENCE_TOKENS on user instead; SINE_TOKENS produces
# a DC-attractor residual mean=+1.13 / std=0.99 that gibbers, while
# SILENCE_TOKENS gives mean=+0.03 / std=1.50, the model's healthy fixed point).
# These are NOT all-zeros; they're the actual token ids that Mimi.encode
# produces for those signals, baked in as constants.
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int32)
SINE_TOKENS    = np.array([430, 1268, 381, 1611, 1095, 1495,   56,  472], dtype=np.int32)


# Per-codebook input delays for PersonaPlex (Moshi-architecture variant).
# Source: `_lm_kwargs["delays"]` in moshi/models/loaders.py. Index layout:
#   [text, moshi_cb0..moshi_cb7, user_cb0..user_cb7]  (17 entries)
# delay=0: the token provided at frame t is read at frame t (synchronous).
# delay=1: the token provided at frame t is read at frame t+1 (i.e. on each
#          frame, the model reads the PREVIOUS frame's value on this channel).
# Without this shift, Q/K cache and the transformer's input distribution are
# off by one frame on 14/17 channels, producing flat/uniform text logits with
# EPAD/PAD ~3 logits BELOW random subword fragments instead of above.
PERSONAPLEX_DELAYS = np.array(
    [0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1], dtype=np.int32)


class TokenDelayer:
    """Stateful per-frame input composer that mimics LMGen.prepare_step_input.

    For each frame the caller supplies the "current" tokens (text, moshi_now,
    user_now). The class returns the K-int32 input vector with delay-shifted
    values: delay=0 channels carry the current value, delay=1 channels carry
    the value that was current on the *previous* call (i.e., what we stashed
    in `prev_moshi` / `prev_user`).

    The delay handling must be applied to BOTH the prefill phases (voice
    prompt, silence spacers, text prompt) and the sampling loop. A single
    instance must persist across all of these, so the prev buffer accumulates
    correctly. Initial buffer values are `zero_token_id = -1`, which the C++
    embed path interprets as "no value provided" (the row lookup is skipped).
    """

    def __init__(self, n_codebooks: int, n_moshi: int,
                 delays: Optional[np.ndarray] = None):
        if delays is None:
            delays = PERSONAPLEX_DELAYS
        self.K = int(n_codebooks)
        self.n_moshi = int(n_moshi)
        self.n_user = max(0, self.K - 1 - self.n_moshi)
        # Slice/extend `delays` so it covers exactly K channels. Pad missing
        # entries with delay=1 (acoustic default) so we degrade safely if the
        # engine reports more codebooks than the canonical layout.
        if delays.size >= self.K:
            self.delays = delays[:self.K].astype(np.int32, copy=True)
        else:
            self.delays = np.concatenate(
                [delays.astype(np.int32, copy=False),
                 np.ones(self.K - delays.size, dtype=np.int32)])
        # `prev_*` is the per-channel "what we provided last call". For first
        # frame, all entries = -1 (skipped by add_row in the C++ embed path),
        # matching LMGen's zero_token_id initial state.
        self.prev_moshi = np.full(self.n_moshi, -1, dtype=np.int32) \
            if self.n_moshi > 0 else np.zeros(0, dtype=np.int32)
        self.prev_user = np.full(self.n_user, -1, dtype=np.int32) \
            if self.n_user > 0 else np.zeros(0, dtype=np.int32)

    def reset(self):
        if self.n_moshi > 0:
            self.prev_moshi.fill(-1)
        if self.n_user > 0:
            self.prev_user.fill(-1)

    def step(self, text_token: int,
             moshi_now: Optional[np.ndarray] = None,
             user_now: Optional[np.ndarray] = None,
             delay_moshi: bool = True,
             delay_user: bool = True) -> np.ndarray:
        """Compose the K-vector for `forward_temporal` and advance the buffer.

        delay_moshi / delay_user toggle whether delay=1 channels read from
        the previous-frame buffer (True) or from this frame's `moshi_now` /
        `user_now` (False). Use True when the caller "provides" tokens (the
        equivalent of LMGen.prepare_step_input writing with provided=True);
        use False when the cache value comes from the model's own depth
        writeback (`provided=False` => depth fills cb1-7 at the same position
        as cb0, so cb0 and cb1-7 share the same source).

        Concretely for our streaming pipeline:
          - prefill:                      delay_moshi=True,  delay_user=True
          - sampling steps t in {0,1}:    delay_moshi=True,  delay_user=True
          - sampling steps t >= 2:        delay_moshi=False, delay_user=True
        See the sampling loop for the rationale (LMGen's prefill prepare
        writes cb1-7 with provided=True at N+1, blocking depth's overwrite at
        sampling step t=0 and t=1; from t=2 onward the cb1-7 cache slot is
        filled by depth's writeback at the same position as cb0).
        """
        tokens = np.zeros(self.K, dtype=np.int32)
        tokens[0] = int(text_token)

        # Normalise the inputs to fixed lengths so we can pad missing entries
        # with -1 (treated as "no value" by the embed path).
        def _coerce(arr: Optional[np.ndarray], n: int) -> np.ndarray:
            if n <= 0:
                return np.zeros(0, dtype=np.int32)
            if arr is None:
                return np.full(n, -1, dtype=np.int32)
            a = np.asarray(arr, dtype=np.int32)
            if a.size >= n:
                return a[:n].copy()
            return np.concatenate(
                [a, np.full(n - a.size, -1, dtype=np.int32)])

        m_now = _coerce(moshi_now, self.n_moshi)
        u_now = _coerce(user_now, self.n_user)

        for k in range(self.n_moshi):
            if delay_moshi and self.delays[1 + k] != 0:
                tokens[1 + k] = self.prev_moshi[k]
            else:
                tokens[1 + k] = m_now[k]
        for k in range(self.n_user):
            if delay_user and self.delays[1 + self.n_moshi + k] != 0:
                tokens[1 + self.n_moshi + k] = self.prev_user[k]
            else:
                tokens[1 + self.n_moshi + k] = u_now[k]

        if self.n_moshi > 0:
            self.prev_moshi[:] = m_now
        if self.n_user > 0:
            self.prev_user[:] = u_now

        return tokens


def make_input_tokens(
    engine: BMOEngine,
    text_token: int,
    moshi_audio: Optional[np.ndarray] = None,
    user_audio: Optional[np.ndarray] = None,
    n_moshi_codebooks: int = DEFAULT_N_MOSHI_CODEBOOKS,
) -> np.ndarray:
    """Compose a single-frame [K] int32 token vector for `forward_temporal`."""
    K = engine.n_codebooks
    n_audio = max(0, K - 1)                                    # total audio codebooks (n_q)
    n_a = max(0, min(int(n_moshi_codebooks), n_audio))         # Moshi slots
    n_u = n_audio - n_a                                        # user slots

    tokens = np.zeros(K, dtype=np.int32)
    tokens[0] = int(text_token)

    if moshi_audio is not None and n_a > 0:
        n = min(int(moshi_audio.size), n_a)
        tokens[1:1 + n] = moshi_audio.astype(np.int32, copy=False)[:n]
    if user_audio is not None and n_u > 0:
        n = min(int(user_audio.size), n_u)
        tokens[1 + n_a:1 + n_a + n] = user_audio.astype(np.int32, copy=False)[:n]

    return tokens


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_top_k(logits: np.ndarray, temp: float, top_k: int) -> int:
    """Numpy-only temperature + top-k sampler. Returns one int token id."""
    logits = logits.astype(np.float64, copy=False)
    if not np.all(np.isfinite(logits)):
        # be defensive: any NaN/Inf in logits collapses softmax to garbage.
        finite = np.where(np.isfinite(logits), logits, -1e9)
        logits = finite

    if temp <= 0.0:
        return int(np.argmax(logits))

    scaled = logits / max(temp, 1e-6)
    if top_k > 0 and top_k < scaled.size:
        # Keep only the top-k logits; mask the rest to -inf.
        idx = np.argpartition(-scaled, top_k - 1)[:top_k]
        masked = np.full_like(scaled, -np.inf)
        masked[idx] = scaled[idx]
        scaled = masked

    m = np.max(scaled)
    probs = np.exp(scaled - m)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.argmax(logits))
    probs = probs / s
    return int(np.random.choice(probs.size, p=probs))


# ---------------------------------------------------------------------------
# Depth strategies (see blueprint section 4C)
# ---------------------------------------------------------------------------

class DepthStrategy:
    name: str = "abstract"

    def reset(self) -> None: ...

    def step(
        self,
        text_token: int,
        transformer_out: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class DummyDepth(DepthStrategy):
    """Primary path: skip depth entirely. moshi_audio_tokens := zeros.

    Cheapest possible validation harness for the C++ Temporal bridge. Audio
    output via Mimi will be silence/garbage but the temporal loop is fully
    exercised end-to-end.
    """
    name = "dummy"

    def __init__(self, dep_q: int):
        self.dep_q = int(dep_q)
        self._zeros = np.zeros(self.dep_q, dtype=np.int32)

    def step(self, text_token, transformer_out):
        return self._zeros


class CppDepth(DepthStrategy):
    """Production path (Phase 4.3): call libbmo.so's bmo_forward_depth.

    Per temporal frame, bmo_forward_depth is invoked dep_q times in sequence:
      cb=0:    prev = sampled text token, lookup via depformer_text_emb
      cb=k>0:  prev = previously sampled audio token at codebook k-1,
               lookup via depformer_emb[k-1]
    The C++ side resets a small cross-codebook KV cache at cb=0 and
    accumulates K/V at slots 0..k as the loop runs, then projects through
    linears[k] to produce per-codebook audio logits.
    """
    name = "cpp"

    def __init__(self, engine: BMOEngine, temp: float = 0.8, top_k: int = 250):
        self.engine = engine
        self.temp = temp
        self.top_k = top_k

    def step(self, text_token, transformer_out):
        prev = int(text_token)
        out = np.empty(self.engine.dep_q, dtype=np.int32)
        for cb in range(self.engine.dep_q):
            logits = self.engine.forward_depth(cb, prev, transformer_out)
            tok = sample_top_k(logits, self.temp, self.top_k)
            out[cb] = tok
            prev = tok
        return out


class PyTorchDepth(DepthStrategy):
    """Fallback path: load *only* depformer.* keys from a .pt with mmap.

    Stub for now -- per blueprint section 4C.2 this should:
      * `torch.load(pt_path, mmap=True, map_location='cpu')`,
      * filter keys starting with 'depformer.',
      * build an nn.Module mirroring the 6-layer depformer architecture,
      * `load_state_dict` with strict=False.
    Memory cost ~300 MB, sidesteps the C++ depth path while it's in Phase 4.3.
    """
    name = "pytorch"

    def __init__(self, pt_path: str, dep_q: int):
        raise NotImplementedError(
            "PyTorchDepth is a stub. To implement (blueprint 4C.2):\n"
            "  1) ckpt = torch.load(pt_path, mmap=True, map_location='cpu')\n"
            "  2) sub = {k.removeprefix('depformer.'): v for k, v in ckpt.items()\n"
            "            if k.startswith('depformer.')}\n"
            "  3) build a 6-layer StreamingTransformer matching the depformer\n"
            "     config, plus depformer_in/depformer_text_emb/depformer_emb/linears\n"
            "  4) load_state_dict(sub, strict=False); .eval(); torch.no_grad()\n"
            "  5) per step, run the same algorithm as LMGen.depformer_step."
        )


def make_depth_strategy(args, engine: BMOEngine) -> DepthStrategy:
    if args.depth_mode == "dummy":
        return DummyDepth(engine.dep_q)
    if args.depth_mode == "cpp":
        return CppDepth(engine, temp=args.temp_audio, top_k=args.topk_audio)
    if args.depth_mode == "pytorch":
        if not args.depformer_pt:
            raise RuntimeError("--depth-mode pytorch requires --depformer-pt PATH")
        return PyTorchDepth(args.depformer_pt, engine.dep_q)
    raise RuntimeError(f"unknown --depth-mode {args.depth_mode!r}")


# ---------------------------------------------------------------------------
# Memory diag (cheap, optional)
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_smoke(args) -> int:
    BMOEngine = _load_engine_class()
    print(f"[smoke] rss_mb={rss_mb():.1f}")
    engine = BMOEngine(args.gguf, n_ctx=args.n_ctx)
    print(f"[smoke] engine: K={engine.n_codebooks} d_embd={engine.n_embd} "
          f"dep_q={engine.dep_q} text_vocab={engine.text_vocab} "
          f"audio_vocab={engine.audio_vocab} layers={engine.n_layers}")
    print(f"[smoke] rss_mb_after_init={rss_mb():.1f}")

    K = engine.n_codebooks
    tokens = np.zeros(K, dtype=np.int32)

    timings = []
    for i in range(args.n_steps):
        t0 = time.perf_counter()
        z, lt = engine.forward_temporal(tokens)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(dt_ms)
        if i < 3 or i == args.n_steps - 1 or (i + 1) % 25 == 0:
            print(f"[smoke] step {i:4d} dt_ms={dt_ms:6.1f} "
                  f"z[:4]={z[:4]} text_argmax={int(np.argmax(lt))} "
                  f"text_max={float(np.max(lt)):.3f} text_finite={bool(np.all(np.isfinite(lt)))}")

    timings = np.asarray(timings, dtype=np.float64)
    print(f"[smoke] done. mean={timings.mean():.1f}ms "
          f"p50={np.median(timings):.1f}ms "
          f"p99={np.percentile(timings, 99):.1f}ms "
          f"min={timings.min():.1f}ms max={timings.max():.1f}ms")
    print(f"[smoke] rss_mb_final={rss_mb():.1f}")

    target_ms = 80.0
    p50 = float(np.median(timings[1:])) if len(timings) > 1 else float(timings[0])
    print(f"[smoke] vs 80ms target: p50={p50:.1f}ms "
          f"({'PASS' if p50 <= target_ms else 'OVER'})")
    return 0 if np.all(np.isfinite(timings)) else 1


def mode_text(args) -> int:
    if not args.tokenizer:
        raise RuntimeError("text mode requires --tokenizer PATH (sentencepiece .model)")

    import sentencepiece as spm
    BMOEngine = _load_engine_class()
    print(f"[text] rss_mb={rss_mb():.1f}")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"[text] tokenizer loaded: vocab={sp.GetPieceSize()}")

    engine = BMOEngine(args.gguf, n_ctx=args.n_ctx)
    print(f"[text] engine: K={engine.n_codebooks} d_embd={engine.n_embd} "
          f"text_vocab={engine.text_vocab} dep_q={engine.dep_q}")
    print(f"[text] rss_mb_after_init={rss_mb():.1f}")

    if sp.GetPieceSize() != engine.text_vocab and \
       sp.GetPieceSize() + 1 != engine.text_vocab:
        print(f"[text] WARN: tokenizer vocab {sp.GetPieceSize()} vs "
              f"engine text_vocab {engine.text_vocab} (off by != 0/1).")

    depth = make_depth_strategy(args, engine)
    print(f"[text] depth_strategy={depth.name}")

    engine.reset()
    n_audio = max(0, engine.n_codebooks - 1)            # total audio channels (n_q)
    n_a = min(int(engine.dep_q), n_audio)               # Moshi/agent slots
    n_u = n_audio - n_a                                  # user-audio slots
    moshi_audio = np.zeros(n_a, dtype=np.int32)
    user_audio = np.zeros(n_u, dtype=np.int32)

    # Prefill the system / user prompt so the KV cache is warmed.
    prompt_text = args.text_prompt or ""
    prompt_ids = sp.EncodeAsIds(prompt_text) if prompt_text else []
    print(f"[text] prompt={prompt_text!r} ids={prompt_ids}")

    last_text_token = int(args.initial_text_token)
    transformer_out = np.zeros(engine.n_embd, dtype=np.float32)
    text_logits = np.zeros(engine.text_vocab, dtype=np.float32)

    t_pref0 = time.perf_counter()
    for tid in prompt_ids:
        toks = make_input_tokens(engine, last_text_token, moshi_audio, user_audio)
        transformer_out, text_logits = engine.forward_temporal(toks)
        last_text_token = int(tid)  # force-feed the prompt
    t_pref_ms = (time.perf_counter() - t_pref0) * 1000.0
    print(f"[text] prefill {len(prompt_ids)} tokens in {t_pref_ms:.1f}ms "
          f"({(t_pref_ms / max(1, len(prompt_ids))):.1f}ms/tok)")

    # Generate args.n_generate tokens, sampling text and using DummyDepth (or
    # whatever) for moshi audio.
    out_ids = []
    timings = []
    t_gen0 = time.perf_counter()
    for step in range(args.n_generate):
        toks = make_input_tokens(engine, last_text_token, moshi_audio, user_audio)
        t0 = time.perf_counter()
        transformer_out, text_logits = engine.forward_temporal(toks)
        timings.append((time.perf_counter() - t0) * 1000.0)

        last_text_token = sample_top_k(text_logits, args.temp_text, args.topk_text)
        out_ids.append(last_text_token)
        moshi_audio = depth.step(last_text_token, transformer_out)

    t_gen_ms = (time.perf_counter() - t_gen0) * 1000.0
    timings_arr = np.asarray(timings, dtype=np.float64)
    decoded = sp.DecodeIds(out_ids) if out_ids else ""
    print(f"[text] generated {args.n_generate} tokens in {t_gen_ms:.1f}ms")
    print(f"[text] forward_temporal: mean={timings_arr.mean():.1f}ms "
          f"p50={np.median(timings_arr):.1f}ms p99={np.percentile(timings_arr,99):.1f}ms")
    print(f"[text] ids={out_ids}")
    print(f"[text] decoded={decoded!r}")
    return 0


def _load_mimi(args, device: str = "cpu"):
    """Locate + load the Mimi codec from the local moshi package.

    Mimi is small enough (~200 MB) to live in PyTorch alongside libbmo.so;
    only the 17 GB LMModel needs to stay in C++.
    """
    if not args.mimi:
        raise RuntimeError(
            "stream mode requires --mimi PATH (Mimi safetensors). "
            "Default file name is 'tokenizer-e351c8d8-checkpoint125.safetensors'."
        )
    from moshi.models import loaders  # noqa: WPS433  -- local package
    mimi = loaders.get_mimi(args.mimi, device=device)
    mimi.set_num_codebooks(DEFAULT_N_MOSHI_CODEBOOKS)
    return mimi


# Special text tokens (matches moshi.offline lines 358-360 + Helium config):
#   id 0 -> 'EPAD' (end-of-pad / no-text-this-frame, the dominant token)
#   id 1 -> 'BOS'
#   id 2 -> 'EOS'
#   id 3 -> 'PAD' (== TEXT_PAD_ID, what we feed when we have nothing to say)
TEXT_SPECIAL_LABELS = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}


def wrap_with_system_tags(text: str) -> str:
    """Match moshi.offline.wrap_with_system_tags exactly.

    The model was trained to see system instructions as
        '<system> ...content... <system>'
    (yes, opening AND closing use the same <system> marker; that is a quirk
    of the PersonaPlex training format, not a typo).
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def decode_text_token(token_id: int, sp) -> str:
    """Render a text-channel token as either a special label or a wordpiece.

    Mirrors moshi.offline lines 351-360. Special ids in (0,3) are mapped
    via TEXT_SPECIAL_LABELS; everything else (including BOS/EOS unless they
    happened to land on 1/2 in this tokenizer) goes through SentencePiece's
    `id_to_piece` with the byte-pair separator stripped.
    """
    if token_id in TEXT_SPECIAL_LABELS:
        return TEXT_SPECIAL_LABELS[token_id]
    if sp is None:
        return f"<id:{int(token_id)}>"
    piece = sp.id_to_piece(int(token_id))
    return piece.replace("\u2581", " ")  # SentencePiece's '▁' -> ASCII space


def _encode_voice_prompt_codes(mimi, voice_prompt_path: str, n_moshi: int) -> np.ndarray:
    """Encode a voice prompt wav -> (T_frames, n_moshi) int32 codebooks.

    PersonaPlex's voice prompts are short (~5-10 s) wavs of the agent's
    speaking voice. Mimi.encode produces (1, K, T_frames); we transpose
    to per-frame layout and clamp to the first n_moshi codebooks (typically 8).
    """
    import torch  # local: only paid when --voice-prompt is set
    wav = _read_wav_24k(voice_prompt_path)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    with torch.no_grad():
        codes_t = mimi.encode(wav_t)  # (1, K_codebooks, T_frames)
    # (1, K, T) -> (T, K) and slice to n_moshi
    codes_np = codes_t[0].transpose(0, 1).cpu().numpy().astype(np.int32)
    if codes_np.shape[1] < n_moshi:
        raise RuntimeError(
            f"voice prompt encoded with only {codes_np.shape[1]} codebooks but "
            f"the model expects {n_moshi}. Did you forget set_num_codebooks(8)?"
        )
    return codes_np[:, :n_moshi]


def _read_wav_24k(path: str):
    """Load a wav file and resample to 24 kHz mono float32 in [-1, 1]."""
    try:
        import soundfile as sf
    except ImportError as ex:
        raise RuntimeError(
            "`soundfile` is required for --voice-prompt / wav IO "
            "(pip install soundfile)."
        ) from ex
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    data = data[:, 0]
    if sr != 24000:
        try:
            from scipy.signal import resample_poly
        except ImportError as ex:
            raise RuntimeError(
                f"input wav is {sr} Hz; resampling to 24000 Hz needs `scipy` "
                f"(pip install scipy). Or supply a 24 kHz wav directly."
            ) from ex
        from math import gcd
        g = gcd(sr, 24000)
        data = resample_poly(data, 24000 // g, sr // g).astype("float32", copy=False)
    return data


def _encode_user_wav_codes_like_offline(mimi, wav_path: str, n_user: int):
    """Encode user-channel WAV the same way as ``moshi.offline.run_inference``.

    Offline loads PCM with ``lm.load_audio`` (``sphn`` read + resample), walks
    fixed-size chunks with ``_iterate_audio(..., pad=True)``, and runs
    ``mimi.encode`` per chunk via ``encode_from_sphn``, with streaming state
    enabled — matching the reference path immediately after
    ``mimi.reset_streaming()`` there.

    Returns ``(codes, meta)`` where ``codes`` is ``(T_frames, n_user)`` int32.
    """
    import torch
    from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
    from moshi.models.lm import load_audio as lm_load_audio
    from moshi.models.lm import _iterate_audio as lm_iterate_audio

    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    frame_size = int(round(sample_rate / frame_rate))

    user_audio = lm_load_audio(str(wav_path), sample_rate)
    n_ch = int(user_audio.shape[0])
    total_samples = int(user_audio.shape[-1])
    n_pcm_chunks = (
        (total_samples + frame_size - 1) // frame_size if total_samples > 0 else 0
    )

    pieces: list[np.ndarray] = []
    with torch.no_grad():
        with mimi.streaming(1):
            mimi.reset_streaming()
            for enc in lm_encode_from_sphn(
                mimi,
                lm_iterate_audio(
                    user_audio, sample_interval_size=frame_size, pad=True
                ),
                max_batch=1,
            ):
                # enc: (1, K, F) -> (F, K)
                kt = enc[0].transpose(0, 1).detach().cpu().numpy().astype(np.int32)
                pieces.append(kt)

    if not pieces:
        codes = np.zeros((0, max(n_user, 1)), dtype=np.int32)
    else:
        codes = np.concatenate(pieces, axis=0)

    enc_k = int(codes.shape[1]) if codes.size else 0
    if enc_k < n_user:
        raise RuntimeError(
            f"user wav encoded with only {enc_k} codebooks but "
            f"the model expects {n_user} user slots."
        )
    codes = codes[:, :n_user]

    meta = {
        "path": str(wav_path),
        "pcm_channels": n_ch,
        "pcm_samples_at_model_sr": total_samples,
        "pcm_duration_s": total_samples / float(sample_rate),
        "frame_size_samples": frame_size,
        "frame_rate_hz": frame_rate,
        "pcm_chunks_zero_padded": n_pcm_chunks,
        "user_codec_frames": int(codes.shape[0]),
        "user_codebooks": int(n_user),
        "load_resample": "moshi.models.lm.load_audio (sphn), same as moshi.offline",
        "chunking": "lm._iterate_audio(..., pad=True) + lm.encode_from_sphn",
    }
    return codes, meta


def _write_wav_24k(path: str, wav: np.ndarray) -> None:
    try:
        import soundfile as sf
    except ImportError as ex:
        raise RuntimeError(
            "stream --output-wav requires `soundfile` (pip install soundfile)."
        ) from ex
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    # Soft clip to [-1, 1] so an over-driven decode doesn't wrap.
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(path, wav, 24000, subtype="PCM_16")


def mode_stream(args) -> int:
    """Full audio-in/audio-out streaming loop with voice/text prompt phases.

    Mirrors `moshi.offline.run_inference` with libbmo.so as the LM:

        Phase 1 (voice prompt prefill)   moshi channels = mimi.encode(voice_wav)
                                         text channel   = PAD
        Phase 2 (audio silence spacer)   all channels   = 0     (~0.5 s)
        Phase 3 (text prompt prefill)    text channel   = <system> tokens
                                         audio channels = 0
        Phase 4 (audio silence spacer)   all channels   = 0     (~0.5 s)
        Phase 5 (sampling loop)          user channels  = Mimi codes from
                                                          ``--input-wav``
                                                          (offline-parity
                                                          chunked encode), else
                                                          SILENCE_TOKENS.
                                         text + moshi   = sampled

    Each phase consumes one KV-cache slot per frame, so total slots used is
    voice_frames + 2*silence + text_frames + n_frames; if that exceeds the
    Jetson n_ctx cap (currently 128 due to a defensive override in bmo.cpp),
    the KV will silently wrap and quality will degrade. We print an explicit
    budget summary up front so it's obvious when the cap needs raising.

    Caveats vs the canonical PyTorch `moshi.offline`:
      * ``TokenDelayer`` applies ``PERSONAPLEX_DELAYS`` across prefill and gen.
      * Single Mimi instance for both encode and decode (offline.py uses two
        with separate streaming state). Should be benign as long as we don't
        interleave encode/decode mid-stream.
    """
    BMOEngine = _load_engine_class()

    # ----- 1. Models + tokenizer -----
    print(f"[stream] rss_mb={rss_mb():.1f}")
    mimi = _load_mimi(args, device="cpu")
    print(f"[stream] mimi loaded: codebooks={mimi.num_codebooks} "
          f"sr={mimi.sample_rate}Hz frame_rate={mimi.frame_rate}Hz")
    print(f"[stream] rss_mb_after_mimi={rss_mb():.1f}")

    engine = BMOEngine(args.gguf, n_ctx=args.n_ctx)
    print(f"[stream] engine: K={engine.n_codebooks} d_embd={engine.n_embd} "
          f"dep_q={engine.dep_q} text_vocab={engine.text_vocab} "
          f"audio_vocab={engine.audio_vocab}")
    print(f"[stream] rss_mb_after_engine={rss_mb():.1f}")

    K = engine.n_codebooks
    n_audio = max(0, K - 1)
    n_moshi = min(DEFAULT_N_MOSHI_CODEBOOKS, n_audio)
    n_user = n_audio - n_moshi
    print(f"[stream] token layout: text=1 moshi={n_moshi} user={n_user} K={K}")

    sp = None
    if args.tokenizer:
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(args.tokenizer)
            print(f"[stream] tokenizer: vocab={sp.GetPieceSize()}")
        except ImportError:
            print("[stream] WARN: sentencepiece not installed; skipping prompt prefill.")
            sp = None

    depth = make_depth_strategy(args, engine)
    print(f"[stream] depth_strategy={depth.name}")
    if depth.name == "dummy":
        print("[stream] WARNING: --depth-mode dummy emits ZERO audio codebooks "
              "every frame. Mimi will decode 100% silence/buzzing AND the "
              "temporal head will see out-of-distribution audio inputs, "
              "producing garbage text. Use --depth-mode cpp for real output.")

    # ----- 2. Pre-encode all audio that prefill needs -----
    voice_codes = None  # (T_voice, n_moshi)
    if args.voice_prompt:
        print(f"[stream] encoding voice prompt: {args.voice_prompt}")
        voice_codes = _encode_voice_prompt_codes(mimi, args.voice_prompt, n_moshi)
        full_t = voice_codes.shape[0]
        if args.voice_prompt_seconds and args.voice_prompt_seconds > 0:
            keep = max(1, int(round(args.voice_prompt_seconds * float(mimi.frame_rate))))
            if voice_codes.shape[0] > keep:
                # Keep the FINAL `keep` frames of the prompt: most voice
                # prompts have leading silence/noise, and the recent context
                # is what the model attends to during generation.
                voice_codes = voice_codes[-keep:]
                print(f"[stream] voice prompt trimmed: {full_t} -> "
                      f"{voice_codes.shape[0]} frames "
                      f"(--voice-prompt-seconds {args.voice_prompt_seconds})")
        print(f"[stream] voice prompt: {voice_codes.shape[0]} frames "
              f"({voice_codes.shape[0] / float(mimi.frame_rate):.2f}s) "
              f"x {voice_codes.shape[1]} codebooks")

    user_codes = None  # (T_user, n_user)
    if args.input_wav and n_user > 0:
        print(f"[stream] encoding user audio (offline-parity): {args.input_wav}")
        user_codes, user_meta = _encode_user_wav_codes_like_offline(
            mimi, args.input_wav, n_user
        )
        for key, val in user_meta.items():
            print(f"[stream] user_audio: {key}={val}")

    text_prompt_ids = []
    if sp is not None and args.text_prompt:
        wrapped = wrap_with_system_tags(args.text_prompt)
        text_prompt_ids = sp.EncodeAsIds(wrapped)
        print(f"[stream] text prompt={wrapped!r} ids={text_prompt_ids}")
    elif args.text_prompt and sp is None:
        print(f"[stream] WARN: --text-prompt set but no --tokenizer; "
              f"text prefill will be skipped.")

    # ----- 3. KV-cache budget -----
    silence_frames = int(round(args.silence_seconds * float(mimi.frame_rate)))
    voice_frames = voice_codes.shape[0] if voice_codes is not None else 0
    text_frames = len(text_prompt_ids)
    n_frames = int(args.n_frames)
    if n_frames <= 0:
        raise RuntimeError("--n-frames must be > 0 for stream mode")

    total_slots = voice_frames + (silence_frames if voice_frames else 0) + \
                  text_frames + (silence_frames if text_frames else 0) + \
                  n_frames
    print(f"[stream] kv budget: voice={voice_frames} silence={silence_frames} "
          f"text={text_frames} silence={silence_frames} gen={n_frames} "
          f"total={total_slots} vs --n-ctx={args.n_ctx}")
    if total_slots > args.n_ctx:
        print(f"[stream] WARN: total prefill+gen ({total_slots}) exceeds n_ctx "
              f"({args.n_ctx}); the KV cache will wrap and quality will suffer. "
              f"Re-run with a higher --n-ctx, or shorten voice/text prompts.")

    # ----- 4. Reset state, run prompt phases -----
    # IMPORTANT: the moshi/user "silence" tokens are NOT all-zeros. They are
    # the specific Mimi codewords [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
    # for silence on the agent side, and we feed the SAME silence codewords on
    # the user side during prefill.
    #
    # NOTE on canonical Moshi divergence: vanilla Moshi's `step_system_prompts`
    # uses a 440 Hz "sine" placeholder (SINE_TOKENS = [430, 1268, 381, 1611,
    # 1095, 1495, 56, 472]) on the user channel for every prefill frame.
    # Empirically (probe_user_channel.log) feeding SINE_TOKENS on user puts
    # PersonaPlex/BMO in a DC-attractor state with residual mean = +1.13 and
    # std = 0.99 for ~every silent prefill frame, KV cache is stuck in that
    # mode for the rest of the run, EPAD/PAD logits get suppressed, and
    # generation produces gibberish wordpieces. Feeding SILENCE_TOKENS on user
    # keeps mean = +0.03 and std = 1.50 -- the model's healthy fixed point.
    # PersonaPlex was fine-tuned with silence on user during prefill, not the
    # 440 Hz sine, so SINE_TOKENS is wrong for this checkpoint.
    engine.reset()

    moshi_silence_tokens = SILENCE_TOKENS[:n_moshi].astype(np.int32, copy=False)
    if n_user > 0:
        user_silence_tokens = SILENCE_TOKENS[:n_user].astype(np.int32, copy=False)
    else:
        user_silence_tokens = np.zeros(0, dtype=np.int32)

    # Single delayer instance shared across prefill phases AND the sampling
    # loop. Its prev_moshi / prev_user buffers carry the per-channel delay=1
    # state correctly across phase boundaries (so e.g. moshi cb1-7 at the
    # first sampling frame is the moshi cb1-7 from the LAST silence-spacer
    # frame, not initial zeros).
    delayer = TokenDelayer(n_codebooks=engine.n_codebooks, n_moshi=n_moshi)
    delay_str = ",".join(str(int(d)) for d in delayer.delays)
    print(f"[stream] delay handling enabled: delays=[{delay_str}]")

    def _prefill_one(text_tok, moshi_in, user_in):
        toks = delayer.step(int(text_tok), moshi_in, user_in)
        engine.forward_temporal(toks)

    # Phase 1: voice prompt -- moshi=voice_codes, user=SILENCE, text=PAD(=zero_text_code).
    # See the SINE_TOKENS divergence note above: PersonaPlex/BMO lands in a DC
    # attractor when fed SINE on user during prefill, but is healthy on
    # SILENCE. That's the only deviation from canonical Moshi prefill.
    if voice_codes is not None:
        t0 = time.perf_counter()
        for f in range(voice_codes.shape[0]):
            _prefill_one(TEXT_PAD_ID, voice_codes[f], user_silence_tokens)
        print(f"[stream] phase1 voice prompt: {voice_codes.shape[0]} frames in "
              f"{(time.perf_counter() - t0) * 1000.0:.1f}ms")

        # Phase 2: silence spacer -- moshi=SILENCE, user=SILENCE, text=PAD.
        t0 = time.perf_counter()
        for _ in range(silence_frames):
            _prefill_one(TEXT_PAD_ID, moshi_silence_tokens, user_silence_tokens)
        print(f"[stream] phase2 silence: {silence_frames} frames in "
              f"{(time.perf_counter() - t0) * 1000.0:.1f}ms")

    # Phase 3: text prompt -- moshi=SILENCE, user=SILENCE, text=current prompt token.
    # No shift: each prompt token is fed as text input on its own frame, exactly
    # matching LMGen._step_text_prompt_core (lm.py lines 1183-1191), with the
    # user-channel divergence noted above.
    if text_prompt_ids:
        t0 = time.perf_counter()
        for tid in text_prompt_ids:
            _prefill_one(int(tid), moshi_silence_tokens, user_silence_tokens)
        print(f"[stream] phase3 text prompt: {text_frames} frames in "
              f"{(time.perf_counter() - t0) * 1000.0:.1f}ms")

        # Phase 4: silence spacer -- same as phase 2.
        t0 = time.perf_counter()
        for _ in range(silence_frames):
            _prefill_one(TEXT_PAD_ID, moshi_silence_tokens, user_silence_tokens)
        print(f"[stream] phase4 silence: {silence_frames} frames in "
              f"{(time.perf_counter() - t0) * 1000.0:.1f}ms")

    # ----- 5. Sampling loop (Phase 5) -----
    # First frame after prefill: text starts at PAD, moshi seeded with the
    # silence codeword pattern (so cb0=948 etc. as the model would expect on
    # a "no audio yet" frame), user is taken from --input-wav if available.
    text_token = TEXT_PAD_ID
    moshi_prev = moshi_silence_tokens.copy()

    moshi_codes_out = np.zeros((n_frames, n_moshi), dtype=np.int32)
    timings_temp = []
    timings_depth = []
    text_pieces: list = []  # for --output-text JSON
    text_id_log: list = []

    t_gen0 = time.perf_counter()
    for t in range(n_frames):
        if user_codes is not None and t < user_codes.shape[0] and n_user > 0:
            user_now = user_codes[t]
        else:
            user_now = user_silence_tokens

        # Delay handling for moshi during sampling:
        #
        # In LMGen, the first two sampling steps (t=0, t=1) read cb1-7 from
        # cache slots that were stamped during the LAST PREFILL STEP's
        # prepare_step_input (provided=True for cb1-7 at positions N and N+1).
        # The depth writeback at sampling step t=0 cannot overwrite those
        # slots because of the ~provided check, so cb1-7 stays = silence at
        # both reads. From step t=2 onward, no prefill prepare reaches that
        # position, provided=False, and depth's writeback at the matching
        # position fills cb1-7 with the SAME-step depth output — i.e. cb0
        # and cb1-7 share a source and there is no further delay shift.
        #
        # We model this by feeding `moshi_now=moshi_prev` (depth from t-1)
        # at every sampling step, but toggling whether the delayer picks
        # cb1-7 from `prev_moshi` (silence carried over from prefill, used
        # at t=0 and t=1) or from this frame's `moshi_now` (depth, used at
        # t>=2).
        delay_moshi_now = (t < 2)
        toks = delayer.step(int(text_token), moshi_prev, user_now,
                            delay_moshi=delay_moshi_now,
                            delay_user=True)

        t0 = time.perf_counter()
        z, lt = engine.forward_temporal(toks)
        timings_temp.append((time.perf_counter() - t0) * 1000.0)

        if args.force_text_pad:
            text_token = TEXT_PAD_ID
        else:
            text_token = int(sample_top_k(lt, args.temp_text, args.topk_text))
        text_id_log.append(text_token)
        text_pieces.append(decode_text_token(text_token, sp))

        # Diagnostic for the "no EPAD/PAD ever" issue: on the first ~6 frames,
        # print the top-5 text logits and the EPAD/PAD logits explicitly. If
        # EPAD/PAD aren't even in the top 100, the bug is upstream of sampling
        # (text head structure / load). If they ARE near the top but argmax is
        # a real wordpiece, the model is just biased and we may need to inspect
        # the prompt phase.
        if t < 6:
            top5 = np.argsort(lt)[::-1][:5]
            top5_str = " ".join(f"{int(i)}({float(lt[i]):+.3f})" for i in top5)
            epad_lg = float(lt[0]) if lt.size > 0 else float('nan')
            bos_lg  = float(lt[1]) if lt.size > 1 else float('nan')
            eos_lg  = float(lt[2]) if lt.size > 2 else float('nan')
            pad_lg  = float(lt[3]) if lt.size > 3 else float('nan')
            print(f"[stream] frame {t:3d} text_logits top5=[{top5_str}] "
                  f"EPAD={epad_lg:+.3f} BOS={bos_lg:+.3f} "
                  f"EOS={eos_lg:+.3f} PAD={pad_lg:+.3f}")

        t0 = time.perf_counter()
        all_audio = depth.step(text_token, z)
        timings_depth.append((time.perf_counter() - t0) * 1000.0)

        moshi_prev = all_audio[:n_moshi].astype(np.int32, copy=False)
        moshi_codes_out[t] = moshi_prev

        # Per-frame log: only print non-PAD/EPAD text events verbosely; print
        # a short tick line otherwise to avoid 100s of pad lines.
        is_special = text_token in TEXT_SPECIAL_LABELS
        if not is_special:
            print(f"[stream] frame {t:4d} text='{text_pieces[-1]}' "
                  f"id={text_token} moshi_cb0={int(moshi_prev[0]):4d} "
                  f"temp_dt={timings_temp[-1]:6.1f}ms "
                  f"depth_dt={timings_depth[-1]:6.1f}ms")
        elif t < 3 or (t + 1) % 25 == 0 or t == n_frames - 1:
            print(f"[stream] frame {t:4d} text='{text_pieces[-1]}' "
                  f"moshi_cb0={int(moshi_prev[0]):4d} "
                  f"temp_dt={timings_temp[-1]:6.1f}ms "
                  f"depth_dt={timings_depth[-1]:6.1f}ms")

    t_gen_ms = (time.perf_counter() - t_gen0) * 1000.0
    timings_temp_arr = np.asarray(timings_temp, dtype=np.float64)
    timings_depth_arr = np.asarray(timings_depth, dtype=np.float64)
    audio_seconds = n_frames / float(mimi.frame_rate)
    n_real_text = sum(1 for tp in text_pieces if tp not in TEXT_SPECIAL_LABELS.values())
    print(f"[stream] generated {n_frames} frames "
          f"({audio_seconds:.2f}s of audio) in {t_gen_ms:.1f}ms "
          f"({t_gen_ms / max(1, n_frames):.1f}ms/frame, "
          f"realtime factor {(t_gen_ms / 1000.0) / max(audio_seconds, 1e-9):.1f}x); "
          f"non-pad text frames={n_real_text}")
    print(f"[stream] temporal: mean={timings_temp_arr.mean():.1f}ms "
          f"p50={np.median(timings_temp_arr):.1f}ms "
          f"p99={np.percentile(timings_temp_arr, 99):.1f}ms")
    print(f"[stream] depth(x{engine.dep_q}): mean={timings_depth_arr.mean():.1f}ms "
          f"p50={np.median(timings_depth_arr):.1f}ms "
          f"p99={np.percentile(timings_depth_arr, 99):.1f}ms")

    # Render concatenated text for human inspection.
    rendered = "".join(text_pieces[i] for i in range(len(text_pieces))
                       if text_pieces[i] not in TEXT_SPECIAL_LABELS.values())
    print(f"[stream] decoded text: {rendered!r}")

    # ----- 6. Output: wav + json -----
    if args.output_text:
        try:
            import json
            with open(args.output_text, "w") as f:
                json.dump(text_pieces, f, ensure_ascii=False)
            print(f"[stream] wrote text tokens to {args.output_text}")
        except Exception as ex:
            print(f"[stream] WARN: failed to write {args.output_text}: {ex}")

    if not args.output_wav:
        print("[stream] no --output-wav given; skipping Mimi decode")
        return 0

    import torch
    print(f"[stream] decoding {n_frames} frames -> {args.output_wav}")
    codes_t = torch.from_numpy(moshi_codes_out.T[None, ...]).long()  # (1, n_moshi, T)
    with torch.no_grad():
        wav_out = mimi.decode(codes_t)  # (1, 1, T_samples)
    wav_out_np = wav_out.squeeze().cpu().numpy()
    _write_wav_24k(args.output_wav, wav_out_np)
    print(f"[stream] wrote {wav_out_np.shape[-1]} samples "
          f"({audio_seconds:.2f}s) to {args.output_wav}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bmo_inference",
        description="Hybrid Python/C++ BMO inference driver (Phase 4.2).",
    )
    p.add_argument("mode", choices=("smoke", "text", "stream"),
                   help="which validation mode to run")
    p.add_argument("--gguf", default=os.environ.get("BMO_GGUF", "bmo_septq_v3.gguf"),
                   help="path to the BMO GGUF (env: BMO_GGUF)")
    p.add_argument("--n-ctx", type=int, default=128,
                   help="KV cache context length passed to bmo_init")
    p.add_argument("--so-path", default=os.environ.get("BMO_SO_PATH"),
                   help="path to libbmo.so (env: BMO_SO_PATH overrides)")

    # smoke
    p.add_argument("--n-steps", type=int, default=100,
                   help="(smoke) number of forward_temporal calls")

    # text
    p.add_argument("--tokenizer", default=os.environ.get("BMO_TOKENIZER"),
                   help="(text/stream) sentencepiece .model path")
    p.add_argument("--text-prompt", default="Hello",
                   help="(text/stream) prompt string to prefill")
    p.add_argument("--n-generate", type=int, default=30,
                   help="(text) number of tokens to autoregressively sample")
    p.add_argument("--initial-text-token", type=int, default=0,
                   help="(text) token id used at the very first step before any "
                        "sampling has happened (default 0 = PAD/EPAD-ish)")
    p.add_argument("--temp-text", type=float, default=0.7)
    p.add_argument("--topk-text", type=int, default=25)

    # depth
    p.add_argument("--depth-mode", choices=("dummy", "pytorch", "cpp"),
                   default="cpp",
                   help=("depth pass strategy (see blueprint 4C). 'cpp' uses "
                         "libbmo.so's forward_depth (production). 'dummy' "
                         "returns zero audio codes -- only for isolating the "
                         "temporal stack; produces silence/buzzing audio AND "
                         "feeds out-of-distribution audio inputs back into the "
                         "temporal head, so text predictions also degenerate."))
    p.add_argument("--depformer-pt",
                   help="(depth-mode=pytorch) path to .pt with depformer.* keys")
    p.add_argument("--temp-audio", type=float, default=0.8)
    p.add_argument("--topk-audio", type=int, default=250)

    # stream
    p.add_argument("--mimi", default=os.environ.get("BMO_MIMI"),
                   help="(stream) Mimi safetensors path "
                        "(env: BMO_MIMI; default file: tokenizer-e351c8d8-checkpoint125.safetensors)")
    p.add_argument("--voice-prompt",
                   help="(stream) wav of the agent's voice (e.g. bmo_621.wav). "
                        "Encoded via Mimi and force-fed into the moshi audio "
                        "channels for `voice_frames` slots before generation, "
                        "matching moshi.offline.step_system_prompts.")
    p.add_argument("--voice-prompt-seconds", type=float, default=0.0,
                   help="(stream) trim the voice prompt to at most this many "
                        "seconds before prefill (we keep the FINAL window of "
                        "the wav, since that's where speech is loudest in most "
                        "voice prompt files). 0 = use the whole file. Useful "
                        "on Jetson where n_ctx is capped at 1024 slots.")
    p.add_argument("--silence-seconds", type=float, default=0.5,
                   help="(stream) length of the all-zero silence spacers placed "
                        "after voice-prompt and after text-prompt prefill, in "
                        "seconds (default 0.5 = 6-7 frames @ 12.5 Hz). Set 0 to "
                        "disable spacers entirely.")
    p.add_argument("--input-wav",
                   help="(stream) user audio file (any sample rate, mono or stereo). "
                        "If omitted, the user channel is fed silence (TTS-only).")
    p.add_argument("--output-wav",
                   help="(stream) destination wav for Mimi-decoded moshi audio "
                        "(24 kHz mono PCM_16). If omitted, decode is skipped.")
    p.add_argument("--output-text",
                   help="(stream) JSON path for the per-frame decoded text "
                        "tokens (mirrors moshi.offline --output-text format: "
                        "list of strings, one per generation frame).")
    p.add_argument("--n-frames", type=int, default=125,
                   help="(stream) number of 12.5 Hz frames to generate "
                        "(default 125 = ~10 s of audio). The total KV cache "
                        "consumption is voice_frames + 2*silence + text_frames + n_frames.")
    p.add_argument("--force-text-pad", action="store_true",
                   help="(stream) clamp the text channel to TEXT_PAD_ID=3 every "
                        "frame instead of sampling. Use only for the dummy TTS "
                        "validation; for real responses, leave it off so the "
                        "model can emit transcribed text alongside audio.")

    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.so_path:
        os.environ["BMO_SO_PATH"] = args.so_path
    if args.seed >= 0:
        np.random.seed(args.seed)
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return mode_smoke(args)
    if args.mode == "text":
        return mode_text(args)
    if args.mode == "stream":
        return mode_stream(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
