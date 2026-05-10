"""bmo_inference.py - Hybrid Python / C++ inference driver for BMO on Jetson.

Phase 4.2 follow-up: instead of patching `moshi.offline` (which would force a
17 GB PyTorch LMModel into 8 GB of unified memory), we orchestrate inference
from Python and keep the language model entirely inside libbmo.so:

    libbmo.so   (via BMOEngine ctypes wrapper)  - Temporal transformer
    sentencepiece                               - Text tokenizer
    MimiModel  (PyTorch, ~200 MB)               - Audio codec   [stream mode]
    DepthStrategy                               - dummy | pytorch | cpp

Three modes, designed to be run *in order* so each stage isolates failures:

    smoke   forward_temporal stress test, all-zero tokens, no I/O.
            Validates: ctypes binding, libbmo.so load, KV alloc, kernel chain.

    text    Text-only autoregressive generation. Dummy moshi/user audio = 0.
            Validates: tokenizer wiring, KV cache across steps, text head.

    stream  Full audio-in/audio-out streaming via Mimi + a depth strategy.
            Currently requires `--depth-mode dummy` until Phase 4.3 lands a
            real bmo_forward_depth.

Examples (all from the personaplex root, with libbmo.so already built):

    PYTHONPATH=./moshi BMO_SO_PATH=$PWD/build_jetson/libbmo.so \\
        python bmo_inference.py smoke \\
            --gguf $PWD/bmo_septq_v3.gguf

    PYTHONPATH=./moshi BMO_SO_PATH=$PWD/build_jetson/libbmo.so \\
        python bmo_inference.py text \\
            --gguf $PWD/bmo_septq_v3.gguf \\
            --tokenizer $PWD/tokenizer_spm_32k_3.model \\
            --text-prompt "Hello, my name is" \\
            --n-generate 30
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
# Moshi convention (per moshi/models/lm.py):
#   tokens[0]              : text channel
#   tokens[1 .. 1 + n_a]   : Moshi/agent audio codebooks (= min(dep_q, n_q) rows)
#   tokens[1 + n_a .. K-1] : user audio codebooks (= n_q - n_a rows)
# where K = engine.n_codebooks (= n_q + 1) and n_q is the number of audio
# channels in the temporal input. We split the K-1 audio slots between agent
# and user with `n_a = min(dep_q, K-1)` so the layout stays correct whether
# dep_q < n_q (full duplex with user audio) or dep_q >= n_q (no user split).

def make_input_tokens(
    engine: BMOEngine,
    text_token: int,
    moshi_audio: Optional[np.ndarray] = None,
    user_audio: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compose a single-frame [K] int32 token vector for `forward_temporal`."""
    K = engine.n_codebooks
    n_audio = max(0, K - 1)                 # total audio codebooks (n_q)
    n_a = min(int(engine.dep_q), n_audio)   # Moshi/agent slots actually present
    n_u = n_audio - n_a                     # user-audio slots actually present

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
    """Final path: call libbmo.so's bmo_forward_depth per codebook.

    Currently a Phase 4.3 stub on the C++ side (rc=10 'not implemented').
    Wired here so the moment that lands, a `--depth-mode cpp` run picks it up.
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


def mode_stream(args) -> int:
    """Full audio-in/audio-out loop. Requires Mimi (PyTorch) + depth strategy.

    Currently only meaningful with `--depth-mode dummy` (Phase 4.2 reality:
    the C++ depth path is a stub, the PyTorch fallback is not yet built).
    """
    raise NotImplementedError(
        "stream mode is intentionally deferred -- see blueprint section 4B/4C.\n"
        "Required pieces (in order):\n"
        "  1) DummyDepth + Mimi.encode/decode round-trip with all-zero moshi\n"
        "     tokens (speech-out is silence, but the loop validates timing).\n"
        "  2) Voice-prompt prefill via Mimi.encode -> forward_temporal.\n"
        "  3) PyTorchDepth (blueprint 4C.2) for real audio output.\n"
        "  4) Switch to CppDepth once Phase 4.3 lands bmo_forward_depth.\n"
        "Until then, use `text` mode to validate the temporal bridge."
    )


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
                   default="dummy",
                   help="depth pass strategy (see blueprint 4C)")
    p.add_argument("--depformer-pt",
                   help="(depth-mode=pytorch) path to .pt with depformer.* keys")
    p.add_argument("--temp-audio", type=float, default=0.8)
    p.add_argument("--topk-audio", type=int, default=250)

    # stream (deferred)
    p.add_argument("--mimi", default=os.environ.get("BMO_MIMI"),
                   help="(stream) Mimi safetensors path")
    p.add_argument("--input-wav",  help="(stream) user audio file")
    p.add_argument("--output-wav", help="(stream) generated audio file")

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
