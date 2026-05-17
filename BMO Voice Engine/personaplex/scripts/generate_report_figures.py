#!/usr/bin/env python3
"""
Generate thesis/report assets in one run: Table 1 + Figures 1–6.

Run on the server from the personaplex repo root (or anywhere with PYTHONPATH):

  python scripts/generate_report_figures.py \\
    --out-dir report_figures \\
    --metadata ../bmo_clips_raw/BMO_SpeechDataset/metadata.csv \\
    --audio-root ../bmo_clips_raw/BMO_SpeechDataset \\
    --septq-ckpt bmo_temporal_half_cushion_max.pt \\
    --zs-json zs_half_cushion_max.json

Canonical input documentation lives in `scripts/README_REPORT_FIGURES.md` (copied to `--out-dir` as `README_DATA_REQUIREMENTS.md` on each run).

If soundfile is missing, Table 1 / Figure 4 still run but duration cells show "—" unless a numeric duration column exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional imports (fail soft where possible)
# ---------------------------------------------------------------------------

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    print("ERROR: numpy is required.", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
except ImportError as e:  # pragma: no cover
    print("ERROR: matplotlib is required.", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import soundfile as sf

    HAS_SF = True
except ImportError:
    HAS_SF = False


# ---------------------------------------------------------------------------
# Table 1 — dataset stats from metadata
# ---------------------------------------------------------------------------

NONVERBAL_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("laugh", (r"\[laugh\]", r"\blaugh\b", r"_laugh_", r"_giggles", r"_giggling", r"_CHUCKLES")),
    ("cry", (r"\[cry\]", r"\bcry\b", r"_cry_", r"_sobbing", r"_sobs")),
    ("gasp", (r"\[gasp\]", r"_gasp_", r"_gasping")),
    ("scream", (r"\[scream\]", r"_scream_", r"_yells")),
    ("grunt", (r"\[grunt\]", r"_grunt_", r"_grunts")),
    ("sigh", (r"\[sigh\]", r"_sigh_", r"_sighs")),
)


def _compile_rules() -> List[Tuple[str, List[re.Pattern]]]:
    out: List[Tuple[str, List[re.Pattern]]] = []
    for name, pats in NONVERBAL_RULES:
        out.append((name, [re.compile(p, re.IGNORECASE) for p in pats]))
    return out


_RULES = _compile_rules()


def classify_clip(filename: str, transcript: str) -> str:
    blob = f"{filename} {transcript}".lower()
    for nv, patterns in _RULES:
        if any(p.search(blob) for p in patterns):
            return nv
    return "verbal"


def _read_metadata_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (fieldnames, rows as dicts). Supports tab or comma CSV.

    Also handles a common malformed-CSV variant used in this project where the
    header is ``filename|text`` (pipe-joined inside a single column, padded with
    trailing commas) and each row is ``<filename>|<transcript>,,,,`` with
    transcripts that themselves contain commas. Parsing such a file as plain
    comma-CSV destroys the transcript; we detect the pipe-header and split on
    the first ``|`` per line instead.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], []

    header_stripped = lines[0].rstrip(",").strip()
    if "|" in header_stripped and "," not in header_stripped:
        parts = [p.strip() for p in header_stripped.split("|")]
        if parts and parts[0].lower() in {"filename", "filepath", "file", "wav", "path", "clip"}:
            fn_col = parts[0].lower()
            tx_col = parts[1].lower() if len(parts) > 1 else "transcript"
            if tx_col == "text":
                tx_col = "transcript"
            rows: List[Dict[str, str]] = []
            for ln in lines[1:]:
                stripped = ln.rstrip(",").rstrip()
                if not stripped:
                    continue
                # Split on the FIRST '|'; transcripts may contain commas and
                # pipes are not used inside the wav basename. Trailing commas
                # are just CSV padding from the source export.
                fn, sep, tx = stripped.partition("|")
                rows.append({fn_col: fn.strip(), tx_col: tx.strip().rstrip(",").rstrip()})
            return [fn_col, tx_col], rows

    delim = "\t" if lines[0].count("\t") >= lines[0].count(",") else ","

    # If first line looks like a header (contains 'filename' or 'path')
    header_like = re.search(r"filename|filepath|path|wav|text|transcript", lines[0], re.I)
    if header_like:
        reader = csv.DictReader(lines, delimiter=delim)
        rows = [dict(r) for r in reader]
        return reader.fieldnames or [], rows

    # No header: try tab with id, file+text combined
    rows: List[Dict[str, str]] = []
    for ln in lines:
        parts = ln.split(delim)
        row: Dict[str, str] = {}
        if len(parts) >= 3:
            row["id"] = parts[0].strip()
            row["filename"] = parts[1].strip()
            row["transcript"] = delim.join(parts[2:]).strip()
        elif len(parts) == 2:
            row["id"] = ""
            row["filename"] = parts[0].strip()
            row["transcript"] = parts[1].strip()
        else:
            row["id"] = ""
            row["filename"] = ln.strip()
            row["transcript"] = ""
        if "|" in row.get("filename", ""):
            fn, _, rest = row["filename"].partition("|")
            row["filename"] = fn.strip()
            row["transcript"] = (rest.strip() + " " + row.get("transcript", "")).strip()
        rows.append(row)
    return list(rows[0].keys()) if rows else [], rows


def _normalize_columns(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Map diverse headers to filename + transcript."""
    if not rows:
        return rows
    keys = {k.lower(): k for k in rows[0]}
    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in keys:
                return keys[c]
            if c.lower() in keys:
                return keys[c.lower()]
        return None

    fn_key = pick("filename", "filepath", "file", "wav", "path", "clip", "audio_file")
    tx_key = pick("transcript", "text", "label", "caption", "utterance")
    out: List[Dict[str, str]] = []
    for r in rows:
        fn = ""
        tx = ""
        if fn_key:
            fn = str(r.get(fn_key, "") or "")
        if tx_key:
            tx = str(r.get(tx_key, "") or "")
        if not fn:
            # single column rows
            for v in r.values():
                if str(v).lower().endswith((".wav", ".flac", ".mp3")):
                    fn = str(v)
                    break
        if "|" in fn:
            a, _, b = fn.partition("|")
            fn, tx = a.strip(), (b.strip() + " " + tx).strip()
        out.append({"filename": fn, "transcript": tx, **{k: v for k, v in r.items() if k not in (fn_key, tx_key)}})
    return out


_AUDIO_BASENAME_INDEX: Dict[str, Dict[str, Path]] = {}


def _build_audio_basename_index(audio_root: Path) -> Dict[str, Path]:
    """Lazily build (and cache) a `{basename: absolute_path}` index for every
    audio file under ``audio_root``, recursing into subdirectories.

    This lets the caller pass any ancestor of the actual wav folder as
    ``--audio-root`` and still resolve filenames by basename. Built once per
    process per resolved root.
    """
    key = str(audio_root.resolve())
    cached = _AUDIO_BASENAME_INDEX.get(key)
    if cached is not None:
        return cached
    idx: Dict[str, Path] = {}
    if audio_root.is_dir():
        for ext in (".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3"):
            for p in audio_root.rglob(f"*{ext}"):
                if p.is_file():
                    idx.setdefault(p.name, p)
    _AUDIO_BASENAME_INDEX[key] = idx
    if not idx:
        print(
            f"[WARN] _build_audio_basename_index: no audio files found under "
            f"{audio_root} (rglob *.wav / *.flac / *.mp3).",
            file=sys.stderr,
        )
    return idx


def _duration_for_row(
    row: Dict[str, str],
    audio_root: Optional[Path],
    duration_key: Optional[str],
) -> Optional[float]:
    if duration_key and duration_key in row:
        try:
            return float(row[duration_key])
        except (TypeError, ValueError):
            pass
    if not audio_root or not HAS_SF:
        return None
    fn = row.get("filename") or ""
    if not fn:
        return None
    # 1) absolute path in metadata wins.
    p = Path(fn)
    if p.is_absolute() and p.is_file():
        try:
            return float(sf.info(str(p)).duration)
        except OSError:
            return None
    # 2) direct child of audio_root (basename and full-relative-path attempts).
    candidates: List[Path] = [audio_root / p.name, audio_root / fn]
    for cand in candidates:
        if cand.is_file():
            try:
                return float(sf.info(str(cand)).duration)
            except OSError:
                return None
    # 3) recursive basename lookup under audio_root (handles e.g. wavs/ nesting).
    idx = _build_audio_basename_index(audio_root)
    found = idx.get(p.name)
    if found is not None and found.is_file():
        try:
            return float(sf.info(str(found)).duration)
        except OSError:
            return None
    return None


def build_table1(
    metadata_csv: Path,
    audio_root: Optional[Path],
    duration_key: Optional[str],
) -> List[Dict[str, Any]]:
    _fieldnames, rows = _read_metadata_rows(metadata_csv)
    rows = _normalize_columns(rows)
    durations: List[Optional[float]] = []
    cats: List[str] = []
    for r in rows:
        cats.append(classify_clip(r.get("filename", ""), r.get("transcript", "")))
        durations.append(_duration_for_row(r, audio_root, duration_key))

    order = ["verbal"] + [nv for nv, _ in NONVERBAL_RULES] + ["total"]
    thresholds = {
        "verbal": "ECAPA-TDNN cosine ≥ 0.50",
        "laugh": "CLAP cosine ≥ 0.78",
        "cry": "CLAP cosine ≥ 0.78",
        "gasp": "CLAP cosine ≥ 0.78",
        "scream": "CLAP cosine ≥ 0.78",
        "grunt": "CLAP cosine ≥ 0.78",
        "sigh": "CLAP cosine ≥ 0.78",
        "total": "—",
    }

    table_rows: List[Dict[str, Any]] = []
    all_ds = [d for d in durations if d is not None and math.isfinite(d) and d > 0]

    for cat in order:
        if cat == "total":
            mask = [True] * len(cats)
            label = "Total"
        else:
            mask = [c == cat for c in cats]
            label = "Verbal" if cat == "verbal" else f"Non-Verbal ({cat})"
        n = sum(mask)
        ds = [durations[i] for i in range(len(mask)) if mask[i] and durations[i] is not None]
        if ds:
            mean_d = float(sum(ds) / len(ds))
            sorted_ds = sorted(ds)
            mid = len(sorted_ds) // 2
            if len(sorted_ds) % 2:
                med_d = float(sorted_ds[mid])
            else:
                med_d = 0.5 * (sorted_ds[mid - 1] + sorted_ds[mid])
        else:
            mean_d = float("nan")
            med_d = float("nan")
        table_rows.append(
            {
                "Category": label,
                "Total Clips": int(n),
                "Mean Duration (s)": mean_d,
                "Median Duration (s)": med_d,
                "Verification Threshold": thresholds[cat if cat != "total" else "total"],
            }
        )

    return table_rows


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def savefig(path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def fig1_memory_bars(out: Path, bf16_gb: float, gguf_gb: float, ceiling_gb: float) -> None:
    labels = [f"BF16 baseline\n{bf16_gb:.2f} GB", f"v12 GGUF\n{gguf_gb:.2f} GB", f"Orin Nano ceiling\n{ceiling_gb:.2f} GB"]
    vals = [bf16_gb, gguf_gb, ceiling_gb]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = range(len(labels))
    ax.barh(y, vals, color=colors, height=0.55)
    ax.axvline(ceiling_gb, color="red", linestyle="--", linewidth=2, label="Hardware ceiling")
    ax.set_xlabel("Memory (GB)")
    ax.set_title("Figure 1 — Memory footprint vs hardware ceiling")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.legend(loc="lower right")
    savefig(out)


def fig2_pipeline(out: Path) -> None:
    stages = [
        "Raw episodes",
        "Vocal separation",
        "Transcription",
        "Speaker verification",
        "Stereo-doubling fix",
        "CLAP non-verbal refinement",
        "AppraisalVector labels",
        "968-clip dataset",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(stages) + 1)
    ax.axis("off")
    ax.set_title("Figure 2 — Data processing pipeline", fontsize=14, pad=16)
    w, h = 7.0, 0.55
    x0 = 1.5
    for i, name in enumerate(stages):
        y = len(stages) - i
        box = FancyBboxPatch(
            (x0, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor="#E8EEF7" if i < len(stages) - 1 else "#D5F5E3",
        )
        ax.add_patch(box)
        ax.text(x0 + w / 2, y, name, ha="center", va="center", fontsize=10)
        if i < len(stages) - 1:
            arr = FancyArrowPatch(
                (x0 + w / 2, y - h / 2 - 0.05),
                (x0 + w / 2, y - 1.0 + h / 2 + 0.05),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color="#333333",
            )
            ax.add_patch(arr)
    savefig(out)


def unpack_uint2_mask(packed: "torch.Tensor", numel: int) -> "torch.Tensor":
    """Unpack packed uint2 tier stream to one tier id per weight element (0–3).

    Must match ``qat_septq.unpack_tier_mask_uint2`` / ``inspect_mask_tier_purity``:
    lane ``i`` fills indices ``i, i+4, i+8, ...``.
    """
    flat = packed.detach().to(dtype=torch.uint8).reshape(-1)
    max_unpacked = int(flat.numel()) * 4
    n = max(0, min(int(numel), max_unpacked))
    if n == 0:
        return torch.zeros(0, dtype=torch.long)
    expanded = torch.zeros(max_unpacked, dtype=torch.uint8)
    for lane in range(4):
        expanded[lane::4] = (flat >> (lane * 2)) & 0b11
    out = expanded[:n].to(torch.long)
    return torch.clamp(out, 0, 3)


def _weight_numel_for_mask_key(
    ckpt: dict, mask_key: str, packed: "torch.Tensor", layer_idx: int
) -> int:
    """Match tier mask key to the same temporal layer's weight tensor in state_dict."""
    sd = ckpt.get("state_dict")
    if not isinstance(sd, dict):
        return 4 * int(packed.numel())
    prefix = f"transformer.layers.{int(layer_idx)}."
    if not mask_key.startswith(prefix):
        return 4 * int(packed.numel())
    candidates = [
        mask_key,
        mask_key + ".weight",
        mask_key.replace(".weight", ""),
        mask_key.replace(".weight", "") + ".weight",
    ]
    for c in candidates:
        t = sd.get(c)
        if torch.is_tensor(t):
            return int(t.numel())
    base = mask_key.split(".")[-1]
    for k, t in sd.items():
        if not k.startswith(prefix):
            continue
        if not torch.is_tensor(t) or t.ndim < 2:
            continue
        if k.endswith(base) or k.endswith(base + ".weight"):
            return int(t.numel())
    return 4 * int(packed.numel())


_TIER_LABELS = ["FP16", "INT8", "INT4", "INT2"]
_TIER_COLORS_HEX = ["#2ca02c", "#ffdd57", "#ff7f0e", "#d62728"]
_TIER_BITS = np.array([16.0, 8.0, 4.0, 2.0], dtype=np.float64)


def _lookup_2d_weight(
    sd: Dict[str, "torch.Tensor"], key: str
) -> Optional["torch.Tensor"]:
    """Resolve a 2-D weight tensor from state_dict for a (possibly base) key.

    Cannot use ``a or b or c`` here: torch tensors raise
    ``RuntimeError: Boolean value of Tensor with more than one value is
    ambiguous`` when evaluated for truthiness. Use explicit ``is None`` /
    ``torch.is_tensor`` checks.
    """
    candidates = [key, key + ".weight", key.replace(".weight", "")]
    seen: set = set()
    for ck in candidates:
        if ck in seen:
            continue
        seen.add(ck)
        w = sd.get(ck)
        if torch.is_tensor(w) and w.ndim == 2:
            return w
    return None


def _pick_representative_module(
    masks: Dict[str, "torch.Tensor"],
    sd: Dict[str, "torch.Tensor"],
    preferred: Sequence[str] = (
        "transformer.layers.0.self_attn.in_proj_weight",
        "transformer.layers.0.gating_linear_in.weight",
        "transformer.layers.0.gating_linear_in",
    ),
) -> Optional[Tuple[str, "torch.Tensor", Tuple[int, int]]]:
    """Return (mask_key, mask_tensor, (rows, cols)) for a representative module.

    Falls back to the largest 2-D weight that has a tier mask if none of the
    preferred keys exist.
    """
    for key in preferred:
        m = masks.get(key)
        if m is None or not torch.is_tensor(m):
            continue
        w = _lookup_2d_weight(sd, key)
        if w is not None:
            return key, m, (int(w.shape[0]), int(w.shape[1]))

    best: Optional[Tuple[str, "torch.Tensor", Tuple[int, int]]] = None
    best_numel = -1
    for key, m in masks.items():
        if not isinstance(key, str) or not torch.is_tensor(m):
            continue
        w = _lookup_2d_weight(sd, key)
        if w is None:
            continue
        n = int(w.shape[0]) * int(w.shape[1])
        if n > best_numel:
            best_numel = n
            best = (key, m, (int(w.shape[0]), int(w.shape[1])))
    return best


def _downsample_tier_fractions(
    tier_grid: np.ndarray,
    target_rows: int = 256,
    target_cols: int = 384,
) -> Tuple[np.ndarray, int, int]:
    """Reduce an (rows, cols) tier grid to (h2, w2, 4) per-tier fractions per tile.

    Returns the fraction array plus the chosen pool height / width (ph, pw) so the
    caller can label the figure ("tile = 48 x 11 weights").
    """
    rows, cols = tier_grid.shape
    ph = max(1, rows // target_rows)
    pw = max(1, cols // target_cols)
    h2 = rows // ph
    w2 = cols // pw
    if h2 == 0 or w2 == 0:
        # Fallback: degenerate sizes, return the raw grid as one-hot fractions.
        one_hot = np.zeros((rows, cols, 4), dtype=np.float32)
        for t in range(4):
            one_hot[..., t] = (tier_grid == t).astype(np.float32)
        return one_hot, 1, 1
    cropped = tier_grid[: h2 * ph, : w2 * pw]
    tile = cropped.reshape(h2, ph, w2, pw)
    tile_area = float(ph * pw)
    fracs = np.stack(
        [(tile == t).sum(axis=(1, 3)).astype(np.float32) / tile_area for t in range(4)],
        axis=-1,
    )
    return fracs, ph, pw


def fig3_tier_heatmap(out: Path, ckpt_path: Path, n_layers: int = 32) -> None:
    """SEPTQ tier-assignment figure.

    Background. The earlier per-(layer, module) "effective bpw" heatmap
    rendered a single uniform value (e.g. 3.72) because the SEPTQ multi-tier
    PTQ in this codebase applies the **same global ratios** (FP16/INT8/INT4/
    INT2 = 0.02/0.12/0.36/0.50) per tensor, so every quantized module has
    identical bpw by construction. That metric has zero per-module variance.

    What actually varies and is interesting for the report is the **spatial
    placement** of the tiers WITHIN a tensor: Hessian-saliency-driven element
    routing puts the FP16 / INT8 budget on the rows / columns where outlier
    activations live, and pushes the rest to INT2. We visualize this for one
    representative module (layer 0 ``self_attn.in_proj_weight`` by default,
    falling back to the largest masked 2-D weight in the checkpoint).
    """
    if not HAS_TORCH:
        _placeholder(out, "Figure 3 — SEPTQ tier heatmap\n(torch missing; install torch)")
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu", mmap=True)
    if not isinstance(ckpt, dict):
        _placeholder(out, "Figure 3 — checkpoint not a dict")
        return
    masks = ckpt.get("tier_masks_uint2")
    if not isinstance(masks, dict) or not masks:
        sm = ckpt.get("septq_meta")
        if isinstance(sm, dict):
            masks = sm.get("tier_masks_uint2")
    if not isinstance(masks, dict) or not masks:
        _placeholder(out, "Figure 3 — no tier_masks_uint2 in checkpoint")
        return
    sd = ckpt.get("state_dict")
    if not isinstance(sd, dict):
        _placeholder(out, "Figure 3 — checkpoint missing state_dict")
        return

    picked = _pick_representative_module(masks, sd)
    if picked is None:
        _placeholder(out, "Figure 3 — no representative 2-D masked module found")
        return
    mask_key, mask_tensor, (rows, cols) = picked

    numel = rows * cols
    flat = unpack_uint2_mask(mask_tensor, numel)
    if int(flat.numel()) != numel:
        _placeholder(
            out,
            f"Figure 3 — tier mask numel mismatch for\n{mask_key}\n"
            f"got {int(flat.numel())} unpacked vs {numel} expected",
        )
        return
    tier_grid = flat.cpu().numpy().reshape(rows, cols).astype(np.int32)
    tier_fracs, pool_h, pool_w = _downsample_tier_fractions(
        tier_grid, target_rows=256, target_cols=384,
    )

    # Global tier composition across ALL temporal-stack masks, for the side panel.
    pat = re.compile(r"^transformer\.layers\.(\d+)\.(.+)$")
    total_counts = np.zeros(4, dtype=np.int64)
    n_modules = 0
    for k, v in masks.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            continue
        if v.dtype != torch.uint8:
            continue
        m = pat.match(k)
        if not m or int(m.group(1)) >= n_layers:
            continue
        wn = _weight_numel_for_mask_key(ckpt, k, v, int(m.group(1)))
        max_unpacked = 4 * int(v.numel())
        n = min(int(wn), max_unpacked)
        if n <= 0:
            continue
        sub = unpack_uint2_mask(v, n).cpu().numpy()
        if sub.size == 0:
            continue
        bc = np.bincount(sub, minlength=4).astype(np.int64)[:4]
        total_counts += bc
        n_modules += 1

    grand_total = float(total_counts.sum())
    if grand_total > 0:
        agg_fracs = total_counts / grand_total
        agg_bpw = float(np.dot(_TIER_BITS, agg_fracs))
    else:
        agg_fracs = np.zeros(4, dtype=np.float64)
        agg_bpw = float("nan")

    # ---- Render: 4-panel per-tier fraction view + stack-wide bar ----
    # Why this layout: with global ratios 2/12/36/50, a "dominant tier" tile
    # reduction makes the 2% FP16 budget invisible (no tile is majority FP16).
    # Showing each tier's LOCAL fraction-per-tile as its own panel exposes
    # exactly where the saliency router clustered the high-precision bits
    # (look for bright rows/cols in the FP16 and INT8 panels).
    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.85],
        height_ratios=[1.0, 0.08],
        wspace=0.18, hspace=0.05,
    )
    pretty_key = mask_key.replace("transformer.layers.", "L").replace(".weight", "")
    fig.suptitle(
        "Figure 3 — SEPTQ within-tensor tier assignment "
        f"(representative module: {pretty_key})\n"
        "Each panel = local fraction of one tier per spatial tile "
        f"({pool_h}\u00d7{pool_w} weights / tile, downsampled from {rows}\u00d7{cols}).",
        fontsize=11, y=1.02,
    )

    # The 0..1 fraction scale is per-panel-relative so the 2% FP16 budget is
    # still visible. We don't share a single colorbar across panels because
    # the dynamic ranges are very different (FP16 might max at 30% locally,
    # INT2 will max at ~100%).
    panel_max = [
        max(0.05, float(tier_fracs[..., i].max())) if tier_fracs.size else 1.0
        for i in range(4)
    ]
    panel_axes = []
    last_im = None
    for i in range(4):
        ax_i = fig.add_subplot(gs[0, i])
        panel_axes.append(ax_i)
        cmap_i = LinearSegmentedColormap.from_list(
            f"white_to_{_TIER_LABELS[i]}",
            ["#ffffff", _TIER_COLORS_HEX[i]],
            N=256,
        )
        last_im = ax_i.imshow(
            tier_fracs[..., i],
            aspect="auto", cmap=cmap_i,
            vmin=0.0, vmax=panel_max[i], interpolation="nearest",
        )
        ax_i.set_title(
            f"{_TIER_LABELS[i]}  ({agg_fracs[i] * 100:.1f}% global)\n"
            f"panel scale 0\u2013{panel_max[i] * 100:.0f}%",
            fontsize=9,
        )
        if i == 0:
            ax_i.set_ylabel(f"output dim (rows / {pool_h})")
        else:
            ax_i.set_yticklabels([])
        ax_i.set_xlabel(f"input dim (cols / {pool_w})")
        # Per-panel colorbar so the per-tier dynamic range is readable.
        cbar = plt.colorbar(last_im, ax=ax_i, fraction=0.04, pad=0.02)
        cbar.set_label("fraction of tile", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    # ---- Side panel: aggregate tier mix across the whole stack ----
    ax_side = fig.add_subplot(gs[0, 4])
    bottom = 0.0
    for i in range(4):
        ax_side.bar(
            [0], [agg_fracs[i]], bottom=[bottom], width=0.6,
            color=_TIER_COLORS_HEX[i], edgecolor="white",
            label=f"{_TIER_LABELS[i]}: {agg_fracs[i] * 100:.1f}%",
        )
        bottom += agg_fracs[i]
    ax_side.set_ylim(0, 1)
    ax_side.set_xlim(-1, 1)
    ax_side.set_xticks([])
    ax_side.set_ylabel("Fraction of elements")
    ax_side.set_title("Stack-wide tier mix", fontsize=10)
    ax_side.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0),
                   fontsize=8, frameon=False)

    subtitle_lines = [f"n modules: {n_modules}"]
    if math.isfinite(agg_bpw):
        subtitle_lines.append(f"mean bpw: {agg_bpw:.2f}")
    ax_side.text(
        0, -0.05, "\n".join(subtitle_lines),
        ha="center", va="top", fontsize=9, transform=ax_side.transAxes,
    )

    savefig(out)


def fig4_duration_hist(out: Path, durations: Sequence[float], bin_min: float, bin_max: float) -> None:
    arr = np.array([d for d in durations if math.isfinite(d) and bin_min <= d <= bin_max], dtype=np.float64)
    if arr.size == 0:
        _placeholder(out, "Figure 4 — no durations (set --audio-root + soundfile or add duration column)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(bin_min, bin_max, 36)
    ax.hist(arr, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.mean(arr)), color="darkred", linestyle="--", label=f"mean={np.mean(arr):.2f}s")
    ax.axvline(float(np.median(arr)), color="orange", linestyle="--", label=f"median={np.median(arr):.2f}s")
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Figure 4 — Acoustic clip duration distribution")
    ax.legend()
    savefig(out)


def fig5_cosine_cascade(out: Path, zs_json: Optional[Path], cascade_csv: Optional[Path]) -> None:
    layers: List[int] = []
    cos: List[float] = []
    if cascade_csv and cascade_csv.is_file():
        with cascade_csv.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                layers.append(int(row["layer"]))
                cos.append(float(row["cosine"]))
    elif zs_json and zs_json.is_file():
        data = json.loads(zs_json.read_text(encoding="utf-8"))
        ls = data.get("layer_summary") or {}
        per = ls.get("per_layer") or []
        for row in sorted(per, key=lambda x: int(x["layer"])):
            layers.append(int(row["layer"]))
            cos.append(float(row["cos_median"]))
    else:
        # illustrative v11-shaped curve (replace with real JSON when available)
        for i in range(32):
            layers.append(i)
            # smooth decay 0.999 -> ~0.686
            t = i / 31.0
            cos.append(0.999 * (1 - t) + 0.686525 * t + 0.02 * math.sin(t * math.pi))
        print(
            "[WARN] Figure 5: using synthetic cascade (provide --zs-json or --cascade-csv for measured data).",
            file=sys.stderr,
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, cos, marker="o", linewidth=1.5, markersize=3, color="#1f77b4")
    ax.axhline(0.997, color="red", linestyle="--", linewidth=1.2, label="Production threshold (0.997)")
    ax.axhline(0.90, color="darkred", linestyle="--", linewidth=1.2, label="QAT abort threshold (0.90)")
    ax.set_xlabel("Temporal layer index")
    ax.set_ylabel("z_s cosine similarity")
    ax.set_title("Figure 5 — v11 cosine cascade (per-layer median z_s)")
    ax.set_xticks(range(0, max(layers) + 1, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    savefig(out)


def fig6_rss_schematic(out: Path, components_gb: Sequence[Tuple[str, float]], ceiling_gb: float) -> None:
    """Stacked area vs pseudo-time (schematic, not measured RSS trace)."""
    labels = [c[0] for c in components_gb]
    vals = np.array([c[1] for c in components_gb], dtype=np.float64)
    t = np.linspace(0, 1, 48)
    ramps = []
    for i, v in enumerate(vals):
        curve = np.minimum(1.0, t * (3.0 + i * 0.4)) * v
        ramps.append(curve)
    stack = np.vstack(ramps)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(t, stack, labels=labels, alpha=0.85)
    ax.axhline(ceiling_gb, color="red", linestyle="--", linewidth=2, label=f"RAM ceiling ({ceiling_gb:.1f} GB)")
    ymax = max(float(vals.sum()), ceiling_gb) * 1.12
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["Init", "Load pools", "KV alloc", "Warmup", "Steady decode"])
    ax.set_ylabel("Resident / allocated mass (GB, schematic)")
    ax.set_title("Figure 6 — RSS / allocation components (current loader; not lazy mmap)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    savefig(out)


def _placeholder(out: Path, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, wrap=True)
    ax.axis("off")
    savefig(out)


def _fmt_dur(x: Any) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(xf):
        return "—"
    return f"{xf:.2f}"


def write_table_files(out_dir: Path, table_rows: List[Dict[str, Any]]) -> None:
    md_path = out_dir / "TABLE1_dataset_overview.md"
    csv_path = out_dir / "TABLE1_dataset_overview.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Category", "Total Clips", "Mean Duration (s)", "Median Duration (s)", "Verification Threshold"],
        )
        w.writeheader()
        for row in table_rows:
            w.writerow(
                {
                    "Category": row["Category"],
                    "Total Clips": row["Total Clips"],
                    "Mean Duration (s)": _fmt_dur(row["Mean Duration (s)"]),
                    "Median Duration (s)": _fmt_dur(row["Median Duration (s)"]),
                    "Verification Threshold": row["Verification Threshold"],
                }
            )

    md = ["# TABLE 1 — Dataset statistical overview", ""]
    md.append(
        "| Category | Total Clips | Mean Duration (s) | Median Duration (s) | Verification Threshold |"
    )
    md.append("| --- | ---: | ---: | ---: | --- |")
    for row in table_rows:
        md.append(
            f"| {row['Category']} | {row['Total Clips']} | {_fmt_dur(row['Mean Duration (s)'])} | "
            f"{_fmt_dur(row['Median Duration (s)'])} | {row['Verification Threshold']} |"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def write_readme(out_dir: Path, _args: argparse.Namespace) -> None:
    """Copy canonical docs next to the script into the output folder."""
    canonical = Path(__file__).resolve().parent / "README_REPORT_FIGURES.md"
    dest = out_dir / "README_DATA_REQUIREMENTS.md"
    if canonical.is_file():
        dest.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        dest.write_text(
            "# Report assets\n\n(Missing `scripts/README_REPORT_FIGURES.md` next to the generator.)\n",
            encoding="utf-8",
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Generate report tables and figures (1–6).")
    p.add_argument("--out-dir", type=Path, default=Path("report_figures"))
    p.add_argument("--metadata", type=Path, help="Path to metadata.csv / TSV for dataset table + histogram.")
    p.add_argument("--audio-root", type=Path, help="Directory containing WAVs referenced by basename in metadata.")
    p.add_argument("--duration-column", type=str, default="", help="Optional column name for duration in seconds.")
    p.add_argument("--hist-min", type=float, default=1.0)
    p.add_argument("--hist-max", type=float, default=12.0)
    p.add_argument("--septq-ckpt", type=Path, help="SEPTQ multitier .pt for Figure 3 tier heatmap.")
    p.add_argument("--zs-json", type=Path, help="verify_septq_zs_drift --save-json output for Figure 5.")
    p.add_argument("--cascade-csv", type=Path, help="Optional two-column CSV: layer,cosine (overrides zs-json).")
    p.add_argument("--bf16-gb", type=float, default=16.7)
    p.add_argument("--gguf-gb", type=float, default=7.69)
    p.add_argument("--ceiling-gb", type=float, default=5.5)
    p.add_argument("--rss-gb-weights", type=float, default=7.2, help="Fig 6: GGUF pools (scalar+big), schematic GB.")
    p.add_argument("--rss-gb-kv", type=float, default=0.5, help="Fig 6: temporal FP16 KV (order-of-magnitude).")
    p.add_argument("--rss-gb-depth-kv", type=float, default=0.002)
    p.add_argument("--rss-gb-work", type=float, default=1.0, help="Fig 6: ggml work arena (Jetson default 1 GiB).")
    p.add_argument("--rss-gb-cuda", type=float, default=0.15, help="Fig 6: staging / misc CUDA host buffers.")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows: List[Dict[str, Any]] = []
    durations_list: List[float] = []

    if args.metadata and args.metadata.is_file():
        table_rows = build_table1(
            args.metadata,
            args.audio_root.resolve() if args.audio_root else None,
            args.duration_column or None,
        )
        write_table_files(out_dir, table_rows)
        # rebuild durations for histogram
        _, rows_raw = _read_metadata_rows(args.metadata)
        rows = _normalize_columns(rows_raw)
        ar_resolved = args.audio_root.resolve() if args.audio_root else None

        # Diagnostic counters so a silent "Figure 4 needs durations" placeholder
        # actually tells us what failed: how many rows have a filename, how many
        # have a numeric duration column, how many resolve to a real file.
        n_total = len(rows)
        n_with_filename = sum(1 for r in rows if r.get("filename"))
        n_via_dur_col = 0
        n_via_audio = 0
        n_unresolved_sample: List[str] = []
        for r in rows:
            d = _duration_for_row(r, ar_resolved, args.duration_column or None)
            if d is not None and math.isfinite(d) and d > 0:
                durations_list.append(d)
                if args.duration_column and args.duration_column in r:
                    try:
                        float(r[args.duration_column])
                        n_via_dur_col += 1
                        continue
                    except (TypeError, ValueError):
                        pass
                n_via_audio += 1
            elif r.get("filename") and len(n_unresolved_sample) < 5:
                n_unresolved_sample.append(r["filename"])

        if not durations_list:
            reasons: List[str] = []
            if n_total == 0:
                reasons.append("metadata has zero rows after parsing")
            elif n_with_filename == 0:
                reasons.append("no row has a 'filename' column after header normalization "
                               "(check that metadata has a column named filename/filepath/path/wav/clip)")
            elif not HAS_SF and not args.duration_column:
                reasons.append("soundfile not importable AND no --duration-column "
                               "(install soundfile, or supply a numeric duration column)")
            elif ar_resolved is None:
                reasons.append("no --audio-root provided")
            elif not ar_resolved.is_dir():
                reasons.append(f"--audio-root {ar_resolved} is not a directory")
            else:
                reasons.append(f"every filename failed to resolve under {ar_resolved} "
                               f"(direct, basename, and recursive lookups). "
                               f"Examples (up to 5): {n_unresolved_sample}")
            print(
                f"[WARN] Figure 4: 0 / {n_total} rows resolved to a duration. "
                f"n_with_filename={n_with_filename}, n_via_dur_col={n_via_dur_col}, "
                f"n_via_audio={n_via_audio}. Reasons: {'; '.join(reasons)}",
                file=sys.stderr,
            )
        else:
            print(
                f"[OK] Figure 4: {len(durations_list)} / {n_total} rows resolved "
                f"(via duration column: {n_via_dur_col}, via audio: {n_via_audio}).",
                file=sys.stderr,
            )
    else:
        print("[WARN] No --metadata; skipping Table 1 and Figure 4.", file=sys.stderr)

    # Each figure is independently sandboxed: a failure in one figure must not
    # prevent subsequent figures from rendering. Errors are captured and replaced
    # with a placeholder PNG plus a stderr trace.
    def _safe_fig(label: str, out_path: Path, fn) -> None:
        try:
            fn()
        except Exception as exc:  # pragma: no cover -- defensive
            import traceback
            print(f"[WARN] {label} failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            _placeholder(out_path, f"{label} failed\n{type(exc).__name__}: {exc}")

    _safe_fig(
        "Figure 1",
        out_dir / "figure1_memory_footprint.png",
        lambda: fig1_memory_bars(out_dir / "figure1_memory_footprint.png",
                                 args.bf16_gb, args.gguf_gb, args.ceiling_gb),
    )
    _safe_fig(
        "Figure 2",
        out_dir / "figure2_data_pipeline.png",
        lambda: fig2_pipeline(out_dir / "figure2_data_pipeline.png"),
    )

    if args.septq_ckpt and args.septq_ckpt.is_file():
        _safe_fig(
            "Figure 3",
            out_dir / "figure3_septq_tier_heatmap.png",
            lambda: fig3_tier_heatmap(out_dir / "figure3_septq_tier_heatmap.png",
                                      args.septq_ckpt),
        )
    else:
        _placeholder(out_dir / "figure3_septq_tier_heatmap.png",
                     "Figure 3 — pass --septq-ckpt\n(e.g. bmo_temporal_half_cushion_max.pt)")

    if durations_list:
        _safe_fig(
            "Figure 4",
            out_dir / "figure4_duration_histogram.png",
            lambda: fig4_duration_hist(out_dir / "figure4_duration_histogram.png",
                                       durations_list, args.hist_min, args.hist_max),
        )
    else:
        _placeholder(out_dir / "figure4_duration_histogram.png",
                     "Figure 4 — need durations\n(--audio-root + soundfile or duration column)")

    _safe_fig(
        "Figure 5",
        out_dir / "figure5_v11_cosine_cascade.png",
        lambda: fig5_cosine_cascade(out_dir / "figure5_v11_cosine_cascade.png",
                                    args.zs_json, args.cascade_csv),
    )

    _safe_fig(
        "Figure 6",
        out_dir / "figure6_rss_schematic.png",
        lambda: fig6_rss_schematic(
            out_dir / "figure6_rss_schematic.png",
            [
                ("GGUF tensor pools (scalar + big)", args.rss_gb_weights),
                ("Temporal KV (FP16, n_ctx-bounded)", args.rss_gb_kv),
                ("Depth KV", args.rss_gb_depth_kv),
                ("ggml work arena", args.rss_gb_work),
                ("CUDA staging / registered", args.rss_gb_cuda),
            ],
            args.ceiling_gb,
        ),
    )

    write_readme(out_dir, args)
    print(f"[OK] Wrote assets under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
