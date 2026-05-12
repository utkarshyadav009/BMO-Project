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
    from matplotlib.colors import BoundaryNorm, ListedColormap
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
    """Return (fieldnames, rows as dicts). Supports tab or comma CSV."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], []

    # Sniff delimiter
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
    p = Path(fn)
    if not p.is_absolute():
        p = audio_root / p.name
    if not p.is_file():
        p = audio_root / fn
    if not p.is_file():
        return None
    try:
        info = sf.info(str(p))
        return float(info.duration)
    except OSError:
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


def fig3_tier_heatmap(out: Path, ckpt_path: Path, n_layers: int = 32) -> None:
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

    pat = re.compile(r"^transformer\.layers\.(\d+)\.(.+)$")
    layer_mod: Dict[int, Dict[str, Tuple[str, "torch.Tensor"]]] = defaultdict(dict)
    for k, v in masks.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            continue
        if v.dtype != torch.uint8:
            continue
        m = pat.match(k)
        if not m:
            continue
        li = int(m.group(1))
        rest = str(m.group(2))
        if li < 0 or li >= n_layers:
            continue
        # collapse duplicate .weight suffix for grouping
        short = rest.replace(".weight", "")
        layer_mod[li][short] = (k, v)

    # union of module names sorted
    all_names = sorted({nm for d in layer_mod.values() for nm in d.keys()})
    if not all_names:
        _placeholder(out, "Figure 3 — no temporal tier keys matched")
        return

    mat = np.full((n_layers, len(all_names)), -1.0)
    for li in range(n_layers):
        for j, name in enumerate(all_names):
            pair = layer_mod[li].get(name)
            if pair is None:
                continue
            full_key, t = pair
            wn = _weight_numel_for_mask_key(ckpt, full_key, t, li)
            max_unpacked = 4 * int(t.numel())
            # Decode only bytes that exist; never index past packed stream.
            numel = min(int(wn), max_unpacked)
            flat = unpack_uint2_mask(t, numel)
            if flat.numel() == 0:
                continue
            bc = torch.bincount(flat, minlength=4)
            tier = int(torch.argmax(bc).item())
            mat[li, j] = float(tier)

    # Discrete tiers 0..3: imshow + ListedColormap + linear norm maps floats onto the
    # whole colormap gradient, so almost everything looks like the top color (INT2/red).
    # BoundaryNorm pins 0,1,2,3 to the four list colors.
    cmap = ListedColormap(["#2ca02c", "#ffdd57", "#ff7f0e", "#d62728"])  # FP16, INT8, INT4, INT2
    cmap.set_bad("#d9d9d9")
    norm = BoundaryNorm(np.arange(-0.5, 4.0, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(max(10, len(all_names) * 0.35), 7))
    mat_masked = np.ma.masked_where(mat < 0, mat)
    im = ax.imshow(mat_masked, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i:02d}" for i in range(n_layers)])
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=55, ha="right", fontsize=7)
    ax.set_title("Figure 3 — SEPTQ tier assignment (dominant tier per module, temporal stack)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["FP16", "INT8", "INT4", "INT2"])
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
        _, rows = _read_metadata_rows(args.metadata)
        rows = _normalize_columns(rows)
        for r in rows:
            d = _duration_for_row(r, args.audio_root.resolve() if args.audio_root else None, args.duration_column or None)
            if d is not None:
                durations_list.append(d)
    else:
        print("[WARN] No --metadata; skipping Table 1 and Figure 4.", file=sys.stderr)

    fig1_memory_bars(out_dir / "figure1_memory_footprint.png", args.bf16_gb, args.gguf_gb, args.ceiling_gb)
    fig2_pipeline(out_dir / "figure2_data_pipeline.png")

    if args.septq_ckpt and args.septq_ckpt.is_file():
        fig3_tier_heatmap(out_dir / "figure3_septq_tier_heatmap.png", args.septq_ckpt)
    else:
        _placeholder(out_dir / "figure3_septq_tier_heatmap.png", "Figure 3 — pass --septq-ckpt\n(e.g. bmo_temporal_half_cushion_max.pt)")

    if durations_list:
        fig4_duration_hist(out_dir / "figure4_duration_histogram.png", durations_list, args.hist_min, args.hist_max)
    else:
        _placeholder(out_dir / "figure4_duration_histogram.png", "Figure 4 — need durations\n(--audio-root + soundfile or duration column)")

    fig5_cosine_cascade(out_dir / "figure5_v11_cosine_cascade.png", args.zs_json, args.cascade_csv)

    fig6_rss_schematic(
        out_dir / "figure6_rss_schematic.png",
        [
            ("GGUF tensor pools (scalar + big)", args.rss_gb_weights),
            ("Temporal KV (FP16, n_ctx-bounded)", args.rss_gb_kv),
            ("Depth KV", args.rss_gb_depth_kv),
            ("ggml work arena", args.rss_gb_work),
            ("CUDA staging / registered", args.rss_gb_cuda),
        ],
        args.ceiling_gb,
    )

    write_readme(out_dir, args)
    print(f"[OK] Wrote assets under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
