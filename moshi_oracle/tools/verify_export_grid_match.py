#!/usr/bin/env python3
"""verify_export_grid_match.py

Export-grid verification for BMO_TIER multi-tier ("tile-region") quantized
gating layers, Candidate C ("Heavy Cushion").

For ONE gating layer, this script compares:

  (A) GROUND TRUTH — the INTEGER quantization levels physically written into
      the shipped GGUF (`qat_heavy_int2.gguf`, `<base>.packed_weights`
      unpacked using `<base>.tile_tiers` / `.n_tiles` / `.tier_offsets`), and

  (B) RECOMPUTED — the same integer levels independently recomputed, from
      scratch, straight from `qat_best.pt`'s own dense (QAT-refined) weight
      for that layer, using the exact affine round-to-grid formula the
      exporter (`export_bmo_gguf.py::pack_tile_region_ffn`, as vendored in
      the server-side `personaplex_repo` -- see PROVENANCE below) applies:

          q = clamp(round(w / scale + zero_point), 0, qmax)

      with `w` taken from `qat_best.pt`'s dense weight (outlier positions
      zeroed first, mirroring the exporter's own `w_bulk`), the SAME
      per-tile tier assignment (`qat_best.pt`'s own `tile_region_metadata`),
      and the SAME scale/zero-point (resolved the same way the exporter
      resolves them: from `septq_meta.per_layer_stats` on the checkpoint
      pointed to by `qat_meta.source_student_quant_meta`, since `qat_best.pt`
      itself carries no `septq_meta`).

  (C) CROSS-CHECK — `qat_septq.MultiTierFakeQuantize` (imported, not
      reimplemented) run on the same dense weight with a per-element tier
      mask expanded from the same tile assignment, to confirm the manual
      per-tile formula in (B) agrees with the project's own official
      fake-quant module at the dequantized-float layer. This directly
      answers "does the GGUF encode the same grid QAT trained on" using the
      project's own reference implementation, not a reimplementation of it.

(A) vs (B) is the primary, literal "bit-for-bit exported quantized values"
comparison the task asked for -- see WHY INTEGER-LEVEL COMPARISON below.
FP16 tier-0 tiles and the sparse FP16 outlier sidecar are also compared,
separately, since they are not on any integer grid.

PROVENANCE (read this before trusting the numbers) -----------------------
The GGUF-export entry point is NOT in this repo. The canonical, tracked copy
of `export_bmo_gguf.py` under `BMO Voice Engine/personaplex/` in THIS repo
implements only an older, unrelated block_size=32 per-element/per-block
scheme (no `tile_tiers`/`n_tiles`/`tier_offsets`/`outlier_indices` at all --
verified by grep, zero hits). The version that actually has a
`pack_tile_region_ffn` function matching the shipped GGUF's tensor layout
lives only in the OTHER repo this project uses for running Python-side
scripts, `/home/jovyan/work/BMO-Project/personaplex_repo/export_bmo_gguf.py`
(1551 lines vs. the canonical copy's 1266 -- a real, uncommitted divergence,
not a copy/paste error on this script's part; see FINDINGS in the report
this script's caller produces). This script does NOT import or execute
that file (to avoid depending on an unversioned, possibly-still-drifting
script with GPU-training side effects in its `__main__`); instead it
re-implements just the ~15-line integer quantization formula from
`pack_tile_region_ffn`, verified line-for-line against a direct read of
that function, and documents that formula inline below
(`recompute_qat_tile_streams`). `qat_best.pt` and `qat_septq.py` /
`apply_septq_multitier.py` (imported read-only, never modified) are used
exactly as instructed.

WHY INTEGER-LEVEL COMPARISON (not dequantized-float) -----------------------
Both `MultiTierFakeQuantize.forward()` (qat_septq.py) and the exporter
compute the SAME affine formula `round(w/scale + zp)` and then either keep
the integer (exporter, for on-device packing) or immediately dequantize it
back to float (fake-quant, for the STE gradient path). Comparing at the
float layer conflates two possible failure modes (wrong integer level vs.
floating-point re-dequantization noise) into one number. Comparing at the
INTEGER level is the literal, stricter reading of "exported quantized
values" the task specifies, and it is what actually ships in the GGUF
(`packed_weights`) -- so it is the ground truth this script treats as
primary. The float-layer MultiTierFakeQuantize cross-check (C) is reported
too, as a secondary corroboration, not the headline number.

READ-ONLY: this script never writes to qat_best.pt, the source PTQ .pt, or
the shipped GGUF. It does not use the GPU (plain tensor ops on CPU; see
report for why CPU was chosen over GPU for this check).

Usage:
    PYTHONPATH=/home/jovyan/work/BMO-Project/personaplex_repo/moshi \\
    /home/jovyan/work/envs/BMO-Project/bin/python \\
        moshi_oracle/tools/verify_export_grid_match.py \\
        [--module-name transformer.layers.0.gating.linear_in.weight]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths (defaults match the confirmed-real files for this session; override
# via CLI flags if verifying a different layer/checkpoint).
# ---------------------------------------------------------------------------

CANONICAL_PERSONAPLEX_DIR = Path(
    "/home/jovyan/work/BMO-Project-Repo/BMO-Project/BMO Voice Engine/personaplex"
)
DEFAULT_MOSHI_PYTHONPATH = Path("/home/jovyan/work/BMO-Project/personaplex_repo/moshi")

DEFAULT_QAT_CKPT = Path(
    "/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/"
    "qat_heavy_int2/qat_best.pt"
)
DEFAULT_GGUF = Path(
    "/home/jovyan/work/BMO-Project-Repo/BMO-Project/moshi_oracle/models_h100_actual/"
    "qat_heavy_int2_dir/qat_heavy_int2.gguf"
)
DEFAULT_MODULE_NAME = "transformer.layers.0.gating.linear_in.weight"

TIER_NAMES = {0: "fp16", 1: "int8", 2: "int4", 3: "int2"}
TIER_QMAX = {1: 255, 2: 15, 3: 3}


def _ensure_imports() -> None:
    """Put the canonical personaplex dir + vendored moshi package on sys.path.

    qat_septq.py / apply_septq_multitier.py are imported read-only, exactly as
    the task instructs -- never modified, never executed as __main__.
    """
    p = str(CANONICAL_PERSONAPLEX_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    if DEFAULT_MOSHI_PYTHONPATH.is_dir():
        mp = str(DEFAULT_MOSHI_PYTHONPATH)
        if mp not in sys.path:
            sys.path.insert(0, mp)


_ensure_imports()

try:
    import gguf  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[FATAL] the 'gguf' python package is required (pip install gguf, or it "
        f"ships in envs/BMO-Project already). Import error: {exc}"
    )

import qat_septq  # noqa: E402  (read-only reference/import only)
import apply_septq_multitier as asm  # noqa: E402  (read-only reference/import only)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_ckpt(path: Path) -> Dict[str, Any]:
    print(f"[LOAD] {path}  ({path.stat().st_size / 1e9:.2f} GB, mmap=True)")
    t0 = time.perf_counter()
    payload = torch.load(str(path), map_location="cpu", mmap=True)
    print(f"[LOAD]   done in {time.perf_counter() - t0:.2f}s")
    if not isinstance(payload, dict):
        raise SystemExit(f"[FATAL] {path}: expected a dict checkpoint, got {type(payload)}")
    return payload


def resolve_stats_source(
    qat_ckpt: Dict[str, Any],
    qat_path: Path,
    override: Optional[Path],
) -> Tuple[Dict[str, Any], str]:
    """Mirror export_bmo_gguf.py::resolve_stats_source_ckpt's resolution order.

    Returns (payload, human-readable description of where it came from).
    """
    if override is not None:
        payload = load_ckpt(override)
        return payload, f"--ptq-ckpt override: {override}"

    sm = qat_ckpt.get("septq_meta")
    tm = qat_ckpt.get("tier_masks_uint2")
    if (
        isinstance(sm, dict)
        and isinstance(sm.get("per_layer_stats"), list)
        and len(sm["per_layer_stats"]) > 0
        and isinstance(tm, dict)
        and tm
    ):
        return qat_ckpt, f"septq_meta.per_layer_stats found directly in {qat_path.name}"

    qm = qat_ckpt.get("qat_meta")
    rel = qm.get("source_student_quant_meta") if isinstance(qm, dict) else None
    if not isinstance(rel, str) or not rel.strip():
        raise SystemExit(
            f"[FATAL] {qat_path}: no septq_meta on the checkpoint itself, and no "
            "qat_meta.source_student_quant_meta to fall back to; pass --ptq-ckpt explicitly."
        )
    src_path = Path(rel)
    if not src_path.is_absolute():
        src_path = (qat_path.parent / rel).resolve()
    if not src_path.is_file():
        raise SystemExit(f"[FATAL] qat_meta.source_student_quant_meta not found: {src_path}")
    payload = load_ckpt(src_path)
    sm2 = payload.get("septq_meta")
    if not isinstance(sm2, dict) or not isinstance(sm2.get("per_layer_stats"), list):
        raise SystemExit(f"[FATAL] {src_path}: missing septq_meta.per_layer_stats")
    return payload, f"qat_meta.source_student_quant_meta -> {src_path}"


def canonical_gguf_base(module_name: str) -> str:
    """Mirror export_bmo_gguf.py::canonical_transformer_multitier_gguf_base."""
    underscore = module_name.replace(".", "_")
    return re.sub(r"^transformer_inner_layers_(\d+)_", r"transformer_layers_\1_", underscore)


# ---------------------------------------------------------------------------
# GGUF ground-truth unpacking
# ---------------------------------------------------------------------------


class GgufTileTensor:
    """Unpacked view of one `.packed_weights` tile-region tensor from the GGUF."""

    def __init__(self, reader: "gguf.GGUFReader", base: str):
        by_name = {t.name: t for t in reader.tensors}

        def need(suffix: str) -> np.ndarray:
            key = f"{base}.{suffix}"
            if key not in by_name:
                raise SystemExit(f"[FATAL] GGUF missing tensor: {key}")
            return by_name[key].data

        self.rows = int(need("rows")[0])
        self.cols = int(need("cols")[0])
        self.packing_version = int(need("packing_version")[0])
        self.n_tiles = need("n_tiles").astype(np.int64)  # [n_fp16, n_int8, n_int4, n_int2]
        self.tier_offsets = need("tier_offsets").astype(np.int64)  # len 5
        self.tile_tiers = need("tile_tiers").view(np.uint8).copy()
        self.scale_int8 = float(need("scale_int8")[0])
        self.zp_int8 = float(need("zp_int8")[0])
        self.scale_int4 = float(need("scale_int4")[0])
        self.zp_int4 = float(need("zp_int4")[0])
        self.scale_int2 = float(need("scale_low")[0])
        self.zp_int2 = float(need("zp_low")[0])
        self.n_outliers = int(need("n_outliers")[0])
        if self.n_outliers > 0:
            self.outlier_indices = need("outlier_indices").astype(np.int64)
            self.outlier_values = need("outlier_values").view(np.float16).copy()
        else:
            self.outlier_indices = np.empty((0,), dtype=np.int64)
            self.outlier_values = np.empty((0,), dtype=np.float16)

        packed = need("packed_weights").view(np.uint8)
        o = self.tier_offsets
        self.fp16_bytes = packed[o[0] : o[1]]
        self.int8_bytes = packed[o[1] : o[2]]
        self.int4_bytes = packed[o[2] : o[3]]
        self.int2_bytes = packed[o[3] : o[4]]

    def tile_shape(self) -> Tuple[int, int]:
        # 64x64 for this project's "ampere" tile-layout target; derived, not hardcoded,
        # from n_tiles_total vs. rows/cols so this stays correct if that ever changes.
        n_tiles_total = int(self.n_tiles.sum())
        # tile grid is square-ish per tier config here (64x64); recover from rows/cols
        # and the known total tile count is not enough alone, so read it from tile_tiers
        # length only as a sanity cross-check; actual geometry passed in by caller.
        return n_tiles_total, n_tiles_total  # placeholder, unused; see unpack_all()

    def unpack_all(
        self, tile_rows: int, tile_cols: int
    ) -> Dict[str, np.ndarray]:
        n_fp16, n_int8, n_int4, n_int2 = (int(x) for x in self.n_tiles)
        elems = tile_rows * tile_cols

        fp16_arr = self.fp16_bytes.view(np.float16).reshape(n_fp16, tile_rows, tile_cols)

        int8_arr = self.int8_bytes.reshape(n_int8, tile_rows, tile_cols)

        int4_flat = np.empty(n_int4 * elems, dtype=np.uint8)
        int4_flat[0::2] = self.int4_bytes & 0x0F
        int4_flat[1::2] = (self.int4_bytes >> 4) & 0x0F
        int4_arr = int4_flat.reshape(n_int4, tile_rows, tile_cols)

        int2_flat = np.empty(n_int2 * elems, dtype=np.uint8)
        int2_flat[0::4] = self.int2_bytes & 0x03
        int2_flat[1::4] = (self.int2_bytes >> 2) & 0x03
        int2_flat[2::4] = (self.int2_bytes >> 4) & 0x03
        int2_flat[3::4] = (self.int2_bytes >> 6) & 0x03

        int2_arr = int2_flat.reshape(n_int2, tile_rows, tile_cols)

        return {"fp16": fp16_arr, "int8": int8_arr, "int4": int4_arr, "int2": int2_arr}


# ---------------------------------------------------------------------------
# qat_best.pt-side recomputation (mirrors export_bmo_gguf.py::pack_tile_region_ffn)
# ---------------------------------------------------------------------------


def recompute_qat_tile_streams(
    w_dense: torch.Tensor,
    tile_tiers: torch.Tensor,
    tile_shape: Tuple[int, int],
    tile_grid: Tuple[int, int],
    outlier_indices: Optional[torch.Tensor],
    scale_int8: float,
    zp_int8: float,
    scale_int4: float,
    zp_int4: float,
    scale_int2: float,
    zp_int2: float,
) -> Dict[str, np.ndarray]:
    """Reproduce pack_tile_region_ffn's tile extraction + quantization exactly.

    Loop order and formula verified line-for-line against
    personaplex_repo/export_bmo_gguf.py::pack_tile_region_ffn (lines ~884-1025
    at the time of this session). Vectorized here (reshape/permute) instead of
    the exporter's per-tile python loop, which is equivalent ONLY when the
    tensor divides evenly into the tile grid (checked by the caller via
    pad_rows == pad_cols == 0; falls back to the literal per-tile loop
    otherwise so this stays correct even off the fast path).
    """
    tile_rows, tile_cols = tile_shape
    n_tiles_row, n_tiles_col = tile_grid
    rows, cols = int(w_dense.shape[0]), int(w_dense.shape[1])

    w_bulk = w_dense.clone()
    if outlier_indices is not None and outlier_indices.numel() > 0:
        w_bulk.reshape(-1)[outlier_indices.to(torch.long)] = 0.0

    n_tiles_total = int(tile_tiers.numel())
    exact_fit = (n_tiles_row * tile_rows == rows) and (n_tiles_col * tile_cols == cols)

    tiers_np = tile_tiers.to(torch.uint8).numpy()

    if exact_fit:
        # Fast vectorized path: (rows, cols) -> (n_tiles_row, n_tiles_col, tile_rows, tile_cols)
        # tile order t_idx = tile_r * n_tiles_col + tile_c, matching the exporter's
        # tile_r = t_idx // n_tiles_col; tile_c = t_idx % n_tiles_col.
        blocks = (
            w_bulk.view(n_tiles_row, tile_rows, n_tiles_col, tile_cols)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(n_tiles_total, tile_rows, tile_cols)
        )
    else:
        print(
            "[WARN] tensor does not divide evenly into the tile grid "
            f"(rows={rows} cols={cols} tile_shape={tile_shape} tile_grid={tile_grid}); "
            "falling back to the exporter's literal per-tile edge-clipped loop.",
            file=sys.stderr,
        )
        blocks_list = []
        for t_idx in range(n_tiles_total):
            tr, tc = divmod(t_idx, n_tiles_col)
            r0, c0 = tr * tile_rows, tc * tile_cols
            tile = torch.zeros((tile_rows, tile_cols), dtype=w_bulk.dtype)
            sub = w_bulk[r0 : r0 + tile_rows, c0 : c0 + tile_cols]
            tile[: sub.shape[0], : sub.shape[1]] = sub
            blocks_list.append(tile)
        blocks = torch.stack(blocks_list, dim=0)

    order = np.argsort(tiers_np, kind="stable")  # groups by tier, preserves ascending t_idx within tier
    tier_sorted = tiers_np[order]

    def _select(tier_code: int) -> torch.Tensor:
        idx = order[tier_sorted == tier_code]
        if idx.size == 0:
            return blocks.new_zeros((0, tile_rows, tile_cols))
        return blocks[torch.from_numpy(idx.astype(np.int64))]

    fp16_tiles = _select(0)
    int8_tiles = _select(1)
    int4_tiles = _select(2)
    int2_tiles = _select(3)

    out: Dict[str, np.ndarray] = {}
    out["fp16"] = fp16_tiles.to(torch.float16).numpy()
    out["int8"] = (
        torch.round(int8_tiles / max(scale_int8, 1e-12) + zp_int8)
        .clamp(0, TIER_QMAX[1])
        .to(torch.uint8)
        .numpy()
    )
    out["int4"] = (
        torch.round(int4_tiles / max(scale_int4, 1e-12) + zp_int4)
        .clamp(0, TIER_QMAX[2])
        .to(torch.uint8)
        .numpy()
    )
    out["int2"] = (
        torch.round(int2_tiles / max(scale_int2, 1e-12) + zp_int2)
        .clamp(0, TIER_QMAX[3])
        .to(torch.uint8)
        .numpy()
    )
    return out


# ---------------------------------------------------------------------------
# Comparison + reporting
# ---------------------------------------------------------------------------


def compare_integer_tier(
    tier_name: str,
    exported: np.ndarray,
    recomputed: np.ndarray,
    scale: float,
) -> Dict[str, Any]:
    if exported.shape != recomputed.shape:
        return {
            "tier": tier_name,
            "shape_mismatch": True,
            "exported_shape": exported.shape,
            "recomputed_shape": recomputed.shape,
        }
    n = int(exported.size)
    if n == 0:
        return {"tier": tier_name, "n_elements": 0, "n_exact_match": 0, "n_mismatch": 0}

    exp_i = exported.astype(np.int32)
    rec_i = recomputed.astype(np.int32)
    delta = exp_i - rec_i
    mismatch = delta != 0
    n_mismatch = int(mismatch.sum())

    hist: Dict[str, int] = {}
    if n_mismatch > 0:
        abs_delta = np.abs(delta[mismatch])
        for bucket, lo, hi in (("1", 1, 2), ("2", 2, 3), ("3", 3, 4), (">=4", 4, None)):
            if hi is None:
                cnt = int((abs_delta >= lo).sum())
            else:
                cnt = int(((abs_delta >= lo) & (abs_delta < hi)).sum())
            hist[f"|delta_q|={bucket}"] = cnt
        dequant_delta = np.abs(delta[mismatch]).astype(np.float64) * scale
        max_dequant_delta = float(dequant_delta.max())
        mean_dequant_delta = float(dequant_delta.mean())
    else:
        max_dequant_delta = 0.0
        mean_dequant_delta = 0.0

    return {
        "tier": tier_name,
        "shape_mismatch": False,
        "n_elements": n,
        "n_exact_match": n - n_mismatch,
        "n_mismatch": n_mismatch,
        "match_rate": (n - n_mismatch) / n,
        "delta_q_histogram": hist,
        "max_dequant_delta": max_dequant_delta,
        "mean_dequant_delta_over_mismatches": mean_dequant_delta,
    }


def compare_fp16_tier(exported: np.ndarray, recomputed: np.ndarray) -> Dict[str, Any]:
    if exported.shape != recomputed.shape:
        return {
            "tier": "fp16",
            "shape_mismatch": True,
            "exported_shape": exported.shape,
            "recomputed_shape": recomputed.shape,
        }
    n = int(exported.size)
    if n == 0:
        return {"tier": "fp16", "n_elements": 0, "n_exact_match": 0, "n_mismatch": 0}
    exp_bits = exported.view(np.uint16)
    rec_bits = recomputed.view(np.uint16)
    mismatch = exp_bits != rec_bits
    n_mismatch = int(mismatch.sum())
    if n_mismatch > 0:
        abs_diff = np.abs(exported.astype(np.float32) - recomputed.astype(np.float32))[mismatch]
        max_abs_diff = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
    return {
        "tier": "fp16",
        "shape_mismatch": False,
        "n_elements": n,
        "n_exact_match": n - n_mismatch,
        "n_mismatch": n_mismatch,
        "match_rate": (n - n_mismatch) / n,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff_over_mismatches": mean_abs_diff,
    }


def compare_outliers(
    gguf_idx: np.ndarray,
    gguf_val: np.ndarray,
    ckpt_idx: np.ndarray,
    ckpt_val_f16: np.ndarray,
) -> Dict[str, Any]:
    n_gguf, n_ckpt = int(gguf_idx.size), int(ckpt_idx.size)
    same_order = (
        n_gguf == n_ckpt
        and bool(np.array_equal(gguf_idx, ckpt_idx))
    )
    if same_order:
        val_mismatch = int((gguf_val.view(np.uint16) != ckpt_val_f16.view(np.uint16)).sum())
        return {
            "n_gguf": n_gguf,
            "n_ckpt": n_ckpt,
            "same_order_indices": True,
            "index_match": True,
            "n_value_mismatch": val_mismatch,
            "n_value_exact_match": n_gguf - val_mismatch,
        }

    # fall back to set/position-independent comparison
    gguf_set = set(gguf_idx.tolist())
    ckpt_set = set(ckpt_idx.tolist())
    idx_match = gguf_set == ckpt_set
    n_value_mismatch = None
    if idx_match:
        g_order = np.argsort(gguf_idx)
        c_order = np.argsort(ckpt_idx)
        val_mismatch = int(
            (
                gguf_val[g_order].view(np.uint16)
                != ckpt_val_f16[c_order].view(np.uint16)
            ).sum()
        )
        n_value_mismatch = val_mismatch
    return {
        "n_gguf": n_gguf,
        "n_ckpt": n_ckpt,
        "same_order_indices": False,
        "index_match": idx_match,
        "n_value_mismatch": n_value_mismatch,
        "n_value_exact_match": (n_gguf - n_value_mismatch) if n_value_mismatch is not None else None,
    }


def fakequant_crosscheck(
    w_dense_f32: torch.Tensor,
    tile_tiers: torch.Tensor,
    tile_shape: Tuple[int, int],
    tile_grid: Tuple[int, int],
    outlier_indices: Optional[torch.Tensor],
    quant_params: Dict[str, float],
    gguf_dequant_bulk: torch.Tensor,
) -> Dict[str, Any]:
    """(C): independent cross-check using qat_septq.MultiTierFakeQuantize directly.

    Expands the SAME per-tile tier assignment to a per-element mask (mirroring
    apply_septq_multitier.expand_2d_tile_tiers), forces outlier positions to
    tier 0 (mirroring qat_septq.force_outlier_tier0_in_masks), packs it with
    apply_septq_multitier.pack_tier_mask_uint2, and runs the project's own
    MultiTierFakeQuantize.forward() -- not a reimplementation -- on the dense
    weight. Reports cosine/rmse of this module's output against the GGUF's
    own dequantized bulk grid (scale*(q_exported-zp) at every non-outlier
    position), i.e. the float-layer analogue of the primary integer check.
    """
    rows, cols = int(w_dense_f32.shape[0]), int(w_dense_f32.shape[1])
    elem_mask = asm.expand_2d_tile_tiers(
        tile_tiers=tile_tiers,
        shape=[rows, cols],
        tile_shape=list(tile_shape),
        tile_grid=list(tile_grid),
        device=torch.device("cpu"),
    )
    if outlier_indices is not None and outlier_indices.numel() > 0:
        elem_mask = elem_mask.clone()
        elem_mask.reshape(-1)[outlier_indices.to(torch.long)] = 0

    packed_mask, _ = asm.pack_tier_mask_uint2(elem_mask)

    fq = qat_septq.MultiTierFakeQuantize(
        tier_mask_packed=packed_mask,
        tier_mask_shape=(rows, cols),
        scale_int8=quant_params["scale_int8"],
        zero_point_int8=quant_params["zero_point_int8"],
        scale_int4=quant_params["scale_int4"],
        zero_point_int4=quant_params["zero_point_int4"],
        scale_int2=quant_params["scale_int2"],
        zero_point_int2=quant_params["zero_point_int2"],
    )
    w_fq = fq(w_dense_f32).detach()

    non_tier0 = (elem_mask != 0)
    if outlier_indices is not None and outlier_indices.numel() > 0:
        outlier_mask = torch.zeros_like(non_tier0)
        outlier_mask.reshape(-1)[outlier_indices.to(torch.long)] = True
        compare_mask = non_tier0 & (~outlier_mask)
    else:
        compare_mask = non_tier0

    a = w_fq[compare_mask].reshape(-1).double()
    b = gguf_dequant_bulk[compare_mask].reshape(-1).double()
    diff = (a - b).abs()
    denom = float(torch.linalg.norm(a) * torch.linalg.norm(b))
    cosine = float((a @ b) / denom) if denom > 0 else float("nan")
    return {
        "n_compared": int(a.numel()),
        "cosine": cosine,
        "max_abs_err": float(diff.max()) if a.numel() else 0.0,
        "mean_abs_err": float(diff.mean()) if a.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff**2))) if a.numel() else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qat-ckpt", type=Path, default=DEFAULT_QAT_CKPT)
    ap.add_argument("--ptq-ckpt", type=Path, default=None, help="Override auto-resolved PTQ stats source.")
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--module-name", type=str, default=DEFAULT_MODULE_NAME)
    args = ap.parse_args()

    qat_path = args.qat_ckpt.resolve()
    gguf_path = args.gguf.resolve()
    module_name = args.module_name

    if not qat_path.is_file():
        raise SystemExit(f"[FATAL] qat checkpoint not found: {qat_path}")
    if not gguf_path.is_file():
        raise SystemExit(f"[FATAL] GGUF not found: {gguf_path}")

    print("=" * 78)
    print("verify_export_grid_match.py")
    print("=" * 78)
    print(f"module_name : {module_name}")
    print(f"qat_ckpt    : {qat_path}")
    print(f"gguf        : {gguf_path}")

    qat_ckpt = load_ckpt(qat_path)

    state_dict = qat_ckpt.get("state_dict")
    if not isinstance(state_dict, dict) or module_name not in state_dict:
        raise SystemExit(f"[FATAL] {qat_path}: state_dict missing key {module_name!r}")
    w_dense = state_dict[module_name].detach()
    print(f"[INFO] dense weight: shape={tuple(w_dense.shape)} dtype={w_dense.dtype}")
    w_dense_f32 = w_dense.to(dtype=torch.float32).contiguous()

    trm = qat_ckpt.get("tile_region_metadata")
    if not isinstance(trm, dict) or module_name not in trm.get("tiles", {}):
        raise SystemExit(f"[FATAL] {qat_path}: tile_region_metadata.tiles missing {module_name!r}")
    tile_info = trm["tiles"][module_name]
    tile_tiers_ckpt = tile_info["tile_tiers"].to(torch.uint8).contiguous()
    tile_shape = tuple(int(x) for x in tile_info["tile_shape"])
    tile_grid = tuple(int(x) for x in tile_info["tile_grid"])
    pad_rows = int(tile_info.get("pad_rows", 0))
    pad_cols = int(tile_info.get("pad_cols", 0))
    print(
        f"[INFO] tile_region_metadata: tile_shape={tile_shape} tile_grid={tile_grid} "
        f"n_tiles_total={int(tile_tiers_ckpt.numel())} pad_rows={pad_rows} pad_cols={pad_cols}"
    )

    om = qat_ckpt.get("outlier_metadata", {})
    outlier_entry = om.get(module_name) if isinstance(om, dict) else None
    if outlier_entry is not None:
        outlier_indices_ckpt = outlier_entry["indices"].to(torch.int64)
        outlier_values_ckpt = outlier_entry["values"].detach()
        print(f"[INFO] outlier_metadata: n_outliers={int(outlier_indices_ckpt.numel())}")
    else:
        outlier_indices_ckpt = None
        outlier_values_ckpt = None
        print("[INFO] outlier_metadata: none for this module")

    stats_payload, stats_desc = resolve_stats_source(qat_ckpt, qat_path, args.ptq_ckpt)
    print(f"[INFO] scale/zero-point source: {stats_desc}")
    quant_params = qat_septq.resolve_multitier_quant_params(module_name, ckpt=stats_payload)
    print(
        "[INFO] resolved quant params: "
        f"scale_int8={quant_params['scale_int8']!r} zp_int8={quant_params['zero_point_int8']!r} "
        f"scale_int4={quant_params['scale_int4']!r} zp_int4={quant_params['zero_point_int4']!r} "
        f"scale_int2={quant_params['scale_int2']!r} zp_int2={quant_params['zero_point_int2']!r}"
    )

    # -------------------------------------------------------------- GGUF side
    base = canonical_gguf_base(module_name)
    print(f"[INFO] GGUF tensor base: {base}")
    reader = gguf.GGUFReader(str(gguf_path))
    gt = GgufTileTensor(reader, base)
    print(
        f"[INFO] GGUF: rows={gt.rows} cols={gt.cols} packing_version={gt.packing_version} "
        f"n_tiles(fp16,int8,int4,int2)={gt.n_tiles.tolist()} n_outliers={gt.n_outliers}"
    )

    if (gt.rows, gt.cols) != tuple(w_dense.shape):
        raise SystemExit(
            f"[FATAL] shape mismatch: GGUF rows/cols=({gt.rows},{gt.cols}) "
            f"vs qat_best.pt weight shape={tuple(w_dense.shape)}"
        )

    tiers_equal = bool(np.array_equal(gt.tile_tiers, tile_tiers_ckpt.numpy()))
    print(f"[CHECK] tile_tiers assignment: GGUF vs qat_best.pt tile_region_metadata -> "
          f"{'EQUAL' if tiers_equal else 'MISMATCH'}")

    exported = gt.unpack_all(tile_shape[0], tile_shape[1])

    # ---------------------------------------------------------- qat_best side
    recomputed = recompute_qat_tile_streams(
        w_dense=w_dense_f32,
        tile_tiers=tile_tiers_ckpt,
        tile_shape=tile_shape,
        tile_grid=tile_grid,
        outlier_indices=outlier_indices_ckpt,
        scale_int8=quant_params["scale_int8"],
        zp_int8=quant_params["zero_point_int8"],
        scale_int4=quant_params["scale_int4"],
        zp_int4=quant_params["zero_point_int4"],
        scale_int2=quant_params["scale_int2"],
        zp_int2=quant_params["zero_point_int2"],
    )

    # --------------------------------------------------------------- compare
    print()
    print("-" * 78)
    print("PRIMARY: integer quantization-level comparison (exported GGUF vs. recomputed from qat_best.pt)")
    print("-" * 78)

    results: Dict[str, Any] = {}
    results["fp16"] = compare_fp16_tier(exported["fp16"], recomputed["fp16"])
    results["int8"] = compare_integer_tier("int8", exported["int8"], recomputed["int8"], quant_params["scale_int8"])
    results["int4"] = compare_integer_tier("int4", exported["int4"], recomputed["int4"], quant_params["scale_int4"])
    results["int2"] = compare_integer_tier("int2", exported["int2"], recomputed["int2"], quant_params["scale_int2"])

    total_elements = 0
    total_exact = 0
    for tier_name in ("fp16", "int8", "int4", "int2"):
        r = results[tier_name]
        if r.get("shape_mismatch"):
            print(f"  tier={tier_name:<5} SHAPE MISMATCH exported={r['exported_shape']} recomputed={r['recomputed_shape']}")
            continue
        n = r["n_elements"]
        total_elements += n
        total_exact += r["n_exact_match"]
        print(
            f"  tier={tier_name:<5} n_elements={n:>10} exact_match={r['n_exact_match']:>10} "
            f"mismatch={r['n_mismatch']:>8} match_rate={r.get('match_rate', float('nan')):.8f}"
        )
        if tier_name != "fp16" and r["n_mismatch"] > 0:
            print(f"      delta_q histogram: {r['delta_q_histogram']}")
            print(
                f"      max_dequant_delta={r['max_dequant_delta']:.6e}  "
                f"mean_dequant_delta_over_mismatches={r['mean_dequant_delta_over_mismatches']:.6e}"
            )
        elif tier_name == "fp16" and r["n_mismatch"] > 0:
            print(f"      max_abs_diff={r['max_abs_diff']:.6e}  mean_abs_diff_over_mismatches={r['mean_abs_diff_over_mismatches']:.6e}")

    print()
    print("-" * 78)
    print("OUTLIER sidecar comparison (exported GGUF vs. qat_best.pt outlier_metadata)")
    print("-" * 78)
    if outlier_indices_ckpt is not None and gt.n_outliers > 0:
        outlier_result = compare_outliers(
            gguf_idx=gt.outlier_indices,
            gguf_val=gt.outlier_values,
            ckpt_idx=outlier_indices_ckpt.numpy(),
            ckpt_val_f16=outlier_values_ckpt.to(torch.float16).numpy(),
        )
        print(f"  n_outliers: gguf={outlier_result['n_gguf']} ckpt={outlier_result['n_ckpt']}")
        print(f"  same_order_indices={outlier_result['same_order_indices']} index_match={outlier_result['index_match']}")
        print(
            f"  n_value_exact_match={outlier_result['n_value_exact_match']} "
            f"n_value_mismatch={outlier_result['n_value_mismatch']}"
        )
        total_elements += outlier_result["n_gguf"]
        total_exact += outlier_result["n_value_exact_match"] or 0
    else:
        outlier_result = None
        print("  no outliers for this module in one or both sources; skipped.")

    print()
    print("-" * 78)
    print("SECONDARY: MultiTierFakeQuantize cross-check (dequantized-float layer, qat_septq.py, imported not reimplemented)")
    print("-" * 78)
    # Build a full dense dequantized-bulk tensor from the GGUF's own exported
    # integer levels, at every element position, for the crosscheck comparison.
    gguf_dequant_bulk = torch.zeros_like(w_dense_f32)
    tile_rows, tile_cols = tile_shape
    n_tiles_col = tile_grid[1]
    order_by_tier: Dict[int, np.ndarray] = {}
    tiers_np = gt.tile_tiers
    for tcode in (0, 1, 2, 3):
        order_by_tier[tcode] = np.nonzero(tiers_np == tcode)[0]
    tier_scale = {1: quant_params["scale_int8"], 2: quant_params["scale_int4"], 3: quant_params["scale_int2"]}
    tier_zp = {1: quant_params["zero_point_int8"], 2: quant_params["zero_point_int4"], 3: quant_params["zero_point_int2"]}
    stream_key = {1: "int8", 2: "int4", 3: "int2"}
    for tcode in (1, 2, 3):
        t_indices = order_by_tier[tcode]
        if t_indices.size == 0:
            continue
        q_vals = torch.from_numpy(exported[stream_key[tcode]].astype(np.float32))
        dq_vals = tier_scale[tcode] * (q_vals - tier_zp[tcode])
        for k, t_idx in enumerate(t_indices.tolist()):
            tr, tc = divmod(t_idx, n_tiles_col)
            r0, c0 = tr * tile_rows, tc * tile_cols
            gguf_dequant_bulk[r0 : r0 + tile_rows, c0 : c0 + tile_cols] = dq_vals[k]

    crosscheck = fakequant_crosscheck(
        w_dense_f32=w_dense_f32,
        tile_tiers=tile_tiers_ckpt,
        tile_shape=tile_shape,
        tile_grid=tile_grid,
        outlier_indices=outlier_indices_ckpt,
        quant_params=quant_params,
        gguf_dequant_bulk=gguf_dequant_bulk,
    )
    print(
        f"  n_compared={crosscheck['n_compared']} cosine={crosscheck['cosine']:.10f} "
        f"max_abs_err={crosscheck['max_abs_err']:.6e} mean_abs_err={crosscheck['mean_abs_err']:.6e} "
        f"rmse={crosscheck['rmse']:.6e}"
    )

    print()
    print("=" * 78)
    all_exact = (total_exact == total_elements) and tiers_equal
    if outlier_result is not None and not outlier_result["index_match"]:
        all_exact = False
    if all_exact:
        print(f"VERDICT: MATCH -- {total_exact}/{total_elements} elements bit-identical "
              f"(bulk grid + fp16 tiles + outliers), tile_tiers assignment equal.")
    else:
        print(
            f"VERDICT: MISMATCH -- {total_exact}/{total_elements} elements bit-identical "
            f"({total_elements - total_exact} mismatched). tile_tiers assignment "
            f"{'equal' if tiers_equal else 'MISMATCH'}."
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
