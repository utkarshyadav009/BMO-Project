#!/usr/bin/env python3
"""Compare PTQ-era per_layer_stats quant scales/zero-points vs GGUF scalar tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_LOCAL_GGUF_PY = _REPO / "llama.cpp" / "gguf-py"
if _LOCAL_GGUF_PY.is_dir():
    sys.path.insert(0, str(_LOCAL_GGUF_PY))

import gguf  # type: ignore  # noqa: E402

from convert_septq_to_fp16 import _get_numpy, _tensor_dict, pick_tensor, read_scalar_f32  # noqa: E402

from export_bmo_gguf import resolve_stats_source_ckpt  # noqa: E402
from qat_septq import get_module_meta  # noqa: E402

_TOL = 1e-6


def _f(meta: dict, *keys: str) -> float | None:
    for k in keys:
        if k not in meta:
            continue
        v = meta[k]
        if v is None:
            continue
        if torch.is_tensor(v):
            return float(v.detach().item())
        return float(v)
    return None


def main() -> None:
    here = _REPO
    ap = argparse.ArgumentParser(description="Verify GGUF scale/zp scalars match PTQ per_layer_stats.")
    ap.add_argument("--qat-ckpt", type=Path, default=here / "qat_septq_final_run" / "qat_best.pt")
    ap.add_argument(
        "--septq-ckpt",
        type=Path,
        default=None,
        help="PTQ multitier .pt (optional; default: resolve like export_bmo_gguf).",
    )
    ap.add_argument("--gguf", type=Path, default=here / "bmo_septq_v3.gguf")
    ap.add_argument("--module-name", type=str, default="transformer.layers.0.self_attn.in_proj_weight")
    ap.add_argument("--gguf-name", type=str, default="transformer_layers_0_self_attn_in_proj_weight")
    args = ap.parse_args()

    qat_path = args.qat_ckpt.resolve()
    gguf_path = args.gguf.resolve()
    if not qat_path.is_file():
        print(f"ERROR: QAT checkpoint not found: {qat_path}", file=sys.stderr)
        sys.exit(2)
    if not gguf_path.is_file():
        print(f"ERROR: GGUF not found: {gguf_path}", file=sys.stderr)
        sys.exit(2)

    qat_payload = torch.load(str(qat_path), map_location="cpu")
    if not isinstance(qat_payload, dict):
        print(f"ERROR: expected dict checkpoint, got {type(qat_payload)}", file=sys.stderr)
        sys.exit(2)

    stats_src = resolve_stats_source_ckpt(qat_payload, qat_path, args.septq_ckpt)
    mod = get_module_meta(stats_src, args.module_name)
    if not isinstance(mod, dict):
        print(f"ERROR: no per_layer_stats entry for module_name={args.module_name!r}", file=sys.stderr)
        sys.exit(2)

    ptq = {
        "scale_low": _f(mod, "quant_scale_low", "quant_scale"),
        "scale_int4": _f(mod, "quant_scale_int4"),
        "scale_int8": _f(mod, "quant_scale_int8"),
        "zp_low": _f(mod, "quant_zero_point_low", "quant_zero_point"),
        "zp_int4": _f(mod, "quant_zero_point_int4"),
        "zp_int8": _f(mod, "quant_zero_point_int8"),
    }

    reader = gguf.GGUFReader(str(gguf_path))
    by_name = _tensor_dict(reader)
    base = str(args.gguf_name)

    def _gg(suffix: str) -> float:
        t = pick_tensor(by_name, base, suffix)
        if t is None:
            return float("nan")
        return read_scalar_f32(_get_numpy(t), float("nan"))

    gg = {
        "scale_low": _gg("scale_low"),
        "scale_int4": _gg("scale_int4"),
        "scale_int8": _gg("scale_int8"),
        "zp_low": _gg("zp_low"),
        "zp_int4": _gg("zp_int4"),
        "zp_int8": _gg("zp_int8"),
    }

    print(f"module_name={args.module_name!r}  gguf_base={base!r}")
    ok = True
    for k in ptq:
        a, b = ptq[k], gg[k]
        if a is None or not np.isfinite(a) or not np.isfinite(b):
            match = False
        else:
            match = abs(float(a) - float(b)) <= _TOL
        ok = ok and match
        tag = "MATCH" if match else "MISMATCH"
        pa = f"{a:.9g}" if a is not None and np.isfinite(a) else str(a)
        pb = f"{b:.9g}" if np.isfinite(b) else str(b)
        print(f"  {k:12} PTQ={pa:>18}  GGUF={pb:>18}  {tag}")

    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
