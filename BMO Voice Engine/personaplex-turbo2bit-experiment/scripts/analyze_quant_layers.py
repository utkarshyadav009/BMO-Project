#!/usr/bin/env python3
"""Analyze NF2+WHT 2-bit vs full-precision tensors in model-turbo2bit.safetensors."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from safetensors import safe_open

META_SUFFIXES = (".packed", ".scales", ".gs", ".np2", ".shape", ".numel")
QUANT_BASE_SUFFIX = ".weight.packed"


def classify_key(key: str) -> tuple[str, str | None]:
    if key.endswith(QUANT_BASE_SUFFIX):
        return "quantized_linear", key[: -len(".packed")]
    if any(key.endswith(s) for s in META_SUFFIXES):
        base = key
        for suffix in META_SUFFIXES:
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                break
        return "quant_metadata", base
    return "full_precision", None


def module_group(module_path: str) -> str:
    if module_path.startswith("text_emb"):
        return "text_emb"
    if module_path.startswith("depformer_emb"):
        return "depformer_emb"
    if module_path.startswith("depformer_in"):
        return "depformer_in"
    if module_path.startswith("depformer"):
        return "depformer"
    if module_path.startswith("transformer"):
        return "temporal_transformer"
    if "self_attn" in module_path:
        return "self_attn"
    if "gating" in module_path:
        return "gating"
    if "linears" in module_path:
        return "linears"
    if "norm" in module_path or ".ln" in module_path:
        return "norm"
    if "emb" in module_path:
        return "embedding"
    return "other"


def tensor_bytes(dtype: str, shape: list[int]) -> int:
    bits_map = {
        "F16": 16,
        "BF16": 16,
        "F32": 32,
        "I8": 8,
        "I16": 16,
        "I32": 32,
        "I64": 64,
        "U8": 8,
    }
    bits = bits_map.get(dtype, 16)
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * bits // 8


def analyze(checkpoint: Path) -> dict:
    quantized_linears: dict[str, dict] = {}
    full_precision: list[dict] = []
    metadata_keys: list[str] = []

    total_bytes = 0
    quant_bytes = 0
    fp_bytes = 0

    with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_slice(key)
            shape = list(tensor.get_shape())
            dtype = tensor.get_dtype()
            nbytes = tensor_bytes(dtype, shape)
            total_bytes += nbytes

            kind, base = classify_key(key)
            if kind == "quantized_linear":
                assert base is not None
                entry = quantized_linears.setdefault(
                    base,
                    {
                        "module_path": base[: -len(".weight")],
                        "precision": "nf2_2bit_wht",
                        "keys": {},
                        "bytes": 0,
                    },
                )
                entry["keys"][key] = {"shape": shape, "dtype": dtype, "bytes": nbytes}
                entry["bytes"] += nbytes
                quant_bytes += nbytes
            elif kind == "quant_metadata":
                metadata_keys.append(key)
                quant_bytes += nbytes
                if base and base in quantized_linears:
                    quantized_linears[base]["keys"][key] = {
                        "shape": shape,
                        "dtype": dtype,
                        "bytes": nbytes,
                    }
                    quantized_linears[base]["bytes"] += nbytes
            else:
                module_path = key.rsplit(".", 1)[0] if "." in key else key
                full_precision.append(
                    {
                        "key": key,
                        "module_path": module_path,
                        "tensor_name": key.split(".")[-1],
                        "shape": shape,
                        "dtype": dtype,
                        "bytes": nbytes,
                        "group": module_group(module_path),
                    }
                )
                fp_bytes += nbytes

    quant_modules = sorted(quantized_linears.values(), key=lambda x: x["module_path"])
    group_quant = Counter(module_group(m["module_path"]) for m in quant_modules)
    group_fp = Counter(item["group"] for item in full_precision)

    fp_by_suffix = Counter(item["tensor_name"] for item in full_precision)

    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_gb": round(checkpoint.stat().st_size / (1024**3), 4),
        "summary": {
            "quantized_linear_modules": len(quant_modules),
            "full_precision_tensors": len(full_precision),
            "quant_metadata_only_keys": len(metadata_keys),
            "model_card_expected_fp_tensors": 77,
            "tensor_storage_bytes": total_bytes,
            "quantized_storage_bytes": quant_bytes,
            "full_precision_storage_bytes": fp_bytes,
        },
        "quantized_by_group": dict(sorted(group_quant.items())),
        "full_precision_by_group": dict(sorted(group_fp.items())),
        "full_precision_by_tensor_name": dict(sorted(fp_by_suffix.items())),
        "quantized_linears": quant_modules,
        "full_precision_tensors": sorted(full_precision, key=lambda x: x["key"]),
    }


def write_markdown(report: dict, out_path: Path) -> None:
    s = report["summary"]
    lines = [
        "# Turbo2bit Quantization Report",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        f"File size: {report['checkpoint_size_gb']} GB",
        "",
        "## Summary",
        "",
        f"- Quantized linear modules (NF2+WHT 2-bit): **{s['quantized_linear_modules']}**",
        f"- Full-precision tensors: **{s['full_precision_tensors']}** "
        f"(model card expects ~{s['model_card_expected_fp_tensors']})",
        f"- Quantized storage: {s['quantized_storage_bytes'] / (1024**3):.3f} GB",
        f"- Full-precision storage: {s['full_precision_storage_bytes'] / (1024**3):.3f} GB",
        "",
        "## Quantized modules by group",
        "",
    ]
    for group, count in report["quantized_by_group"].items():
        lines.append(f"- {group}: {count}")
    lines.extend(["", "## Full-precision tensors by group", ""])
    for group, count in report["full_precision_by_group"].items():
        lines.append(f"- {group}: {count}")
    lines.extend(["", "## Full-precision tensor names", ""])
    for name, count in report["full_precision_by_tensor_name"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Sample quantized modules (first 20)", ""])
    for item in report["quantized_linears"][:20]:
        lines.append(f"- `{item['module_path']}` ({item['bytes'] / 1024:.1f} KB packed+meta)")
    lines.extend(["", "## Sample full-precision tensors (first 30)", ""])
    for item in report["full_precision_tensors"][:30]:
        lines.append(
            f"- `{item['key']}` shape={item['shape']} dtype={item['dtype']}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models" / "personaplex-7b-turbo2bit" / "model-turbo2bit.safetensors",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=root / "reports" / "quant_layer_manifest.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=root / "reports" / "quant_layer_report.md",
    )
    args = parser.parse_args()

    report = analyze(args.checkpoint)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.md_out)

    s = report["summary"]
    print(f"Quantized linear modules: {s['quantized_linear_modules']}")
    print(f"Full-precision tensors:   {s['full_precision_tensors']}")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
