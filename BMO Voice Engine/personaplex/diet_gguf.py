#!/usr/bin/env python3
"""
Strip accidental dense temporal weights from a BMO GGUF.

The v12 export contains both dense temporal matrices and the packed SEPTQ
artifacts. The runtime uses the packed artifacts, so this script drops only the
dense temporal weights and writes a lean Jetson-friendly GGUF.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_GGUF_PY = REPO_ROOT / "llama.cpp" / "gguf-py"
if LOCAL_GGUF_PY.exists():
    sys.path.insert(0, str(LOCAL_GGUF_PY))

import gguf  # noqa: E402


DEFAULT_INPUT = Path("bmo_weights_v12.gguf")
DEFAULT_OUTPUT = Path("bmo_jetson_lean.gguf")

PACKED_TEMPORAL_RE = re.compile(
    r"^transformer_layers_\d+_"
    r"(?:self_attn_in_proj|self_attn_out_proj|gating_linear_in|gating_linear_out)_weight\."
    r"(?:packed_weights|packed_mask|fp16_indices|fp16_values|idx2_start|idx4_start|idx8_start|"
    r"n_2bit_bytes|n_4bit_bytes|n_8bit_bytes|scale_low|scale_int4|scale_int8|"
    r"zp_low|zp_int4|zp_int8|rows|cols|bias)$"
)
TEMPORAL_NORM_RE = re.compile(
    r"^transformer_layers_\d+_(?:norm[12]_weight|norm[12]\.weight|attn_norm\.weight|ffn_norm\.weight)$"
)
TEMPORAL_DENSE_OUT_RE = re.compile(
    r"^transformer_layers_\d+_self_attn_out_proj_(?:weight|bias)$"
)
TEMPORAL_DENSE_ALIAS_RE = re.compile(
    r"^transformer_layers_\d+_"
    r"(?:self_attn_in_proj|self_attn_out_proj|gating_linear_in|gating_linear_out)_weight$"
)
TEMPORAL_ORIGINAL_RE = re.compile(r"^transformer\.layers\.\d+\..*weight$")
DEPTH_RUNTIME_RE = re.compile(r"^(?:depformer_layers_\d+_|depformer_(?:in|emb)\.\d+\.weight)")
DEPTH_NORM_DOT_RE = re.compile(r"^depformer\.layers\.\d+\.norm[12]\.weight$")
CODEBOOK_RE = re.compile(r"^(?:emb|linears)\.\d+\.weight$")

DIRECT_KEEP = {
    "depformer_text_emb.weight",
    "text_emb.weight",
    "text_linear.weight",
    "text_linear.bias",
    "token_embedding",
    "output_head",
    "out_norm_weight",
}

SMALL_METADATA_TENSOR_BYTES = 1 << 20
EXPECTED_PACKED_KINDS = [
    "self_attn_in_proj",
    "gating_linear_in",
    "gating_linear_out",
]


def should_keep_tensor(name: str, n_bytes: int) -> bool:
    if name in DIRECT_KEEP:
        return True
    if PACKED_TEMPORAL_RE.match(name):
        return True
    if TEMPORAL_NORM_RE.match(name):
        return True
    if TEMPORAL_DENSE_OUT_RE.match(name):
        return True
    if DEPTH_RUNTIME_RE.match(name):
        return True
    if DEPTH_NORM_DOT_RE.match(name):
        return True
    if CODEBOOK_RE.match(name):
        return True

    # Keep tiny scalar/config tensors, but never keep original temporal aliases.
    if n_bytes <= SMALL_METADATA_TENSOR_BYTES and not name.startswith("transformer.layers."):
        return True

    return False


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        # GGUFWriter writes architecture during construction. GGUF.* fields are
        # virtual/internal reader fields and should not be serialized.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()

        if field.name == gguf.Keys.General.ALIGNMENT:
            writer.add_custom_alignment(int(value))
        else:
            writer.add_key_value(field.name, value, val_type, sub_type=sub_type)


def select_tensors(reader: gguf.GGUFReader) -> tuple[list[gguf.ReaderTensor], list[gguf.ReaderTensor]]:
    kept_tensors = []
    dropped_tensors = []
    for tensor in reader.tensors:
        if should_keep_tensor(tensor.name, tensor.n_bytes):
            kept_tensors.append(tensor)
        else:
            dropped_tensors.append(tensor)
    return kept_tensors, dropped_tensors


def copy_tensors(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
    kept_tensors: list[gguf.ReaderTensor],
    dropped_tensors: list[gguf.ReaderTensor],
) -> tuple[int, int, int, int]:
    kept_count = 0
    dropped_count = 0
    kept_bytes = 0
    dropped_bytes = 0
    dropped_names = {tensor.name for tensor in dropped_tensors}

    for tensor in reader.tensors:
        if tensor.name in dropped_names:
            dropped_count += 1
            dropped_bytes += tensor.n_bytes
            continue

        kept_count += 1
        kept_bytes += tensor.n_bytes
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            raw_dtype=tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for idx, tensor in enumerate(kept_tensors, start=1):
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        if idx % 100 == 0 or idx == kept_count:
            print(f"[diet_gguf] wrote {idx}/{kept_count} tensors", flush=True)

    return kept_count, dropped_count, kept_bytes, dropped_bytes


def tensor_category(name: str, n_bytes: int) -> str:
    if PACKED_TEMPORAL_RE.match(name):
        return "temporal packed artifacts"
    if TEMPORAL_DENSE_ALIAS_RE.match(name):
        return "temporal dense aliases"
    if TEMPORAL_ORIGINAL_RE.match(name):
        return "original transformer weights"
    if TEMPORAL_NORM_RE.match(name):
        return "temporal norms"
    if name.startswith("depformer_layers_") or name.startswith("depformer.layers."):
        return "depth stack"
    if (
        name.startswith("depformer_in.")
        or name.startswith("depformer_emb.")
        or CODEBOOK_RE.match(name)
        or name in DIRECT_KEEP
    ):
        return "embeddings and heads"
    if n_bytes <= SMALL_METADATA_TENSOR_BYTES:
        return "small metadata tensors"
    return "other large tensors"


def original_dense_name_for_base(base: str) -> str | None:
    m = re.match(r"^transformer_layers_(\d+)_(.+)_weight$", base)
    if not m:
        return None
    layer = m.group(1)
    kind = m.group(2)
    mapping = {
        "self_attn_in_proj": f"transformer.layers.{layer}.self_attn.in_proj_weight",
        "self_attn_out_proj": f"transformer.layers.{layer}.self_attn.out_proj.weight",
        "gating_linear_in": f"transformer.layers.{layer}.gating.linear_in.weight",
        "gating_linear_out": f"transformer.layers.{layer}.gating.linear_out.weight",
    }
    return mapping.get(kind)


def tensor_scalar_int(tensor: gguf.ReaderTensor | None, default: int = 0) -> int:
    if tensor is None:
        return default
    try:
        return int(tensor.data.reshape(-1)[0])
    except Exception:
        return default


def packed_component_name(base: str, name: str) -> str:
    if name == base + ".bias":
        return "bias"
    if not name.startswith(base + "."):
        return "unknown"
    return name[len(base) + 1:]


def audit_gguf(reader: gguf.GGUFReader, largest: int) -> None:
    tensors_by_name = {tensor.name: tensor for tensor in reader.tensors}
    total_bytes = sum(tensor.n_bytes for tensor in reader.tensors)

    print("[audit] file tensor payload:", fmt_gib(total_bytes), f"({len(reader.tensors)} tensors)")
    print("[audit] category totals:")
    category_bytes: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)
    for tensor in reader.tensors:
        cat = tensor_category(tensor.name, tensor.n_bytes)
        category_bytes[cat] += tensor.n_bytes
        category_counts[cat] += 1
    for cat, n_bytes in sorted(category_bytes.items(), key=lambda item: item[1], reverse=True):
        print(f"  {fmt_gib(n_bytes):>9}  {category_counts[cat]:5d}  {cat}")

    print(f"[audit] largest {largest} tensors:")
    for tensor in sorted(reader.tensors, key=lambda t: t.n_bytes, reverse=True)[:largest]:
        shape = "x".join(str(x) for x in tensor.shape)
        print(f"  {fmt_gib(tensor.n_bytes):>9}  {shape:<18}  {tensor.tensor_type.name:<6}  {tensor.name}")

    packed_bases = sorted(
        tensor.name[: -len(".packed_weights")]
        for tensor in reader.tensors
        if tensor.name.startswith("transformer_layers_") and tensor.name.endswith(".packed_weights")
    )
    print(f"[audit] temporal packed bases found: {len(packed_bases)}")

    expected_bases = [
        f"transformer_layers_{layer}_{kind}_weight"
        for layer in range(32)
        for kind in EXPECTED_PACKED_KINDS
    ]
    missing_expected = [base for base in expected_bases if base + ".packed_weights" not in tensors_by_name]
    unexpected_packed = [base for base in packed_bases if base not in expected_bases]
    print(f"[audit] expected packed bases: {len(expected_bases)}")
    if missing_expected:
        print(f"[audit] MISSING expected packed bases: {len(missing_expected)}")
        for base in missing_expected:
            dense = tensors_by_name.get(base)
            dense_txt = f" dense_present={fmt_gib(dense.n_bytes)}" if dense is not None else " dense_present=no"
            print(f"  missing {base}{dense_txt}")
    else:
        print("[audit] expected packed coverage: complete")
    if unexpected_packed:
        print(f"[audit] unexpected packed bases: {len(unexpected_packed)}")
        for base in unexpected_packed[:40]:
            print(f"  unexpected {base}")

    required_suffixes = [
        ".packed_weights",
        ".packed_mask",
        ".fp16_indices",
        ".fp16_values",
        ".rows",
        ".cols",
    ]
    packed_total = 0
    dense_f16_equiv_total = 0
    dense_alias_total = 0
    original_dense_total = 0
    missing_required = []
    bases_with_dense_alias = []
    bases_with_original_dense = []
    component_totals: dict[str, int] = defaultdict(int)
    packed_ratios = []

    for base in packed_bases:
        missing = [suffix for suffix in required_suffixes if base + suffix not in tensors_by_name]
        if missing:
            missing_required.append((base, missing))

        base_packed_bytes = 0
        for name, tensor in tensors_by_name.items():
            if name == base + ".bias" or name.startswith(base + "."):
                base_packed_bytes += tensor.n_bytes
                component_totals[packed_component_name(base, name)] += tensor.n_bytes
        packed_total += base_packed_bytes

        rows = tensor_scalar_int(tensors_by_name.get(base + ".rows"))
        cols = tensor_scalar_int(tensors_by_name.get(base + ".cols"))
        dense_f16_bytes = rows * cols * 2
        dense_f16_equiv_total += dense_f16_bytes
        if dense_f16_bytes > 0 and base_packed_bytes > 0:
            packed_ratios.append((base, dense_f16_bytes / base_packed_bytes, dense_f16_bytes, base_packed_bytes))

        dense_alias = tensors_by_name.get(base)
        if dense_alias is not None:
            dense_alias_total += dense_alias.n_bytes
            bases_with_dense_alias.append((base, dense_alias.n_bytes, base_packed_bytes))

        original_name = original_dense_name_for_base(base)
        original_dense = tensors_by_name.get(original_name) if original_name else None
        if original_dense is not None:
            original_dense_total += original_dense.n_bytes
            bases_with_original_dense.append((base, original_name, original_dense.n_bytes, base_packed_bytes))

    print("[audit] temporal packed payload:", fmt_gib(packed_total))
    print("[audit] temporal dense-f16 equivalent for packed bases:", fmt_gib(dense_f16_equiv_total))
    if packed_total > 0:
        print(f"[audit] effective packed compression vs f16: {dense_f16_equiv_total / packed_total:.2f}x")
    print("[audit] packed component totals:")
    for component, n_bytes in sorted(component_totals.items(), key=lambda item: item[1], reverse=True):
        print(f"  {fmt_gib(n_bytes):>9}  {component}")
    print("[audit] temporal dense aliases next to packed bases:", len(bases_with_dense_alias), fmt_gib(dense_alias_total))
    print("[audit] original transformer dense weights next to packed bases:", len(bases_with_original_dense), fmt_gib(original_dense_total))

    if missing_required:
        print("[audit] WARNING: packed bases with missing required components:")
        for base, missing in missing_required[:40]:
            print(f"  {base}: missing {', '.join(missing)}")
    else:
        print("[audit] packed component check: all packed bases have required components")

    if bases_with_dense_alias:
        print("[audit] largest dense aliases that can be dropped:")
        for base, dense_bytes, packed_bytes in sorted(bases_with_dense_alias, key=lambda x: x[1], reverse=True)[:20]:
            ratio = dense_bytes / max(1, packed_bytes)
            print(f"  {fmt_gib(dense_bytes):>9} dense vs {fmt_gib(packed_bytes):>9} packed  ratio={ratio:6.1f}  {base}")

    if bases_with_original_dense:
        print("[audit] largest original dense weights that can be dropped:")
        for base, original_name, dense_bytes, packed_bytes in sorted(bases_with_original_dense, key=lambda x: x[2], reverse=True)[:20]:
            ratio = dense_bytes / max(1, packed_bytes)
            print(f"  {fmt_gib(dense_bytes):>9} dense vs {fmt_gib(packed_bytes):>9} packed  ratio={ratio:6.1f}  {original_name}")

    if packed_ratios:
        print("[audit] weakest packed compression ratios:")
        for base, ratio, dense_bytes, packed_bytes in sorted(packed_ratios, key=lambda item: item[1])[:20]:
            print(f"  ratio={ratio:5.2f}x  f16={fmt_gib(dense_bytes):>9} packed={fmt_gib(packed_bytes):>9}  {base}")


def print_plan(kept_tensors: list[gguf.ReaderTensor], dropped_tensors: list[gguf.ReaderTensor]) -> None:
    kept_bytes = sum(t.n_bytes for t in kept_tensors)
    dropped_bytes = sum(t.n_bytes for t in dropped_tensors)
    print(f"[diet_gguf] would keep: {len(kept_tensors)} tensors ({fmt_gib(kept_bytes)})")
    print(f"[diet_gguf] would drop: {len(dropped_tensors)} tensors ({fmt_gib(dropped_bytes)})")
    print("[diet_gguf] largest dropped tensors:")
    for tensor in sorted(dropped_tensors, key=lambda t: t.n_bytes, reverse=True)[:40]:
        print(f"  drop {fmt_gib(tensor.n_bytes):>9}  {tensor.name}")


def fmt_gib(nbytes: int) -> str:
    return f"{nbytes / (1024 ** 3):.2f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a lean BMO GGUF for Jetson deployment.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="replace output if it already exists")
    parser.add_argument("--list-drops", action="store_true", help="print the keep/drop plan and exit")
    parser.add_argument("--audit", action="store_true", help="print GGUF size/category and packed-vs-dense diagnostics")
    parser.add_argument("--largest", type=int, default=30, help="number of largest tensors to print with --audit")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[diet_gguf] input not found: {args.input}", file=sys.stderr)
        return 2
    if args.output.exists() and not args.overwrite and not args.list_drops:
        print(f"[diet_gguf] output already exists: {args.output} (use --overwrite)", file=sys.stderr)
        return 2

    print(f"[diet_gguf] reading {args.input}")
    reader = gguf.GGUFReader(args.input, "r")
    if args.audit:
        audit_gguf(reader, args.largest)
        return 0

    kept_tensors, dropped_tensors = select_tensors(reader)
    if args.list_drops:
        print_plan(kept_tensors, dropped_tensors)
        return 0

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise RuntimeError("input GGUF is missing general.architecture")

    arch = arch_field.contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    try:
        copy_metadata(reader, writer)
        kept_count, dropped_count, kept_bytes, dropped_bytes = copy_tensors(
            reader,
            writer,
            kept_tensors,
            dropped_tensors,
        )
    finally:
        writer.close()

    output_size = args.output.stat().st_size
    print("[diet_gguf] complete")
    print(f"[diet_gguf] kept tensors:    {kept_count} ({fmt_gib(kept_bytes)} raw tensor data)")
    print(f"[diet_gguf] dropped tensors: {dropped_count} ({fmt_gib(dropped_bytes)} raw tensor data)")
    print(f"[diet_gguf] output size:     {fmt_gib(output_size)} at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
