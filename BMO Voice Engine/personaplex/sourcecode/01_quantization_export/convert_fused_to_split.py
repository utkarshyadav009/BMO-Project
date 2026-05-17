import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from moshi.models import loaders
from moshi.models.lm import LMModel


def parse_dtype(name: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError("--dtype must be one of: bfloat16, float32")


def resolve_local_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def is_safetensors(path: Path) -> bool:
    return path.suffix in {".safetensors", ".sft", ".sfts"}


def read_config_override_from_payload(path: Path):
    if is_safetensors(path):
        return None
    with open(path, "rb") as handle:
        loaded_obj = torch.load(handle, map_location="cpu")
    if not isinstance(loaded_obj, dict):
        return None
    cfg = loaded_obj.get("config_override")
    if isinstance(cfg, dict):
        return cfg
    return None


def get_input_state_keys(path: Path) -> list[str]:
    if is_safetensors(path):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return list(handle.keys())

    with open(path, "rb") as handle:
        loaded_obj = torch.load(handle, map_location="cpu")

    if isinstance(loaded_obj, dict) and isinstance(loaded_obj.get("state_dict"), dict):
        return list(loaded_obj["state_dict"].keys())
    if isinstance(loaded_obj, dict):
        return list(loaded_obj.keys())
    raise TypeError(f"Unsupported checkpoint payload type: {type(loaded_obj)}")


def build_lm_kwargs(checkpoint_path: Path) -> dict:
    lm_kwargs = dict(loaders._lm_kwargs)
    lm_kwargs["dep_q"] = 16
    cfg = read_config_override_from_payload(checkpoint_path)
    if isinstance(cfg, dict):
        lm_kwargs.update(cfg)
    return lm_kwargs


def convert_checkpoint(input_path: Path, output_path: Path, dtype: torch.dtype, device: str, copy_missing_weights: bool) -> int:
    start = time.perf_counter()
    input_keys = get_input_state_keys(input_path)

    print(f"[INFO] Input checkpoint: {input_path}")
    print(f"[INFO] Output checkpoint: {output_path}")
    print(f"[INFO] device={device} dtype={dtype} copy_missing_weights={copy_missing_weights}")

    model = loaders.get_moshi_lm(
        str(input_path),
        copy_missing_weights=copy_missing_weights,
        device=device,
        dtype=dtype,
        cpu_offload=False,
    )
    model.eval()

    converted = {
        key: tensor.detach().to(device="cpu").contiguous()
        for key, tensor in model.state_dict().items()
    }

    output_keys = list(converted.keys())
    input_key_set = set(input_keys)
    output_key_set = set(output_keys)
    added_keys = sorted(output_key_set - input_key_set)
    removed_keys = sorted(input_key_set - output_key_set)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    print(f"[INFO] Input key count: {len(input_keys)}")
    print(f"[INFO] Output key count: {len(output_keys)}")
    print(f"[INFO] Added keys: {len(added_keys)}")
    print(f"[INFO] Removed keys: {len(removed_keys)}")
    if added_keys:
        print(f"[INFO] First added key: {added_keys[0]}")
    if removed_keys:
        print(f"[INFO] First removed key: {removed_keys[0]}")

    # Strict-load verification against a freshly initialized LMModel.
    lm_kwargs = build_lm_kwargs(input_path)
    verify_model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)
    verify_sd = load_file(str(output_path), device="cpu")

    try:
        verify_model.load_state_dict(verify_sd, strict=True, assign=True)
    except RuntimeError as exc:
        print(f"[RESULT] strict_load = FAIL")
        print(f"[ERROR] {exc}")
        return 1

    elapsed = time.perf_counter() - start
    print("[RESULT] strict_load = PASS")
    print(f"[RESULT] output = {output_path}")
    print(f"[RESULT] elapsed_sec = {elapsed:.3f}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fused checkpoint weights into post-load-hook flat state_dict format."
    )
    parser.add_argument("--input", default="v5_step1500.safetensors")
    parser.add_argument("--out", default="v5_step1500_split.safetensors")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-copy-missing-weights",
        action="store_true",
        help="Disable loader fallback that copies missing depformer groups 0..7 -> 8..15.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_path = resolve_local_path(root, args.input)
    output_path = resolve_local_path(root, args.out)
    dtype = parse_dtype(args.dtype)

    if not input_path.exists():
        print(f"[ERROR] input checkpoint not found: {input_path}")
        sys.exit(1)

    rc = convert_checkpoint(
        input_path=input_path,
        output_path=output_path,
        dtype=dtype,
        device=str(args.device),
        copy_missing_weights=not bool(args.no_copy_missing_weights),
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
