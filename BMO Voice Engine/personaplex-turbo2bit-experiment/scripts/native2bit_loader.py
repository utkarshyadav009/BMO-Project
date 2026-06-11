"""Experiment-local loader for PersonaPlex Turbo2bit (NF2+WHT native 2-bit) checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from moshi.models.loaders import get_personaplex_lm_kwargs
from moshi.models.lm import LMModel

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = EXPERIMENT_ROOT / "models" / "personaplex-7b-turbo2bit"
DEFAULT_WEIGHT = DEFAULT_MODEL_DIR / "model-turbo2bit.safetensors"
DEFAULT_LINEAR2BIT = DEFAULT_MODEL_DIR / "linear2bit.py"


def _load_linear2bit_module(linear2bit_path: Path | None = None):
    path = Path(linear2bit_path or DEFAULT_LINEAR2BIT)
    if not path.is_file():
        raise FileNotFoundError(f"linear2bit.py not found: {path}")
    spec = importlib.util.spec_from_file_location("linear2bit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import linear2bit from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["linear2bit"] = module
    spec.loader.exec_module(module)
    return module


def is_turbo2bit_checkpoint(path: Path | str) -> bool:
    path = Path(path)
    if not path.is_file():
        return False

    config = path.parent / "config.json"
    if config.is_file():
        import json

        try:
            data = json.loads(config.read_text(encoding="utf-8"))
            if data.get("quantization") == "turbo2bit":
                return True
        except json.JSONDecodeError:
            pass

    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if key.endswith(".weight.packed"):
                return True
    return False


def get_moshi_lm_native2bit(
    filename: Path | str | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    linear2bit_path: Path | str | None = None,
) -> LMModel:
    """Load PersonaPlex LM with native 2-bit Linear2bit modules."""
    weight_path = Path(filename or DEFAULT_WEIGHT)
    if not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    linear2bit = _load_linear2bit_module(
        Path(linear2bit_path) if linear2bit_path else None
    )
    lm_kwargs = get_personaplex_lm_kwargs()
    model = LMModel(device="meta", dtype=dtype, **lm_kwargs)
    state = load_file(str(weight_path), device="cpu")
    model = linear2bit.replace_linears_with_2bit(
        model, state, device=device, dtype=dtype
    )
    model.eval()
    return model


def patch_moshi_loaders() -> None:
    """Monkey-patch moshi.models.loaders.get_moshi_lm for native-2bit env runs."""
    import moshi.models.loaders as loaders

    original = loaders.get_moshi_lm

    def patched_get_moshi_lm(filename, *args, **kwargs):
        if filename is not None and is_turbo2bit_checkpoint(filename):
            return get_moshi_lm_native2bit(
                filename,
                device=kwargs.get("device", "cuda"),
                dtype=kwargs.get("dtype", torch.bfloat16),
            )
        return original(filename, *args, **kwargs)

    loaders.get_moshi_lm = patched_get_moshi_lm
