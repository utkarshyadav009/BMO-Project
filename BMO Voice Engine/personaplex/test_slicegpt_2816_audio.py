import argparse
import io
import json
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch

from moshi.models.lm import LMModel
from moshi.offline import run_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="SliceGPT d_new=2816 audio generation via clean moshi.offline path"
    )
    parser.add_argument(
        "--slicegpt-ckpt",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/bmo_slicegpt_2816_qinit.pt",
    )
    parser.add_argument("--voice-prompt", default="bmo_621.wav")
    parser.add_argument(
        "--voice-prompt-dir",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/",
    )
    parser.add_argument(
        "--input-wav",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/tellmeajoke_padded.wav",
    )
    parser.add_argument("--text-prompt", default="You are BMO.")
    parser.add_argument(
        "--mimi-weight",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/tokenizer_spm_32k_3.model",
    )
    parser.add_argument("--out-wav", default="/tmp/slicegpt_2816_test/output.wav")
    parser.add_argument("--out-text", default="/tmp/slicegpt_2816_test/output.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _extract_first_30_text_tokens(combined_output: str):
    return _extract_text_tokens(combined_output)[:30]


def _extract_text_tokens(combined_output: str):
    return re.findall(r"text token '([^']*)'", combined_output)


def _load_slicegpt_dense_model(ckpt_path, device, dtype):
    ckpt_path = str(Path(ckpt_path).resolve())
    root = Path(__file__).resolve().parent

    try:
        payload = torch.load(ckpt_path, map_location="cpu", mmap=True)
    except RuntimeError as exc:
        if "PytorchStreamReader failed locating file" in str(exc):
            raise RuntimeError(
                f"Checkpoint appears corrupted or incomplete: {ckpt_path} ({exc})"
            ) from exc
        raise

    assert isinstance(payload, dict), "SliceGPT checkpoint must be a dict payload"
    assert "state_dict" in payload and isinstance(payload["state_dict"], dict), "Missing dict payload['state_dict']"
    assert "config_override" in payload and isinstance(payload["config_override"], dict), "Missing dict payload['config_override']"

    state_dict = payload["state_dict"]
    config_override = payload["config_override"]

    config_path = (root / "bmo_config.json").resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    cfg.update(config_override)
    cfg.pop("model_type", None)

    with torch.device("meta"):
        model = LMModel(dtype=dtype, **cfg)

    model.load_state_dict(state_dict, strict=False, assign=True)
    model.to(device=device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    temporal_inner_dim = int(config_override.get("temporal_inner_dim", -1))
    return model, temporal_inner_dim


def main():
    args = parse_args()

    slicegpt_ckpt = str(Path(args.slicegpt_ckpt).resolve())
    output_wav = str(Path(args.out_wav).resolve())

    temporal_inner_dim = -1
    output_wav_bytes = 0
    first_30_text_tokens = []
    passed = False

    error_message = None
    captured_output = ""

    try:
        assert os.environ.get("BMO_ENABLE_RUNTIME_PATCHES") != "1", (
            "BMO_ENABLE_RUNTIME_PATCHES=1 is not allowed for test_slicegpt_2816_audio.py"
        )
        assert os.environ.get("MOSHI_ENABLE_INT4") != "1", (
            "MOSHI_ENABLE_INT4=1 is not allowed for test_slicegpt_2816_audio.py"
        )

        out_wav_path = Path(args.out_wav).resolve()
        out_text_path = Path(args.out_text).resolve()
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        out_text_path.parent.mkdir(parents=True, exist_ok=True)

        preloaded_model, temporal_inner_dim = _load_slicegpt_dense_model(
            args.slicegpt_ckpt,
            device=str(args.device),
            dtype=torch.bfloat16,
        )

        import moshi.models.loaders as _loaders

        _original_get_moshi_lm = _loaders.get_moshi_lm

        def _preloaded_get_moshi_lm(*_args, **_kwargs):
            return preloaded_model

        _loaders.get_moshi_lm = _preloaded_get_moshi_lm
        restore_error = None

        try:
            run_stdout = io.StringIO()
            run_stderr = io.StringIO()
            with redirect_stdout(run_stdout), redirect_stderr(run_stderr):
                with torch.no_grad():
                    run_inference(
                        input_wav=str(Path(args.input_wav).resolve()),
                        output_wav=str(out_wav_path),
                        output_text=str(out_text_path),
                        text_prompt=str(args.text_prompt),
                        voice_prompt_path=str((Path(args.voice_prompt_dir).resolve() / args.voice_prompt).resolve()),
                        tokenizer_path=str(Path(args.tokenizer).resolve()),
                        moshi_weight=str(Path(args.slicegpt_ckpt).resolve()),
                        mimi_weight=str(Path(args.mimi_weight).resolve()),
                        hf_repo="nvidia/personaplex-7b-v1",
                        device=str(args.device),
                        seed=1234,
                        temp_audio=0.8,
                        temp_text=0.7,
                        topk_audio=250,
                        topk_text=25,
                        greedy=False,
                        save_voice_prompt_embeddings=False,
                        cpu_offload=False,
                    )

            captured_output = run_stdout.getvalue() + "\n" + run_stderr.getvalue()
        finally:
            try:
                _loaders.get_moshi_lm = _original_get_moshi_lm
                if _loaders.get_moshi_lm is not _original_get_moshi_lm:
                    raise RuntimeError("Failed to restore moshi.models.loaders.get_moshi_lm")
            except Exception as exc:
                restore_error = exc

        if restore_error is not None:
            raise RuntimeError(str(restore_error))

        all_text_tokens = _extract_text_tokens(captured_output)
        first_30_text_tokens = all_text_tokens[:30]

        if out_wav_path.exists():
            output_wav_bytes = int(out_wav_path.stat().st_size)

        if temporal_inner_dim <= 0:
            raise RuntimeError("temporal_inner_dim missing in checkpoint config_override")

        if output_wav_bytes <= 100_000:
            raise RuntimeError("Output WAV missing or too small (<100 KB)")

        if len(all_text_tokens) == 0:
            raise RuntimeError("Token stream is empty")

        special_tokens = {"PAD", "EPAD", "BOS", "EOS"}
        if all(token in special_tokens for token in all_text_tokens):
            raise RuntimeError("Token stream contains only special tokens (PAD/EPAD/BOS/EOS)")

        passed = True

    except Exception as exc:
        error_message = str(exc)
        passed = False

    if error_message:
        print(f"[ERROR] {error_message}")
        if captured_output.strip():
            print("[ERROR] run_inference output (tail):")
            tail = captured_output.strip().splitlines()[-40:]
            for line in tail:
                print(line)

    print(f"[RESULT] slicegpt_ckpt = {slicegpt_ckpt}")
    print(f"[RESULT] temporal_inner_dim = {int(temporal_inner_dim)}")
    print(f"[RESULT] output_wav = {output_wav}")
    print(f"[RESULT] output_wav_bytes = {int(output_wav_bytes)}")
    print(f"[RESULT] first_30_text_tokens = {first_30_text_tokens}")
    print(f"[RESULT] {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
