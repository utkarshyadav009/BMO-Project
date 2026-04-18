import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _print_results(quant_check: str, hybrid_prompt_check: str, output_wav_bytes: int, passed: bool) -> None:
    print(f"[RESULT] quant_check = {quant_check}")
    print(f"[RESULT] hybrid_prompt_check = {hybrid_prompt_check}")
    print(f"[RESULT] output_wav_bytes = {int(output_wav_bytes)}")
    print(f"[RESULT] {'PASS' if passed else 'FAIL'}")


def main() -> int:
    quant_check = "CLEAN"
    hybrid_prompt_check = "MISSING"
    output_wav_bytes = 0
    passed = False

    root = Path(__file__).resolve().parent

    # Hard guard rails: this must validate vanilla offline behavior only.
    if os.environ.get("BMO_ENABLE_RUNTIME_PATCHES") == "1":
        print("[ERROR] BMO_ENABLE_RUNTIME_PATCHES=1 is not allowed for test_offline_regression.py", file=sys.stderr)
        quant_check = "FOUND"
        _print_results(quant_check, hybrid_prompt_check, output_wav_bytes, passed)
        return 1

    if os.environ.get("MOSHI_ENABLE_INT4") == "1":
        print("[ERROR] MOSHI_ENABLE_INT4=1 is not allowed for test_offline_regression.py", file=sys.stderr)
        quant_check = "FOUND"
        _print_results(quant_check, hybrid_prompt_check, output_wav_bytes, passed)
        return 1

    moshi_weight = _resolve_path(root, os.environ.get("BMO_OFFLINE_MOSHI_WEIGHT", "bmo_slicegpt_4096_identity.pt"))
    mimi_weight = _resolve_path(root, os.environ.get("BMO_OFFLINE_MIMI_WEIGHT", "tokenizer-e351c8d8-checkpoint125.safetensors"))
    tokenizer_path = _resolve_path(root, os.environ.get("BMO_OFFLINE_TOKENIZER", "tokenizer_spm_32k_3.model"))
    input_wav = _resolve_path(root, os.environ.get("BMO_OFFLINE_INPUT_WAV", "tellmeajoke_padded.wav"))
    voice_prompt_dir = _resolve_path(root, os.environ.get("BMO_OFFLINE_VOICE_PROMPT_DIR", "."))
    voice_prompt = os.environ.get("BMO_OFFLINE_VOICE_PROMPT", "bmo_621.wav")
    text_prompt = os.environ.get("BMO_OFFLINE_TEXT_PROMPT", "Tell me a joke.")
    device = os.environ.get("BMO_OFFLINE_DEVICE", "cuda")
    timeout_sec = int(os.environ.get("BMO_OFFLINE_TIMEOUT_SEC", "420"))

    voice_prompt_path = (voice_prompt_dir / voice_prompt).resolve()

    required_paths = {
        "moshi weight": moshi_weight,
        "mimi weight": mimi_weight,
        "tokenizer": tokenizer_path,
        "input wav": input_wav,
        "voice prompt": voice_prompt_path,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        print("[ERROR] Missing required regression assets:", file=sys.stderr)
        for item in missing:
            print(f"[ERROR]   {item}", file=sys.stderr)
        _print_results(quant_check, hybrid_prompt_check, output_wav_bytes, passed)
        return 1

    with tempfile.TemporaryDirectory(prefix="bmo_offline_regression_") as tmp_dir:
        tmp = Path(tmp_dir)
        output_wav = tmp / "offline_out.wav"
        output_text = tmp / "offline_out.json"

        cmd = [
            sys.executable,
            "-m",
            "moshi.offline",
            "--input-wav",
            str(input_wav),
            "--output-wav",
            str(output_wav),
            "--output-text",
            str(output_text),
            "--text-prompt",
            text_prompt,
            "--voice-prompt",
            voice_prompt,
            "--voice-prompt-dir",
            str(voice_prompt_dir),
            "--tokenizer",
            str(tokenizer_path),
            "--moshi-weight",
            str(moshi_weight),
            "--mimi-weight",
            str(mimi_weight),
            "--device",
            device,
            "--seed",
            "1234",
        ]

        if os.environ.get("BMO_OFFLINE_CPU_OFFLOAD", "0") == "1":
            cmd.append("--cpu-offload")

        env = dict(os.environ)
        moshi_pythonpath = str((root / "moshi").resolve())
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            moshi_pythonpath if not existing_pythonpath else f"{moshi_pythonpath}{os.pathsep}{existing_pythonpath}"
        )
        env.pop("BMO_ENABLE_RUNTIME_PATCHES", None)
        env.pop("MOSHI_ENABLE_INT4", None)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            print(
                f"[ERROR] moshi.offline timed out after {timeout_sec}s: {exc}",
                file=sys.stderr,
            )
            _print_results(quant_check, hybrid_prompt_check, output_wav_bytes, passed)
            return 1

        # Regression sentinels: if these appear, quantization path leaked back in.
        if ("Replaced" in combined_output) or ("INT4 Quantization" in combined_output):
            quant_check = "FOUND"

        # Hybrid System Prompt flow must stay intact.
        if ("Done loading voice prompt." in combined_output) and ("Done loading text prompt." in combined_output):
            hybrid_prompt_check = "OK"

        if output_wav.exists():
            output_wav_bytes = int(output_wav.stat().st_size)

        passed = bool(
            proc.returncode == 0
            and quant_check == "CLEAN"
            and hybrid_prompt_check == "OK"
            and output_wav_bytes > 100_000
        )

        if not passed:
            print(f"[ERROR] moshi.offline return code: {proc.returncode}", file=sys.stderr)
            if proc.stdout:
                print("[ERROR] subprocess stdout:", file=sys.stderr)
                print(proc.stdout, file=sys.stderr)
            if proc.stderr:
                print("[ERROR] subprocess stderr:", file=sys.stderr)
                print(proc.stderr, file=sys.stderr)

    _print_results(quant_check, hybrid_prompt_check, output_wav_bytes, passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
