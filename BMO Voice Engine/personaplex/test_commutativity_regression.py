import math
import os
import re
import subprocess
import sys
from pathlib import Path


def _parse_max_abs(text: str) -> float:
    match = re.search(r"max_abs\s*=\s*([0-9eE+\-.]+|nan|inf|-inf)", text, flags=re.IGNORECASE)
    if not match:
        return float("nan")
    try:
        token = match.group(1).strip().lower()
        if token == "nan":
            return float("nan")
        if token == "inf":
            return float("inf")
        if token == "-inf":
            return float("-inf")
        return float(token)
    except Exception:
        return float("nan")


def _parse_allclose(text: str) -> bool:
    match = re.search(r"allclose\s*=\s*(True|False|true|false)", text)
    if not match:
        return False
    return match.group(1).lower() == "true"


def main():
    if os.environ.get("BMO_ENABLE_RUNTIME_PATCHES") == "1":
        print("BMO_ENABLE_RUNTIME_PATCHES=1 is not allowed for this regression test.")
        print("[RESULT] max_abs = nan")
        print("[RESULT] allclose = False")
        print("[RESULT] FAIL")
        sys.exit(1)

    import test_rtx_edge  # noqa: F401

    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "test_rmsnorm_commutativity.py",
        "--atol",
        "1e-3",
        "--rtol",
        "1e-3",
        "--headwise-q-basis",
        "true",
        "--absorb-rms-alpha",
        "true",
        "--layer-idx",
        "0",
        "--basis-source",
        "norm1",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
    )

    merged_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    max_abs = _parse_max_abs(merged_output)
    allclose = _parse_allclose(merged_output)

    if proc.returncode != 0:
        print(f"[INFO] commutativity subprocess returncode={proc.returncode}")
        if proc.stdout:
            print("[INFO] commutativity subprocess stdout:")
            print(proc.stdout.rstrip())
        if proc.stderr:
            print("[INFO] commutativity subprocess stderr:")
            print(proc.stderr.rstrip())

    ok = bool(proc.returncode == 0 and allclose and not math.isnan(max_abs) and max_abs < 1e-3)

    print(f"[RESULT] max_abs = {max_abs}")
    print(f"[RESULT] allclose = {allclose}")
    print(f"[RESULT] {'PASS' if ok else 'FAIL'}")

    if ok:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
