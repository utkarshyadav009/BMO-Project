import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent

    files = [
        "train_lqec.py",
        "generate_lqec_manifest.py",
        "lqec_manifest.json",
        "bmo_temporal_int4_base.pt",
        "v5_step1500.safetensors",
        "tokenizer_spm_32k_3.model",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tellmeajoke_padded.wav",
        "bmo_config.json",
        "test_rtx_edge.py",
        "apply_awq_scales.py",
        "extract_awq_scales.py",
    ]

    present = []
    missing = []
    for rel in files:
        p = root / rel
        if p.exists():
            present.append(p)
        else:
            missing.append(rel)

    bundle_path = root / "lqec_h100_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=0) as zf:
        for p in present:
            zf.write(p, arcname=p.name)

    report = {
        "bundle": str(bundle_path),
        "included_count": len(present),
        "missing_count": len(missing),
        "included": [p.name for p in present],
        "missing": missing,
    }

    report_path = root / "lqec_h100_bundle_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Wrote bundle: {bundle_path}")
    print(f"[INFO] Included files: {len(present)}")
    if missing:
        print(f"[WARN] Missing files: {len(missing)}")
        for m in missing:
            print(f"  - {m}")
    print(f"[INFO] Report: {report_path}")


if __name__ == "__main__":
    main()
