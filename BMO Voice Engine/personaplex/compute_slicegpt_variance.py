import argparse
import json
import random
from pathlib import Path

import sentencepiece
import torch
import torch.nn as nn

# Reuse runtime patches used by existing experiments.
import test_rtx_edge  # noqa: F401
from moshi import offline
from moshi.models import loaders
from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio


class CovAccumulator:
    """Accumulates X^T X and sample count to compute empirical covariance."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.sum_xx = torch.zeros((self.dim, self.dim), dtype=torch.float64, device="cpu")
        self.sum_x = torch.zeros((self.dim,), dtype=torch.float64, device="cpu")
        self.num_rows = 0

    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return

        x = x.detach().to(dtype=torch.float64)
        n = int(x.shape[0])
        xx = torch.matmul(x.T, x)
        sx = x.sum(dim=0)
        self.sum_xx += xx.to(device="cpu", dtype=torch.float64)
        self.sum_x += sx.to(device="cpu", dtype=torch.float64)
        self.num_rows += n

    def covariance(self) -> torch.Tensor:
        if self.num_rows <= 1:
            raise RuntimeError("No samples were collected for covariance estimation")
        n = float(self.num_rows)
        # True empirical covariance with centering.
        centered_ss = self.sum_xx - torch.outer(self.sum_x, self.sum_x) / n
        cov = centered_ss / float(self.num_rows - 1)
        return 0.5 * (cov + cov.T)


def parse_layers(raw: str) -> list[int]:
    layers = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        layers.append(int(token))
    if not layers:
        raise ValueError("--layers must contain at least one index")
    return sorted(set(layers))


def parse_targets(raw: str) -> list[float]:
    targets = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Variance target must be in (0,1), got {value}")
        targets.append(value)
    if not targets:
        raise ValueError("--targets must contain at least one value")
    return sorted(set(targets))


def parse_manifest_paths(raw: str, root: Path) -> list[Path]:
    manifests: list[Path] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        path = Path(token)
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        manifests.append(path)
    return manifests


def resolve_dataset_root(raw: str, root: Path) -> Path | None:
    if not raw.strip():
        return None
    path = Path(raw.strip())
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_optional_path(raw: str, root: Path) -> Path | None:
    if not raw.strip():
        return None
    path = Path(raw.strip())
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_dataset_audio_path(raw_path: str, dataset_root: Path | None, fallback_root: Path) -> Path:
    raw_norm = raw_path.replace("\\", "/")
    path = Path(raw_path)

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((fallback_root / path).resolve())
        candidates.append((Path.cwd() / path).resolve())

    if dataset_root is not None:
        marker = "/bmo_dataset/"
        if marker in raw_norm:
            tail = raw_norm.split(marker, 1)[1]
            candidates.append((dataset_root / tail).resolve())

        audio_marker = "/audio/"
        if audio_marker in raw_norm:
            tail = raw_norm.split(audio_marker, 1)[1]
            candidates.append((dataset_root / "audio" / tail).resolve())

        candidates.append((dataset_root / path.name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else path


def load_sidecar_transcript(audio_path: Path) -> str:
    transcript, _duration = load_sidecar_metadata(audio_path)
    return transcript


def load_sidecar_metadata(audio_path: Path) -> tuple[str, float | None]:
    sidecar = audio_path.with_suffix(".json")
    if not sidecar.exists():
        return "", None

    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return "", None

    transcript = ""
    duration = None
    if isinstance(obj, dict):
        for key in ("transcript", "text", "utterance"):
            value = obj.get(key, "")
            if isinstance(value, str) and value.strip():
                transcript = value.strip()
                break

        raw_duration = obj.get("duration")
        if raw_duration is not None:
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                duration = None

    return transcript, duration


def collect_dataset_clips_from_directory(
    dataset_dir: Path,
    recursive: bool,
    min_duration: float,
) -> tuple[list[dict], dict]:
    clips: list[dict] = []
    seen_paths: set[str] = set()
    stats = {
        "found_wav": 0,
        "duration_filtered": 0,
        "kept": 0,
    }

    iterator = dataset_dir.rglob("*") if recursive else dataset_dir.iterdir()
    for path in iterator:
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        stats["found_wav"] += 1

        resolved = path.resolve()
        resolved_key = str(resolved)
        if resolved_key in seen_paths:
            continue
        seen_paths.add(resolved_key)

        transcript, sidecar_duration = load_sidecar_metadata(resolved)
        duration = sidecar_duration

        if duration is not None and duration < min_duration:
            stats["duration_filtered"] += 1
            continue

        clips.append(
            {
                "audio_path": resolved,
                "duration": duration,
                "transcript": transcript,
                "manifest": None,
            }
        )
        stats["kept"] += 1

    return clips, stats


def collect_dataset_clips(
    manifests: list[Path],
    dataset_root: Path | None,
    fallback_root: Path,
    min_duration: float,
) -> tuple[list[dict], dict]:
    clips: list[dict] = []
    seen_paths: set[str] = set()
    stats = {
        "lines": 0,
        "missing_path": 0,
        "duration_filtered": 0,
        "missing_file": 0,
        "invalid_json": 0,
        "kept": 0,
    }

    for manifest in manifests:
        with open(manifest, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    print(f"[WARN] Invalid JSON in {manifest}:{line_no}")
                    continue

                if not isinstance(obj, dict):
                    stats["missing_path"] += 1
                    continue

                raw_path = ""
                for key in ("path", "audio_path", "audio_filepath", "wav_path", "file_path"):
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        raw_path = value.strip()
                        break

                if not raw_path:
                    stats["missing_path"] += 1
                    continue

                duration = float(obj.get("duration", 0.0) or 0.0)
                if duration < min_duration:
                    stats["duration_filtered"] += 1
                    continue

                resolved_audio = resolve_dataset_audio_path(raw_path, dataset_root, fallback_root)
                if not resolved_audio.exists():
                    stats["missing_file"] += 1
                    continue

                resolved_key = str(resolved_audio)
                if resolved_key in seen_paths:
                    continue
                seen_paths.add(resolved_key)

                transcript = obj.get("transcript") if isinstance(obj.get("transcript"), str) else ""
                if not transcript:
                    transcript = load_sidecar_transcript(resolved_audio)

                clips.append(
                    {
                        "audio_path": resolved_audio,
                        "duration": duration,
                        "transcript": transcript,
                        "manifest": manifest,
                    }
                )
                stats["kept"] += 1

    return clips, stats


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _unwrap_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    return None


def _infer_model_input_device(model: nn.Module, fallback_device: str | torch.device) -> torch.device:
    try:
        if hasattr(model, "emb") and len(model.emb) > 0 and hasattr(model.emb[0], "weight"):
            return model.emb[0].weight.device
    except Exception:
        pass

    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    return torch.device(fallback_device)


def _safe_model_to_device(model: nn.Module, device: str, model_name: str) -> nn.Module:
    try:
        model = model.to(device)
    except Exception as e:
        print(f"[WARN] Could not fully move {model_name} to {device}: {e}")
    return model


def build_forced_tokens(
    model,
    mimi,
    tokenizer,
    input_wav: Path,
    text_prompt: str,
    steps: int,
    device: str,
) -> torch.Tensor:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    sample_pcm = load_audio(str(input_wav), mimi.sample_rate)
    samples = _iterate_audio(sample_pcm, frame_size, max_len=steps, pad=True)
    encoded_iter = encode_from_sphn(mimi, samples, max_batch=1)

    text_ids = tokenizer.encode(offline.wrap_with_system_tags(text_prompt))

    k_codebooks = model.num_codebooks
    audio_pad = int(model.card)
    text_pad = 0

    forced = torch.full((steps, k_codebooks), audio_pad, dtype=torch.long)
    forced[:, 0] = text_pad

    for t in range(steps):
        if t < len(text_ids):
            forced[t, 0] = int(text_ids[t])

        try:
            frame_codes = next(encoded_iter)  # [1, K_audio, F]
            audio_codes = frame_codes[:, :, 0].to(dtype=torch.long).cpu()[0]
            n_audio = min(k_codebooks - 1, int(audio_codes.numel()))
            forced[t, 1 : 1 + n_audio] = audio_codes[:n_audio]
        except StopIteration:
            pass

    return forced.to(device)


def attach_covariance_hooks(model: nn.Module, layer_indices: list[int], covariances: dict[str, CovAccumulator]):
    hooks = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            out = _unwrap_tensor(output)
            if out is None or out.ndim < 2:
                return
            dim = int(out.shape[-1])
            if name not in covariances:
                covariances[name] = CovAccumulator(dim)
            covariances[name].update(out.reshape(-1, dim))

        return hook

    n_layers = len(model.transformer.layers)
    for idx in layer_indices:
        if idx < 0 or idx >= n_layers:
            raise IndexError(f"Layer index out of range: {idx} (valid: 0..{n_layers-1})")

        layer = model.transformer.layers[idx]
        for norm_name in ("norm1", "norm2"):
            norm_module = getattr(layer, norm_name, None)
            if norm_module is None:
                print(f"[WARN] transformer.layers.{idx}.{norm_name} not found; skipping")
                continue
            hook_name = f"layer_{idx}_{norm_name}"
            hooks.append(norm_module.register_forward_hook(make_hook(hook_name)))
            print(f"[INFO] Hooked: {hook_name}")

    return hooks


@torch.no_grad()
def run_calibration_pass(model: nn.Module, forced_tokens: torch.Tensor, mode: str, device: str) -> None:
    model_input_device = _infer_model_input_device(model, fallback_device=device)

    if mode == "full":
        seq = forced_tokens.transpose(0, 1).unsqueeze(0).contiguous().to(dtype=torch.long)
        if seq.device != model_input_device:
            seq = seq.to(model_input_device, non_blocking=True)
        model.forward_codes(seq)
        return

    steps = int(forced_tokens.shape[0])
    for t in range(steps):
        seq_prefix = forced_tokens[: t + 1].clone().detach().to(dtype=torch.long).contiguous()
        seq = seq_prefix.transpose(0, 1).unsqueeze(0).contiguous()
        if seq.device != model_input_device:
            seq = seq.to(model_input_device, non_blocking=True)
        model.forward_codes(seq)


def eigendecompose_covariance(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=0.0).flip(dims=(0,))
    eigvecs = eigvecs.flip(dims=(1,))
    return eigvals, eigvecs


def summarize_eigenspectrum(eigvals: torch.Tensor, targets: list[float], num_rows: int) -> dict:

    total = eigvals.sum()
    if float(total) <= 0.0:
        raise RuntimeError("Non-positive total variance encountered")

    explained = torch.cumsum(eigvals, dim=0) / total
    embed_dim = int(eigvals.shape[0])

    dims_by_target = {}
    for target in targets:
        idx = int(torch.searchsorted(explained, torch.tensor(target, dtype=explained.dtype)).item())
        dim_needed = min(embed_dim, idx + 1)
        dims_by_target[f"{target:.2f}"] = {
            "dim": dim_needed,
            "embed_dim": embed_dim,
            "ratio": float(dim_needed / embed_dim),
            "percent": float((dim_needed / embed_dim) * 100.0),
        }

    return {
        "embed_dim": embed_dim,
        "num_rows": int(num_rows),
        "total_variance": float(total.item()),
        "dims_by_target": dims_by_target,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute SliceGPT variance retention ratios from Moshi activations")
    parser.add_argument("--model", default="v5_step1500.safetensors")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--text-prompt", default="Tell me a joke.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--layers", default="0,15,31", help="Comma-separated transformer layer indices")
    parser.add_argument("--targets", default="0.90,0.95,0.99", help="Comma-separated variance targets in (0,1)")
    parser.add_argument(
        "--dataset-jsonl",
        default="",
        help="Comma-separated dataset JSONL manifests. If provided, probe runs across multiple clips.",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional bmo_dataset root used to remap audio paths from manifests.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Directory containing wav/json clip pairs (alternative to --dataset-jsonl).",
    )
    parser.add_argument(
        "--dataset-dir-recursive",
        action="store_true",
        help="Recursively scan --dataset-dir for wav files.",
    )
    parser.add_argument("--max-clips", type=int, default=50, help="Maximum clips to process from manifests; <=0 means all")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Skip clips shorter than this duration (seconds)")
    parser.add_argument(
        "--use-clip-transcript",
        action="store_true",
        help="Use transcript from manifest/sidecar as text prompt when available.",
    )
    parser.add_argument("--progress-every", type=int, default=5, help="Print calibration progress every N clips")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["prefix", "full"], default="prefix")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-json", default="slicegpt_variance_report.json")
    parser.add_argument("--eigenvectors-out", default="bmo_slicegpt_eigenvectors.pt")
    parser.add_argument("--eigenvectors-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip-eigenvectors", action="store_true", help="Skip saving Q matrices and only save JSON")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    targets = parse_targets(args.targets)

    root = Path(__file__).resolve().parent
    model_path = (root / args.model).resolve()
    mimi_weight = (root / args.mimi_weight).resolve()
    tokenizer_path = (root / args.tokenizer).resolve()
    input_wav = (root / args.input_wav).resolve()
    out_json = (root / args.out_json).resolve()
    eigenvectors_out = (root / args.eigenvectors_out).resolve()
    dataset_manifests = parse_manifest_paths(args.dataset_jsonl, root)
    dataset_root = resolve_dataset_root(args.dataset_root, root)
    dataset_dir = resolve_optional_path(args.dataset_dir, root)

    for path in [model_path, mimi_weight, tokenizer_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    if dataset_manifests:
        for manifest in dataset_manifests:
            if not manifest.exists():
                raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    if dataset_dir is not None and not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if dataset_manifests and dataset_dir is not None:
        print("[WARN] Both --dataset-jsonl and --dataset-dir provided; using --dataset-jsonl mode.")

    if not dataset_manifests and dataset_dir is None:
        if not input_wav.exists():
            raise FileNotFoundError(f"Required file not found: {input_wav}")

    if dataset_root is not None and not dataset_root.exists():
        print(f"[WARN] dataset_root does not exist: {dataset_root}")

    print(f"[INFO] Loading BF16 teacher on {args.device}: {model_path}")
    model = loaders.get_moshi_lm(str(model_path), device=args.device, dtype=torch.bfloat16, cpu_offload=False)
    model = _safe_model_to_device(model, args.device, "teacher")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if args.layers.strip().lower() == "all":
        layers = list(range(len(model.transformer.layers)))
    else:
        layers = parse_layers(args.layers)

    print(f"[INFO] Loading Mimi for token extraction: {mimi_weight}")
    mimi = loaders.get_mimi(str(mimi_weight), args.device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    clip_specs: list[dict] = []
    dataset_stats = None
    if dataset_manifests:
        collected, dataset_stats = collect_dataset_clips(
            manifests=dataset_manifests,
            dataset_root=dataset_root,
            fallback_root=root,
            min_duration=args.min_duration,
        )
        if not collected:
            raise RuntimeError("No valid dataset clips found. Check manifest paths and dataset_root.")

        rng = random.Random(args.seed)
        rng.shuffle(collected)
        if args.max_clips > 0:
            collected = collected[: args.max_clips]
        clip_specs = collected

        print(
            "[INFO] Dataset mode enabled: "
            f"manifests={len(dataset_manifests)} clips_selected={len(clip_specs)} "
            f"(lines={dataset_stats['lines']} kept={dataset_stats['kept']} missing_file={dataset_stats['missing_file']})"
        )
    elif dataset_dir is not None:
        collected, dataset_stats = collect_dataset_clips_from_directory(
            dataset_dir=dataset_dir,
            recursive=args.dataset_dir_recursive,
            min_duration=args.min_duration,
        )
        if not collected:
            raise RuntimeError("No valid WAV clips found in --dataset-dir")

        rng = random.Random(args.seed)
        rng.shuffle(collected)
        if args.max_clips > 0:
            collected = collected[: args.max_clips]
        clip_specs = collected

        print(
            "[INFO] Dataset-dir mode enabled: "
            f"dir={dataset_dir} clips_selected={len(clip_specs)} "
            f"(found_wav={dataset_stats['found_wav']} kept={dataset_stats['kept']})"
        )
    else:
        clip_specs = [
            {
                "audio_path": input_wav,
                "duration": None,
                "transcript": "",
                "manifest": None,
            }
        ]

    print(f"[INFO] Attaching covariance hooks for layers: {layers}")

    covariances: dict[str, CovAccumulator] = {}
    hooks = attach_covariance_hooks(model, layers, covariances)

    processed_clips = 0
    skipped_clips = 0
    progress_every = max(1, args.progress_every)

    try:
        print(
            f"[INFO] Running calibration pass in {args.mode} mode for "
            f"{len(clip_specs)} clip(s), steps_per_clip={args.steps}"
        )

        for idx, clip in enumerate(clip_specs, start=1):
            clip_path: Path = clip["audio_path"]
            prompt = args.text_prompt
            if args.use_clip_transcript and isinstance(clip.get("transcript"), str) and clip["transcript"].strip():
                prompt = clip["transcript"].strip()

            try:
                forced_tokens = build_forced_tokens(
                    model=model,
                    mimi=mimi,
                    tokenizer=tokenizer,
                    input_wav=clip_path,
                    text_prompt=prompt,
                    steps=args.steps,
                    device=args.device,
                )
                run_calibration_pass(model=model, forced_tokens=forced_tokens, mode=args.mode, device=args.device)
                processed_clips += 1
            except Exception as e:
                skipped_clips += 1
                print(f"[WARN] Skipping clip {clip_path}: {e}")

            if idx % progress_every == 0 or idx == len(clip_specs):
                print(
                    f"[INFO] Calibration progress: {idx}/{len(clip_specs)} "
                    f"(processed={processed_clips}, skipped={skipped_clips})"
                )
    finally:
        for h in hooks:
            h.remove()

    if processed_clips <= 0:
        raise RuntimeError("No clips were successfully processed for covariance accumulation")

    print("[INFO] Computing eigenspectrum summaries")
    report = {
        "model": str(model_path),
        "device": args.device,
        "steps": args.steps,
        "mode": args.mode,
        "layers": layers,
        "targets": targets,
        "dataset_jsonl": [str(p) for p in dataset_manifests],
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else None,
        "processed_clips": processed_clips,
        "skipped_clips": skipped_clips,
        "selected_clips": [str(c["audio_path"]) for c in clip_specs],
        "dataset_stats": dataset_stats,
        "results": {},
    }

    save_q = not args.skip_eigenvectors
    eigenvectors_payload = {
        "model": str(model_path),
        "layers": layers,
        "dtype": args.eigenvectors_dtype,
        "steps": args.steps,
        "mode": args.mode,
        "results": {},
    }
    save_dtype = parse_dtype(args.eigenvectors_dtype)

    for name in sorted(covariances.keys()):
        cov = covariances[name].covariance()
        eigvals, eigvecs = eigendecompose_covariance(cov)
        stats = summarize_eigenspectrum(eigvals, targets, covariances[name].num_rows)
        report["results"][name] = stats

        print(f"\n[{name}] embed_dim={stats['embed_dim']} rows={stats['num_rows']}")
        for target in targets:
            key = f"{target:.2f}"
            row = stats["dims_by_target"][key]
            print(
                f"  target={target:.2f} -> dim={row['dim']}/{row['embed_dim']} "
                f"({row['percent']:.2f}%)"
            )

        if stats["num_rows"] < stats["embed_dim"]:
            print(
                f"  [WARN] rows<{stats['embed_dim']} may under-constrain a {stats['embed_dim']}-dim covariance estimate."
            )

        if save_q:
            eigenvectors_payload["results"][name] = {
                "Q": eigvecs.to(device="cpu", dtype=save_dtype).contiguous(),
                "eigenvalues": eigvals.to(device="cpu", dtype=torch.float32).contiguous(),
                "num_rows": int(covariances[name].num_rows),
            }

    has_dim99_target = any(abs(t - 0.99) < 1e-8 for t in targets)
    if has_dim99_target:
        print("\n=== dim_99 Ratios ===")
        dim99_rows = []
        for name in sorted(report["results"].keys()):
            if "0.99" not in report["results"][name]["dims_by_target"]:
                continue
            row = report["results"][name]["dims_by_target"]["0.99"]
            print(f"{name}: {row['dim']}/{row['embed_dim']} ({row['percent']:.2f}%)")
            dim99_rows.append((name, row))

        if dim99_rows:
            worst_name, worst_row = max(dim99_rows, key=lambda x: x[1]["dim"])
            report["worst_dim99"] = {
                "layer": worst_name,
                "dim": worst_row["dim"],
                "embed_dim": worst_row["embed_dim"],
                "ratio": worst_row["ratio"],
                "percent": worst_row["percent"],
            }
            print(
                f"\n[INFO] Worst-case dim_99: {worst_name} -> "
                f"{worst_row['dim']}/{worst_row['embed_dim']} ({worst_row['percent']:.2f}%)"
            )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[INFO] Saved variance report JSON: {out_json}")

    if save_q:
        torch.save(eigenvectors_payload, eigenvectors_out)
        print(f"[INFO] Saved eigenvector payload: {eigenvectors_out}")


if __name__ == "__main__":
    main()
