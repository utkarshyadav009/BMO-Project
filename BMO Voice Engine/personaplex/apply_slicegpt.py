import argparse
import json
from pathlib import Path
import time

import torch

try:
    import sentencepiece as spm
except Exception:
    spm = None

from moshi import offline
from moshi.models import loaders
from moshi.models.lm import LMModel, _iterate_audio, encode_from_sphn, load_audio


def parse_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_compute_dtype(name: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered == "float32":
        return torch.float32
    if lowered == "float64":
        return torch.float64
    raise ValueError(f"Unsupported compute dtype: {name}")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_eigenspectrum_entry(eig_payload: dict, layer_idx: int, norm_name: str) -> tuple[str, dict]:
    key = f"layer_{layer_idx}_{norm_name}"
    results = eig_payload.get("results", {})
    if key not in results:
        raise KeyError(f"Missing eigenspectrum entry: {key}")
    entry = results[key]
    if not isinstance(entry, dict):
        raise TypeError(f"Invalid eigenspectrum payload for {key}")
    return key, entry


def build_headwise_q_matrix(
    q_full: torch.Tensor,
    eigvals: torch.Tensor,
    d_old: int,
    d_new: int,
    num_heads: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if d_old % num_heads != 0:
        raise ValueError(f"d_old={d_old} must be divisible by num_heads={num_heads}")
    if d_new % num_heads != 0:
        raise ValueError(
            f"d_new={d_new} must be divisible by num_heads={num_heads} when headwise basis is enabled"
        )

    head_old = d_old // num_heads
    head_new = d_new // num_heads

    q_full = q_full.to(device=device, dtype=compute_dtype)
    eigvals = eigvals.to(device=device, dtype=compute_dtype).view(-1)
    if eigvals.shape[0] != q_full.shape[1]:
        raise ValueError(
            f"eigenvalue count mismatch: expected {q_full.shape[1]}, got {eigvals.shape[0]}"
        )

    # Reconstruct per-head covariance blocks from Q * sqrt(lambda) without building full covariance.
    weighted = q_full * torch.sqrt(torch.clamp_min(eigvals, 0.0)).unsqueeze(0)
    q_head = torch.zeros((d_old, d_new), device=device, dtype=compute_dtype)
    for h in range(num_heads):
        r0 = h * head_old
        r1 = r0 + head_old
        c0 = h * head_new
        c1 = c0 + head_new

        cov_h = weighted[r0:r1, :] @ weighted[r0:r1, :].T
        _, vecs_h = torch.linalg.eigh(cov_h)
        vecs_h = torch.flip(vecs_h[:, -head_new:], dims=[1])
        q_head[r0:r1, c0:c1] = vecs_h

    return q_head


def load_q_matrix(
    eig_payload: dict,
    layer_idx: int,
    norm_name: str,
    d_old: int,
    d_new: int,
    device: torch.device,
    *,
    num_heads: int,
    headwise_q_basis: bool,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    key, entry = get_eigenspectrum_entry(eig_payload, layer_idx, norm_name)

    q = entry.get("Q")
    if not torch.is_tensor(q):
        raise TypeError(f"Invalid Q matrix for {key}")
    if q.ndim != 2:
        raise ValueError(f"Q matrix must be rank-2 for {key}, got {q.shape}")
    if q.shape[0] != d_old:
        raise ValueError(f"Q matrix row size mismatch for {key}: expected {d_old}, got {q.shape[0]}")
    if q.shape[1] < d_new:
        raise ValueError(f"Q matrix too narrow for requested d_new={d_new} at {key}")

    if not headwise_q_basis:
        return q[:, :d_new].to(device=device, dtype=compute_dtype)

    eigvals = entry.get("eigenvalues")
    if not torch.is_tensor(eigvals):
        raise TypeError(f"Missing eigenvalues for headwise basis at {key}")

    return build_headwise_q_matrix(
        q,
        eigvals,
        d_old,
        d_new,
        num_heads,
        device,
        compute_dtype,
    )


def compress_norm_alpha(alpha: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
    a = alpha.detach().to(device=qk.device, dtype=qk.dtype).view(-1)
    # Best diagonal fit in compressed basis for diag(alpha) @ Qk.
    projected = a.unsqueeze(1) * qk
    out = torch.linalg.vector_norm(projected, dim=0)
    return out.view(1, 1, -1)


def pad_or_trim_rows(x: torch.Tensor, target_rows: int) -> torch.Tensor:
    if x.shape[0] == target_rows:
        return x
    out = torch.zeros((target_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    keep = min(target_rows, x.shape[0])
    out[:keep] = x[:keep]
    return out


def pad_or_trim_cols(x: torch.Tensor, target_cols: int) -> torch.Tensor:
    if x.shape[1] == target_cols:
        return x
    out = torch.zeros((x.shape[0], target_cols), dtype=x.dtype, device=x.device)
    keep = min(target_cols, x.shape[1])
    out[:, :keep] = x[:, :keep]
    return out


def tensor_nbytes_gb(state_dict: dict) -> float:
    total = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total / 1e9


def parse_bridge_spec(spec: str) -> tuple[int, str]:
    raw = spec.strip()
    if raw.startswith("layer_"):
        parts = raw.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid bridge layer spec: {spec}")
        return int(parts[1]), parts[2]

    parts = raw.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid bridge layer spec: {spec}")
    return int(parts[0]), parts[1]


def resolve_local_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


AUDIO_FILE_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


def collect_audio_files(root: Path, dataset_dir: str, max_files: int = 0) -> list[Path]:
    dataset_path = resolve_local_path(root, dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"High-density dataset directory not found: {dataset_path}")

    files = []
    for path in dataset_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in AUDIO_FILE_SUFFIXES:
            files.append(path.resolve())

    files.sort()
    if max_files and max_files > 0:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No audio files found under: {dataset_path}")
    return files


def build_forced_tokens(model, steps: int, seed: int, batch_size: int = 1) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    k = int(model.num_codebooks)
    text_card = int(model.text_card)
    audio_card = int(model.card)

    text_tokens = torch.randint(0, text_card, (steps, batch_size, 1), generator=g, dtype=torch.long)
    audio_tokens = torch.randint(0, audio_card + 1, (steps, batch_size, k - 1), generator=g, dtype=torch.long)
    return torch.cat([text_tokens, audio_tokens], dim=2)


@torch.no_grad()
def _encode_audio_frames_with_mimi(
    mimi,
    wav_path: Path,
    max_steps: int,
    *,
    pad_audio: bool = False,
) -> list[torch.Tensor]:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    sample_pcm = load_audio(str(wav_path), mimi.sample_rate)
    samples = _iterate_audio(sample_pcm, frame_size, max_len=max_steps, pad=bool(pad_audio))
    encoded_iter = encode_from_sphn(mimi, samples, max_batch=1)

    frames = []
    for _ in range(max_steps):
        try:
            frame_codes = next(encoded_iter)  # [1, K_audio, F]
        except StopIteration:
            break
        frames.append(frame_codes[:, :, 0].to(dtype=torch.long).cpu()[0])
    return frames


@torch.no_grad()
def build_real_forced_tokens(
    model,
    *,
    root: Path,
    steps: int,
    batch_size: int,
    extract_device: torch.device,
    input_wav: str,
    voice_prompt_wav: str,
    text_prompt: str,
    mimi_weight: str,
    tokenizer_path: str,
    voice_ratio: float,
    high_density_ls: bool,
    dataset_dir: str,
    min_keep_rows: int,
    max_audio_files: int,
    allow_audio_reuse: bool,
) -> torch.Tensor:
    if spm is None:
        raise RuntimeError(
            "Real-token calibration requested but sentencepiece is unavailable. "
            "Install sentencepiece or set --fit-token-source random."
        )

    mimi_path = resolve_local_path(root, mimi_weight)
    tok_path = resolve_local_path(root, tokenizer_path)
    in_wav_path = resolve_local_path(root, input_wav)

    if not mimi_path.exists():
        raise FileNotFoundError(f"Mimi checkpoint not found: {mimi_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tok_path}")
    if not in_wav_path.exists():
        raise FileNotFoundError(f"Calibration input wav not found: {in_wav_path}")

    voice_wav_path = None
    voice_prompt_wav = str(voice_prompt_wav).strip()
    if voice_prompt_wav:
        voice_wav_path = resolve_local_path(root, voice_prompt_wav)
        if not voice_wav_path.exists():
            raise FileNotFoundError(f"Calibration voice prompt wav not found: {voice_wav_path}")

    print(
        "[INFO] Building real-token LS stream: "
        f"input_wav={in_wav_path} voice_prompt_wav={voice_wav_path} "
        f"high_density={bool(high_density_ls)}"
    )
    print(f"[INFO] Loading Mimi for LS token extraction: {mimi_path}")

    mimi = loaders.get_mimi(str(mimi_path), extract_device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    tokenizer = spm.SentencePieceProcessor(str(tok_path))

    base_audio_paths: list[Path] = []
    if voice_wav_path is not None:
        base_audio_paths.append(voice_wav_path)
    base_audio_paths.append(in_wav_path)

    dense_audio_paths: list[Path] = []
    if bool(high_density_ls):
        all_dense = collect_audio_files(root, dataset_dir, max_files=int(max_audio_files))
        excluded = {p.resolve() for p in base_audio_paths}
        dense_audio_paths = [p for p in all_dense if p.resolve() not in excluded]
        print(
            "[INFO] High-density LS audio pool: "
            f"dataset_dir={resolve_local_path(root, dataset_dir)} total_files={len(all_dense)} "
            f"usable_files={len(dense_audio_paths)}"
        )

    target_steps = int(steps)
    if bool(high_density_ls):
        target_steps = max(target_steps, int(min_keep_rows))

    audio_frames: list[torch.Tensor] = []
    try:
        if not bool(high_density_ls):
            if voice_wav_path is not None and target_steps > 1:
                voice_steps = int(round(float(voice_ratio) * float(target_steps)))
                voice_steps = max(1, min(voice_steps, target_steps - 1))
                input_steps = max(1, target_steps - voice_steps)
                audio_frames.extend(
                    _encode_audio_frames_with_mimi(mimi, voice_wav_path, voice_steps, pad_audio=False)
                )
                audio_frames.extend(
                    _encode_audio_frames_with_mimi(mimi, in_wav_path, input_steps, pad_audio=False)
                )
            else:
                audio_frames.extend(_encode_audio_frames_with_mimi(mimi, in_wav_path, target_steps, pad_audio=False))
        else:
            ordered_paths = list(base_audio_paths)
            ordered_paths.extend(dense_audio_paths)
            if not ordered_paths:
                raise RuntimeError("High-density LS requested but no audio paths are available")

            pass_idx = 0
            while len(audio_frames) < target_steps:
                before = len(audio_frames)
                for wav_path in ordered_paths:
                    if len(audio_frames) >= target_steps:
                        break
                    remaining = target_steps - len(audio_frames)
                    encoded = _encode_audio_frames_with_mimi(mimi, wav_path, remaining, pad_audio=False)
                    if encoded:
                        audio_frames.extend(encoded)

                pass_idx += 1
                if len(audio_frames) >= target_steps:
                    break
                if len(audio_frames) == before:
                    break
                if not bool(allow_audio_reuse):
                    break

            if audio_frames and len(audio_frames) < target_steps:
                base_frames = list(audio_frames)
                idx = 0
                while len(audio_frames) < target_steps:
                    audio_frames.append(base_frames[idx % len(base_frames)])
                    idx += 1

            print(
                "[INFO] High-density LS stream coverage: "
                f"target_steps={target_steps} built_frames={len(audio_frames)} passes={pass_idx}"
            )
    finally:
        del mimi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    k = int(model.num_codebooks)
    audio_pad = int(model.card)
    text_pad = 0

    forced_base = torch.full((steps, k), audio_pad, dtype=torch.long)
    forced_base[:, 0] = text_pad

    text_ids = tokenizer.encode(offline.wrap_with_system_tags(text_prompt))
    for t in range(min(steps, len(text_ids))):
        forced_base[t, 0] = int(text_ids[t])

    for t in range(min(steps, len(audio_frames))):
        audio_codes = audio_frames[t]
        n_audio = min(k - 1, int(audio_codes.numel()))
        forced_base[t, 1 : 1 + n_audio] = audio_codes[:n_audio]

    forced = forced_base.unsqueeze(1).repeat(1, batch_size, 1)
    if batch_size > 1:
        stride = max(1, steps // batch_size)
        for b in range(batch_size):
            forced[:, b, :] = forced_base.roll(shifts=b * stride, dims=0)

    audio_tokens = forced_base[:, 1:]
    pad_ratio = float((audio_tokens == audio_pad).float().mean().item()) if audio_tokens.numel() > 0 else 1.0
    nonpad_text = int((forced_base[:, 0] != text_pad).sum().item())
    print(
        "[INFO] Real-token LS stream stats: "
        f"steps={steps} batch={batch_size} text_nonpad={nonpad_text} "
        f"audio_pad_ratio={pad_ratio:.3f}"
    )
    return forced


def infer_model_input_device(model, fallback: torch.device) -> torch.device:
    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    b = next(model.buffers(), None)
    if b is not None:
        return b.device
    return fallback


@torch.no_grad()
def fit_output_projection_least_squares(
    src_model,
    tgt_model,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    ridge: float,
    forced_tokens: torch.Tensor | None = None,
    mask_audio_pad: bool = False,
    audio_pad_token: int | None = None,
    center_before_solve: bool = True,
    calibrate_gain: bool = True,
    gain_mode: str = "scalar",
    gain_clamp_max: float = 0.0,
    apply_bias_correction: bool = True,
    rank1_mean_shift: bool = True,
) -> int:
    out_norm = getattr(src_model, "out_norm", None)
    out_proj = getattr(getattr(tgt_model, "transformer", None), "output_proj", None)
    if out_norm is None or out_proj is None or not hasattr(out_proj, "weight"):
        print("[WARN] Output-proj LS fit skipped: required modules not found")
        return 0

    d_old = int(out_proj.weight.shape[0])
    d_new = int(out_proj.weight.shape[1])

    accum_device = torch.device(device)
    src_input_device = infer_model_input_device(src_model, fallback=accum_device)
    tgt_input_device = infer_model_input_device(tgt_model, fallback=torch.device("cpu"))

    if tgt_input_device.type == "cpu" and src_input_device.type == "cuda":
        try:
            tgt_model.to(src_input_device)
            tgt_input_device = infer_model_input_device(tgt_model, fallback=src_input_device)
            print(f"[INFO] Output-proj LS moved target model to {tgt_input_device} for faster calibration")
        except RuntimeError as exc:
            torch.cuda.empty_cache()
            print(
                "[WARN] Output-proj LS could not move target model to CUDA; "
                f"keeping CPU target ({type(exc).__name__}: {exc})"
            )

    print(
        "[INFO] Output-proj LS devices: "
        f"src={src_input_device} tgt={tgt_input_device} accum={accum_device}"
    )

    gram_zz = torch.zeros((d_new, d_new), device=accum_device, dtype=torch.float32)
    gram_xz = torch.zeros((d_old, d_new), device=accum_device, dtype=torch.float32)
    sum_x = torch.zeros((d_old,), device=accum_device, dtype=torch.float32)
    sum_x2 = torch.zeros((d_old,), device=accum_device, dtype=torch.float32)
    sum_z = torch.zeros((d_new,), device=accum_device, dtype=torch.float32)
    sum_x_sq = 0.0
    total_rows = 0
    masked_rows = 0

    src_cache = []
    tgt_cache = []

    def src_pre_hook(_module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            src_cache.append(x.detach().float().reshape(-1, d_old))

    def tgt_pre_hook(_module, inputs):
        if not inputs:
            return
        z = inputs[0]
        if torch.is_tensor(z):
            tgt_cache.append(z.detach().float().reshape(-1, d_new))

    h_src = out_norm.register_forward_pre_hook(src_pre_hook)
    h_tgt = out_proj.register_forward_pre_hook(tgt_pre_hook)

    total_steps = int(steps)
    sample_count = 0

    if forced_tokens is None:
        forced_tokens = build_forced_tokens(src_model, total_steps, int(seed), batch_size=int(batch_size))
    else:
        forced_tokens = forced_tokens.detach().to(dtype=torch.long, device="cpu")
        if forced_tokens.ndim == 2:
            forced_tokens = forced_tokens.unsqueeze(1)
        if forced_tokens.ndim != 3:
            raise ValueError(f"forced_tokens must have rank 3 [steps, batch, K], got shape={forced_tokens.shape}")
        if int(forced_tokens.shape[2]) != int(src_model.num_codebooks):
            raise ValueError(
                "forced_tokens codebook mismatch: "
                f"expected K={int(src_model.num_codebooks)}, got {int(forced_tokens.shape[2])}"
            )
        if int(forced_tokens.shape[0]) < total_steps:
            raise ValueError(
                "forced_tokens too short: "
                f"required at least {total_steps}, got {int(forced_tokens.shape[0])}"
            )
        if int(forced_tokens.shape[1]) == 1 and int(batch_size) > 1:
            forced_tokens = forced_tokens.repeat(1, int(batch_size), 1)
        elif int(forced_tokens.shape[1]) != int(batch_size):
            raise ValueError(
                "forced_tokens batch mismatch: "
                f"expected batch={int(batch_size)}, got {int(forced_tokens.shape[1])}"
            )
        forced_tokens = forced_tokens[:total_steps]

    speech_keep_mask = None
    if bool(mask_audio_pad):
        if audio_pad_token is None:
            inferred_pad = int(getattr(src_model, "card", -1))
            if inferred_pad >= 0:
                audio_pad_token = inferred_pad
        if audio_pad_token is None:
            print("[WARN] Output-proj LS pad-mask requested but audio pad token is unknown; disabling mask")
        elif int(forced_tokens.shape[2]) <= 1:
            print("[WARN] Output-proj LS pad-mask requested but no audio codebooks found; disabling mask")
        else:
            audio_tokens = forced_tokens[:, :, 1:]
            speech_keep_mask = (audio_tokens != int(audio_pad_token)).any(dim=2)
            keep_rows = int(speech_keep_mask.sum().item())
            all_rows = int(speech_keep_mask.numel())
            keep_ratio = float(keep_rows / max(1, all_rows))
            print(
                "[INFO] Output-proj LS pad-mask enabled: "
                f"audio_pad_token={int(audio_pad_token)} keep_rows={keep_rows}/{all_rows} "
                f"keep_ratio={keep_ratio:.3f}"
            )

    forced_tokens_src = forced_tokens
    if forced_tokens_src.device != src_input_device:
        forced_tokens_src = forced_tokens_src.to(src_input_device, non_blocking=True)

    if tgt_input_device == src_input_device:
        forced_tokens_tgt = forced_tokens_src
    else:
        forced_tokens_tgt = forced_tokens
        if forced_tokens_tgt.device != tgt_input_device:
            forced_tokens_tgt = forced_tokens_tgt.to(tgt_input_device, non_blocking=True)

    progress_interval = max(1, total_steps // 16)
    t_start = time.perf_counter()

    def map_step_mask(step_mask: torch.Tensor, n_rows: int, expected_batch: int, out_device: torch.device) -> torch.Tensor:
        step_mask = step_mask.detach().to(dtype=torch.bool, device="cpu").view(-1)
        b = int(step_mask.numel())
        if n_rows <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=out_device)

        if b == n_rows:
            return step_mask.to(device=out_device)
        if b > 0 and n_rows % b == 0:
            return step_mask.repeat_interleave(n_rows // b).to(device=out_device)
        if b > 0 and b % n_rows == 0:
            group = b // n_rows
            return step_mask.view(n_rows, group).any(dim=1).to(device=out_device)
        if n_rows == 1:
            return torch.tensor([bool(step_mask.any().item())], dtype=torch.bool, device=out_device)

        keep_ratio = float(step_mask.float().mean().item()) if b > 0 else 0.0
        keep_n = int(round(keep_ratio * float(n_rows)))
        keep_n = max(0, min(n_rows, keep_n))
        out = torch.zeros((n_rows,), dtype=torch.bool, device=out_device)
        if keep_n > 0:
            out[:keep_n] = True
        return out

    try:
        with src_model.streaming(batch_size=int(batch_size)), tgt_model.streaming(batch_size=int(batch_size)):
            for t in range(total_steps):
                seq_src = forced_tokens_src[t].unsqueeze(-1).contiguous()
                seq_tgt = forced_tokens_tgt[t].unsqueeze(-1).contiguous()

                src_model.forward_codes(seq_src)
                tgt_model.forward_codes(seq_tgt)

                step_idx = t + 1
                if (step_idx % progress_interval == 0) or (step_idx == total_steps):
                    elapsed = max(1e-6, time.perf_counter() - t_start)
                    iter_per_sec = step_idx / elapsed
                    eta_sec = (total_steps - step_idx) / max(iter_per_sec, 1e-6)
                    print(
                        "[INFO] Output-proj LS progress: "
                        f"{step_idx}/{total_steps} ({100.0 * step_idx / total_steps:.1f}%) "
                        f"samples={sample_count} iter_per_sec={iter_per_sec:.2f} eta_sec={eta_sec:.1f}"
                    )

                if not src_cache or not tgt_cache:
                    continue

                x = src_cache.pop()
                z = tgt_cache.pop()
                n = min(int(x.shape[0]), int(z.shape[0]))
                if n <= 0:
                    continue

                x = x[:n]
                z = z[:n]
                total_rows += n

                if speech_keep_mask is not None:
                    row_mask = map_step_mask(speech_keep_mask[t], n, int(batch_size), x.device)
                    keep_n = int(row_mask.sum().item())
                    masked_rows += n - keep_n
                    if keep_n <= 0:
                        continue
                    x = x[row_mask]
                    z = z[row_mask]

                if x.device != accum_device:
                    x = x.to(accum_device, non_blocking=True)
                if z.device != accum_device:
                    z = z.to(accum_device, non_blocking=True)
                gram_zz.add_(z.T @ z)
                gram_xz.add_(x.T @ z)
                sum_x.add_(x.sum(dim=0))
                sum_x2.add_((x * x).sum(dim=0))
                sum_z.add_(z.sum(dim=0))
                sum_x_sq += float((x * x).sum().item())
                sample_count += int(x.shape[0])
    finally:
        h_src.remove()
        h_tgt.remove()

    if sample_count <= 0:
        print("[WARN] Output-proj LS fit collected zero samples; skipping")
        return 0

    if sample_count < d_new:
        print(
            "[WARN] Output-proj LS is underdetermined: "
            f"samples={sample_count} < d_new={d_new}; applying stronger regularization"
        )

    mean_x = sum_x / float(sample_count)
    mean_z = sum_z / float(sample_count)

    gram_zz_centered = gram_zz - float(sample_count) * torch.outer(mean_z, mean_z)
    gram_xz_centered = gram_xz - float(sample_count) * torch.outer(mean_x, mean_z)
    gram_zz_centered = 0.5 * (gram_zz_centered + gram_zz_centered.T)

    if center_before_solve:
        solve_zz = gram_zz_centered
        solve_xz = gram_xz_centered
    else:
        solve_zz = gram_zz
        solve_xz = gram_xz

    reg_scale = float((torch.trace(solve_zz).clamp_min(0.0) / max(1, d_new)).item())
    reg_boost = max(1.0, float(d_new) / float(max(1, sample_count)))
    reg = float(ridge) * reg_boost * (reg_scale + 1e-6)
    eye = torch.eye(d_new, device=accum_device, dtype=torch.float32)

    # Solve (Z^T Z + reg I) * W^T = Z^T X in normal-equation form.
    solved_t = torch.linalg.solve(solve_zz + reg * eye, solve_xz.T)
    fitted_w = solved_t.T.contiguous()

    gm = str(gain_mode).strip().lower()
    if not calibrate_gain:
        gm = "none"
    if gm not in {"none", "scalar", "per_channel"}:
        raise ValueError(f"Unsupported gain_mode: {gain_mode}")

    gamma = 1.0
    gain_vec = None

    teacher_center_energy = sum_x_sq - float(sample_count) * float(torch.dot(mean_x, mean_x).item())
    teacher_center_energy = max(teacher_center_energy, 1e-8)

    student_projected_cov = fitted_w @ gram_zz_centered @ fitted_w.T
    student_center_energy = float(torch.trace(student_projected_cov).item())
    student_center_energy = max(student_center_energy, 1e-8)

    if gm == "scalar":
        gamma = (teacher_center_energy / student_center_energy) ** 0.5
        fitted_w.mul_(float(gamma))
    elif gm == "per_channel":
        teacher_var = (sum_x2 / float(sample_count)) - (mean_x * mean_x)
        teacher_var = torch.clamp(teacher_var, min=0.0)

        cov_zz = gram_zz_centered / float(max(1, sample_count - 1))
        cov_zz = 0.5 * (cov_zz + cov_zz.T)

        student_var = torch.sum((fitted_w @ cov_zz) * fitted_w, dim=1)
        student_var = torch.clamp(student_var, min=0.0)

        std_teacher = torch.sqrt(teacher_var + 1e-8)
        std_student = torch.sqrt(student_var + 1e-8)
        gain_vec = std_teacher / torch.clamp_min(std_student, 1e-8)

        if float(gain_clamp_max) > 1.0:
            clamp_max = float(gain_clamp_max)
            clamp_min = 1.0 / clamp_max
            gain_vec = torch.clamp(gain_vec, min=clamp_min, max=clamp_max)

        fitted_w = fitted_w * gain_vec.unsqueeze(1)

    fitted_bias = None
    if apply_bias_correction:
        out_proj_bias = getattr(out_proj, "bias", None)
        if isinstance(out_proj_bias, torch.Tensor):
            mean_student_projected = torch.mv(fitted_w, mean_z)
            fitted_bias = mean_x - mean_student_projected
        elif rank1_mean_shift:
            mean_student_projected = torch.mv(fitted_w, mean_z)
            delta_mean = mean_x - mean_student_projected
            mean_z_norm2 = float(torch.dot(mean_z, mean_z).item())
            if mean_z_norm2 > 1e-12:
                # Apply a minimal rank-1 correction so W * mean(z) matches mean(x) without a bias term.
                fitted_w.add_(torch.outer(delta_mean, mean_z) / mean_z_norm2)
            else:
                print("[WARN] Output-proj LS rank-1 mean-shift skipped: mean(z) norm is near zero")
        else:
            print("[WARN] Output-proj LS bias correction requested but output_proj has no bias parameter")

    with torch.no_grad():
        out_proj.weight.copy_(fitted_w.to(device=out_proj.weight.device, dtype=out_proj.weight.dtype))
        if fitted_bias is not None:
            out_proj.bias.copy_(fitted_bias.to(device=out_proj.bias.device, dtype=out_proj.bias.dtype))

    kept_ratio = float(sample_count / max(1, total_rows))
    if speech_keep_mask is not None:
        print(
            "[INFO] Output-proj LS masked rows: "
            f"kept={sample_count} masked={masked_rows} total={total_rows} keep_ratio={kept_ratio:.3f}"
        )

    teacher_center_norm = max(teacher_center_energy, 0.0) ** 0.5 if calibrate_gain else 0.0
    student_center_norm = (
        max(student_center_energy, 0.0) ** 0.5
        if calibrate_gain
        else 0.0
    )
    bias_norm = float(torch.linalg.vector_norm(fitted_bias).item()) if fitted_bias is not None else 0.0
    gain_min = float(gain_vec.min().item()) if gain_vec is not None else gamma
    gain_max = float(gain_vec.max().item()) if gain_vec is not None else gamma
    gain_mean = float(gain_vec.mean().item()) if gain_vec is not None else gamma

    print(
        "[INFO] Output-proj LS fit applied: "
        f"samples={sample_count} batch={batch_size} ridge={ridge:.2e} "
        f"reg_boost={reg_boost:.3f} effective_reg={reg:.6e} "
        f"centered={bool(center_before_solve)} gain_mode={gm} gain={gamma:.6f} "
        f"gain_min={gain_min:.6f} gain_max={gain_max:.6f} gain_mean={gain_mean:.6f} "
        f"teacher_center_norm={teacher_center_norm:.4f} student_center_norm={student_center_norm:.4f} "
        f"bias_norm={bias_norm:.4f}"
    )
    return sample_count


def get_rms_alpha_or_ones(
    norm_module,
    d_old: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    alpha = getattr(norm_module, "alpha", None)
    if isinstance(alpha, torch.Tensor):
        return alpha.detach().to(device=device, dtype=compute_dtype).view(-1)
    return torch.ones((d_old,), device=device, dtype=compute_dtype)


def set_norm_alpha_one_if_present(norm_module):
    alpha = getattr(norm_module, "alpha", None)
    if isinstance(alpha, torch.Tensor):
        with torch.no_grad():
            alpha.fill_(1.0)


def rotate_attention_with_basis(
    src_in_blocks: torch.Tensor,
    src_out_proj: torch.Tensor,
    q_basis: torch.Tensor,
    alpha: torch.Tensor,
    alpha_scale_for_weights: float,
    *,
    rope_safe_quarot: bool,
    rope_safe_v_mode: str = "one-sided",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate attention weights into the compressed basis.

    In RoPE-safe identity mode, q/k are input-side-only so RoPE acts on native
    q/k coordinates. Value/output branch can be either:
    - one-sided: v input-side-only, out output-side-only
    - two-sided: v/out both-sided conjugation in rotated space
    """
    w_q = src_in_blocks[0] * alpha.unsqueeze(0) * alpha_scale_for_weights
    w_k = src_in_blocks[1] * alpha.unsqueeze(0) * alpha_scale_for_weights
    w_v = src_in_blocks[2] * alpha.unsqueeze(0) * alpha_scale_for_weights

    if rope_safe_quarot:
        if q_basis.shape[0] != q_basis.shape[1]:
            raise ValueError(
                "RoPE-safe QuaRot mode requires square basis (identity mode): "
                f"got q shape={tuple(q_basis.shape)}"
            )
        v_mode = str(rope_safe_v_mode).strip().lower()
        if v_mode not in {"one-sided", "two-sided"}:
            raise ValueError(
                "Invalid rope_safe_v_mode: "
                f"{rope_safe_v_mode}. Expected one of ['one-sided', 'two-sided']"
            )
        # Input-side-only fusion for q/k/v so attention projections consume rotated residual
        # but produce native q/k channels where RoPE is defined.
        dst_q = w_q @ q_basis
        dst_k = w_k @ q_basis
        if v_mode == "two-sided":
            dst_v = q_basis.T @ w_v @ q_basis
            dst_out_proj = q_basis.T @ src_out_proj @ q_basis
        else:
            dst_v = w_v @ q_basis
            # Output-side-only projection maps native attention output back to rotated residual basis.
            dst_out_proj = q_basis.T @ src_out_proj
        dst_in_proj = torch.cat([dst_q, dst_k, dst_v], dim=0)
        return dst_in_proj, dst_out_proj

    dst_q = q_basis.T @ w_q @ q_basis
    dst_k = q_basis.T @ w_k @ q_basis
    dst_v = q_basis.T @ w_v @ q_basis
    dst_in_proj = torch.cat([dst_q, dst_k, dst_v], dim=0)
    dst_out_proj = q_basis.T @ src_out_proj @ q_basis
    return dst_in_proj, dst_out_proj


def main():
    parser = argparse.ArgumentParser(description="Apply uniform SliceGPT compression to temporal transformer")
    parser.add_argument("--bf16", default="v5_step1500.safetensors")
    parser.add_argument("--eigenvectors", default="bmo_slicegpt_eigenvectors.pt")
    parser.add_argument("--config", default="bmo_config.json")
    parser.add_argument("--out", default="bmo_slicegpt_2560.pt")
    parser.add_argument("--d-new", type=int, default=2560)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bridge-in-layer", default="0_norm1")
    parser.add_argument("--bridge-out-layer", default="31_norm2")
    parser.add_argument(
        "--single-q-basis",
        type=parse_bool,
        default=True,
        help="If true, use one basis per layer for both attn and MLP branches.",
    )
    parser.add_argument(
        "--basis-source",
        default="norm1",
        choices=["norm1", "norm2"],
        help="Source hook to derive the layer basis when --single-q-basis is true.",
    )
    parser.add_argument(
        "--absorb-rms-alpha",
        type=parse_bool,
        default=True,
        help="Absorb RMSNorm alpha into input-side weights before basis projection.",
    )
    parser.add_argument(
        "--global-q-basis",
        type=parse_bool,
        default=True,
        help=(
            "If true with --single-q-basis, use one global basis across all layers "
            "to avoid inter-layer basis handoff mismatch."
        ),
    )
    parser.add_argument(
        "--global-q-layer",
        type=int,
        default=None,
        help="Optional layer index for global basis source. Defaults to bridge-in layer index.",
    )
    parser.add_argument(
        "--global-q-norm",
        choices=["norm1", "norm2"],
        default=None,
        help="Optional norm hook for global basis source. Defaults to --basis-source.",
    )
    parser.add_argument(
        "--headwise-q-basis",
        type=parse_bool,
        default=True,
        help=(
            "If true, rebuild Q as block-diagonal per attention head to preserve "
            "head partition structure."
        ),
    )
    parser.add_argument(
        "--attn-rope-mode",
        choices=["none", "auto", "quarot"],
        default="auto",
        help=(
            "RoPE handling for attention rotations: none uses dense Q conjugation, "
            "auto enables QuaRot fusion in identity mode only, quarot forces QuaRot fusion "
            "(requires d_new == d_old)."
        ),
    )
    parser.add_argument(
        "--rope-safe-v-mode",
        choices=["one-sided", "two-sided"],
        default="one-sided",
        help=(
            "Value/output transform mode when --attn-rope-mode enables RoPE-safe path: "
            "one-sided uses v=input-side and out=output-side; two-sided uses both-sided "
            "conjugation for v/out."
        ),
    )
    parser.add_argument(
        "--rotation-math-dtype",
        choices=["float32", "float64"],
        default="float32",
        help=(
            "Compute dtype used for SliceGPT basis/weight rotation math before final checkpoint cast. "
            "Use float64 to reduce numerical drift in identity diagnostics."
        ),
    )
    parser.add_argument(
        "--global-q-bridge-out",
        type=parse_bool,
        default=False,
        help=(
            "If true, tie output bridge basis to global Q. If false, keep bridge-out layer basis "
            "for final decode."
        ),
    )
    parser.add_argument(
        "--rms-dim-compensation",
        type=parse_bool,
        default=False,
        help=(
            "Apply sqrt(d_old/d_new) compensation when absorbing RMS alpha (and reciprocal when "
            "copying compressed alpha)."
        ),
    )
    parser.add_argument(
        "--preserve-head-dim",
        type=parse_bool,
        default=True,
        help=(
            "If true, keep original attention head dimension by adjusting temporal inner num_heads "
            "when possible."
        ),
    )
    parser.add_argument(
        "--inner-num-heads",
        type=int,
        default=0,
        help="Optional explicit num_heads for compressed inner transformer (0 means auto).",
    )
    parser.add_argument(
        "--fit-output-proj-ls",
        type=parse_bool,
        default=False,
        help="If true, fit output_proj with least squares against BF16 activations.",
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        default=256,
        help="Number of forced-token steps used for output_proj least-squares fitting.",
    )
    parser.add_argument(
        "--fit-batch-size",
        type=int,
        default=8,
        help="Batch size for forced-token streams used in output_proj least-squares fitting.",
    )
    parser.add_argument(
        "--fit-seed",
        type=int,
        default=1234,
        help="Seed for forced-token stream used by output_proj least-squares fitting.",
    )
    parser.add_argument(
        "--fit-ridge",
        type=float,
        default=1e-4,
        help="Ridge coefficient for output_proj least-squares fitting.",
    )
    parser.add_argument(
        "--fit-token-source",
        choices=["real", "random"],
        default="real",
        help="Token source for bridge LS calibration.",
    )
    parser.add_argument(
        "--fit-input-wav",
        default="tellmeajoke_padded.wav",
        help="Primary user audio wav used for real-token LS calibration.",
    )
    parser.add_argument(
        "--fit-voice-prompt-wav",
        default="bmo_621.wav",
        help="Voice prompt wav used for real-token LS calibration (empty disables).",
    )
    parser.add_argument(
        "--fit-text-prompt",
        default="Tell me a joke.",
        help="Text prompt used when building real-token LS calibration streams.",
    )
    parser.add_argument(
        "--fit-mimi-weight",
        default="tokenizer-e351c8d8-checkpoint125.safetensors",
        help="Mimi checkpoint used for real-token LS calibration streams.",
    )
    parser.add_argument(
        "--fit-tokenizer",
        default="tokenizer_spm_32k_3.model",
        help="Tokenizer model used for real-token LS calibration streams.",
    )
    parser.add_argument(
        "--fit-voice-ratio",
        type=float,
        default=0.25,
        help="Fraction of calibration steps reserved for voice prompt frames in real-token mode.",
    )
    parser.add_argument(
        "--fit-mask-audio-pad",
        type=parse_bool,
        default=True,
        help="If true, drop all-pad audio rows when solving bridge LS to avoid silence bias.",
    )
    parser.add_argument(
        "--fit-center-solve",
        type=parse_bool,
        default=True,
        help="If true, solve LS on centered teacher/student activations.",
    )
    parser.add_argument(
        "--fit-gain-calibration",
        type=parse_bool,
        default=True,
        help="If true, apply scalar norm calibration after bridge LS solve.",
    )
    parser.add_argument(
        "--fit-bias-correction",
        type=parse_bool,
        default=True,
        help="If true, apply mean-shift correction via output_proj bias when available.",
    )
    parser.add_argument(
        "--high-density-ls",
        type=parse_bool,
        default=False,
        help="If true, build a long multi-file real-token stream for bridge LS calibration.",
    )
    parser.add_argument(
        "--fit-dataset-dir",
        default="bmo_dataset_clean",
        help="Dataset directory scanned recursively for high-density LS audio clips.",
    )
    parser.add_argument(
        "--fit-min-keep-rows",
        type=int,
        default=20000,
        help="Minimum active rows targeted by high-density LS mode.",
    )
    parser.add_argument(
        "--fit-max-audio-files",
        type=int,
        default=0,
        help="Maximum number of dataset audio files for high-density LS (0 means all).",
    )
    parser.add_argument(
        "--fit-allow-audio-reuse",
        type=parse_bool,
        default=True,
        help="If true, high-density LS cycles through dataset clips until enough frames are built.",
    )
    parser.add_argument(
        "--fit-per-channel-gain",
        type=parse_bool,
        default=False,
        help="If true, apply diagonal per-output-channel gain instead of a single scalar gain.",
    )
    parser.add_argument(
        "--fit-gain-clamp-max",
        type=float,
        default=0.0,
        help="Optional max absolute gain multiplier for per-channel gain (<=1 disables clamping).",
    )
    parser.add_argument(
        "--fit-rank1-mean-shift",
        type=parse_bool,
        default=True,
        help="If true and output_proj has no bias, apply rank-1 mean-shift correction to weights.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if int(args.fit_steps) <= 0:
        raise ValueError(f"fit-steps must be > 0, got {args.fit_steps}")
    if int(args.fit_batch_size) <= 0:
        raise ValueError(f"fit-batch-size must be > 0, got {args.fit_batch_size}")
    if float(args.fit_voice_ratio) < 0.0 or float(args.fit_voice_ratio) >= 1.0:
        raise ValueError(f"fit-voice-ratio must be in [0, 1), got {args.fit_voice_ratio}")
    if int(args.fit_min_keep_rows) <= 0:
        raise ValueError(f"fit-min-keep-rows must be > 0, got {args.fit_min_keep_rows}")
    if int(args.fit_max_audio_files) < 0:
        raise ValueError(f"fit-max-audio-files must be >= 0, got {args.fit_max_audio_files}")
    if float(args.fit_gain_clamp_max) < 0.0:
        raise ValueError(f"fit-gain-clamp-max must be >= 0, got {args.fit_gain_clamp_max}")

    torch_dtype = parse_dtype(args.dtype)
    work_device = torch.device(args.device)

    root = Path(__file__).resolve().parent
    bf16_path = (root / args.bf16).resolve()
    eigen_path = (root / args.eigenvectors).resolve()
    config_path = (root / args.config).resolve()
    out_path = (root / args.out).resolve()

    for p in (bf16_path, eigen_path, config_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print(f"[INFO] Loading source BF16 model: {bf16_path}")
    src_model = loaders.get_moshi_lm(
        str(bf16_path),
        device=work_device,
        dtype=torch.bfloat16,
        cpu_offload=False,
    )
    src_model.eval()
    for param in src_model.parameters():
        param.requires_grad = False

    print(f"[INFO] Loading eigenvectors: {eigen_path}")
    eig_payload = torch.load(str(eigen_path), map_location="cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.pop("model_type", None)
    d_old = int(cfg["dim"])
    orig_num_heads = int(cfg["num_heads"])
    if d_old % orig_num_heads != 0:
        raise ValueError(f"dim={d_old} must be divisible by num_heads={orig_num_heads}")
    orig_head_dim = d_old // orig_num_heads
    d_new = int(args.d_new)
    if d_new <= 0 or d_new > d_old:
        raise ValueError(f"d_new must be in (0, {d_old}], got {d_new}")
    if args.headwise_q_basis and d_new % orig_num_heads != 0:
        raise ValueError(
            f"d_new={d_new} must be divisible by num_heads={orig_num_heads} when --headwise-q-basis is enabled"
        )

    if args.inner_num_heads and args.inner_num_heads > 0:
        inner_num_heads = int(args.inner_num_heads)
    elif args.preserve_head_dim and d_new % orig_head_dim == 0:
        inner_num_heads = d_new // orig_head_dim
    else:
        inner_num_heads = orig_num_heads

    if d_new % inner_num_heads != 0:
        raise ValueError(
            f"d_new={d_new} must be divisible by inner_num_heads={inner_num_heads}"
        )
    if args.headwise_q_basis and inner_num_heads != orig_num_heads:
        raise ValueError(
            "--headwise-q-basis requires inner_num_heads == original num_heads; "
            f"got inner={inner_num_heads}, original={orig_num_heads}. "
            "Disable headwise basis or set --inner-num-heads to original."
        )

    num_layers = len(src_model.transformer.layers)
    identity_rotation_mode = d_new == d_old

    attn_rope_mode = str(args.attn_rope_mode).strip().lower()
    rope_safe_v_mode = str(args.rope_safe_v_mode).strip().lower()
    rotation_math_dtype = parse_compute_dtype(args.rotation_math_dtype)
    use_rope_safe_quarot = False
    if attn_rope_mode == "quarot":
        if not identity_rotation_mode:
            raise ValueError(
                "--attn-rope-mode quarot currently requires d_new == d_old. "
                f"Got d_new={d_new}, d_old={d_old}."
            )
        use_rope_safe_quarot = True
    elif attn_rope_mode == "auto":
        use_rope_safe_quarot = bool(identity_rotation_mode)

    print(
        "[INFO] Attention RoPE mode: "
        f"mode={attn_rope_mode} quarot_active={bool(use_rope_safe_quarot)} "
        f"rope_safe_v_mode={rope_safe_v_mode}"
    )
    print(f"[INFO] Rotation math dtype: {str(rotation_math_dtype).replace('torch.', '')}")

    # Preserve original gating hidden width for easier weight transfer.
    hidden_old = int(src_model.transformer.layers[0].gating.linear_out.weight.shape[1])
    temporal_dim_feedforward = int((3 * hidden_old) // 2)
    if (2 * temporal_dim_feedforward) // 3 != hidden_old:
        raise RuntimeError(
            "Unable to derive temporal_dim_feedforward that preserves hidden width; "
            f"hidden_old={hidden_old} temporal_dim_feedforward={temporal_dim_feedforward}"
        )

    cfg_override = dict(cfg)
    cfg_override["temporal_inner_dim"] = d_new
    cfg_override["temporal_dim_feedforward"] = temporal_dim_feedforward
    cfg_override["num_heads"] = inner_num_heads
    cfg_override["force_temporal_projected"] = bool(identity_rotation_mode)

    print(
        "[INFO] Building compressed target architecture: "
        f"outer_dim={d_old} inner_dim={d_new} temporal_dim_feedforward={temporal_dim_feedforward} "
        f"orig_heads={orig_num_heads} inner_heads={inner_num_heads} "
        f"orig_head_dim={orig_head_dim} inner_head_dim={d_new // inner_num_heads}"
    )
    tgt_model = LMModel(device="cpu", dtype=torch_dtype, **cfg_override)
    tgt_model.eval()
    for param in tgt_model.parameters():
        param.requires_grad = False

    src_sd = src_model.state_dict()
    tgt_sd = tgt_model.state_dict()

    # Copy non-temporal weights directly (heads, depformer, embeddings, etc.).
    copied_direct = 0
    skipped_direct = 0
    for key, value in src_sd.items():
        if key.startswith("transformer."):
            continue
        if key in tgt_sd and tgt_sd[key].shape == value.shape:
            copied = value.detach().cpu()
            if torch.is_tensor(copied) and copied.is_floating_point():
                copied = copied.to(dtype=torch_dtype)
            tgt_sd[key] = copied
            copied_direct += 1
        else:
            skipped_direct += 1

    tgt_model.load_state_dict(tgt_sd, strict=False, assign=True)
    print(f"[INFO] Direct-copy weights: copied={copied_direct} skipped={skipped_direct}")

    # Bridge setup: 4096 -> 2560 -> 4096.
    bridge_in_idx, in_norm = parse_bridge_spec(args.bridge_in_layer)
    bridge_out_idx, out_norm = parse_bridge_spec(args.bridge_out_layer)

    global_q_layer = bridge_in_idx if args.global_q_layer is None else int(args.global_q_layer)
    global_q_norm = args.basis_source if args.global_q_norm is None else args.global_q_norm

    q_global = None
    if args.single_q_basis and args.global_q_basis:
        q_global = load_q_matrix(
            eig_payload,
            global_q_layer,
            global_q_norm,
            d_old,
            d_new,
            work_device,
            num_heads=orig_num_heads,
            headwise_q_basis=bool(args.headwise_q_basis),
            compute_dtype=rotation_math_dtype,
        )

    if q_global is not None:
        q_in = q_global
        if args.global_q_bridge_out:
            q_out = q_global
        else:
            q_out = load_q_matrix(
                eig_payload,
                bridge_out_idx,
                out_norm,
                d_old,
                d_new,
                work_device,
                num_heads=orig_num_heads,
                headwise_q_basis=bool(args.headwise_q_basis),
                compute_dtype=rotation_math_dtype,
            )
    else:
        q_in = load_q_matrix(
            eig_payload,
            bridge_in_idx,
            in_norm,
            d_old,
            d_new,
            work_device,
            num_heads=orig_num_heads,
            headwise_q_basis=bool(args.headwise_q_basis),
            compute_dtype=rotation_math_dtype,
        )
        q_out = load_q_matrix(
            eig_payload,
            bridge_out_idx,
            out_norm,
            d_old,
            d_new,
            work_device,
            num_heads=orig_num_heads,
            headwise_q_basis=bool(args.headwise_q_basis),
            compute_dtype=rotation_math_dtype,
        )

    if identity_rotation_mode:
        # Full-width sanity gate must decode with the same basis used by input projection.
        q_out = q_in
        if not bool(args.global_q_bridge_out):
            print("[INFO] Full-width identity mode: forcing bridge-out basis to match bridge-in basis")

    q_in = q_in.to(device=work_device, dtype=rotation_math_dtype)
    q_out = q_out.to(device=work_device, dtype=rotation_math_dtype)

    if not hasattr(tgt_model.transformer, "input_proj") or not hasattr(tgt_model.transformer, "output_proj"):
        raise RuntimeError(
            "Target transformer is missing input/output projection bridges. "
            "Enable force_temporal_projected in LMModel config for full-width sanity mode."
        )

    with torch.no_grad():
        tgt_model.transformer.input_proj.weight.copy_(q_in.T.detach().cpu().to(tgt_model.transformer.input_proj.weight.dtype))
        if identity_rotation_mode:
            # Linear uses y = x @ W^T, so decode with W=Q to realize y = z @ Q^T.
            tgt_model.transformer.output_proj.weight.copy_(
                q_out.detach().cpu().to(tgt_model.transformer.output_proj.weight.dtype)
            )
            print("[INFO] Full-width identity mode: output bridge set analytically to Q_final")
        else:
            tgt_model.transformer.output_proj.weight.copy_(
                q_out.detach().cpu().to(tgt_model.transformer.output_proj.weight.dtype)
            )

    src_layers = src_model.transformer.layers
    tgt_layers = tgt_model.transformer.layers
    if len(src_layers) != len(tgt_layers):
        raise RuntimeError(f"Layer count mismatch: src={len(src_layers)} tgt={len(tgt_layers)}")

    print(f"[INFO] Compressing temporal layers: {len(src_layers)}")
    for i in range(len(src_layers)):
        if args.single_q_basis:
            if q_global is not None:
                q1 = q_global
                q2 = q_global
            else:
                q_layer = load_q_matrix(
                    eig_payload,
                    i,
                    args.basis_source,
                    d_old,
                    d_new,
                    work_device,
                    num_heads=orig_num_heads,
                    headwise_q_basis=bool(args.headwise_q_basis),
                    compute_dtype=rotation_math_dtype,
                )
                q1 = q_layer
                q2 = q_layer
        else:
            q1 = load_q_matrix(
                eig_payload,
                i,
                "norm1",
                d_old,
                d_new,
                work_device,
                num_heads=orig_num_heads,
                headwise_q_basis=bool(args.headwise_q_basis),
                compute_dtype=rotation_math_dtype,
            )
            q2 = load_q_matrix(
                eig_payload,
                i,
                "norm2",
                d_old,
                d_new,
                work_device,
                num_heads=orig_num_heads,
                headwise_q_basis=bool(args.headwise_q_basis),
                compute_dtype=rotation_math_dtype,
            )

        src_layer = src_layers[i]
        dst_layer = tgt_layers[i]
        norm_dim_scale = (float(d_old) / float(d_new)) ** 0.5

        with torch.no_grad():
            if args.absorb_rms_alpha:
                alpha1 = get_rms_alpha_or_ones(
                    src_layer.norm1,
                    d_old,
                    work_device,
                    rotation_math_dtype,
                )
                alpha2 = get_rms_alpha_or_ones(
                    src_layer.norm2,
                    d_old,
                    work_device,
                    rotation_math_dtype,
                )
                set_norm_alpha_one_if_present(dst_layer.norm1)
                set_norm_alpha_one_if_present(dst_layer.norm2)
                alpha_scale_for_weights = norm_dim_scale if args.rms_dim_compensation else 1.0
            else:
                alpha1 = torch.ones((d_old,), device=work_device, dtype=rotation_math_dtype)
                alpha2 = torch.ones((d_old,), device=work_device, dtype=rotation_math_dtype)
                alpha_scale_for_weights = 1.0
                alpha_scale_for_norm = (float(d_new) / float(d_old)) ** 0.5 if args.rms_dim_compensation else 1.0
                if hasattr(src_layer.norm1, "alpha") and hasattr(dst_layer.norm1, "alpha"):
                    a1 = compress_norm_alpha(src_layer.norm1.alpha, q1)
                    dst_layer.norm1.alpha.copy_((a1 * alpha_scale_for_norm).cpu().to(dst_layer.norm1.alpha.dtype))
                if hasattr(src_layer.norm2, "alpha") and hasattr(dst_layer.norm2, "alpha"):
                    a2 = compress_norm_alpha(src_layer.norm2.alpha, q2)
                    dst_layer.norm2.alpha.copy_((a2 * alpha_scale_for_norm).cpu().to(dst_layer.norm2.alpha.dtype))

            # Attention in-proj: 3 blocks of [D, D] -> [3*d_new, d_new].
            src_in_proj = src_layer.self_attn.in_proj_weight.detach().to(
                device=work_device,
                dtype=rotation_math_dtype,
            )
            src_in_blocks = src_in_proj.view(3, d_old, d_old)

            # Attention out-proj: [D, D] -> [d_new, d_new].
            src_out_proj = src_layer.self_attn.out_proj.weight.detach().to(
                device=work_device,
                dtype=rotation_math_dtype,
            )

            dst_in_proj, dst_out_proj = rotate_attention_with_basis(
                src_in_blocks,
                src_out_proj,
                q1,
                alpha1,
                alpha_scale_for_weights,
                rope_safe_quarot=bool(use_rope_safe_quarot),
                rope_safe_v_mode=rope_safe_v_mode,
            )

            if i == 0 and bool(use_rope_safe_quarot):
                expected_in = tuple(dst_layer.self_attn.in_proj_weight.shape)
                expected_out = tuple(dst_layer.self_attn.out_proj.weight.shape)
                got_in = tuple(dst_in_proj.shape)
                got_out = tuple(dst_out_proj.shape)
                print(
                    "[INFO] RoPE-safe attention shape sanity (layer 0): "
                    f"in_proj got={got_in} expected={expected_in} "
                    f"out_proj got={got_out} expected={expected_out}"
                )
                if got_in != expected_in or got_out != expected_out:
                    raise RuntimeError(
                        "RoPE-safe attention shape mismatch at layer 0: "
                        f"in got={got_in} expected={expected_in}; "
                        f"out got={got_out} expected={expected_out}"
                    )

            dst_layer.self_attn.in_proj_weight.copy_(
                dst_in_proj.cpu().to(dst_layer.self_attn.in_proj_weight.dtype)
            )
            dst_layer.self_attn.out_proj.weight.copy_(
                dst_out_proj.cpu().to(dst_layer.self_attn.out_proj.weight.dtype)
            )

            # Gating linear_in: [2H, D] -> [2H, d_new] (preserve hidden width when possible).
            src_lin_in = src_layer.gating.linear_in.weight.detach().to(
                device=work_device,
                dtype=rotation_math_dtype,
            )
            src_lin_in = src_lin_in * alpha2.unsqueeze(0) * alpha_scale_for_weights
            dst_lin_in = src_lin_in @ q2
            dst_lin_in = pad_or_trim_rows(dst_lin_in, dst_layer.gating.linear_in.weight.shape[0])
            dst_layer.gating.linear_in.weight.copy_(
                dst_lin_in.cpu().to(dst_layer.gating.linear_in.weight.dtype)
            )

            # Gating linear_out: [D, H] -> [d_new, H].
            src_lin_out = src_layer.gating.linear_out.weight.detach().to(
                device=work_device,
                dtype=rotation_math_dtype,
            )
            dst_lin_out = q2.T @ src_lin_out
            dst_lin_out = pad_or_trim_cols(dst_lin_out, dst_layer.gating.linear_out.weight.shape[1])
            dst_layer.gating.linear_out.weight.copy_(
                dst_lin_out.cpu().to(dst_layer.gating.linear_out.weight.dtype)
            )

    fit_steps_effective = int(args.fit_steps)

    if args.fit_output_proj_ls:
        if identity_rotation_mode:
            print("[INFO] Skipping output-proj LS fit: d_new equals d_old (identity rotation mode)")
        else:
            if bool(args.high_density_ls) and args.fit_token_source != "real":
                raise ValueError("--high-density-ls requires --fit-token-source real")

            if bool(args.high_density_ls):
                target_steps = int(args.fit_min_keep_rows) + 256
                if fit_steps_effective < target_steps:
                    print(
                        "[INFO] High-density LS raised fit-steps to satisfy keep-row target: "
                        f"requested={fit_steps_effective} effective={target_steps}"
                    )
                    fit_steps_effective = target_steps

            fit_forced_tokens = None
            if args.fit_token_source == "real":
                fit_forced_tokens = build_real_forced_tokens(
                    src_model,
                    root=root,
                    steps=int(fit_steps_effective),
                    batch_size=int(args.fit_batch_size),
                    extract_device=work_device,
                    input_wav=args.fit_input_wav,
                    voice_prompt_wav=args.fit_voice_prompt_wav,
                    text_prompt=args.fit_text_prompt,
                    mimi_weight=args.fit_mimi_weight,
                    tokenizer_path=args.fit_tokenizer,
                    voice_ratio=float(args.fit_voice_ratio),
                    high_density_ls=bool(args.high_density_ls),
                    dataset_dir=args.fit_dataset_dir,
                    min_keep_rows=int(args.fit_min_keep_rows),
                    max_audio_files=int(args.fit_max_audio_files),
                    allow_audio_reuse=bool(args.fit_allow_audio_reuse),
                )

            gain_mode = "none"
            if bool(args.fit_gain_calibration):
                gain_mode = "per_channel" if (bool(args.fit_per_channel_gain) or bool(args.high_density_ls)) else "scalar"

            print(
                "[INFO] Calibrating output projection (bridge) with least squares: "
                f"steps={fit_steps_effective} batch={args.fit_batch_size} "
                f"seed={args.fit_seed} ridge={args.fit_ridge} token_source={args.fit_token_source} "
                f"mask_audio_pad={args.fit_mask_audio_pad} center={args.fit_center_solve} "
                f"gain={args.fit_gain_calibration} gain_mode={gain_mode} bias={args.fit_bias_correction} "
                f"high_density={args.high_density_ls}"
            )
            fit_output_projection_least_squares(
                src_model,
                tgt_model,
                steps=int(fit_steps_effective),
                batch_size=int(args.fit_batch_size),
                seed=int(args.fit_seed),
                device=work_device,
                ridge=float(args.fit_ridge),
                forced_tokens=fit_forced_tokens,
                mask_audio_pad=bool(args.fit_mask_audio_pad),
                audio_pad_token=int(src_model.card),
                center_before_solve=bool(args.fit_center_solve),
                calibrate_gain=bool(args.fit_gain_calibration),
                gain_mode=gain_mode,
                gain_clamp_max=float(args.fit_gain_clamp_max),
                apply_bias_correction=bool(args.fit_bias_correction),
                rank1_mean_shift=bool(args.fit_rank1_mean_shift),
            )

    export_sd = {}
    for key, value in tgt_model.state_dict().items():
        t = value.detach().cpu().contiguous()
        if t.is_floating_point():
            t = t.to(dtype=torch_dtype)
        export_sd[key] = t

    payload = {
        "state_dict": export_sd,
        "config_override": cfg_override,
        "model_mode": "slicegpt_dense",
        "force_dense": True,
        "slicegpt_meta": {
            "source_checkpoint": str(bf16_path),
            "eigenvectors": str(eigen_path),
            "d_old": d_old,
            "d_new": d_new,
            "num_heads": inner_num_heads,
            "orig_num_heads": orig_num_heads,
            "orig_head_dim": orig_head_dim,
            "inner_head_dim": d_new // inner_num_heads,
            "num_layers": num_layers,
            "temporal_dim_feedforward": temporal_dim_feedforward,
            "bridge_in_layer": args.bridge_in_layer,
            "bridge_out_layer": args.bridge_out_layer,
            "single_q_basis": bool(args.single_q_basis),
            "basis_source": args.basis_source,
            "absorb_rms_alpha": bool(args.absorb_rms_alpha),
            "global_q_basis": bool(args.global_q_basis),
            "global_q_layer": global_q_layer,
            "global_q_norm": global_q_norm,
            "headwise_q_basis": bool(args.headwise_q_basis),
            "attn_rope_mode": attn_rope_mode,
            "attn_rope_quarot_active": bool(use_rope_safe_quarot),
            "rope_safe_v_mode": rope_safe_v_mode,
            "rotation_math_dtype": str(rotation_math_dtype).replace("torch.", ""),
            "global_q_bridge_out": bool(args.global_q_bridge_out),
            "rms_dim_compensation": bool(args.rms_dim_compensation),
            "preserve_head_dim": bool(args.preserve_head_dim),
            "inner_num_heads": inner_num_heads,
            "fit_output_proj_ls": bool(args.fit_output_proj_ls),
            "fit_steps": int(fit_steps_effective),
            "fit_steps_requested": int(args.fit_steps),
            "fit_batch_size": int(args.fit_batch_size),
            "fit_seed": int(args.fit_seed),
            "fit_ridge": float(args.fit_ridge),
            "fit_token_source": args.fit_token_source,
            "fit_input_wav": args.fit_input_wav,
            "fit_voice_prompt_wav": args.fit_voice_prompt_wav,
            "fit_text_prompt": args.fit_text_prompt,
            "fit_mimi_weight": args.fit_mimi_weight,
            "fit_tokenizer": args.fit_tokenizer,
            "fit_voice_ratio": float(args.fit_voice_ratio),
            "fit_mask_audio_pad": bool(args.fit_mask_audio_pad),
            "fit_center_solve": bool(args.fit_center_solve),
            "fit_gain_calibration": bool(args.fit_gain_calibration),
            "fit_per_channel_gain": bool(args.fit_per_channel_gain),
            "fit_gain_clamp_max": float(args.fit_gain_clamp_max),
            "fit_bias_correction": bool(args.fit_bias_correction),
            "fit_rank1_mean_shift": bool(args.fit_rank1_mean_shift),
            "high_density_ls": bool(args.high_density_ls),
            "fit_dataset_dir": args.fit_dataset_dir,
            "fit_min_keep_rows": int(args.fit_min_keep_rows),
            "fit_max_audio_files": int(args.fit_max_audio_files),
            "fit_allow_audio_reuse": bool(args.fit_allow_audio_reuse),
            "identity_rotation_mode": bool(identity_rotation_mode),
        },
    }

    src_temporal = {
        k: v.detach().cpu() for k, v in src_sd.items() if k.startswith("transformer.") and torch.is_tensor(v)
    }
    dst_temporal = {
        k: v for k, v in export_sd.items() if k.startswith("transformer.") and torch.is_tensor(v)
    }

    src_temporal_gb = tensor_nbytes_gb(src_temporal)
    dst_temporal_gb = tensor_nbytes_gb(dst_temporal)

    torch.save(payload, out_path)

    print(f"[INFO] Saved SliceGPT checkpoint: {out_path}")
    print(f"[INFO] Temporal bytes (approx): src={src_temporal_gb:.3f} GB -> dst={dst_temporal_gb:.3f} GB")
    if src_temporal_gb > 0:
        ratio = dst_temporal_gb / src_temporal_gb
        delta_pct = (ratio - 1.0) * 100.0
        if ratio <= 1.0:
            print(f"[INFO] Temporal reduction: {(1.0 - ratio) * 100.0:.2f}%")
        else:
            print(f"[INFO] Temporal growth: {delta_pct:.2f}%")


if __name__ == "__main__":
    main()
