import argparse
import gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sentencepiece as spm
except Exception:
    spm = None

from moshi import offline
from moshi.models import loaders
from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn


def parse_dtype(name: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError(f"Invalid dtype: {name}")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_local_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


@torch.no_grad()
def _encode_audio_frames_with_mimi(mimi, wav_path: Path, max_steps: int) -> list[torch.Tensor]:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    sample_pcm = load_audio(str(wav_path), mimi.sample_rate)
    samples = _iterate_audio(sample_pcm, frame_size, max_len=max_steps, pad=False)
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
def build_forced_tokens(
    model,
    steps: int,
    device: str,
    *,
    root: Path | None = None,
    input_wav: str = "tellmeajoke_padded.wav",
    voice_prompt_wav: str = "bmo_621.wav",
    text_prompt: str = "Tell me a joke.",
    mimi_weight: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
    tokenizer_path: str = "tokenizer_spm_32k_3.model",
    voice_ratio: float = 0.25,
) -> torch.Tensor:
    if spm is None:
        raise RuntimeError(
            "sentencepiece is required for real-token drift evaluation. "
            "Install sentencepiece to build the Mimi/tokenizer prompt stream."
        )

    root_path = Path(__file__).resolve().parent if root is None else root
    input_wav_path = resolve_local_path(root_path, input_wav)
    voice_prompt_wav = str(voice_prompt_wav).strip()
    voice_wav_path = resolve_local_path(root_path, voice_prompt_wav) if voice_prompt_wav else None
    mimi_path = resolve_local_path(root_path, mimi_weight)
    tok_path = resolve_local_path(root_path, tokenizer_path)

    required_assets = [input_wav_path, mimi_path, tok_path]
    if voice_wav_path is not None:
        required_assets.append(voice_wav_path)

    for req in required_assets:
        if not req.exists():
            raise FileNotFoundError(f"Required calibration asset not found: {req}")

    print(
        "[INFO] Building in-distribution forced tokens: "
        f"voice_prompt={(voice_wav_path.name if voice_wav_path is not None else 'disabled')} "
        f"input_wav={input_wav_path.name}"
    )

    mimi = loaders.get_mimi(str(mimi_path), device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    tokenizer = spm.SentencePieceProcessor(str(tok_path))

    voice_steps = 0
    if voice_wav_path is not None and int(steps) > 1:
        voice_steps = int(round(float(voice_ratio) * float(steps)))
        voice_steps = max(1, min(voice_steps, int(steps) - 1))
    input_steps = max(1, int(steps) - voice_steps)

    audio_frames = []
    if voice_steps > 0 and voice_wav_path is not None:
        audio_frames.extend(_encode_audio_frames_with_mimi(mimi, voice_wav_path, voice_steps))
    audio_frames.extend(_encode_audio_frames_with_mimi(mimi, input_wav_path, input_steps))

    if len(audio_frames) < int(steps):
        refill_paths = [input_wav_path]
        if voice_wav_path is not None:
            refill_paths.insert(0, voice_wav_path)

        refill_idx = 0
        while len(audio_frames) < int(steps):
            remaining = int(steps) - len(audio_frames)
            wav_path = refill_paths[refill_idx % len(refill_paths)]
            refill_idx += 1
            encoded = _encode_audio_frames_with_mimi(mimi, wav_path, remaining)
            if not encoded:
                break
            audio_frames.extend(encoded)

    del mimi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    k = int(model.num_codebooks)
    audio_pad = int(model.card)
    text_pad = 0

    forced = torch.full((steps, k), audio_pad, dtype=torch.long)
    forced[:, 0] = text_pad

    text_ids = tokenizer.encode(offline.wrap_with_system_tags(text_prompt))
    for t in range(min(int(steps), len(text_ids))):
        forced[t, 0] = int(text_ids[t])

    for t in range(min(int(steps), len(audio_frames))):
        audio_codes = audio_frames[t]
        n_audio = min(k - 1, int(audio_codes.numel()))
        forced[t, 1 : 1 + n_audio] = audio_codes[:n_audio]

    audio_tokens = forced[:, 1:]
    pad_ratio = float((audio_tokens == audio_pad).float().mean().item()) if audio_tokens.numel() > 0 else 1.0
    nonpad_text = int((forced[:, 0] != text_pad).sum().item())
    print(
        "[INFO] Forced stream stats: "
        f"steps={steps} text_nonpad={nonpad_text} audio_pad_ratio={pad_ratio:.3f}"
    )
    return forced


def register_post_bridge_hook(model):
    cache = []

    out_norm = getattr(model, "out_norm", None)
    if out_norm is None:
        return cache, None

    def pre_hook(_module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            cache.append(x.detach().float().reshape(-1, x.shape[-1]))

    handle = out_norm.register_forward_pre_hook(pre_hook)
    return cache, handle


def _unwrap_tensor_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    return None


def _capture_stage_tensor(cache: dict[str, torch.Tensor], key: str, output):
    y = _unwrap_tensor_output(output)
    if torch.is_tensor(y):
        cache[key] = y.detach().float().cpu().contiguous()


def get_temporal_layers(model) -> list[nn.Module]:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return []

    inner = getattr(transformer, "inner", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)

    layers = getattr(transformer, "layers", None)
    if layers is not None:
        return list(layers)

    return []


def get_student_unrotation_matrix(model) -> torch.Tensor | None:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return None

    input_proj = getattr(transformer, "input_proj", None)
    if input_proj is None or not hasattr(input_proj, "weight"):
        return None

    # For row-vector convention and Linear forward y = x @ W^T:
    # z = x @ Q where W = Q^T, so x = z @ W.
    return input_proj.weight.detach().float().cpu().contiguous()


class LoRAAdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int, alpha: float):
        super().__init__()
        self.base = base_layer

        for p in self.base.parameters():
            p.requires_grad = False

        if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
            in_features = int(base_layer.in_features)
            out_features = int(base_layer.out_features)
        else:
            w = base_layer.weight
            out_features, in_features = int(w.shape[0]), int(w.shape[1])

        self.r = int(r)
        self.scaling = float(alpha) / float(r)

        base_param = next(self.base.parameters(), None)
        dev = base_param.device if base_param is not None else torch.device("cpu")
        dt = torch.bfloat16

        self.lora_A = nn.Parameter(torch.zeros(self.r, in_features, dtype=dt, device=dev), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r, dtype=dt, device=dev), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lora = x if x.dtype == self.lora_A.dtype else x.to(self.lora_A.dtype)
        base_out = self.base(x_lora)
        delta = (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling
        if delta.dtype != base_out.dtype:
            delta = delta.to(base_out.dtype)
        return base_out + delta


def _get_parent_and_attr(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_ckpt(model: nn.Module, lora_ckpt_path: Path):
    ckpt = torch.load(str(lora_ckpt_path), map_location="cpu")
    rank = int(ckpt["rank"])
    alpha = float(ckpt["alpha"])
    wrapped_modules = list(ckpt["wrapped_modules"])
    lora_state = ckpt["lora_state_dict"]

    # Wrap target modules
    for mod_name in wrapped_modules:
        parent, attr = _get_parent_and_attr(model, mod_name)
        base_layer = getattr(parent, attr)
        setattr(parent, attr, LoRAAdapterLinear(base_layer, r=rank, alpha=alpha))

    # Load LoRA weights
    missing = []
    for mod_name in wrapped_modules:
        mod = dict(model.named_modules()).get(mod_name, None)
        if mod is None or not isinstance(mod, LoRAAdapterLinear):
            missing.append(mod_name)
            continue

        key_a = f"{mod_name}.lora_A"
        key_b = f"{mod_name}.lora_B"
        if key_a not in lora_state or key_b not in lora_state:
            missing.append(mod_name)
            continue

        with torch.no_grad():
            mod.lora_A.copy_(lora_state[key_a].to(device=mod.lora_A.device, dtype=mod.lora_A.dtype))
            mod.lora_B.copy_(lora_state[key_b].to(device=mod.lora_B.device, dtype=mod.lora_B.dtype))

    if missing:
        print(f"[WARN] LoRA missing/incomplete for {len(missing)} modules.")
    print(f"[INFO] Applied LoRA checkpoint: {lora_ckpt_path.name} (wrapped={len(wrapped_modules)})")


@torch.no_grad()
def run_pair_rollout(
    bf16_ckpt: str,
    int4_ckpt: str,
    *,
    steps: int,
    seed: int,
    device: str,
    teacher_forced: bool,
    lora_ckpt: str | None,
    input_wav: str,
    voice_prompt_wav: str,
    text_prompt: str,
    mimi_weight: str,
    tokenizer_path: str,
    voice_ratio: float,
    teacher_dtype: str,
    student_dtype: str,
    layer_ladder: bool,
    layer_ladder_step: int,
    layer_ladder_unrotate_student: bool,
    layer_stage_probe: bool,
):
    teacher_dt = parse_dtype(teacher_dtype)
    student_dt = None if str(student_dtype).strip().lower() == "auto" else parse_dtype(student_dtype)

    print(f"\n[RUN] Loading model: {bf16_ckpt}")
    teacher = loaders.get_moshi_lm(
        bf16_ckpt,
        device=device,
        dtype=teacher_dt,
        cpu_offload=False,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"\n[RUN] Loading model: {int4_ckpt}")
    student = loaders.get_moshi_lm(
        int4_ckpt,
        device=device,
        dtype=student_dt,
        cpu_offload=False,
    )
    if lora_ckpt is not None:
        apply_lora_ckpt(student, Path(lora_ckpt).resolve())
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    print(
        "[INFO] Eval dtypes: "
        f"teacher={str(teacher_dt).replace('torch.', '')} "
        f"student={str(student_dt).replace('torch.', '') if student_dt is not None else 'auto'}"
    )

    if int(teacher.num_codebooks) != int(student.num_codebooks):
        raise RuntimeError(
            f"num_codebooks mismatch: teacher={teacher.num_codebooks} student={student.num_codebooks}"
        )

    forced_tokens = build_forced_tokens(
        teacher,
        int(steps),
        device,
        input_wav=input_wav,
        voice_prompt_wav=voice_prompt_wav,
        text_prompt=text_prompt,
        mimi_weight=mimi_weight,
        tokenizer_path=tokenizer_path,
        voice_ratio=float(voice_ratio),
    )
    k = int(teacher.num_codebooks)

    t_bridge_cache, t_bridge_handle = register_post_bridge_hook(teacher)
    s_bridge_cache, s_bridge_handle = register_post_bridge_hook(student)

    layer_ladder_rows = []
    layer_stage_rows = []
    use_layer_ladder = bool(layer_ladder)
    ladder_step = int(layer_ladder_step)
    ladder_layers = 0
    teacher_layers = []
    student_layers = []
    student_unrot = None

    if use_layer_ladder:
        teacher_layers = get_temporal_layers(teacher)
        student_layers = get_temporal_layers(student)
        if not teacher_layers or not student_layers:
            print(
                "[WARN] Layer ladder requested but temporal layers are unavailable; disabling ladder probe"
            )
            use_layer_ladder = False
        else:
            ladder_layers = min(len(teacher_layers), len(student_layers))
            if len(teacher_layers) != len(student_layers):
                print(
                    "[WARN] Layer ladder layer-count mismatch: "
                    f"teacher={len(teacher_layers)} student={len(student_layers)} using={ladder_layers}"
                )

            if bool(layer_ladder_unrotate_student):
                student_unrot = get_student_unrotation_matrix(student)
                if student_unrot is None:
                    print(
                        "[WARN] Layer ladder unrotation requested but student.input_proj is unavailable"
                    )
                else:
                    print(
                        "[INFO] Layer ladder unrotation matrix: "
                        f"shape={tuple(student_unrot.shape)}"
                    )

            print(
                "[INFO] Layer ladder enabled: "
                f"step={ladder_step} layers={ladder_layers} "
                f"unrotate_student={bool(layer_ladder_unrotate_student and student_unrot is not None)}"
            )

    student_prev_text = int(forced_tokens[0, 0].item())
    per_step = []

    try:
        with teacher.streaming(batch_size=1), student.streaming(batch_size=1):
            for t in range(int(steps)):
                teacher_codes = forced_tokens[t].clone()
                student_codes = forced_tokens[t].clone()
                if not bool(teacher_forced) and t > 0:
                    student_codes[0] = int(student_prev_text)

                teacher_in_text = int(teacher_codes[0].item())
                student_in_text = int(student_codes[0].item())

                teacher_seq = teacher_codes.view(1, k, 1).to(device)
                student_seq = student_codes.view(1, k, 1).to(device)

                teacher_layer_cache = None
                student_layer_cache = None
                layer_handles = []
                capture_layer_step = use_layer_ladder and int(t) == ladder_step and ladder_layers > 0
                if capture_layer_step:
                    teacher_layer_cache = [None for _ in range(ladder_layers)]
                    student_layer_cache = [None for _ in range(ladder_layers)]
                    teacher_stage_cache = [dict() for _ in range(ladder_layers)]
                    student_stage_cache = [dict() for _ in range(ladder_layers)]

                    def make_hook(cache_ref, idx):
                        def hook(_module, _inputs, output):
                            y = _unwrap_tensor_output(output)
                            if torch.is_tensor(y):
                                cache_ref[idx] = y.detach().float().cpu().contiguous()

                        return hook

                    def make_stage_hook(cache_ref, idx, stage_key):
                        def hook(_module, _inputs, output):
                            _capture_stage_tensor(cache_ref[idx], stage_key, output)

                        return hook

                    for idx in range(ladder_layers):
                        teacher_layer = teacher_layers[idx]
                        student_layer = student_layers[idx]

                        layer_handles.append(teacher_layer.register_forward_hook(make_hook(teacher_layer_cache, idx)))
                        layer_handles.append(student_layer.register_forward_hook(make_hook(student_layer_cache, idx)))

                        if bool(layer_stage_probe):
                            if hasattr(teacher_layer, "norm1") and hasattr(student_layer, "norm1"):
                                layer_handles.append(
                                    teacher_layer.norm1.register_forward_hook(
                                        make_stage_hook(teacher_stage_cache, idx, "norm1")
                                    )
                                )
                                layer_handles.append(
                                    student_layer.norm1.register_forward_hook(
                                        make_stage_hook(student_stage_cache, idx, "norm1")
                                    )
                                )

                            if hasattr(teacher_layer, "self_attn") and hasattr(student_layer, "self_attn"):
                                layer_handles.append(
                                    teacher_layer.self_attn.register_forward_hook(
                                        make_stage_hook(teacher_stage_cache, idx, "self_attn")
                                    )
                                )
                                layer_handles.append(
                                    student_layer.self_attn.register_forward_hook(
                                        make_stage_hook(student_stage_cache, idx, "self_attn")
                                    )
                                )

                            if hasattr(teacher_layer, "norm2") and hasattr(student_layer, "norm2"):
                                layer_handles.append(
                                    teacher_layer.norm2.register_forward_hook(
                                        make_stage_hook(teacher_stage_cache, idx, "norm2")
                                    )
                                )
                                layer_handles.append(
                                    student_layer.norm2.register_forward_hook(
                                        make_stage_hook(student_stage_cache, idx, "norm2")
                                    )
                                )

                            teacher_gating = getattr(teacher_layer, "gating", None)
                            student_gating = getattr(student_layer, "gating", None)
                            if isinstance(teacher_gating, nn.Module) and isinstance(student_gating, nn.Module):
                                if not isinstance(teacher_gating, nn.ModuleList) and not isinstance(student_gating, nn.ModuleList):
                                    layer_handles.append(
                                        teacher_gating.register_forward_hook(
                                            make_stage_hook(teacher_stage_cache, idx, "gating")
                                        )
                                    )
                                    layer_handles.append(
                                        student_gating.register_forward_hook(
                                            make_stage_hook(student_stage_cache, idx, "gating")
                                        )
                                    )

                try:
                    _, teacher_text_logits = teacher.forward_codes(teacher_seq)
                    _, student_text_logits = student.forward_codes(student_seq)
                finally:
                    for handle in layer_handles:
                        handle.remove()

                if capture_layer_step and teacher_layer_cache is not None and student_layer_cache is not None:
                    for idx in range(ladder_layers):
                        teacher_act = teacher_layer_cache[idx]
                        student_act = student_layer_cache[idx]
                        if teacher_act is None or student_act is None:
                            continue

                        teacher_flat = teacher_act.reshape(-1, teacher_act.shape[-1])
                        student_flat = student_act.reshape(-1, student_act.shape[-1])

                        if bool(layer_ladder_unrotate_student) and student_unrot is not None:
                            if int(student_flat.shape[1]) == int(student_unrot.shape[0]):
                                student_cmp = student_flat @ student_unrot
                            else:
                                student_cmp = student_flat
                        else:
                            student_cmp = student_flat

                        rows = min(int(teacher_flat.shape[0]), int(student_cmp.shape[0]))
                        dims = min(int(teacher_flat.shape[1]), int(student_cmp.shape[1]))
                        if rows <= 0 or dims <= 0:
                            continue

                        teacher_vec = teacher_flat[:rows, :dims].reshape(-1)
                        student_vec = student_cmp[:rows, :dims].reshape(-1)
                        diff = teacher_vec - student_vec

                        layer_ladder_rows.append(
                            {
                                "step": int(t),
                                "layer": int(idx),
                                "cos": float(F.cosine_similarity(teacher_vec, student_vec, dim=0).item()),
                                "max_abs": float(diff.abs().max().item()),
                                "mse": float(F.mse_loss(teacher_vec, student_vec).item()),
                            }
                        )

                        if bool(layer_stage_probe):
                            teacher_stages = teacher_stage_cache[idx]
                            student_stages = student_stage_cache[idx]
                            for stage_name in ("norm1", "self_attn", "norm2", "gating"):
                                if stage_name not in teacher_stages or stage_name not in student_stages:
                                    continue

                                t_stage = teacher_stages[stage_name].reshape(-1, teacher_stages[stage_name].shape[-1])
                                s_stage = student_stages[stage_name].reshape(-1, student_stages[stage_name].shape[-1])

                                if bool(layer_ladder_unrotate_student) and student_unrot is not None:
                                    if int(s_stage.shape[1]) == int(student_unrot.shape[0]):
                                        s_stage_cmp = s_stage @ student_unrot
                                    else:
                                        s_stage_cmp = s_stage
                                else:
                                    s_stage_cmp = s_stage

                                stage_rows = min(int(t_stage.shape[0]), int(s_stage_cmp.shape[0]))
                                stage_dims = min(int(t_stage.shape[1]), int(s_stage_cmp.shape[1]))
                                if stage_rows <= 0 or stage_dims <= 0:
                                    continue

                                t_stage_vec = t_stage[:stage_rows, :stage_dims].reshape(-1)
                                s_stage_vec = s_stage_cmp[:stage_rows, :stage_dims].reshape(-1)
                                stage_diff = t_stage_vec - s_stage_vec

                                layer_stage_rows.append(
                                    {
                                        "step": int(t),
                                        "layer": int(idx),
                                        "stage": stage_name,
                                        "cos": float(F.cosine_similarity(t_stage_vec, s_stage_vec, dim=0).item()),
                                        "max_abs": float(stage_diff.abs().max().item()),
                                        "mse": float(F.mse_loss(t_stage_vec, s_stage_vec).item()),
                                    }
                                )

                teacher_logits = teacher_text_logits[:, 0, 0, :].float().cpu().contiguous().view(-1)
                student_logits = student_text_logits[:, 0, 0, :].float().cpu().contiguous().view(-1)

                teacher_top1 = int(torch.argmax(teacher_logits, dim=-1).item())
                student_top1 = int(torch.argmax(student_logits, dim=-1).item())
                student_prev_text = student_top1

                if not t_bridge_cache or not s_bridge_cache:
                    raise RuntimeError("Post-bridge caches are empty; out_norm hooks did not capture activations")

                teacher_post = t_bridge_cache.pop().cpu().contiguous()
                student_post = s_bridge_cache.pop().cpu().contiguous()
                n = min(int(teacher_post.shape[0]), int(student_post.shape[0]))
                if n <= 0:
                    raise RuntimeError("Invalid post-bridge activation shape captured by hooks")

                teacher_vec = teacher_post[:n].reshape(-1)
                student_vec = student_post[:n].reshape(-1)

                post_diff = teacher_vec - student_vec
                logits_diff = teacher_logits - student_logits

                cos = F.cosine_similarity(teacher_vec, student_vec, dim=0).item()
                mse = F.mse_loss(teacher_vec, student_vec).item()
                max_abs_post_bridge = torch.max(torch.abs(post_diff)).item()
                max_abs_logits = torch.max(torch.abs(logits_diff)).item()

                kl_div = F.kl_div(
                    F.log_softmax(student_logits.unsqueeze(0), dim=-1),
                    F.softmax(teacher_logits.unsqueeze(0), dim=-1),
                    reduction="batchmean",
                ).item()

                topk = 5
                t_topk = set(torch.topk(teacher_logits, k=topk).indices.tolist())
                s_topk = set(torch.topk(student_logits, k=topk).indices.tolist())
                overlap_pct = (100.0 * len(t_topk.intersection(s_topk))) / float(topk)

                per_step.append(
                    {
                        "step": t,
                        "cos": cos,
                        "mse": mse,
                        "max_abs_post_bridge": max_abs_post_bridge,
                        "max_abs_logits": max_abs_logits,
                        "kl_div": kl_div,
                        "top5_overlap_pct": overlap_pct,
                        "top1_same": int(teacher_top1 == student_top1),
                        "bf16_top1": teacher_top1,
                        "int4_top1": student_top1,
                        "teacher_in_text": teacher_in_text,
                        "student_in_text": student_in_text,
                    }
                )
    finally:
        if t_bridge_handle is not None:
            t_bridge_handle.remove()
        if s_bridge_handle is not None:
            s_bridge_handle.remove()

        del teacher
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return per_step, layer_ladder_rows, layer_stage_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", "--bf16-ckpt", dest="bf16", default="v5_step1500.safetensors")
    parser.add_argument("--int4", "--int4-ckpt", dest="int4", default="bmo_mixed_precision.pt")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--catastrophic-cos", type=float, default=0.95)
    parser.add_argument("--catastrophic-mse", type=float, default=0.1)
    parser.add_argument("--lora-ckpt", type=str, default=None, help="Optional LoRA checkpoint (train_lqec.py output)")
    parser.add_argument("--report-step", type=int, default=63, help="Print detailed metrics for this rollout step")
    parser.add_argument(
        "--runtime-patch",
        type=parse_bool,
        default=True,
        help=(
            "If true, import test_rtx_edge to apply runtime monkey patches. "
            "If false, use the native loader/attention path for A/B diagnosis."
        ),
    )
    parser.add_argument(
        "--teacher-forced",
        type=parse_bool,
        default=False,
        help="If true, student uses teacher forced input tokens each step. If false, student self-feeds text token.",
    )
    parser.add_argument("--input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--voice-prompt-wav", default="bmo_621.wav")
    parser.add_argument("--text-prompt", default="Tell me a joke.")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--voice-ratio", type=float, default=0.25)
    parser.add_argument(
        "--teacher-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Loader dtype for teacher model.",
    )
    parser.add_argument(
        "--student-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Loader dtype for student model (auto keeps checkpoint/native default).",
    )
    parser.add_argument(
        "--layer-ladder",
        type=parse_bool,
        default=False,
        help="If true, capture per-layer cosine ladder at --layer-ladder-step.",
    )
    parser.add_argument(
        "--layer-ladder-step",
        type=int,
        default=0,
        help="Rollout step index for per-layer ladder capture.",
    )
    parser.add_argument(
        "--layer-ladder-unrotate-student",
        type=parse_bool,
        default=True,
        help="If true, map student inner-layer activations back via transformer.input_proj weight.",
    )
    parser.add_argument(
        "--layer-stage-probe",
        type=parse_bool,
        default=False,
        help="If true, capture stage-level ladders per layer (norm1/self_attn/norm2/gating).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if float(args.voice_ratio) < 0.0 or float(args.voice_ratio) >= 1.0:
        raise ValueError(f"voice-ratio must be in [0, 1), got {args.voice_ratio}")
    if bool(args.layer_ladder):
        if int(args.layer_ladder_step) < 0:
            raise ValueError(f"layer-ladder-step must be >= 0, got {args.layer_ladder_step}")
        if int(args.layer_ladder_step) >= int(args.steps):
            raise ValueError(
                f"layer-ladder-step must be < steps ({args.steps}), got {args.layer_ladder_step}"
            )

    if bool(args.runtime_patch):
        # Importing this module applies the same runtime monkey patches used in test runs.
        import test_rtx_edge  # noqa: F401
    else:
        print("[INFO] Runtime patch disabled: using native loader/attention path.")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this drift probe")

    per_step, layer_ladder_rows, layer_stage_rows = run_pair_rollout(
        args.bf16,
        args.int4,
        steps=int(args.steps),
        seed=int(args.seed),
        device=args.device,
        teacher_forced=bool(args.teacher_forced),
        lora_ckpt=args.lora_ckpt,
        input_wav=args.input_wav,
        voice_prompt_wav=args.voice_prompt_wav,
        text_prompt=args.text_prompt,
        mimi_weight=args.mimi_weight,
        tokenizer_path=args.tokenizer,
        voice_ratio=float(args.voice_ratio),
        teacher_dtype=args.teacher_dtype,
        student_dtype=args.student_dtype,
        layer_ladder=bool(args.layer_ladder),
        layer_ladder_step=int(args.layer_ladder_step),
        layer_ladder_unrotate_student=bool(args.layer_ladder_unrotate_student),
        layer_stage_probe=bool(args.layer_stage_probe),
    )

    mean_cos = sum(x["cos"] for x in per_step) / len(per_step)
    mean_mse = sum(x["mse"] for x in per_step) / len(per_step)
    mean_kl = sum(x["kl_div"] for x in per_step) / len(per_step)
    mean_top5 = sum(x["top5_overlap_pct"] for x in per_step) / len(per_step)

    worst_cos = min(per_step, key=lambda x: x["cos"])
    worst_mse = max(per_step, key=lambda x: x["mse"])
    top1_match_rate = sum(x["top1_same"] for x in per_step) / len(per_step)

    catastrophic_steps = [
        x for x in per_step if x["cos"] < args.catastrophic_cos or x["mse"] > args.catastrophic_mse
    ]

    if 0 <= args.report_step < len(per_step):
        row = per_step[args.report_step]
        print("\n=== REPORT STEP ===")
        print(
            f"step={row['step']:02d} cos={row['cos']:.6f} mse={row['mse']:.6f} "
            f"max_abs_post_bridge={row['max_abs_post_bridge']:.6f} "
            f"max_abs_logits={row['max_abs_logits']:.6f} "
            f"kl_div={row['kl_div']:.6f} top5_overlap_pct={row['top5_overlap_pct']:.2f} "
            f"top1_same={row['top1_same']} bf16_top1={row['bf16_top1']} int4_top1={row['int4_top1']} "
            f"teacher_in_text={row['teacher_in_text']} student_in_text={row['student_in_text']}"
        )

    print("\n=== ROLLOUT DRIFT SUMMARY ===")
    print(f"Steps: {args.steps}")
    print(f"Teacher forced mode: {bool(args.teacher_forced)}")
    print(f"Mean cosine: {mean_cos:.6f}")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Mean KL(student||teacher): {mean_kl:.6f}")
    print(f"Mean top-5 overlap: {mean_top5:.2f}%")
    print(f"Top-1 agreement: {top1_match_rate * 100:.2f}%")
    print(
        "Worst cosine: "
        f"step={worst_cos['step']} cos={worst_cos['cos']:.6f} mse={worst_cos['mse']:.6f} "
        f"max_abs_post_bridge={worst_cos['max_abs_post_bridge']:.6f} "
        f"max_abs_logits={worst_cos['max_abs_logits']:.6f}"
    )
    print(
        "Worst MSE: "
        f"step={worst_mse['step']} mse={worst_mse['mse']:.6f} cos={worst_mse['cos']:.6f} "
        f"max_abs_post_bridge={worst_mse['max_abs_post_bridge']:.6f} "
        f"max_abs_logits={worst_mse['max_abs_logits']:.6f}"
    )
    print(
        f"Catastrophic steps (cos<{args.catastrophic_cos} or mse>{args.catastrophic_mse}): {len(catastrophic_steps)}"
    )

    print("\n=== FIRST 20 STEPS ===")
    for row in per_step[:20]:
        print(
            f"step={row['step']:02d} cos={row['cos']:.6f} mse={row['mse']:.6f} "
            f"max_abs_post_bridge={row['max_abs_post_bridge']:.6f} "
            f"max_abs_logits={row['max_abs_logits']:.6f} kl_div={row['kl_div']:.6f} "
            f"top5_overlap_pct={row['top5_overlap_pct']:.2f} top1_same={row['top1_same']} "
            f"bf16_top1={row['bf16_top1']} int4_top1={row['int4_top1']}"
        )

    print("\n=== WORST 10 STEPS BY COSINE ===")
    for row in sorted(per_step, key=lambda x: x["cos"])[:10]:
        print(
            f"step={row['step']:02d} cos={row['cos']:.6f} mse={row['mse']:.6f} "
            f"max_abs_post_bridge={row['max_abs_post_bridge']:.6f} "
            f"max_abs_logits={row['max_abs_logits']:.6f} kl_div={row['kl_div']:.6f} "
            f"top5_overlap_pct={row['top5_overlap_pct']:.2f} top1_same={row['top1_same']} "
            f"bf16_top1={row['bf16_top1']} int4_top1={row['int4_top1']}"
        )

    if layer_ladder_rows:
        print("\n=== LAYER COSINE LADDER ===")
        for row in layer_ladder_rows:
            print(
                f"step={row['step']:02d} layer={row['layer']:02d} "
                f"cos={row['cos']:.6f} max_abs={row['max_abs']:.6f} mse={row['mse']:.6f}"
            )
    elif bool(args.layer_ladder):
        print("\n=== LAYER COSINE LADDER ===")
        print("No ladder rows captured.")

    if layer_stage_rows:
        print("\n=== LAYER STAGE LADDER ===")
        for row in layer_stage_rows:
            print(
                f"step={row['step']:02d} layer={row['layer']:02d} stage={row['stage']} "
                f"cos={row['cos']:.6f} max_abs={row['max_abs']:.6f} mse={row['mse']:.6f}"
            )
    elif bool(args.layer_stage_probe):
        print("\n=== LAYER STAGE LADDER ===")
        print("No stage rows captured.")


if __name__ == "__main__":
    main()
