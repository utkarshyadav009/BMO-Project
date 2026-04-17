import os
import sys
import traceback

import torch
import torch.nn.functional as F

from moshi.models import loaders
from verify_int4_rollout_drift import (
    build_forced_tokens,
    get_student_unrotation_matrix,
    get_temporal_layers,
)


def _unwrap_tensor_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    return None


@torch.no_grad()
def _capture_step_layer_output_and_cache_class(model, forced_tokens, step_idx, layer_idx, device):
    layers = get_temporal_layers(model)
    if not layers:
        raise RuntimeError("Temporal layers are unavailable")
    if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
        raise ValueError(f"Invalid layer_idx={layer_idx}; expected [0, {len(layers) - 1}]")

    captured = {"y": None}
    cache_class = "None"

    def hook(_module, _inputs, output):
        y = _unwrap_tensor_output(output)
        if torch.is_tensor(y):
            captured["y"] = y.detach().to(dtype=torch.float32).cpu().contiguous()

    handle = layers[int(layer_idx)].register_forward_hook(hook)
    k = int(model.num_codebooks)

    try:
        with model.streaming(batch_size=1):
            for t in range(int(step_idx) + 1):
                seq = forced_tokens[t].view(1, k, 1).to(device)
                model.forward_codes(seq)

                # Streaming state can be cleared at context exit; sample cache class per step.
                attn0 = getattr(layers[0], "self_attn", None)
                state = getattr(attn0, "_streaming_state", None) if attn0 is not None else None
                kv_cache = getattr(state, "kv_cache", None) if state is not None else None
                if kv_cache is not None:
                    cache_class = type(kv_cache).__name__
    finally:
        handle.remove()

    if captured["y"] is None:
        raise RuntimeError("Failed to capture layer output at requested step")

    return captured["y"], cache_class


@torch.no_grad()
def main():
    if os.environ.get("BMO_ENABLE_RUNTIME_PATCHES") == "1":
        print("BMO_ENABLE_RUNTIME_PATCHES=1 is not allowed for this regression test.")
        sys.exit(1)

    cache_class = "None"
    step1_l31_cos = float("nan")
    ok = False

    try:
        device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for test_identity_regression.py")

        print("[INFO] Native-path mode: skipping test_rtx_edge runtime monkey patches.")

        teacher = loaders.get_moshi_lm(
            "v5_step1500.safetensors",
            device=device,
            dtype=torch.float32,
            cpu_offload=False,
        )
        student = loaders.get_moshi_lm(
            "bmo_slicegpt_4096_identity_quarot_fix1.pt",
            device=device,
            dtype=torch.float32,
            cpu_offload=False,
        )

        teacher.eval()
        student.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        for p in student.parameters():
            p.requires_grad = False

        student_unrot = get_student_unrotation_matrix(student)
        if student_unrot is not None:
            print(
                "[INFO] Student unrotation matrix loaded: "
                f"shape={tuple(student_unrot.shape)}"
            )
        else:
            print("[INFO] Student unrotation matrix unavailable; using raw student layer output.")

        forced_tokens = build_forced_tokens(
            teacher,
            steps=2,
            device=device,
            input_wav="tellmeajoke_padded.wav",
            voice_prompt_wav="bmo_621.wav",
            text_prompt="Tell me a joke.",
            mimi_weight="tokenizer-e351c8d8-checkpoint125.safetensors",
            tokenizer_path="tokenizer_spm_32k_3.model",
            voice_ratio=0.25,
        )

        teacher_layers = get_temporal_layers(teacher)
        student_layers = get_temporal_layers(student)
        n_layers = min(len(teacher_layers), len(student_layers))
        if n_layers <= 0:
            raise RuntimeError("No temporal layers found on teacher/student")

        layer_idx = min(31, n_layers - 1)

        teacher_l31, teacher_cache_class = _capture_step_layer_output_and_cache_class(
            teacher,
            forced_tokens,
            step_idx=1,
            layer_idx=layer_idx,
            device=device,
        )
        student_l31, student_cache_class = _capture_step_layer_output_and_cache_class(
            student,
            forced_tokens,
            step_idx=1,
            layer_idx=layer_idx,
            device=device,
        )

        cache_class = student_cache_class if student_cache_class != "None" else teacher_cache_class

        if student_unrot is not None:
            unrot = student_unrot.to(device=student_l31.device, dtype=torch.float32)
            student_l31 = student_l31 @ unrot

        t_vec = teacher_l31.reshape(-1)
        s_vec = student_l31.reshape(-1)
        step1_l31_cos = float(F.cosine_similarity(t_vec, s_vec, dim=0).item())

        native_cache_ok = cache_class in {"RingKVCache", "KVCache"}
        ok = bool(native_cache_ok and step1_l31_cos > 0.999)
    except Exception as exc:
        print(f"[INFO] identity regression exception: {exc}")
        traceback.print_exc()
        ok = False

    print(f"[RESULT] cache_class = {cache_class}")
    print(f"[RESULT] step1_L31_cos = {step1_l31_cos:.6f}")
    print(f"[RESULT] {'PASS' if ok else 'FAIL'}")

    if ok:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
