import os
import sys
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

REPO_DIR = Path("/home/jovyan/work/BMO-Project/personaplex_repo")
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "moshi"))

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

from qat_septq import unpack_tier_mask_uint2
from verify_septq_zs_drift import run_zs_drift, summarize_steps, summarize_layer_steps

TEACHER_CKPT = "/home/jovyan/work/BMO-Project/personaplex_repo/_keep/v5_step1500_split.safetensors"
QAT_BEST_CKPT = "/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/qat_heavy_int2/qat_best.pt"
BASE_PTQ_CKPT = "/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/bmo_tr_heavy_int2.pt"

INPUT_WAV = str(REPO_DIR / "tellmeajoke_padded.wav")
VOICE_PROMPT_WAV = str(REPO_DIR / "bmo_621.wav")
MIMI_WEIGHT = str(REPO_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors")
TOKENIZER = str(REPO_DIR / "tokenizer_spm_32k_3.model")

GATING_PREFIXES = [f"transformer.layers.{i}.gating" for i in range(31)]

def get_gating_tensor_names(state_dict):
    names = []
    for k in state_dict.keys():
        if any(k.startswith(p) for p in GATING_PREFIXES) and k.endswith(".weight"):
            names.append(k)
    return sorted(names)

def quantize_block32_tier_map_vectorized(weight, mask, block_size=32):
    R, C = weight.shape
    w_flat = weight.detach().float()
    m_flat = mask.detach()
    
    n_blocks = (R * C) // block_size
    w_blocks = w_flat.view(n_blocks, block_size)
    m_blocks = m_flat.view(n_blocks, block_size)
    
    w_out_blocks = w_blocks.clone()
    
    is_t1 = (m_blocks == 1)
    is_t2 = (m_blocks == 2)
    is_t3 = (m_blocks == 3)

    if is_t1.any():
        w_t1_max = torch.where(is_t1, w_blocks, torch.tensor(-1e9)).max(dim=1, keepdim=True).values
        w_t1_min = torch.where(is_t1, w_blocks, torch.tensor(1e9)).min(dim=1, keepdim=True).values
        has_t1 = is_t1.any(dim=1, keepdim=True)
        scale1 = torch.where(has_t1 & (w_t1_max > w_t1_min), (w_t1_max - w_t1_min) / 3.0, torch.tensor(1.0))
        zp1 = torch.where(has_t1 & (w_t1_max > w_t1_min), torch.round(-w_t1_min / scale1), torch.tensor(0.0))
        q1 = torch.clamp(torch.round(w_blocks / scale1 + zp1), 0, 3)
        deq1 = scale1 * (q1 - zp1)
        w_out_blocks = torch.where(is_t1, deq1, w_out_blocks)

    if is_t2.any():
        w_t2_max = torch.where(is_t2, w_blocks, torch.tensor(-1e9)).max(dim=1, keepdim=True).values
        w_t2_min = torch.where(is_t2, w_blocks, torch.tensor(1e9)).min(dim=1, keepdim=True).values
        has_t2 = is_t2.any(dim=1, keepdim=True)
        scale2 = torch.where(has_t2 & (w_t2_max > w_t2_min), (w_t2_max - w_t2_min) / 15.0, torch.tensor(1.0))
        zp2 = torch.where(has_t2 & (w_t2_max > w_t2_min), torch.round(-w_t2_min / scale2), torch.tensor(0.0))
        q2 = torch.clamp(torch.round(w_blocks / scale2 + zp2), 0, 15)
        deq2 = scale2 * (q2 - zp2)
        w_out_blocks = torch.where(is_t2, deq2, w_out_blocks)

    if is_t3.any():
        w_t3_max = torch.where(is_t3, w_blocks, torch.tensor(-1e9)).max(dim=1, keepdim=True).values
        w_t3_min = torch.where(is_t3, w_blocks, torch.tensor(1e9)).min(dim=1, keepdim=True).values
        has_t3 = is_t3.any(dim=1, keepdim=True)
        scale3 = torch.where(has_t3 & (w_t3_max > w_t3_min), (w_t3_max - w_t3_min) / 255.0, torch.tensor(1.0))
        zp3 = torch.where(has_t3 & (w_t3_max > w_t3_min), torch.round(-w_t3_min / scale3), torch.tensor(0.0))
        q3 = torch.clamp(torch.round(w_blocks / scale3 + zp3), 0, 255)
        deq3 = scale3 * (q3 - zp3)
        w_out_blocks = torch.where(is_t3, deq3, w_out_blocks)

    return w_out_blocks.view(R, C).to(weight.dtype)


def quantize_q4_0(weight, block_size=32):
    R, C = weight.shape
    w_flat = weight.detach().float()
    n_blocks = (R * C) // block_size
    w_blocks = w_flat.view(n_blocks, block_size)
    scales = torch.max(torch.abs(w_blocks), dim=1, keepdim=True).values.clamp_min(1e-8) / 7.0
    q = torch.clamp(torch.round(w_blocks / scales), -8, 7)
    return (q * scales).view(R, C).to(weight.dtype)


def quantize_q3_k(weight, block_size=256, sub_block_size=16):
    R, C = weight.shape
    w_flat = weight.detach().float()
    n_super_blocks = (R * C) // block_size
    w_super = w_flat.view(n_super_blocks, 16, sub_block_size)
    
    super_max = torch.max(torch.abs(w_flat.view(n_super_blocks, block_size)), dim=1).values.clamp_min(1e-8).view(n_super_blocks, 1, 1)
    super_scale = super_max / 3.5
    
    sub_max = torch.max(torch.abs(w_super), dim=2, keepdim=True).values.clamp_min(1e-8)
    sub_scales_6bit = torch.clamp(torch.round(sub_max / super_scale * 63.0 / 3.5), 1, 63) * (super_scale * 3.5 / 63.0)
    
    q = torch.clamp(torch.round(w_super / sub_scales_6bit), -4, 3)
    return (q * sub_scales_6bit).view(R, C).to(weight.dtype)


def quantize_q2_k(weight, block_size=256, sub_block_size=16):
    R, C = weight.shape
    w_flat = weight.detach().float()
    n_super_blocks = (R * C) // block_size
    w_super = w_flat.view(n_super_blocks, 16, sub_block_size)
    
    sub_min = torch.min(w_super, dim=2, keepdim=True).values
    sub_max = torch.max(w_super, dim=2, keepdim=True).values
    range_sub = (sub_max - sub_min).clamp_min(1e-8)
    
    sub_scales = range_sub / 3.0
    q = torch.clamp(torch.round((w_super - sub_min) / sub_scales), 0, 3)
    return (sub_scales * q + sub_min).view(R, C).to(weight.dtype)


def evaluate_candidate(candidate_type, mix_name, mix_config, out_temp_path, qat_template, base_ckpt):
    print(f"\n=======================================================", flush=True)
    print(f"[PREPARE] Building {candidate_type} ({mix_name}) -> {out_temp_path}", flush=True)
    print(f"=======================================================", flush=True)
    
    # Create fresh state_dict copy from qat_template
    state_dict = {k: v.clone() for k, v in qat_template["state_dict"].items()}
    
    if candidate_type == "Candidate A Proxy":
        tier_masks = base_ckpt["tier_masks_uint2"]
        gating_names = get_gating_tensor_names(state_dict)
        for name in gating_names:
            w = state_dict[name]
            mask_packed = tier_masks[name]
            mask_uint2 = unpack_tier_mask_uint2(mask_packed, target_shape=(w.shape[0], w.shape[1]))
            state_dict[name] = quantize_block32_tier_map_vectorized(w, mask_uint2, block_size=32)
        print(f"[BUILD] Re-quantized {len(gating_names)} gating tensors with Block-32 scales.", flush=True)
    elif candidate_type == "Candidate B PTQ":
        modified_count = 0
        for layer_idx in range(31):
            qtype = mix_config.get(layer_idx, 'Q2_K')
            prefix = f"transformer.layers.{layer_idx}.gating"
            for role in ["linear_in.weight", "linear_out.weight"]:
                tname = f"{prefix}.{role}"
                if tname in state_dict:
                    w = state_dict[tname]
                    if qtype == 'Q2_K':
                        w_q = quantize_q2_k(w)
                    elif qtype == 'Q3_K':
                        w_q = quantize_q3_k(w)
                    elif qtype == 'Q4_0':
                        w_q = quantize_q4_0(w)
                    else:
                        raise ValueError(f"Unknown qtype {qtype}")
                    state_dict[tname] = w_q
                    modified_count += 1
        print(f"[BUILD] Re-quantized {modified_count} gating tensors for Candidate B ({mix_name}).", flush=True)

    # Save to temp path
    eval_payload = {k: v for k, v in qat_template.items() if k != "state_dict"}
    eval_payload["state_dict"] = state_dict
    torch.save(eval_payload, out_temp_path)
    print(f"[SAVED] Temp checkpoint saved to {out_temp_path}.", flush=True)

    # Run z_s evaluation
    print(f"[EVAL] Starting z_s drift probe for {candidate_type} - {mix_name}...", flush=True)
    payload = run_zs_drift(
        teacher_ckpt=TEACHER_CKPT,
        student_ckpt=out_temp_path,
        steps=125,
        seed=1234,
        device="cuda:0",
        input_wav=INPUT_WAV,
        voice_prompt_wav=VOICE_PROMPT_WAV,
        text_prompt="Tell me a joke.",
        mimi_weight=MIMI_WEIGHT,
        tokenizer_path=TOKENIZER,
        voice_ratio=0.25,
        teacher_dtype="bfloat16",
        student_dtype="auto",
    )
    summary = summarize_steps(payload["per_step"])
    layer_summary = summarize_layer_steps(payload["per_step_layer"], cliff_threshold=0.995)
    
    res = {
        "candidate": candidate_type,
        "label": f"{candidate_type} ({mix_name})",
        "cos_median": summary["cos_median"],
        "cos_min": summary["cos_min"],
        "cos_mean": summary["cos_mean"],
        "mse_median": summary["mse_median"],
        "max_abs_median": summary["max_abs_median"],
        "first_layer_below": layer_summary["first_layer_below_threshold"],
        "drift_mode": layer_summary["drift_mode_hint"],
    }
    print(f"[RESULT] {res['label']} -> cos_median={res['cos_median']:.6f} cos_min={res['cos_min']:.6f} cos_mean={res['cos_mean']:.6f} first_below={res['first_layer_below']}", flush=True)
    
    # Cleanup temp file
    if os.path.exists(out_temp_path):
        os.remove(out_temp_path)
        
    return res


def main():
    os.makedirs("/tmp/bmo_probes", exist_ok=True)
    temp_ckpt_path = "/tmp/bmo_probes/temp_eval_ckpt.pt"

    print("[INIT] Loading QAT best template & Base PTQ metadata into host RAM...", flush=True)
    qat_template = torch.load(QAT_BEST_CKPT, map_location="cpu")
    base_ckpt = torch.load(BASE_PTQ_CKPT, map_location="cpu")
    print("[INIT] Loaded templates successfully.", flush=True)

    results = []

    # 1. Candidate A Proxy Probe
    res_a = evaluate_candidate(
        candidate_type="Candidate A Proxy",
        mix_name="Block-32 Scales",
        mix_config={},
        out_temp_path=temp_ckpt_path,
        qat_template=qat_template,
        base_ckpt=base_ckpt,
    )
    results.append(res_a)

    # 2. Candidate B Mix 1 (Uniform Q2_K)
    b1_cfg = {i: 'Q2_K' for i in range(31)}
    res_b1 = evaluate_candidate(
        candidate_type="Candidate B PTQ",
        mix_name="Mix 1 (Uniform Q2_K)",
        mix_config=b1_cfg,
        out_temp_path=temp_ckpt_path,
        qat_template=qat_template,
        base_ckpt=base_ckpt,
    )
    results.append(res_b1)

    # 3. Candidate B Mix 2 (Sensitivity-tuned Q4_0 / Q3_K / Q2_K)
    b2_cfg = {}
    for i in range(31):
        if i <= 4:
            b2_cfg[i] = 'Q4_0'
        elif i <= 20:
            b2_cfg[i] = 'Q3_K'
        else:
            b2_cfg[i] = 'Q2_K'
    res_b2 = evaluate_candidate(
        candidate_type="Candidate B PTQ",
        mix_name="Mix 2 (Sensitivity-tuned Q4/Q3/Q2)",
        mix_config=b2_cfg,
        out_temp_path=temp_ckpt_path,
        qat_template=qat_template,
        base_ckpt=base_ckpt,
    )
    results.append(res_b2)

    # 4. Candidate B Mix 3 (High-fidelity Q4_0 / Q3_K)
    b3_cfg = {}
    for i in range(31):
        if i <= 14:
            b3_cfg[i] = 'Q4_0'
        else:
            b3_cfg[i] = 'Q3_K'
    res_b3 = evaluate_candidate(
        candidate_type="Candidate B PTQ",
        mix_name="Mix 3 (High-fidelity Q4/Q3)",
        mix_config=b3_cfg,
        out_temp_path=temp_ckpt_path,
        qat_template=qat_template,
        base_ckpt=base_ckpt,
    )
    results.append(res_b3)

    print("\n\n=======================================================", flush=True)
    print("FINAL Z_S QUALITY PROBES COMPARISON TABLE", flush=True)
    print("=======================================================", flush=True)
    print(f"{'Config / Candidate':<42} | {'Median z_s':<10} | {'Min z_s':<10} | {'Mean z_s':<10} | {'First Cliff Layer':<15}", flush=True)
    print("-" * 97, flush=True)
    print(f"{'Baseline (Shipped QAT heavy int2)':<42} | {0.889420:<10.6f} | {0.871562:<10.6f} | {0.888165:<10.6f} | {3:<15}", flush=True)
    for r in results:
        print(f"{r['label']:<42} | {r['cos_median']:<10.6f} | {r['cos_min']:<10.6f} | {r['cos_mean']:<10.6f} | {r['first_layer_below']:<15}", flush=True)

    out_summary_path = "/tmp/bmo_probes/probe_results.json"
    with open(out_summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_summary_path}", flush=True)

if __name__ == "__main__":
    main()
