import argparse
import copy
from pathlib import Path

import torch

from moshi.models import loaders

from apply_slicegpt import (
    get_rms_alpha_or_ones,
    load_q_matrix,
    parse_bool,
    set_norm_alpha_one_if_present,
)


def rotate_layer_weights(src_layer, dst_layer, q: torch.Tensor, absorb_rms_alpha: bool):
    d_old = int(q.shape[0])
    if absorb_rms_alpha:
        alpha1 = get_rms_alpha_or_ones(src_layer.norm1, d_old, q.device)
        alpha2 = get_rms_alpha_or_ones(src_layer.norm2, d_old, q.device)
        set_norm_alpha_one_if_present(dst_layer.norm1)
        set_norm_alpha_one_if_present(dst_layer.norm2)
    else:
        alpha1 = torch.ones((d_old,), device=q.device, dtype=torch.float32)
        alpha2 = torch.ones((d_old,), device=q.device, dtype=torch.float32)

    with torch.no_grad():
        src_in_proj = src_layer.self_attn.in_proj_weight.detach().to(device=q.device, dtype=torch.float32)
        src_in_blocks = src_in_proj.view(3, d_old, d_old)
        dst_blocks = []
        for b in range(3):
            w_block = src_in_blocks[b] * alpha1.unsqueeze(0)
            dst_blocks.append(q.T @ w_block @ q)
        dst_layer.self_attn.in_proj_weight.copy_(
            torch.cat(dst_blocks, dim=0).to(dst_layer.self_attn.in_proj_weight.dtype)
        )

        src_out_proj = src_layer.self_attn.out_proj.weight.detach().to(device=q.device, dtype=torch.float32)
        dst_out_proj = q.T @ src_out_proj @ q
        dst_layer.self_attn.out_proj.weight.copy_(dst_out_proj.to(dst_layer.self_attn.out_proj.weight.dtype))

        src_lin_in = src_layer.gating.linear_in.weight.detach().to(device=q.device, dtype=torch.float32)
        src_lin_in = src_lin_in * alpha2.unsqueeze(0)
        dst_lin_in = src_lin_in @ q
        dst_layer.gating.linear_in.weight.copy_(dst_lin_in.to(dst_layer.gating.linear_in.weight.dtype))

        src_lin_out = src_layer.gating.linear_out.weight.detach().to(device=q.device, dtype=torch.float32)
        dst_lin_out = q.T @ src_lin_out
        dst_layer.gating.linear_out.weight.copy_(dst_lin_out.to(dst_layer.gating.linear_out.weight.dtype))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="RMSNorm commutativity sanity test for SliceGPT rotation")
    parser.add_argument("--bf16", default="v5_step1500.safetensors")
    parser.add_argument("--eigenvectors", default="bmo_slicegpt_eigenvectors.pt")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--basis-source", choices=["norm1", "norm2"], default="norm1")
    parser.add_argument("--absorb-rms-alpha", type=parse_bool, default=True)
    parser.add_argument("--headwise-q-basis", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    root = Path(__file__).resolve().parent
    bf16_path = (root / args.bf16).resolve()
    eig_path = (root / args.eigenvectors).resolve()

    print(f"[INFO] Loading model: {bf16_path}")
    model = loaders.get_moshi_lm(str(bf16_path), device=device, dtype=torch.float32, cpu_offload=False)
    model.eval()

    eig_payload = torch.load(str(eig_path), map_location="cpu")

    d_old = int(model.dim)
    num_heads = int(model.transformer.layers[0].self_attn.num_heads)

    q = load_q_matrix(
        eig_payload,
        int(args.layer_idx),
        args.basis_source,
        d_old,
        d_old,
        device,
        num_heads=num_heads,
        headwise_q_basis=bool(args.headwise_q_basis),
    ).to(device=device, dtype=torch.float32)

    src_layer = model.transformer.layers[int(args.layer_idx)]
    base_layer = copy.deepcopy(src_layer).to(device=device, dtype=torch.float32).eval()
    rot_layer = copy.deepcopy(src_layer).to(device=device, dtype=torch.float32).eval()

    rotate_layer_weights(base_layer, rot_layer, q, absorb_rms_alpha=bool(args.absorb_rms_alpha))

    x = torch.randn(
        int(args.batch_size),
        int(args.seq_len),
        d_old,
        dtype=torch.float32,
        device=device,
    )
    x_rot = torch.matmul(x, q)

    out_base = base_layer(x)
    out_rot = rot_layer(x_rot)
    out_rot_back = torch.matmul(out_rot, q.T)

    diff = out_base - out_rot_back
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    ok = torch.allclose(out_base, out_rot_back, atol=float(args.atol), rtol=float(args.rtol))

    print(f"[INFO] layer={args.layer_idx} basis={args.basis_source} absorb_rms_alpha={bool(args.absorb_rms_alpha)}")
    print(f"[INFO] max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} atol={args.atol} rtol={args.rtol}")
    print(f"[INFO] allclose={ok}")

    if not ok:
        raise AssertionError(
            "Commutativity check failed: out_base !~= out_rot @ Q^T. "
            f"max_abs={max_abs:.6e}"
        )


if __name__ == "__main__":
    main()
