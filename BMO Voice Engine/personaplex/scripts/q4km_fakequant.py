# Q4_K / Q4_K_M block layout (GGML / llama.cpp-style, 256 weights per block).
#
# Per super-block (256 consecutive elements):
#   Bytes 0-1:    d     (fp16 LE)
#   Bytes 2-3:    dmin  (fp16 LE)
#   Bytes 4-15:   scales[12] — packed 8×(6-bit scale, 6-bit min)
#   Bytes 16-143: qs[128] — 256 nibbles (low nibble = even index)
#
# Dequant: sub-block j, index i = 32*j+t:
#   (sc, mn) = unpack_k4(scales, j);  dl = d*(sc-32); ml = dmin*(mn-32); w[i] = dl*nibble+ml

from __future__ import annotations

import torch
import torch.nn as nn

BLOCK = 256
SUB = 32
NSUB = 8
BLOCK_BYTES = 144


def _unpack_scales_k4(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """scales (B, 12) uint8 -> sc, mn each (B, 8) int64."""
    sc = torch.zeros(scales.shape[0], 8, dtype=torch.int64, device=scales.device)
    mn = torch.zeros_like(sc)
    sl = scales.long()
    for j in range(4):
        sc[:, j] = sl[:, j] & 63
        mn[:, j] = sl[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (sl[:, j + 4] & 0x0F) | ((sl[:, j - 4] >> 6) << 4)
        mn[:, j] = (sl[:, j + 4] >> 4) | ((sl[:, j] >> 6) << 4)
    return sc, mn


def _pack_scales_k4(sc: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    """sc, mn (B, 8) int64 -> (B, 12) uint8."""
    B = sc.shape[0]
    out = torch.zeros(B, 12, dtype=torch.uint8, device=sc.device)
    sc = sc.long() & 63
    mn = mn.long() & 63
    for j in range(4):
        out[:, j] = (out[:, j] & 0xC0) | sc[:, j]
        out[:, j + 4] = (out[:, j + 4] & 0xC0) | mn[:, j]
    for j in range(4, 8):
        out[:, j + 4] = (out[:, j + 4] & 0xF0) | (sc[:, j] & 0x0F)
        out[:, j - 4] = (out[:, j - 4] & 0x3F) | ((sc[:, j] >> 4) << 6)
        out[:, j + 4] = (out[:, j + 4] & 0x0F) | ((mn[:, j] & 0x0F) << 4)
        out[:, j] = (out[:, j] & 0x3F) | ((mn[:, j] >> 4) << 6)
    return out


def _float_to_f16_bytes(x: torch.Tensor) -> torch.Tensor:
    """x (B,) float32 -> (B, 2) uint8 LE fp16."""
    h = x.to(torch.float16).view(torch.uint16).to(torch.int32)
    lo = (h & 255).to(torch.uint8)
    hi = ((h >> 8) & 255).to(torch.uint8)
    return torch.stack([lo, hi], dim=1)


def _f16_bytes_to_float(b2: torch.Tensor) -> torch.Tensor:
    """b2 (B, 2) uint8 -> (B,) float32."""
    u = b2[:, 0].to(torch.int32) | (b2[:, 1].to(torch.int32) << 8)
    return u.to(torch.uint16).view(torch.float16).float()


def _quantize_blocks_batched(wb: torch.Tensor) -> torch.Tensor:
    """wb (B, 256) float32 -> packed (B, 144) uint8."""
    B = wb.shape[0]
    device = wb.device
    w = wb.view(B, NSUB, SUB)
    mins = w.amin(dim=2)
    maxs = w.amax(dim=2)
    span = (maxs - mins).clamp_min(1e-8)
    qf = ((w - mins.unsqueeze(-1)) / span.unsqueeze(-1) * 15.0).round().clamp(0.0, 15.0)
    a = span / 15.0
    b = mins
    d_f = a.mean(dim=1).clamp_min(1e-6)
    dmin_f = b.abs().mean(dim=1).clamp_min(1e-8)
    sc = ((a / d_f.unsqueeze(1)) + 32.0).round().clamp(1.0, 62.0).long()
    mn = ((b / dmin_f.unsqueeze(1)) + 32.0).round().clamp(0.0, 63.0).long()

    def rebuild_q() -> torch.Tensor:
        dl = d_f.unsqueeze(1) * (sc.float() - 32.0)
        ml = dmin_f.unsqueeze(1) * (mn.float() - 32.0)
        return ((w - ml.unsqueeze(-1)) / (dl.unsqueeze(-1) + 1e-12)).round().clamp(0.0, 15.0)

    qf = rebuild_q()
    for _ in range(8):
        t = (sc.float() - 32.0).unsqueeze(-1)
        num_a = (t * (qf * w)).sum(dim=(1, 2))
        den_a = (t * t * (qf * qf)).sum(dim=(1, 2)).clamp_min(1e-12)
        d_f = (num_a / den_a).clamp_min(1e-8)
        dl = d_f.unsqueeze(1) * (sc.float() - 32.0)
        resid = w - dl.unsqueeze(-1) * qf
        tmn = (mn.float() - 32.0).unsqueeze(-1)
        num_b = (tmn * resid).sum(dim=(1, 2))
        den_b = ((mn.float() - 32.0) ** 2 * float(SUB)).sum(dim=1).clamp_min(1e-12)
        dmin_f = num_b / den_b
        qf = rebuild_q()

    head_d = _float_to_f16_bytes(d_f)
    head_m = _float_to_f16_bytes(dmin_f)
    scales_b = _pack_scales_k4(sc, mn)
    nib = qf.round().clamp(0, 15).long()
    nib = nib.view(B, BLOCK)
    low = nib[:, 0::2].to(torch.uint8)
    high = nib[:, 1::2].to(torch.uint8)
    qs_b = low | (high << 4)
    return torch.cat([head_d, head_m, scales_b, qs_b], dim=1)


def _dequantize_blocks_batched(pb: torch.Tensor) -> torch.Tensor:
    """pb (B, 144) uint8 -> (B, 256) float32."""
    d = _f16_bytes_to_float(pb[:, 0:2])
    dmin = _f16_bytes_to_float(pb[:, 2:4])
    sc, mn = _unpack_scales_k4(pb[:, 4:16])
    dl = d.unsqueeze(1) * (sc.float() - 32.0)
    ml = dmin.unsqueeze(1) * (mn.float() - 32.0)
    qs = pb[:, 16:144].long()
    t = torch.arange(BLOCK, device=pb.device)
    nib = (qs[:, t // 2] >> (4 * (t % 2))) & 0xF
    nib = nib.view(pb.shape[0], NSUB, SUB).float()
    out = dl.unsqueeze(-1) * nib + ml.unsqueeze(-1)
    return out.reshape(pb.shape[0], BLOCK)


def quantize_q4km(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    if weight.size(-1) % BLOCK != 0:
        raise ValueError(f"last dim must be divisible by {BLOCK}, got {weight.size(-1)}")
    rows, cols = weight.shape
    dev = weight.device
    wb = weight.detach().float().reshape(-1, BLOCK)
    chunks: list[torch.Tensor] = []
    step = 4096
    for s in range(0, wb.shape[0], step):
        chunks.append(_quantize_blocks_batched(wb[s : s + step].to(dev)))
    return torch.cat(chunks, dim=0).reshape(-1).to(torch.uint8).cpu()


def dequantize_q4km(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8")
    rows, cols = shape
    if cols % BLOCK != 0:
        raise ValueError(f"shape[1] must be divisible by {BLOCK}")
    n_blk = rows * (cols // BLOCK)
    if packed.numel() != n_blk * BLOCK_BYTES:
        raise ValueError(
            f"packed size mismatch: have {packed.numel()} need {n_blk * BLOCK_BYTES} for shape {shape}"
        )
    dev = packed.device
    pb = packed.view(n_blk, BLOCK_BYTES)
    outs: list[torch.Tensor] = []
    step = 4096
    for s in range(0, n_blk, step):
        outs.append(_dequantize_blocks_batched(pb[s : s + step].to(dev)))
    out = torch.cat(outs, dim=0).reshape(rows, cols)
    return out.to(torch.float16)


class Q4KMFakeQuantize(nn.Module):
    """Parametrization: weight → quantize → dequantize (STE)."""

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if w.dim() != 2 or w.size(-1) % BLOCK != 0:
            raise RuntimeError(
                f"Q4KMFakeQuantize expects 2D weight with last dim % {BLOCK} == 0, got {tuple(w.shape)}"
            )
        with torch.no_grad():
            dq = dequantize_q4km(quantize_q4km(w), w.shape).to(device=w.device, dtype=w.dtype)
        return w + (dq - w).detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4096, 4096, dtype=torch.float16)
    q = quantize_q4km(x)
    y = dequantize_q4km(q, (4096, 4096))
    xf = x.float().reshape(-1)
    yf = y.float().reshape(-1)
    cos = float(torch.sum(xf * yf) / (torch.norm(xf) * torch.norm(yf) + 1e-30))
    max_abs = float((xf - yf).abs().max().item())
    print(f"cosine={cos:.6f} max_abs_err={max_abs:.6g}")
