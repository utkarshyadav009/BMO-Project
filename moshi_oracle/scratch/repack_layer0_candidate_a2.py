# WARNING: DEFECTIVE-DO-NOT-USE
# This repack tool exported Candidate-A2 trial payloads without zero-points.
# Both CUDA kernel microbenchmarks and independent NumPy evaluation measured
# rel_l2 ~0.39 error against ground truth. The emitted layer0_a2_*.bin files
# are defective and marked DEFECTIVE-DO-NOT-USE.

import os
import sys
import json
import struct
import numpy as np
import torch
from pathlib import Path

REPO_DIR = Path("/home/jovyan/work/BMO-Project/personaplex_repo")
sys.path.insert(0, str(REPO_DIR))

import gguf

QAT_BEST_CKPT = "/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/qat_heavy_int2/qat_best.pt"
GGUF_PATH = "/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/qat_heavy_int2.gguf"
MODELS_DIR = Path("/home/jovyan/work/BMO-Project-Repo/BMO-Project/moshi_oracle/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def pack_6bit_indices_vectorized(scale_indices):
    # scale_indices: shape (128,) uint8 in 0..63
    # Pack 4 6-bit values into 3 bytes -> 32 groups of 4 indices = 96 bytes
    s = scale_indices.reshape(32, 4)
    i0 = s[:, 0] & 0x3F
    i1 = s[:, 1] & 0x3F
    i2 = s[:, 2] & 0x3F
    i3 = s[:, 3] & 0x3F
    
    b0 = i0 | ((i1 & 0x03) << 6)
    b1 = ((i1 >> 2) & 0x0F) | ((i2 & 0x0F) << 4)
    b2 = ((i2 >> 4) & 0x03) | ((i3 & 0x3F) << 2)
    
    bytes_arr = np.stack([b0, b1, b2], axis=1).astype(np.uint8)
    return bytes_arr.tobytes() # 96 bytes

def repack_tensor_a2_fast(tensor_name, qat_sd, gguf_tensors):
    print(f"\n[REPACK] Processing {tensor_name} (vectorized)...", flush=True)
    w_orig = qat_sd[tensor_name].float().numpy() # [R, C]
    R, C = w_orig.shape
    
    tile_tiers_gguf = gguf_tensors[f"{tensor_name.replace('.', '_')}.tile_tiers"].data
    n_tile_rows = R // 64
    n_tile_cols = C // 64
    tile_tiers = tile_tiers_gguf.reshape(n_tile_rows, n_tile_cols)
    
    outlier_indices = gguf_tensors[f"{tensor_name.replace('.', '_')}.outlier_indices"].data
    outlier_values = gguf_tensors[f"{tensor_name.replace('.', '_')}.outlier_values"].data
    
    # CSR outliers
    n_outliers = len(outlier_indices)
    outlier_rows = (outlier_indices // C).astype(np.int32)
    outlier_cols = (outlier_indices % C).astype(np.uint16)
    
    sort_perm = np.argsort(outlier_rows)
    outlier_rows = outlier_rows[sort_perm]
    outlier_cols = outlier_cols[sort_perm]
    outlier_vals_fp16 = outlier_values[sort_perm].astype(np.float16)
    
    csr_offsets = np.zeros(R + 1, dtype=np.uint32)
    np.add.at(csr_offsets, outlier_rows + 1, 1)
    np.cumsum(csr_offsets, out=csr_offsets)
    
    dequant_w_a2 = np.zeros((R, C), dtype=np.float64)
    
    col_perms = np.zeros((n_tile_rows, n_tile_cols), dtype=np.uint16)
    band_stream_offsets = np.zeros((n_tile_rows, 4), dtype=np.uint32)
    band_stream_bytes_list = []
    
    n_levels_dict = {1: 3.0, 2: 15.0, 3: 255.0, 0: 65535.0}
    
    for b in range(n_tile_rows):
        row_start = b * 64
        row_end = row_start + 64
        tiers_b = tile_tiers[b]
        
        sorted_cols = []
        for target_tier in [1, 2, 3, 0]:
            for c in range(n_tile_cols):
                if tiers_b[c] == target_tier:
                    sorted_cols.append(c)
        
        col_perms[b] = np.array(sorted_cols, dtype=np.uint16)
        
        n_t1 = sum(1 for c in sorted_cols if tiers_b[c] == 1)
        n_t2 = sum(1 for c in sorted_cols if tiers_b[c] == 2)
        n_t3 = sum(1 for c in sorted_cols if tiers_b[c] == 3)
        n_t0 = sum(1 for c in sorted_cols if tiers_b[c] == 0)
        
        t1_tile_bytes = 100 + 1024
        t2_tile_bytes = 100 + 2048
        t3_tile_bytes = 100 + 4096
        t0_tile_bytes = 100 + 8192
        
        off_t1 = 0
        off_t2 = off_t1 + n_t1 * t1_tile_bytes
        off_t3 = off_t2 + n_t2 * t2_tile_bytes
        off_t0 = off_t3 + n_t3 * t3_tile_bytes
        band_stream_offsets[b] = [off_t1, off_t2, off_t3, off_t0]
        
        band_buffer = bytearray()
        
        for c in sorted_cols:
            tier = tiers_b[c]
            col_start = c * 64
            col_end = col_start + 64
            
            w_tile = w_orig[row_start:row_end, col_start:col_end] # [64, 64]
            w_blocks = w_tile.reshape(128, 32) # 128 blocks of 32 elements
            
            n_lev = n_levels_dict[tier]
            w_min = w_blocks.min(axis=1, keepdims=True)
            w_max = w_blocks.max(axis=1, keepdims=True)
            w_diff = np.maximum(w_max - w_min, 1e-8)
            
            scales = w_diff / n_lev
            zps = np.round(-w_min / scales)
            qs = np.clip(np.round(w_blocks / scales + zps), 0, n_lev).astype(np.int32)
            
            if tier != 0:
                deq_blocks = scales * (qs - zps)
            else:
                deq_blocks = w_blocks.astype(np.float64)
                
            dequant_w_a2[row_start:row_end, col_start:col_end] = deq_blocks.reshape(64, 64)
            
            s_flat = scales.ravel()
            s_min = float(s_flat.min())
            s_max = float(s_flat.max())
            s_range = max(s_max - s_min, 1e-8)
            
            s_idx = np.clip(np.round((s_flat - s_min) / s_range * 63.0), 0, 63).astype(np.uint8)
            packed_scales_96b = pack_6bit_indices_vectorized(s_idx)
            
            # Write 100-byte scale header
            band_buffer.extend(struct.pack("<e", np.float16(s_min)))
            band_buffer.extend(struct.pack("<e", np.float16(s_range)))
            band_buffer.extend(packed_scales_96b)
            
            # Write weight payload dp4a interleaved
            q_2d = qs.reshape(64, 64)
            if tier == 1:
                # 2 bits/elem: pack 4 contiguous K-elements (columns) per byte
                q_4 = q_2d.reshape(64, 16, 4)
                packed = (q_4[:, :, 0] & 0x03) | \
                         ((q_4[:, :, 1] & 0x03) << 2) | \
                         ((q_4[:, :, 2] & 0x03) << 4) | \
                         ((q_4[:, :, 3] & 0x03) << 6)
                band_buffer.extend(packed.astype(np.uint8).tobytes())
            elif tier == 2:
                # 4 bits/elem: pack 2 contiguous K-elements (columns) per byte
                q_2 = q_2d.reshape(64, 32, 2)
                packed = (q_2[:, :, 0] & 0x0F) | \
                         ((q_2[:, :, 1] & 0x0F) << 4)
                band_buffer.extend(packed.astype(np.uint8).tobytes())
            elif tier == 3:
                # 8 bits/elem
                band_buffer.extend(q_2d.astype(np.uint8).tobytes())
            elif tier == 0:
                # FP16
                band_buffer.extend(w_tile.astype(np.float16).tobytes())

        band_stream_bytes_list.append(bytes(band_buffer))
        
    # Apply CSR outliers
    for idx in range(n_outliers):
        r_out = outlier_rows[idx]
        c_out = outlier_cols[idx]
        v_out = float(outlier_vals_fp16[idx])
        dequant_w_a2[r_out, c_out] = v_out
        
    print(f"[REPACK] Layer 0 {tensor_name} repacked successfully.", flush=True)
    return {
        "R": R,
        "C": C,
        "n_tile_rows": n_tile_rows,
        "n_tile_cols": n_tile_cols,
        "n_outliers": n_outliers,
        "col_perms": col_perms,
        "band_stream_offsets": band_stream_offsets,
        "band_stream_bytes_list": band_stream_bytes_list,
        "csr_offsets": csr_offsets,
        "outlier_cols": outlier_cols,
        "outlier_vals_fp16": outlier_vals_fp16,
        "dequant_w_a2": dequant_w_a2,
    }


def serialize_a2_binary(repack_data, out_path):
    print(f"[SERIALIZE] Emitting binary payload -> {out_path}", flush=True)
    with open(out_path, "wb") as f:
        magic = 0x41324d4f # "OMA2"
        reserved = b"\x00" * 36
        header = struct.pack(
            "<IIIIIII",
            magic,
            repack_data["R"],
            repack_data["C"],
            64,
            repack_data["n_tile_rows"],
            repack_data["n_tile_cols"],
            repack_data["n_outliers"],
        ) + reserved
        f.write(header)
        
        f.write(repack_data["col_perms"].tobytes())
        f.write(repack_data["band_stream_offsets"].tobytes())
        
        for band_bytes in repack_data["band_stream_bytes_list"]:
            f.write(band_bytes)
            
        f.write(repack_data["csr_offsets"].tobytes())
        f.write(repack_data["outlier_cols"].tobytes())
        f.write(repack_data["outlier_vals_fp16"].tobytes())
        
    file_size = os.path.getsize(out_path)
    print(f"[SERIALIZE] Emitted {out_path} ({file_size:,} bytes / {file_size/1024**2:.2f} MiB).", flush=True)


def main():
    print("[INIT] Loading QAT best PyTorch checkpoint & GGUF reader...", flush=True)
    qat_ckpt = torch.load(QAT_BEST_CKPT, map_location="cpu")
    qat_sd = qat_ckpt["state_dict"]
    
    reader = gguf.GGUFReader(GGUF_PATH)
    gguf_tensors = {t.name: t for t in reader.tensors}
    
    # 1. Repack linear_in
    data_in = repack_tensor_a2_fast("transformer.layers.0.gating.linear_in.weight", qat_sd, gguf_tensors)
    bin_in_path = MODELS_DIR / "layer0_a2_in.bin"
    serialize_a2_binary(data_in, bin_in_path)
    
    # 2. Repack linear_out
    data_out = repack_tensor_a2_fast("transformer.layers.0.gating.linear_out.weight", qat_sd, gguf_tensors)
    bin_out_path = MODELS_DIR / "layer0_a2_out.bin"
    serialize_a2_binary(data_out, bin_out_path)
    
    # 3. CPU Double-Accumulator Reference Verification (Fixed Seed x)
    print("\n[VERIFY] Generating CPU double-accumulator reference vector y = W * x...", flush=True)
    np.random.seed(1783708826)
    
    x_in = np.random.randn(4096).astype(np.float64)
    y_in_ref = np.dot(data_in["dequant_w_a2"], x_in)
    
    x_out = np.random.randn(11264).astype(np.float64)
    y_out_ref = np.dot(data_out["dequant_w_a2"], x_out)
    
    ref_payload = {
        "seed": 1783708826,
        "linear_in": {
            "shape_w": [data_in["R"], data_in["C"]],
            "x_in_head": x_in[:10].tolist(),
            "x_in_norm": float(np.linalg.norm(x_in)),
            "y_in_ref_head": y_in_ref[:10].tolist(),
            "y_in_ref_min": float(y_in_ref.min()),
            "y_in_ref_max": float(y_in_ref.max()),
            "y_in_ref_norm": float(np.linalg.norm(y_in_ref)),
        },
        "linear_out": {
            "shape_w": [data_out["R"], data_out["C"]],
            "x_out_head": x_out[:10].tolist(),
            "x_out_norm": float(np.linalg.norm(x_out)),
            "y_out_ref_head": y_out_ref[:10].tolist(),
            "y_out_ref_min": float(y_out_ref.min()),
            "y_out_ref_max": float(y_out_ref.max()),
            "y_out_ref_norm": float(np.linalg.norm(y_out_ref)),
        },
        "full_x_in": x_in.tolist(),
        "full_y_in_ref": y_in_ref.tolist(),
        "full_x_out": x_out.tolist(),
        "full_y_out_ref": y_out_ref.tolist(),
    }
    
    ref_path = MODELS_DIR / "layer0_a2_ref.json"
    with open(ref_path, "w") as f:
        json.dump(ref_payload, f, indent=2)
    print(f"[VERIFY] Saved CPU double-accumulator reference -> {ref_path}", flush=True)
    print(f"  linear_in  y_norm: {ref_payload['linear_in']['y_in_ref_norm']:.6f}")
    print(f"  linear_out y_norm: {ref_payload['linear_out']['y_out_ref_norm']:.6f}")
    print("\n[COMPLETE] Candidate-A2 repacking and verification complete!", flush=True)

if __name__ == "__main__":
    main()
