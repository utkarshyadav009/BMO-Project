import os
import sys
import time
import threading
import psutil
import subprocess
import gc
import zipfile
import torch
import importlib
import bitsandbytes as bnb


# Must be set before importing moshi modules so @torch_compile_lazy becomes a no-op
# and CUDAGraphed wrappers run eagerly for Params4bit compatibility.
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["NO_CUDA_GRAPH"] = "1"
# --- 1. THE MONKEY PATCH (JETSON MEMORY SAVER) ---
from moshi.models import loaders
from moshi.models.lm import LMModel
import moshi.offline
from moshi.modules.gating import ActivationGating

moshi_compile_utils = importlib.import_module("moshi.utils.compile")
moshi_gating = importlib.import_module("moshi.modules.gating")


def _rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def _resolve_cuda_device(device="cuda:0"):
    if not torch.cuda.is_available():
        return None
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return device
        return torch.device("cuda:0")
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.device(device)
    return torch.device("cuda:0")


def _print_memory_snapshot(label, device="cuda:0"):
    message = f"[JETSON][MEM] {label}: rss={_rss_gb():.2f} GB"
    cuda_device = _resolve_cuda_device(device)
    if cuda_device is not None:
        allocated = torch.cuda.memory_allocated(cuda_device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(cuda_device) / (1024 ** 3)
        message += (
            f" | cuda_allocated={allocated:.2f} GB"
            f" cuda_reserved={reserved:.2f} GB"
        )
    print(message)


def _print_linear_and_parameter_audit(model, *, threshold_mb=1.0, top_k=30):
    rows = []
    dtype_totals_mb = {}
    type_totals_mb = {}

    for name, param in model.named_parameters():
        if param is None:
            continue
        size_mb = (param.numel() * param.element_size()) / 1e6
        dtype_name = str(param.dtype).replace("torch.", "")
        type_name = type(param).__name__
        device_name = str(param.device)

        dtype_totals_mb[dtype_name] = dtype_totals_mb.get(dtype_name, 0.0) + size_mb
        type_totals_mb[type_name] = type_totals_mb.get(type_name, 0.0) + size_mb

        if size_mb >= float(threshold_mb):
            rows.append(
                {
                    "name": name,
                    "ptype": type_name,
                    "dtype": dtype_name,
                    "size_mb": size_mb,
                    "device": device_name,
                }
            )

    rows.sort(key=lambda item: item["size_mb"], reverse=True)

    linear4bit_count = sum(1 for module in model.modules() if isinstance(module, bnb.nn.Linear4bit))
    linear_count = sum(1 for module in model.modules() if isinstance(module, torch.nn.Linear))

    print(
        "[JETSON][AUDIT] linear modules: "
        f"Linear4bit={linear4bit_count} "
        f"torch.nn.Linear={linear_count}"
    )

    print(
        "[JETSON][AUDIT] parameters >= "
        f"{float(threshold_mb):.2f} MB (top {top_k} by size):"
    )
    if not rows:
        print("[JETSON][AUDIT]   (none)")
    else:
        for row in rows[: int(top_k)]:
            print(
                "[JETSON][AUDIT]   "
                f"size_mb={row['size_mb']:.2f} "
                f"dtype={row['dtype']} "
                f"ptype={row['ptype']} "
                f"device={row['device']} "
                f"name={row['name']}"
            )

    print("[JETSON][AUDIT] per-dtype totals (MB):")
    for dtype_name, total_mb in sorted(dtype_totals_mb.items(), key=lambda item: item[1], reverse=True):
        print(f"[JETSON][AUDIT]   dtype={dtype_name} total_mb={total_mb:.2f}")

    print("[JETSON][AUDIT] per-parameter-type totals (MB):")
    for type_name, total_mb in sorted(type_totals_mb.items(), key=lambda item: item[1], reverse=True):
        print(f"[JETSON][AUDIT]   ptype={type_name} total_mb={total_mb:.2f}")


def _print_state_dict_remaining(state_dict, *, top_k=20):
    remaining_rows = []
    total_mb = 0.0
    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        size_mb = (tensor.numel() * tensor.element_size()) / 1e6
        total_mb += size_mb
        remaining_rows.append((size_mb, str(tensor.dtype).replace("torch.", ""), key))

    remaining_rows.sort(key=lambda item: item[0], reverse=True)
    print(
        "[JETSON][RAM] remaining state_dict tensors after load: "
        f"count={len(remaining_rows)} total_mb={total_mb:.2f}"
    )
    for size_mb, dtype_name, key in remaining_rows[: int(top_k)]:
        print(
            "[JETSON][RAM]   "
            f"size_mb={size_mb:.2f} dtype={dtype_name} key={key}"
        )


def _instrumented_warmup(mimi, other_mimi, lm_gen, device, frame_size):
    # Mirror moshi.offline.warmup but add targeted VRAM checkpoints.
    cuda_device = _resolve_cuda_device(device)
    printed_first_step = False

    if cuda_device is not None:
        _print_memory_snapshot("warmup entry", cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            if not printed_first_step:
                _print_memory_snapshot("right before first lm_gen.step in warmup", cuda_device or device)
                printed_first_step = True
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        _print_memory_snapshot("warmup end current", cuda_device)
        max_allocated = torch.cuda.max_memory_allocated(cuda_device) / (1024 ** 3)
        max_reserved = torch.cuda.max_memory_reserved(cuda_device) / (1024 ** 3)
        print(
            "[JETSON][MEM] warmup peak: "
            f"cuda_max_allocated={max_allocated:.2f} GB "
            f"cuda_max_reserved={max_reserved:.2f} GB"
        )


moshi.offline.warmup = _instrumented_warmup


def _patched_activation_gating_forward(self, x: torch.Tensor):
    # Use module calls so bnb Linear4bit executes its own quantized matmul path.
    x = self.linear_in(x)
    bsz, tlen, _ = x.shape
    x = x.view(bsz, tlen, 2, -1)
    x = self.activation(x[..., 0, :]) * x[..., 1, :]
    return self.linear_out(x)


ActivationGating.forward = _patched_activation_gating_forward

original_get_moshi_lm = loaders.get_moshi_lm

def load_and_compress_layer_by_layer(module, state_dict, device="cuda:0", prefix=""):
    """
    For every module in the tree:
      1) Load this module's OWN direct params/buffers from state_dict and place on `device`.
         (Catches raw params like StreamingMultiheadAttention.in_proj_weight that aren't
         wrapped in an nn.Linear and aren't on a "leaf" module.)
      2) For child Linear layers (excluding out_proj), swap in bnb Linear4bit and quantize.
      3) Recurse into all other children.
    """
    # ---- (1) This module's own direct parameters ----
    for pname, param in list(module.named_parameters(recurse=False)):
        full_pkey = f"{prefix}{pname}"
        source_tensor = state_dict.pop(full_pkey, None)
        if source_tensor is not None:
            param.data = (
                source_tensor.data.clone().contiguous()
                .to(device=device, dtype=param.dtype)
            )
        elif param.device.type == "cpu":
            # No state_dict entry but param exists -> still move to device.
            param.data = param.data.to(device)

    # ---- (1b) This module's own direct buffers ----
    for bname, buf in list(module.named_buffers(recurse=False)):
        if buf is None:
            continue
        full_bkey = f"{prefix}{bname}"
        source_buffer = state_dict.pop(full_bkey, None)
        if source_buffer is not None:
            module._buffers[bname] = (
                source_buffer.data.clone().contiguous()
                .to(device=device, dtype=buf.dtype)
            )
        elif buf.device.type == "cpu":
            module._buffers[bname] = buf.to(device)

    # ---- (2) + (3) Children ----
    for name, child in module.named_children():
        full_name = f"{prefix}{name}"

        if isinstance(child, torch.nn.Linear) and "out_proj" not in full_name:
            has_bias = child.bias is not None
            new_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
            )

            w_key = f"{full_name}.weight"
            source_weight = state_dict.pop(w_key, None)
            if source_weight is not None:
                # Keep on CPU here. The .to(device) below is what triggers
                # Params4bit.cuda() -> quantize_4bit -> 4-bit on GPU.
                new_layer.weight = bnb.nn.Params4bit(
                    source_weight.data.clone().contiguous(),
                    requires_grad=False,
                    quant_type="nf4",
                )

            if has_bias:
                b_key = f"{full_name}.bias"
                source_bias = state_dict.pop(b_key, None)
                if source_bias is not None:
                    new_layer.bias = torch.nn.Parameter(
                        source_bias.data.clone().contiguous().to(torch.bfloat16)
                    )

            new_layer = new_layer.to(device)

            if getattr(new_layer.weight, "quant_state", None) is None:
                raise RuntimeError(
                    f"[{full_name}] Params4bit was NOT quantized "
                    f"(weight on {new_layer.weight.device}). "
                    f"Check bitsandbytes version / source tensor placement."
                )

            setattr(module, name, new_layer)
        else:
            load_and_compress_layer_by_layer(
                child, state_dict, device, full_name + "."
            )
                        
def jetson_get_moshi_lm(weight_path, copy_missing_weights=False, device='cpu', dtype=torch.bfloat16, cpu_offload=False):
    resolved_weight_path = os.path.abspath(weight_path)
    checkpoint_size_gb = os.path.getsize(resolved_weight_path) / (1024 ** 3)
    checkpoint_is_zip = zipfile.is_zipfile(resolved_weight_path)

    print("\n[JETSON] 1. Reading weights via SSD memory-map (System RAM stays at ~0 GB)...")
    print(
        "[JETSON][RAM] checkpoint: "
        f"path={resolved_weight_path} "
        f"size={checkpoint_size_gb:.2f} GB "
        f"zipfile={checkpoint_is_zip}"
    )
    _print_memory_snapshot("before torch.load", "cuda:0")

    ckpt = torch.load(resolved_weight_path, map_location='cpu', mmap=True)
    _print_memory_snapshot("after torch.load(mmap=True)", "cuda:0")
    if not checkpoint_is_zip:
        print(
            "[JETSON][RAM][WARN] checkpoint is not zipfile-serialized; "
            "torch mmap may be ineffective for this file format."
        )

    state_dict = ckpt['state_dict']
    
    # THE FIX: Ensure the skeleton matches the 16-codebook architecture
    lm_kwargs = loaders._lm_kwargs.copy()
    lm_kwargs["dep_q"] = 16 
    if 'config_override' in ckpt and ckpt['config_override']:
        lm_kwargs.update(ckpt['config_override'])
        
    print("[JETSON] 2. Building empty Moshi skeleton...")
    model = LMModel(device='cpu', dtype=dtype, **lm_kwargs)
    
    target_device = device if device != 'cpu' else 'cuda:0'
    print("[JETSON] 3. Compressing weights to 4-bit VRAM layer-by-layer...")
    load_and_compress_layer_by_layer(model, state_dict, device=target_device)
    _print_state_dict_remaining(state_dict, top_k=20)
    _print_linear_and_parameter_audit(model, threshold_mb=1.0, top_k=30)

    _print_memory_snapshot("after layer-by-layer load (pre-cleanup)", target_device)

    # Release checkpoint-owned CPU references before inference starts.
    del state_dict
    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _print_memory_snapshot("after ckpt/state_dict cleanup", target_device)
    
    print("[JETSON] Model successfully compressed and loaded into VRAM!\n")
    model.eval()
    return model

loaders.get_moshi_lm = jetson_get_moshi_lm

# --- 2. THE PROFILER ---
BASE = "."
# THE FIX: Point to the final merged file
MODEL = "bmo_jetson_ready.pt" 
MIMI = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOK = "tokenizer_spm_32k_3.model"
VOICE = "bmo_621.wav"
VOICE_DIR = BASE
INPUT_WAV = "silence.wav"
OUT_DIR = "outputs/profiler_run"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "Explain airplane flight in detail. Please provide a very long and detailed explanation."
GPU_ID = "1"


def print_gpu_process_snapshot(gpu_id):
    smi_cmd = (
        f"nvidia-smi --id={gpu_id} "
        "--query-compute-apps=pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    try:
        out = subprocess.check_output(smi_cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            print(f"[JETSON][GPU] active compute processes on GPU {gpu_id}:")
            for line in out.splitlines():
                print(f"[JETSON][GPU]   {line.strip()}")
        else:
            print(f"[JETSON][GPU] no active compute processes on GPU {gpu_id} before run")
    except Exception as exc:
        print(f"[JETSON][GPU][WARN] could not query nvidia-smi process list: {exc}")

keep_monitoring = True
peak_vram_mb = 0
peak_ram_mb = 0

def monitor_memory(pid):
    global keep_monitoring, peak_vram_mb, peak_ram_mb
    try:
        process = psutil.Process(pid)
    except:
        return
    while keep_monitoring:
        try:
            ram_mb = process.memory_info().rss / (1024 * 1024)
            if ram_mb > peak_ram_mb: peak_ram_mb = ram_mb
            
            smi_cmd = f"nvidia-smi --id={GPU_ID} --query-gpu=memory.used --format=csv,nounits,noheader"
            res = subprocess.check_output(smi_cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            vram_mb = int(res.strip())
            if vram_mb > peak_vram_mb: peak_vram_mb = vram_mb
        except:
            pass
        time.sleep(0.1)

print("\n=== BMO JETSON DEPLOYMENT PROFILER ===")
print(f"Target Model : {MODEL}")
print(f"Target GPU   : {GPU_ID}")
print("======================================\n")

sys.argv = [
    "moshi.offline",
    "--moshi-weight", MODEL,
    "--mimi-weight", MIMI,
    "--tokenizer", TOK,
    "--voice-prompt", VOICE,
    "--voice-prompt-dir", VOICE_DIR,
    "--input-wav", INPUT_WAV,
    "--text-prompt", PROMPT,
    "--output-wav", f"{OUT_DIR}/jetson_profiler.wav",
    "--output-text", f"{OUT_DIR}/jetson_profiler.json",
    "--device", "cuda:0"
]

start_time = time.time()
monitor_thread = threading.Thread(target=monitor_memory, args=(os.getpid(),))
monitor_thread.start()

print(
    "[JETSON] compile controls: "
    f"NO_TORCH_COMPILE={os.environ.get('NO_TORCH_COMPILE')} "
    f"NO_CUDA_GRAPH={os.environ.get('NO_CUDA_GRAPH')} "
    f"torch_compile_lazy={moshi_compile_utils.torch_compile_lazy} "
    f"gating_wrapped={hasattr(moshi_gating.gating_forward_kernel, '__wrapped__')} "
    f"gating_forward={ActivationGating.forward.__name__}"
)
print_gpu_process_snapshot(GPU_ID)

moshi.offline.main()
_print_memory_snapshot("after offline.main", "cuda:0")

keep_monitoring = False
monitor_thread.join()
elapsed_time = time.time() - start_time

print("\n=== JETSON PROFILING RESULTS ===")
print(f"Generation Time : {elapsed_time:.2f} seconds")
print(f"Peak System RAM : {peak_ram_mb / 1024:.2f} GB")
print(f"Peak GPU VRAM   : {peak_vram_mb / 1024:.2f} GB")
print("================================\n")

if (peak_vram_mb / 1024) < 7.5 and (peak_ram_mb / 1024) < 7.5:
    print("[SUCCESS] The deployment pipeline is safe for the Jetson Orin Nano!")
else:
    print("[WARNING] The footprint exceeds 8GB. Review KV cache limits.")