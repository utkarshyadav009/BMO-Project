import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
from moshi.models.lm import LMModel

with open('bmo_config.json', 'r') as f:
    cfg = json.load(f)
cfg.pop('model_type', None)

with torch.device('meta'):
    model = LMModel(**cfg)

SKIP_MODULES = {"text_emb", "audio_emb", "out_norm", "audio_heads", "text_heads", "extra_heads", "mimi"}

def _swap(module, prefix=''):
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if 'depformer' in full_name or any(skip in full_name for skip in SKIP_MODULES):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, child_name, bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type='nf4'
            ))
        else:
            _swap(child, full_name)

_swap(model)

sd = torch.load('bmo_mixed_precision.pt', map_location='cpu', mmap=True)

# Identify quantized weights by metadata marker
quant_suffix = '.weight.quant_state.bitsandbytes__nf4'
quant_bases = [k[:-len('.quant_state.bitsandbytes__nf4')] for k in sd.keys() if k.endswith(quant_suffix)]

print('quantized weight groups:', len(quant_bases))

# Build dense dict by removing quantized metadata + packed weights for those bases
dense_sd = {}
quant_base_set = set(quant_bases)
for k, v in sd.items():
    # drop quant metadata
    if k.endswith('.absmax') or k.endswith('.quant_map') or k.endswith('.nested_absmax') or k.endswith('.nested_quant_map') or k.endswith('.quant_state.bitsandbytes__nf4'):
        continue
    # drop packed weight tensors for quantized modules; these will be handled manually
    if k.endswith('.weight') and k in quant_base_set:
        continue
    dense_sd[k] = v

print('dense keys to load:', len(dense_sd))

# load dense first
incompat = model.load_state_dict(dense_sd, strict=False, assign=True)
print('dense missing:', len(incompat.missing_keys), 'unexpected:', len(incompat.unexpected_keys))

# assign prequantized weights per module
modules = dict(model.named_modules())
loaded_prequant = 0
for base in quant_bases:
    module_name = base[:-len('.weight')]
    module = modules.get(module_name)
    if module is None or not isinstance(module, bnb.nn.Linear4bit):
        continue
    packed = sd[base]
    stats = {}
    prefix = base + '.'
    for sk, sv in sd.items():
        if sk.startswith(prefix):
            stats[sk[len(prefix):]] = sv
    module.weight = Params4bit.from_prequantized(
        packed,
        stats,
        requires_grad=False,
        device='cpu',
        module=module,
    )
    loaded_prequant += 1

print('loaded prequant weights:', loaded_prequant)

# quick materialize test
model.cuda()
print('model.cuda() ok')
