import json, os, torch, bitsandbytes as bnb
from moshi.models.lm import LMModel
import torch.nn as nn

# load config
with open('bmo_config.json') as f:
    cfg = json.load(f)
    cfg.pop('model_type', None)

# build on meta
with torch.device('meta'):
    model = LMModel(**cfg)

# replace Linear with bnb Linear4bit (same as patched_get_moshi)
SKIP_MODULES = {"text_emb", "audio_emb", "out_norm", "audio_heads", "text_heads", "extra_heads", "mimi"}

def _swap(module, prefix=''):
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if any(skip in full_name for skip in SKIP_MODULES):
            continue
        if isinstance(child, nn.Linear):
            int4_layer = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=child.bias is not None, compute_dtype=torch.bfloat16, quant_type='nf4')
            setattr(module, child_name, int4_layer)
        else:
            _swap(child, full_name)
_swap(model)

ckpt_path = 'bmo_mixed_precision.pt'
print('Loading checkpoint', ckpt_path)
ckpt = torch.load(ckpt_path, map_location='cpu')
if 'state_dict' in ckpt:
    sd = ckpt['state_dict']
else:
    sd = ckpt

# Prune quant-metadata: keep keys that end exactly with '.weight' or '.bias', and also keep any keys that are not quant-meta
pruned = {}
for k, v in sd.items():
    if k.endswith('.weight') or k.endswith('.bias'):
        pruned[k] = v
    else:
        # keep non-quant-meta keys (like emb.*, norm alphas)
        if not (k.endswith('.absmax') or k.endswith('quant_map') or 'quant_state.bitsandbytes' in k or 'nested_absmax' in k or 'nested_quant_map' in k):
            pruned[k] = v

print('Original keys:', len(sd), 'Pruned keys:', len(pruned))
# attempt load
try:
    model.load_state_dict(pruned, assign=True)
    print('load_state_dict succeeded (assign=True)')
except Exception as e:
    print('load_state_dict failed:')
    print(e)

