import sys
import torch
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else 'bmo_mixed_precision.pt'
print('Inspecting', path)
state = torch.load(path, map_location='cpu')

if isinstance(state, dict) and 'state_dict' in state:
    sd = state['state_dict']
else:
    sd = state

print('Top-level type:', type(sd))
keys = list(sd.keys())
print('Total keys:', len(keys))
print('\nFirst 40 keys:')
for k in keys[:40]:
    v = sd[k]
    try:
        print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}, size_bytes={getattr(v, 'numel', lambda:None)() if hasattr(v, 'numel') else None}")
    except Exception as e:
        print(f"  {k}: <error getting info: {e}>")

# Count suffixes
suffix_counts = Counter(k.split('.')[-1] for k in keys)
print('\nSuffix counts (most common 20):')
for s, c in suffix_counts.most_common(20):
    print(f"  {s}: {c}")

# Find quant-related keys
quant_keys = [k for k in keys if ('absmax' in k or 'quant_map' in k or 'quant_state' in k or 'bitsandbytes' in k or '.weight' in k and getattr(sd[k], 'ndim', None) == 1)]
print(f'\nFound {len(quant_keys)} quant-related keys (showing up to 60):')
for k in quant_keys[:60]:
    v = sd[k]
    print(f'  {k}: type={type(v)}, shape={getattr(v, "shape", None)}')

# Show shapes distribution for `.weight` keys
weight_shapes = Counter()
for k in keys:
    if k.endswith('.weight'):
        v = sd[k]
        shape = getattr(v, 'shape', None)
        weight_shapes[shape] += 1

print('\nDistinct shapes among .weight keys (top 20):')
for shape, count in weight_shapes.most_common(20):
    print(f'  {shape}: {count}')

print('\nDone')
