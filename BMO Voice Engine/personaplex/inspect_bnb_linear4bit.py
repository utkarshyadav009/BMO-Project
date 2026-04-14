import bitsandbytes as bnb
import torch
m = bnb.nn.Linear4bit(1024, 2048, bias=True, compute_dtype=torch.bfloat16, quant_type='nf4')
print('Linear4bit state_dict keys:')
for k, v in m.state_dict().items():
    print(' ', k, getattr(v, 'shape', None), type(v))
print('\nAttributes around weight:')
for attr in ['weight','qweight','scale','scales','zeros','qzeros','qweight_zero_point','weight_dtype','packed']:
    if hasattr(m, attr): print(' ', attr, type(getattr(m, attr)))
