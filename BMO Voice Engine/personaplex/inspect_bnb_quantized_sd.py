import torch
import bitsandbytes as bnb

print('cuda available:', torch.cuda.is_available())
lin = bnb.nn.Linear4bit(16, 8, bias=False, quant_type='nf4', compute_dtype=torch.bfloat16)
# force quantization path by moving to cuda then running one forward
if torch.cuda.is_available():
    lin = lin.cuda()
    x = torch.randn(2, 16, device='cuda', dtype=torch.bfloat16)
    _ = lin(x)

sd = lin.state_dict()
print('state_dict keys:', list(sd.keys()))
for k, v in sd.items():
    print(k, type(v), getattr(v, 'shape', None), getattr(v, 'dtype', None))
