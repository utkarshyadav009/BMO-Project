import torch
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit

assert torch.cuda.is_available(), 'CUDA required'

src = bnb.nn.Linear4bit(16, 8, bias=False, quant_type='nf4', compute_dtype=torch.bfloat16).cuda()
_ = src(torch.randn(1, 16, device='cuda', dtype=torch.bfloat16))
sd = src.state_dict()

w = sd['weight'].clone()
stats_prefixed = {k: v for k, v in sd.items() if k.startswith('weight.')}
stats_unprefixed = {k[len('weight.'):]: v for k, v in stats_prefixed.items()}

for name, stats in [('prefixed', stats_prefixed), ('unprefixed', stats_unprefixed)]:
    dst = bnb.nn.Linear4bit(16, 8, bias=False, quant_type='nf4', compute_dtype=torch.bfloat16).cuda()
    try:
        p = Params4bit.from_prequantized(w, stats, requires_grad=False, device='cuda', module=dst)
        dst.weight = p
        y = dst(torch.randn(1, 16, device='cuda', dtype=torch.bfloat16))
        print(name, 'OK', y.shape)
    except Exception as e:
        print(name, 'FAIL', repr(e))
