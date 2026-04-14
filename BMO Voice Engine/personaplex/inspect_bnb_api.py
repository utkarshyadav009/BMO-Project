import inspect
import torch
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit

print('bnb version:', bnb.__version__)
print('Params4bit has from_prequantized:', hasattr(Params4bit, 'from_prequantized'))
print('Params4bit.__new__ sig:', inspect.signature(Params4bit.__new__))
if hasattr(Params4bit, 'from_prequantized'):
    print('Params4bit.from_prequantized sig:', inspect.signature(Params4bit.from_prequantized))

m = bnb.nn.Linear4bit(8, 4, bias=False, quant_type='nf4', compute_dtype=torch.bfloat16)
print('Linear4bit state_dict keys:', list(m.state_dict().keys()))
print('weight type:', type(m.weight))
print('weight.quant_state attr exists:', hasattr(m.weight, 'quant_state'))
