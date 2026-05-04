import torch
ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
sd = ckpt.get("state_dict") or ckpt
print("linear_in:", sd.get("transformer.layers.0.gating.linear_in.weight", torch.tensor([])).shape)
print("linear_out:", sd.get("transformer.layers.0.gating.linear_out.weight", torch.tensor([])).shape)
print("linear_in q_weight:", sd.get("transformer.layers.0.gating.linear_in.q_weight", torch.tensor([])).shape)
print("linear_out q_weight:", sd.get("transformer.layers.0.gating.linear_out.q_weight", torch.tensor([])).shape)
