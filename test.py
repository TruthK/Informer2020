import math
import torch.nn.functional as F

import torch

max_len = 5000
d_model = 512
pe = torch.zeros(max_len, d_model).float()
pe.require_grad = False

position = torch.arange(0, max_len).float().unsqueeze(1)
div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

pe = pe.unsqueeze(0)

v = torch.randn(4, 4)
v = torch.flatten(torch.sum(v,0))
print(v.shape)
repeat_time = int(d_model / v.numel()) + 1
v = v.repeat(repeat_time)[:d_model]
print(v)
print(v/torch.max(v))
