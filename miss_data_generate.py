# import torch
# import numpy as np
#
#
# # torch.einsum("bhls,bshd->blhd", A, values)
# def my_batch_mul(tensor_a, tensor_b, adj=None):
#     B, H, L, S = tensor_a.shape
#     b, s, h, d = tensor_b.shape
#     out = torch.empty((B, L, h, d))
#     # 自由索引外循环
#     # 这个例子是 i,j和l
#     for b_item in range(0, B):
#         for l_item in range(0, L):
#             for h_item in range(0, H):
#                 for d_item in range(0, d):
#                     # 求和索引内循环
#                     # 这个例子是 k
#                     sum_result = 0
#                     for s_item in range(0, s):
#                         sum_result += tensor_a[b_item, h_item, l_item, s_item] * tensor_b[
#                             b_item, s_item, h_item, d_item]
#                     out[b_item, l_item, h_item, d_item] = sum_result
#     return out
#
#
# a = torch.randn(32, 8, 96, 96).cuda()
# b = torch.randn(32, 96, 8, 64).cuda()
# torch_ein_out = torch.einsum("bhls,bshd->blhd", a, b)
# out = my_batch_mul(a, b)
#
# print("is np_out == torch_ein_out ?", torch.allclose(torch_ein_out, out))

from utils.tools import *
import pandas as pd
import warnings
import torch
import random

df = pd.read_csv('data/electricity/electricity.csv')
n = df[df.columns[1:]].applymap(lambda x: x if random.randint(0, 100) > 5 else -1)
df = pd.concat([df[df.columns[0]], n], axis=1)
df.to_csv('data/electricity/electricity_miss.csv', index=False)