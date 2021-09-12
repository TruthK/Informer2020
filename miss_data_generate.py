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
import numpy as np

from utils.tools import *
import pandas as pd
import warnings
import torch
import random
import numpy as np

df = pd.read_csv('data/electricity/electricity.csv')
cols_name=df.columns.values.tolist()
# rnd = np.random.RandomState(666)  # 定义一个随机数种子
# n_samples_missing = int(df.shape[0] * 0.1)  # 大概有30%的数据将要被我们擦除掉。计算出要擦除的个数
# print('we will drop {} values.'.format(n_samples_missing))

# missing_samples_list = rnd.randint(low=0, high=df.shape[0] - 1, size=n_samples_missing)

# res = df[df.columns[1:]].values
# print(missing_samples_list)
# res[missing_samples_list] = np.NaN
n = df[df.columns[1:]].applymap(lambda x: x if random.randint(0, 100) > 10 else np.NaN)
# res = pd.DataFrame(res)
df_res = pd.concat([df[df.columns[0]], n], axis=1)
df_res.columns = cols_name
print(df_res)
df_res.to_csv('data/electricity/electricity_miss_10.csv', index=False)
