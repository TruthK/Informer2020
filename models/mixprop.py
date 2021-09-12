import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = x.permute(0, 2, 1)
        x = torch.einsum('cwl,vw->cvl', (x, A))
        x = x.transpose(1, 2)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=3, padding=2, dilation=2, padding_mode='circular',
                                   bias=bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.mlp(x).transpose(1, 2)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        # 16 16 2 0.3 0.05
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        x = x.float()
        adj = adj.float()
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class imp(nn.Module):
    def __init__(self, c_in, adj):
        # 16 16 2 0.3 0.05
        super(imp, self).__init__()
        self.adj = adj
        self.mlp = linear(c_in, c_in)
        self.lin = nn.Linear(c_in, c_in, bias=True)

    def forward(self, x):
        self.adj = self.adj.float().to(x.device)
        x = torch.tanh(x)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        x = torch.einsum('cwl,vw->cvl', (x, self.adj))
        x = x.transpose(1, 2)
        return self.lin(x)
