import torch
import torch.nn as nn
import einops
import math
from torch_geometric.nn import MessageNorm
from tsl.nn.layers.base import Dense


class DenseFiLMConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.LeakyReLU(), order=1):
        super(DenseFiLMConv, self).__init__()
        self.act = act
        self.out_channels = out_channels
        self.order = order
        
        self.lin_skip = nn.Linear(in_channels, out_channels)
        self.film_skip = nn.Linear(in_channels, out_channels * 2)
        
        self.lins = nn.ModuleList([])
        self.films = nn.ModuleList([])
    
        self.lin = nn.Linear(in_channels, out_channels)
        self.film = nn.Linear(in_channels, out_channels * 2)
        self.msg_norm = MessageNorm(True)
        
        self.reset_parameters()
        
        
    def forward(self, x, adj):
        beta, gamma = self.film_skip(x).split(self.out_channels, dim=-1)
        skip = self.lin_skip(x)
        skip = gamma * skip + beta
        skip = self.act(skip)
        
        # # message passing
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        beta, gamma = self.film(x).split(self.out_channels, dim=-1)
        x = self.lin(x)
        x = gamma * x + beta
        
        for _ in range(self.order):
            msg = torch.einsum('btnf, mn -> btmf', x, adj)
            x = self.msg_norm(x, msg)
        x = self.act(x)
        return x + skip
        
        
    def reset_parameters(self):
        self.reset_weights(self.lin_skip)
        self.reset_weights(self.film_skip)
        self.reset_weights(self.lin)
        self.reset_weights(self.film)
            
    def reset_weights(self, layer):
        torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        
    def reset_mlp(self, mlp):
        for layer in mlp:
            if isinstance(layer, (nn.Sequential)):
                self.reset_mlp(layer)
            if isinstance(layer, Dense):
                self.reset_weights(layer.affinity)
            if hasattr(layer, 'weight'):
                self.reset_weights(layer)
    