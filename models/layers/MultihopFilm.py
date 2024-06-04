import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, RENet, MixHopConv, MessagePassing, MessageNorm
from torch_geometric.nn.norm import MessageNorm
from typing import Any
from torch.utils.checkpoint import checkpoint
import einops

class MultiHopFiLM(MessagePassing):
    def __init__(self, in_channels, out_channels, n_iterations, act=nn.LeakyReLU()):
        super(MultiHopFiLM,self).__init__()
        self.out_channels = out_channels
        self.act = act
        self.lin_skip = nn.Linear(in_channels, out_channels)
        self.film_skip = nn.Linear(in_channels, out_channels * 2)
        
        self.n_iterations = n_iterations
        
        # self.lins = nn.ModuleList([])
        self.films = nn.ModuleList([nn.Linear(out_channels, out_channels*2)])
        self.lin = nn.Linear(in_channels, out_channels)
        for _ in range(n_iterations-1):
            # self.lins.append(nn.Linear(in_channels, out_channels))
            self.films.append(nn.Linear(out_channels, out_channels*2))
            
        # self.gru = nn.GRU(out_channels, out_channels, 1, batch_first=True, dropout=0.3, bidirectional=False)
        self.msg_norm = MessageNorm(True)
        
    def forward(self, x, edge_index):
        # FiLM residual
        beta, gamma = self.film_skip(x).split(self.out_channels, dim=-1)
        skip = self.act(gamma * self.lin_skip(x) + beta)
        
        # h = None
        x = self.lin(x)
        for film in self.films:
            beta, gamma = film(x).split(self.out_channels, dim=-1)
            # x = gamma * x + beta
            
            msg = self.propagate(edge_index, x=x, beta=beta, gamma=gamma)
            
            skip = skip + self.msg_norm(x, msg)
            
            # b = msg.size(0)
            # stacked = einops.rearrange(msg, 'b t n f -> (b n) t f')
            # res, h = checkpoint(self.gru, stacked,h, use_reentrant=True)
            # x = einops.rearrange(res, '(b n) t f -> b t n f', b=b)
        return x
    
    def message(self, x_j, beta_i, gamma_i):
        return self.act(gamma_i * x_j + beta_i)