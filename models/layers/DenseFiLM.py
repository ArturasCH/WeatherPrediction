import torch
import torch.nn as nn
import einops
import math
from torch_geometric.nn import MessageNorm
from tsl.nn.blocks import ResidualMLP
from tsl.nn.layers.base import Dense
from torch.utils.checkpoint import checkpoint

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
        # std, mean = torch.std_mean(skip, dim=-1, keepdim=True, unbiased=False)
        # skip = gamma * ((skip - (mean * theta)) / std) + beta
        skip = gamma * skip + beta
        skip = self.act(skip)
        
        # # message passing
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        beta, gamma = self.film(x).split(self.out_channels, dim=-1)
        x = self.lin(x)
        # std, mean = torch.std_mean(x, dim=-1, keepdim=True, unbiased=False)
        # standardized = ((x - (mean * theta)) / std)
        x = gamma * x + beta
        # x = self.act(x)
        # x = torch.einsum('btnf, an -> btaf', x, adj)
        
        # x = einops.rearrange(x, 'b t n f -> b f n t')
        # x = torch.einsum('ncvl, wv -> ncwl', x, adj)
        
        # return einops.rearrange(x, 'b c n s -> b s n c') + skip
        # return einops.rearrange(x, 'b c n s -> b s n c')
        
        # no reorder
        for _ in range(self.order):
            msg = torch.einsum('btnf, mn -> btmf', x, adj)
            x = self.msg_norm(x, msg)
        x = self.act(x)
        # std, mean = torch.std_mean(x, dim=-2, keepdim=True, unbiased=False)
        # gamma, beta = self.norm_lin(x).split(self.out_channels, dim=-1)
        # x = gamma * ((x - mean) / std) + beta
        # x = checkpoint(self.mlp_out, x, use_reentrant=True)
        return x + skip
        
        
    def reset_parameters(self):
        self.reset_weights(self.lin_skip)
        self.reset_weights(self.film_skip)
        self.reset_weights(self.lin)
        self.reset_weights(self.film)
        # self.reset_mlp(self.mlp_out.layers)
        # self.reset_mlp(self.mlp_out.skip_connections)
        # self.reset_weights(self.mlp_out.readout)
        # self.reset_weights(self.norm_lin)
        # for film, lin in zip(self.films, self.lins):
        #     self.reset_weights(film)
        #     self.reset_weights(lin)
            
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
    
class DenseFiLMConvTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.LeakyReLU(), order=2):
        super(DenseFiLMConvTemporal, self).__init__()
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
        # std, mean = torch.std_mean(skip, dim=-1, keepdim=True, unbiased=False)
        # skip = gamma * ((skip - (mean * theta)) / std) + beta
        skip = gamma * skip + beta
        skip = self.act(skip)
        
        # # message passing
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        beta, gamma = self.film(x).split(self.out_channels, dim=-1)
        x = self.lin(x)
        # std, mean = torch.std_mean(x, dim=-1, keepdim=True, unbiased=False)
        # standardized = ((x - (mean * theta)) / std)
        x = gamma * x + beta
        # x = self.act(x)
        # x = torch.einsum('btnf, an -> btaf', x, adj)
        
        # x = einops.rearrange(x, 'b t n f -> b f n t')
        # x = torch.einsum('ncvl, wv -> ncwl', x, adj)
        
        # return einops.rearrange(x, 'b c n s -> b s n c') + skip
        # return einops.rearrange(x, 'b c n s -> b s n c')
        
        # no reorder
        for _ in range(self.order):
            # msg = torch.einsum('ncvl, wv -> ncwl', x, adj) #message passing with last time dim
            msg = torch.einsum('nvcl, wv -> nwcl', x, adj) #message passing with last time dim alt
            x = self.msg_norm(x, msg)
        x = self.act(x)
        # std, mean = torch.std_mean(x, dim=-2, keepdim=True, unbiased=False)
        # gamma, beta = self.norm_lin(x).split(self.out_channels, dim=-1)
        # x = gamma * ((x - mean) / std) + beta
        # x = checkpoint(self.mlp_out, x, use_reentrant=True)
        return x + skip
        
        
    def reset_parameters(self):
        self.reset_weights(self.lin_skip)
        self.reset_weights(self.film_skip)
        self.reset_weights(self.lin)
        self.reset_weights(self.film)
        # self.reset_mlp(self.mlp_out.layers)
        # self.reset_mlp(self.mlp_out.skip_connections)
        # self.reset_weights(self.mlp_out.readout)
        # self.reset_weights(self.norm_lin)
        # for film, lin in zip(self.films, self.lins):
        #     self.reset_weights(film)
        #     self.reset_weights(lin)
            
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
    
