import torch
import math
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tsl.nn.blocks import MLP
from torch_geometric.nn.norm import GraphNorm

from model.layers.DenseFiLM import DenseFiLMConv
from model.layers.GraphNorm import GraphNorm

# CompiledModule = torch.compile(model, mode='reduce-overhead')
class DenseGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=8, k=3, activation=nn.ReLU()):
        super(DenseGraphConv, self).__init__()
        self.graph_conv = DenseFiLMConv(in_channels, out_channels, act=activation)
        self.norm = GraphNorm(out_channels)
        
        
    def forward(self, x, edge_index, edge_attr=None,):
        y = self.norm(checkpoint(self.graph_conv, x, edge_index, use_reentrant=True))
        return torch.cat([y, x], dim=-1)
    

class DenseBlock(nn.Module):
    # current training 32/2
    # try out 64/3
    def __init__(self,
                 in_channels,
                 out_channels,
                 growth_rate=64,
                 edge_dim=8,
                 num_relations=3,
                 n_blocks=3,
                 activation=nn.ReLU()
                 ):
        super(DenseBlock, self).__init__()
        
        self.dense_convs = nn.ModuleList([
            DenseFiLMConv(in_channels, out_channels, act=activation)
            ])
        size = out_channels
        self.norm = GraphNorm(out_channels)
        self.reduction = MLP(
            input_size=size,
            hidden_size=size*2,
            output_size=out_channels,
            exog_size=None,
            n_layers=3,
            activation='leaky_relu',
            dropout=0.0
            
        )
        
        self.reset_parameters()
        
        
        
    def forward(self, x, edge_index, edge_attr=None):
        
        for conv in self.dense_convs:
            x=checkpoint(conv,x,edge_index, use_reentrant=True)
        out = (checkpoint(self.reduction, x, use_reentrant=True))
        return self.norm(out)
    
    def reset_parameters(self):
        self.reduction.reset_parameters()

    def reset_weights(self, mlp):
        for layer in mlp:
            if hasattr(layer, 'weight'):
                torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                
    def reset_mlp(self, mlp):
        for layer in mlp:
            if isinstance(layer, (nn.Sequential)):
                self.reset_mlp(layer)
            if isinstance(layer, Dense):
                self.reset_weights(layer.affinity)
            if hasattr(layer, 'weight'):
                self.reset_weights(layer)