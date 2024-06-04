import torch
import math
from torch.nn import Module, Parameter, init

class LearnableWeight(Module):
    def __init__(self, n_nodes: int, learnable_weight_dim: int) -> None:
        super(LearnableWeight, self).__init__()
        self.weights = Parameter(torch.empty(n_nodes, learnable_weight_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        
    def forward(self, x):
        # batch, time, nodes, learnable_weights
        b,t, _, _  = x.size()
        return torch.cat([x, self.weights.expand(b,t, -1, -1)], dim=-1)