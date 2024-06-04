import torch
import math
import torch.nn as nn
from torch_geometric.nn.inits import ones, zeros

class GraphNorm(nn.Module):
    def __init__(self, feature_size):
        super(GraphNorm, self).__init__()
        self.gamma = nn.Parameter(torch.empty(feature_size))
        self.beta = nn.Parameter(torch.empty(feature_size))
        self.alpha = nn.Parameter(torch.empty(feature_size))
        
        self.reset_parameters()
        
    def forward(self, x):
        # x = b t n f
        std, mean = torch.std_mean(x, dim=-2, keepdim=True, unbiased=False)
        
        return self.gamma * ((x - (self.alpha * mean)) / std) + self.beta
        
    def reset_parameters(self):
        ones(self.gamma)
        zeros(self.beta)
        ones(self.alpha)