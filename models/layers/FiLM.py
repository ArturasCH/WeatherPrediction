import torch
import math
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, in_features, out_features):
        super(FiLM, self).__init__()
        # self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features)
        self.film = nn.Linear(in_features, out_features * 2)
        self.act = nn.LeakyReLU()
        
        self.reset_parameters()
        
    def forward(self, x):
        gamma, beta = self.film(x).split(self.out_features, dim=-1)
        x = self.lin(x)
        return self.act(gamma * x + beta)
    
    def reset_parameters(self):
        self.reset_weights(self.lin)
        self.reset_weights(self.film)
    
    def reset_weights(self, layer):
        torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))