import torch
import torch.nn as nn

class ReGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReGLU, self).__init__()
        self.glu_weights = nn.Linear(in_channels, out_channels * 2)
        
        
    def forward(self, x):
        out, gate = torch.tensor_split(self.glu_weights(x), 2, dim=-1)
        
        return out + torch.relu(gate)