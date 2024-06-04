import torch

class UnbounTanh(torch.nn.Module):
    def __init__(self, scale_factor=0.1, learnable=False):
        super(UnbounTanh, self).__init__()
        if learnable:
            self.scale_factor = torch.nn.Parameter(torch.tensor(scale_factor))
        else:
            self.scale_factor = torch.tensor(scale_factor)
            
            
    def forward(self, x):
        return torch.tanh(x) + (x * self.scale_factor)
    