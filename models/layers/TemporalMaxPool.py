import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class TemporalMaxPool(nn.Module):
    def __init__(self) -> None:
        super(TemporalMaxPool, self).__init__()
        self.time_to_last = Rearrange('b t n f -> b f n t')
        self.time_to_its_place = Rearrange('b f n t -> b t n f')
        self.pool = torch.nn.MaxPool1d(2,2, ceil_mode=True)
        
    def forward(self, x):
        x = self.time_to_last(x)
        x = self.pool_every_batch(x)
        return self.time_to_its_place(x)
    
    def pool_every_batch(self, x):
        batches = []
        for graph in x:
            batches.append(self.pool(graph))
        return torch.stack(batches)