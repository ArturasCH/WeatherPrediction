import torch
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj, OptTensor

class GATBatched(GATv2Conv):
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        if x.dim() == 2:
            return super(GATBatched, self).forward(x, edge_index, edge_attr)
        
        stack = []
        for graph in range(x.size(0)):
            stack.append(super(GATBatched, self).forward(x[graph], edge_index, edge_attr))
            
        return torch.stack(stack)