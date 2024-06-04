import torch
import torch.nn as nn
from torch_geometric.nn import DMoNPooling

class BatchedDmonPool(nn.Module):
    def __init__(self, input_size, k, dropout=0.3) -> None:
        super(BatchedDmonPool, self).__init__()
        self.dmon = DMoNPooling(input_size, k, dropout=dropout)
        
    def forward(self, x, adj):
        b = x.size(0)
        batches = []
        coarse_adj = 0
        _spectral_loss = 0
        _ortho_loss = 0
        _cluster_loss = 0
        for temporal_graph in x:
            cluster_assignment, pooled_node_feature_matrix, coarse_sym_normed_adj, spectral_loss, ortho_loss, cluster_loss = self.dmon(temporal_graph, adj)
            _spectral_loss += spectral_loss / b
            _ortho_loss += ortho_loss / b
            _cluster_loss += cluster_loss / b
            coarse_adj += coarse_sym_normed_adj / b
            batches.append(pooled_node_feature_matrix)
        return torch.stack(batches), coarse_adj.mean(dim=0), _spectral_loss, _ortho_loss, _cluster_loss