import torch
import numpy as np
from typing import Union, List, Optional
from torch import Tensor
from torch import nn
from torch_geometric.utils import dropout_adj, to_undirected
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
# from lib.sgp_preprocessing import sgp_spatial_embedding


class SGPSpatialEncoder(nn.Module):
    def __init__(self,
                 receptive_field,
                 bidirectional,
                 undirected,
                 global_attr,
                 add_self_loops=False):
        super(SGPSpatialEncoder, self).__init__()
        self.receptive_field = receptive_field
        self.bidirectional = bidirectional
        self.undirected = undirected
        self.add_self_loops = add_self_loops
        self.global_attr = global_attr

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(-2)
        out = sgp_spatial_embedding(x,
                                    num_nodes=num_nodes,
                                    edge_index=edge_index,
                                    edge_weight=edge_weight,
                                    k=self.receptive_field,
                                    bidirectional=self.bidirectional,
                                    undirected=self.undirected,
                                    add_self_loops=self.add_self_loops)
        if self.global_attr:
            g = torch.ones_like(x) * x.mean(-2, keepdim=True)
            out.append(g)
        return torch.cat(out, -1)
    
    
def sgp_spatial_embedding(x,
                          num_nodes,
                          edge_index,
                          edge_weight=None,
                          k=2,
                          undirected=False,
                          add_self_loops=False,
                          remove_self_loops=False,
                          bidirectional=False,
                          one_hot_encoding=False,
                          dropout_rate=0.):
    # x [batch, node, features]

    # subsample operator
    edge_index, edge_weight = dropout_adj(edge_index, edge_weight,
                                          p=dropout_rate,
                                          num_nodes=num_nodes)

    # to undirected
    if undirected:
        assert bidirectional is False
        edge_index, edge_weight = to_undirected(edge_index, edge_weight,
                                                num_nodes)

    # get adj
    adj = preprocess_adj(edge_index, edge_weight,
                         num_nodes=num_nodes,
                         gcn_norm=undirected,
                         set_diag=add_self_loops,
                         remove_diag=remove_self_loops)

    if one_hot_encoding:
        ids = torch.eye(num_nodes, dtype=x.dtype, device=x.device)
        ids = ids.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([x, ids], dim=-1)

    # preprocessing of features
    res = [x]
    for _ in range(k):
        x = adj @ x
        res.append(x)

    if bidirectional:
        res_bwd = sgp_spatial_embedding(res[0],
                                        num_nodes,
                                        edge_index[[1, 0]],
                                        edge_weight,
                                        k=k,
                                        undirected=False,
                                        add_self_loops=add_self_loops,
                                        remove_self_loops=remove_self_loops,
                                        bidirectional=False,
                                        one_hot_encoding=False,
                                        dropout_rate=0)
        res += res_bwd[1:]
    return res

def preprocess_adj(edge_index: Adj, edge_weight: OptTensor = None,
                   num_nodes: Optional[int] = None,
                   gcn_norm: bool = False,
                   set_diag: bool = True,
                   remove_diag: bool = False) -> SparseTensor:
    # convert numpy to torch
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
        if edge_weight is not None:
            edge_weight = torch.from_numpy(edge_weight)

    if isinstance(edge_index, Tensor):
        # transpose
        col, row = edge_index
        adj = SparseTensor(row=row, col=col, value=edge_weight,
                           sparse_sizes=(num_nodes, num_nodes))
    elif isinstance(edge_index, SparseTensor):
        adj = edge_index
    else:
        raise RuntimeError("Edge index must be (edge_index, edge_weight) tuple "
                           "or SparseTensor.")

    if set_diag:
        adj = adj.set_diag()
    elif remove_diag:
        adj = adj.remove_diag()

    if gcn_norm:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    else:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        adj = deg_inv.view(-1, 1) * adj

    return adj