from typing import Optional

import torch
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
        n_iterations = 2,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout
        self.out_channels = out_channels
        self.rearrange_pre = Rearrange('t n f -> n t f')
        self.rearrange_post = Rearrange('n t f -> t n f')
        self.n_iterations = n_iterations

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(in_channels, out_channels, False)
        self.lin2_2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        x = self.rearrange_pre(x.squeeze())
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        # propagate_type: (x: Tensor, alpha: Tensor)
        out = x
        for i in range(self.n_iterations):
            out = self.propagate(edge_index, x=out, alpha=alpha)
        out = out + self.bias
        
        out = self.rearrange_post(out)
        return out.unsqueeze(0)

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        timesteps = x_j.size(-2)
        edge_att_dim = edge_attr.size(-1)
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr.unsqueeze(1).expand(-1, timesteps, edge_att_dim)], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        if x_j.size(-1) == self.out_channels:
            return self.lin2_2(x_j) * alpha.unsqueeze(-1)
        return self.lin2(x_j) * alpha.unsqueeze(-1)
    
from typing import Optional

import torch
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class GATEConv2(MessagePassing):
    # a variant for that fucked up GRU
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
        n_iterations = 2,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout
        self.out_channels = out_channels
        # self.rearrange_pre = Rearrange('t n f -> n t f')
        # self.rearrange_post = Rearrange('n t f -> t n f')
        self.n_iterations = n_iterations

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(in_channels, out_channels, False)
        self.lin2_2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        # x - b, n , f
        # x = self.rearrange_pre(x.squeeze())
        x = x.squeeze()
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        # propagate_type: (x: Tensor, alpha: Tensor)
        out = x
        for i in range(self.n_iterations):
            out = self.propagate(edge_index, x=out, alpha=alpha)
        out = out + self.bias
        
        # out = self.rearrange_post(out)
        return out.unsqueeze(0)

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        timesteps = x_j.size(-2)
        edge_att_dim = edge_attr.size(-1)
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr.unsqueeze(-1)], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        if x_j.size(-1) == self.out_channels:
            return self.lin2_2(x_j) * alpha.unsqueeze(-1)
        return self.lin2(x_j) * alpha.unsqueeze(-1)