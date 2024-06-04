import copy
from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ModuleList, ReLU

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    PairTensor,
)
from torch_geometric.nn.norm import MessageNorm



class EdgeFiLMConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            act: Optional[Callable] = ReLU(),
            aggr: str = 'max',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.film = nn.Linear(in_channels, out_channels * 2)
        self.lin = nn.Linear(in_channels, out_channels)
        if in_channels != out_channels:
            self.residual_projection = nn.Linear(in_channels, out_channels)
        else:
            self.residual_projection = nn.Identity()
        
        self.out_transform = nn.Linear(in_channels * 2, out_channels)
        self.msg_norm = MessageNorm(True)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.film)
        reset(self.lin)
        reset(self.residual_projection)
        reset(self.out_transform)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        residual = x

        beta, gamma = self.film(x).split(self.out_channels, dim=-1)

        x = self.lin(x)
        x = gamma * x + beta
        msg = self.propagate(edge_index, x=x)
        x =  self.msg_norm(x, msg)

        return x + self.residual_projection(residual)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j - neighbours
        # x_i - source nodes
        
        return self.act(self.out_transform(torch.cat([x_i ,x_j - x_i], dim=-1)))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}')
