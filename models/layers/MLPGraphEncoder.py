# Bipartite graph support?
from typing import Callable, Optional, Union
import torch
import einops
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.nn.models import MLP
from tsl.nn.blocks import ResidualMLP
from torch_geometric.nn.norm import GraphNorm

class MLPGraphEncoder(MessagePassing):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 edge_in_channels,
                 edge_hidden_channels,
                 edge_out_channels,
                 n_node_layers=2,
                 n_edge_layers=2,
                 aggr: str = 'add',
                 **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        # MLP(
        #     in_channels = in_dim_node + in_dim_edge,
        #     out_channels = in_dim_node,
        #     hidden_channels = hidden_dim,
        #     num_layers = hidden_layers,
        #     norm=None
        # )
        self.edge_mlp = ResidualMLP(
                input_size=edge_in_channels,
                hidden_size=edge_hidden_channels,
                output_size=edge_out_channels,
                n_layers=n_edge_layers,
                activation='leaky_relu',
                dropout=0.3,
                parametrized_skip=True,
            )
        
        self.node_mlp = ResidualMLP(
                input_size=in_channels,
                hidden_size=hidden_channels,
                output_size=out_channels,
                n_layers=n_node_layers,
                activation='leaky_relu',
                dropout=0.3,
                parametrized_skip=True,
            )
        
        self.output_projection = ResidualMLP(
                input_size=out_channels*2 + edge_out_channels,
                hidden_size=hidden_channels,
                output_size=out_channels,
                n_layers=n_node_layers,
                activation='leaky_relu',
                dropout=0.3,
                parametrized_skip=True,
            )
        
        self.graph_norm = GraphNorm(out_channels, 1e-7)
        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # self.edge_mlp.reset_parameters()
        # self.node_mlp.reset_parameters()
        # self.output_projection.reset_parameters()
    
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        x_in = x[0]
        x_out = x[1]
        
        # b = x_in.size(0)
        
        x_in = checkpoint(self.node_mlp,x_in)
        x_out = checkpoint(self.node_mlp,x_out)
        edge_attr = self.edge_updater(edge_index, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=(x_in, x_out), edge_attr=edge_attr, size=None)
        # output = checkpoint(self.output_projection,out)
        output = out
        return output, edge_attr
        # for_gn = einops.rearrange(output,'b n f -> (b n) f')
        # normed = self.graph_norm(for_gn)
        # return einops.rearrange(normed, '(b n) f -> b n f', b=b)
        
    def message(self, x_i, x_j, edge_attr):
        batch_size = x_i.size(0)
        batched_edge_attr = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
        out = torch.cat([x_j, x_i, batched_edge_attr], dim=-1)
        return checkpoint(self.output_projection,out)
    
    def edge_update(self, edge_attr):
        return checkpoint(self.edge_mlp,edge_attr)
        
        
        