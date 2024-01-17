from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor, set_diag
from einops import repeat

from tsl.nn.functional import sparse_softmax

import snntorch as snn


class GSATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dim: int = -2,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        return_last: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=dim, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.return_last = return_last

        if self.concat:
            self.head_channels = self.out_channels // self.heads
            assert self.head_channels * self.heads == self.out_channels, \
                "`out_channels` must be divisible by `heads`."
        else:
            self.head_channels = self.out_channels

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        # if isinstance(in_channels, int):
        self.synaptic_src = snn.Synaptic(
            alpha=0.9,
            beta=0.8,
            learn_alpha=True,
            learn_beta=True,
            learn_threshold=True,
            # init_hidden=True,
            # output=True
        )
        self.lin_src = Linear(in_channels,
                                heads * self.head_channels,
                                bias=False,
                                weight_initializer='glorot')
        self.lin_dst = self.lin_src
        self.synaptic_dist = snn.Synaptic(
            alpha=0.9,
            beta=0.8,
            learn_alpha=True,
            learn_beta=True,
            learn_threshold=True,
            # init_hidden=True,
            # output=True
        )
        # else:
        #     self.lin_src = Linear(in_channels[0],
        #                           heads * self.head_channels,
        #                           False,
        #                           weight_initializer='glorot')
        #     self.lin_dst = Linear(in_channels[1],
        #                           heads * self.head_channels,
        #                           False,
        #                           weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, self.head_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, self.head_channels))

        if edge_dim is not None:
            self.edge_synaptic = snn.Synaptic(
                alpha=0.9,
                beta=0.8,
                learn_alpha=True,
                learn_beta=True,
                learn_threshold=True,
                # init_hidden=True,
                # output=True,
            )
            self.lin_edge = Linear(edge_dim,
                                   heads * self.head_channels,
                                   bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(
                torch.Tensor(1, heads, self.head_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.head_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                edge_attr: OptTensor = None,
                need_weights: bool = False):
        """"""
        node_dim = self.node_dim
        self.node_dim = (node_dim + x.dim()) if node_dim < 0 else node_dim
        b, t, n, c = x.size()

        N, H, C = n, self.heads, self.head_channels
        if not self.return_last:
            output = []
        syn, mem = self.synaptic_src.init_synaptic()
        if self.edge_synaptic is not None:
            syn_e, membrane_e = self.edge_synaptic.init_synaptic()
        # syn, membrane_pot = synaptic.init_synaptic()
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        # if isinstance(x, Tensor):
        # print("x size", x.size())
        for timestep in range(t):
            data_at_time = x[:, timestep, : , :]
            spike, syn, mem = self.synaptic_src(data_at_time, syn, mem)
            x_src = x_dst = self.lin_src(mem).view(*data_at_time.shape[:-1], H, C)
            # else:  # Tuple of source and target node features:
            #     x_src, x_dst = x
            #     x_src = self.lin_src(x_src).view(*x_src.shape[:-1], H, C)
            #     if x_dst is not None:
            #         x_dst = self.lin_dst(x_dst).view(*x_dst.shape[:-1], H, C)

            x_node_features = (x_src, x_dst)

            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)

            if self.add_self_loops:
                if isinstance(edge_index, Tensor):
                    edge_index, edge_attr = remove_self_loops(
                        edge_index, edge_attr)
                    edge_index, edge_attr = add_self_loops(
                        edge_index,
                        edge_attr,
                        fill_value=self.fill_value,
                        num_nodes=N)
                elif isinstance(edge_index, SparseTensor):
                    if self.edge_dim is None:
                        edge_index = set_diag(edge_index)
                    else:
                        raise NotImplementedError(
                            "The usage of 'edge_attr' and 'add_self_loops' "
                            "simultaneously is currently not yet supported for "
                            "'edge_index' in a 'SparseTensor' form")

            # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
            alpha, syn_e, membrane_e = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, syn_e=syn_e, membrane_e=membrane_e)

            # propagate_type: (x: OptPairTensor, alpha: Tensor)
            out = self.propagate(edge_index, x=x_node_features, alpha=alpha, size=(N, N))

            if self.concat:
                out = out.view(*out.shape[:-2], self.out_channels)
            else:
                out = out.mean(dim=-2)

            if self.bias is not None:
                out += self.bias

            if need_weights:
                # alpha rearrange: [... e ... h] -> [e ... h]
                alpha = torch.movedim(alpha, self.node_dim, 0)
                if isinstance(edge_index, Tensor):
                    alpha = (edge_index, alpha)
                elif isinstance(edge_index, SparseTensor):
                    alpha = edge_index.set_value(alpha, layout='coo')
            else:
                alpha = None

            self.node_dim = node_dim
            if not self.return_last:
                output.append(out)
            else:
                output = out

        if self.return_last:
            return repeat(output, 'b n f -> b t n f', t=1), alpha
        else:
            return torch.stack(output, dim=1), alpha

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int], membrane_e=None, syn_e=None) -> Tensor:
        """"""
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            # edge_attr = self.lin_edge(edge_attr)
            spike, syn, mem = self.edge_synaptic(edge_attr, syn_e, membrane_e)
            edge_attr = self.lin_edge(mem)
            edge_attr = edge_attr.view(-1, self.heads, self.head_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            shape = [1] * (alpha.ndim - 1) + [self.heads]
            shape[self.node_dim] = alpha_edge.size(0)
            alpha = alpha + alpha_edge.view(shape)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = sparse_softmax(alpha,
                               index,
                               num_nodes=size_i,
                               ptr=ptr,
                               dim=self.node_dim)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha, syn, mem

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        """"""
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
