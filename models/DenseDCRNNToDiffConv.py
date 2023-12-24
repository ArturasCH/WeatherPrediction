from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

# from tsl.nn.blocks.encoders import DenseDCRNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation


from tsl.nn.layers.graph_convs.dense_graph_conv import DenseGraphConvOrderK
from tsl.nn.layers.recurrent import DenseDCRNNCell

from tsl.nn.blocks.encoders.recurrent.base import RNNBase


class DenseDCRNN(RNNBase):
    """Dense implementation of the Diffusion Convolutional Recurrent Neural
    Network from the paper `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    (Li et al., ICLR 2018).

    In this implementation, the adjacency matrix is dense and the convolution is
    performed with matrix multiplication.

    Args:
        input_size: Size of the input.
        hidden_size: Number of units in the hidden state.
        n_layers: Number of layers.
        k: Size of the diffusion kernel.
        root_weight: Whether to learn a separate transformation for the central
            node.
    """
    _n_states = 1

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 k: int = 2,
                 root_weight: bool = False,
                 device=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.device = device
        rnn_cells = [
            DenseDCRNNCell(input_size if i == 0 else hidden_size,
                           hidden_size,
                           k=k,
                           root_weight=root_weight) for i in range(n_layers)
        ]
        super(DenseDCRNN, self).__init__(rnn_cells, cat_states_layers,
                                         return_only_last_state)

    def forward(self, x: Tensor, adj, h: Optional[Tensor] = None, **kwargs):
        """"""
        support = DenseGraphConvOrderK.compute_support(adj, device=self.device)
        return super(DenseDCRNN, self).forward(x, h=h, support=support)



class DenseDCRNNThenDiffConvModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 temporal_layers: int = 1,
                 gnn_kernel: int = 2,
                 adjacency = None,
                 device=None):
        super(DenseDCRNNThenDiffConvModel, self).__init__()
        self.adjacency = torch.Tensor(adjacency)
        self.device = device
        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        # self.time_nn = RNN(input_size=hidden_size,
        #                    hidden_size=hidden_size,
        #                    n_layers=rnn_layers,
        #                    cell='gru',
        #                    return_only_last_state=True)
        self.time_nn = DenseDCRNN(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  n_layers=temporal_layers,
                                  cat_states_layers=True,
                                  return_only_last_state=True,
                                  device=device)
        
        self.space_nn = DiffConv(in_channels=hidden_size * temporal_layers,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb, self.adjacency)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index, edge_weight)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon