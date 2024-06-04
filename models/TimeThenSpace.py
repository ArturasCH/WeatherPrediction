import torch.nn as nn
import torch
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation
from tsl.nn.blocks import ResidualMLP

class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)
        
        self.attribute_encoder = ResidualMLP(
            input_size=1,
            hidden_size=hidden_size,
            output_size=1,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=False,
        )

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index)  # spatial processing
        x_out = self.decoder(nn.functional.elu(z))  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon
    
    def get_edge_attr(self, edge_weight):
        
        # edge_attr = (torch.sin(edge_weight) + torch.cos(edge_weight))
        # res = self.attribute_encoder(edge_attr)
        return self.attribute_encoder(torch.sin(edge_weight).unsqueeze(-1)).squeeze()