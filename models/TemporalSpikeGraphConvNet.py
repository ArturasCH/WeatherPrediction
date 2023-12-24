import torch
import torch.nn as nn
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange
from snntorch import utils

from .layers.TemporalSpike import TemporalSpike
from .layers.SynapticSpike import SynapticSpike

class TemporalSpikeGraphConvNet(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 temporal_layers: int = 1,
                 gnn_kernel: int = 2,
                 use_spike_for_output = True,
                 timesteps = 112
                 ) -> None:
        super(TemporalSpikeGraphConvNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_spike_for_output = use_spike_for_output
        self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        # start with 1 layer, time then space model lookalike
        # self.time_nn = TemporalSpike(hidden_size=hidden_size, return_last=False)
        self.time_nn = SynapticSpike(n_nodes=n_nodes, hidden_size=hidden_size, return_last=True, timesteps=timesteps)
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)
        
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
        
    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        # utils.reset(self.time_nn)
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        # h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        # s, m = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        s, m, n = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        if self.use_spike_for_output:
            z = self.space_nn(s, edge_index, edge_weight)  # spatial processing for spikes
        else:
            z = self.space_nn(n, edge_index, edge_weight)  # spatial processing for membrane potentials
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon
        
        