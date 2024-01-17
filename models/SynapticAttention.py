import torch
import torch.nn as nn
from tsl.nn.layers import NodeEmbedding, DenseGraphConvOrderK
from tsl.nn.blocks.decoders import MLPDecoder
from einops.layers.torch import Rearrange
from snntorch import utils
from torch.nn import functional as F
from typing_extensions import Literal

from .layers.GSATConv import GSATConv

class SynapticAttention(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 256,
                 ff_size: int = 256,
                 gnn_kernel: int = 2,
                 output_type: Literal["spike", "synaptic_current", "membrane_potential"] = "spike",
                 temporal_reduction_factor = 2,
                 number_of_blocks = 2,
                 ) -> None:
        super(SynapticAttention, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_type = output_type
        self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        # self.node_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        assert n_nodes is not None
        self.source_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.target_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
            
        dense_sconvs = []
        skip_connections = []
        attention = []
        
        for i in range(number_of_blocks):
            is_last = i == number_of_blocks - 1
            attention.append(GSATConv(in_channels=hidden_size, out_channels=hidden_size, dim = 1, edge_dim=1, heads=8, return_last=is_last))
            dense_sconvs.append(
                    DenseGraphConvOrderK(input_size=hidden_size,
                                         output_size=hidden_size,
                                         support_len=1,
                                         order=2, # spatial kernel size
                                         include_self=False,
                                         channel_last=True))
            skip_connections.append(nn.Linear(hidden_size, ff_size))

        self.attention = nn.ModuleList(attention)
        self.dense_sconvs = nn.ModuleList(dense_sconvs)
        self.skip_connections = nn.ModuleList(skip_connections)
        
        self.readout = nn.Sequential(
            nn.ReLU(),
            MLPDecoder(input_size=ff_size,
                       hidden_size=2 * ff_size,
                       output_size=input_size,
                       horizon=horizon,
                       activation='relu'))
        # self.decoder = nn.Linear(hidden_size, input_size * horizon)
        # self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
        
        
    def get_learned_adj(self):
        logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
        adj = torch.softmax(logits, dim=1)
        return adj
        
    def forward(self, x, edge_index, edge_weight):
        assert not torch.isnan(x).any()
        
        if len(self.dense_sconvs):
            adj_z = self.get_learned_adj()
            
        # x: [batch time nodes features]
        # utils.reset(self.temporal)
        x = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        # x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        # spatial_aggregation = self.spatial(x_emb, edge_index, edge_weight)
        # spatial_aggregation2 = self.spatial(spatial_aggregation, edge_index, edge_weight)
        # for time, space in zip(self.temporal, self.spatial):
        #     utils.reset(time)
        #     x_emb = time(x_emb)
        #     assert not torch.isnan(x_emb).any()
        #     x_emb = space(x_emb, edge_index, edge_weight)
        # z, _ = self.attention(torch.cat((x_emb, spatial_aggregation, spatial_aggregation2), dim=-1), edge_index, edge_weight)
        # residual rather than dense
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for i, (attention, skip_connection) in enumerate(zip(self.attention, self.skip_connections)):
            res = x
            processed, _ = attention(x, edge_index, edge_weight)
            skip_processed = skip_connection(processed)
            out = skip_processed + out[:, -x.size(1):]
            if len(self.dense_sconvs):
                x = processed + self.dense_sconvs[i](x, adj_z)
            x = x + res[:, -x.size(1):]
        # z, _ = self.attention(x_enc, edge_index, edge_weight)
        # assert not torch.isnan(z).any()
        # z, _ = self.attention2(z, edge_index, edge_weight)
        # assert not torch.isnan(z).any()
        # x_out = self.decoder(out)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        # x_horizon = self.rearrange(x_out)
        # return x_horizon
        return self.readout(out)
        
        