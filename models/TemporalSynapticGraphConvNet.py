import torch
import torch.nn as nn
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding, DenseGraphConvOrderK, DiffConv, Norm
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks.encoders.mlp import MLP
from einops.layers.torch import Rearrange
from snntorch import utils
from typing_extensions import Literal
from torch.utils.checkpoint import checkpoint

from .layers.SynapticChain import SynapticChain

class TemporalSynapticGraphConvNet(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 256,
                 ff_size: int = 256,
                 gnn_kernel: int = 2,
                 output_type: Literal["spike", "synaptic_current", "membrane_potential"] = "spike",
                 temporal_reduction_factor = 2,
                 number_of_blocks = 2,
                 number_of_temporal_steps = 3,
                 dropout: float = 0.3
                 ) -> None:
        super(TemporalSynapticGraphConvNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_type = output_type
        
        self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        # self.node_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        assert n_nodes is not None
        self.source_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.target_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        # start with 1 layer, time then space model lookalike
        # self.time_nn = TemporalSpike(hidden_size=hidden_size, return_last=False)
        temporal = []
        spatial = []
        dense_sconvs = []
        skip_connections = []
        norms = []
        for i in range(number_of_blocks):
            # is_last = i == number_of_blocks - 1
            is_last = False
            time_nn = SynapticChain(
                hidden_size=hidden_size,
                return_last=is_last,
                output_type=output_type,
                n_layers=number_of_temporal_steps,
                temporal_reduction_factor=temporal_reduction_factor
                )
            space_nn = DiffConv(
                            # in_channels=hidden_size * ((3-1)**2),
                            in_channels=hidden_size,
                            out_channels=hidden_size,
                            k=gnn_kernel)
            dense_sconvs.append(
                    DenseGraphConvOrderK(input_size=hidden_size,
                                         output_size=hidden_size,
                                         support_len=1,
                                         order=2, # spatial kernel size
                                         include_self=False,
                                         channel_last=True))
            skip_connections.append(nn.Linear(hidden_size, ff_size))
            temporal.append(time_nn)
            spatial.append(space_nn)
            norms.append(Norm('batch', hidden_size))
        
        # self.time_nn = SynapticChain(
        #     hidden_size=hidden_size,
        #     return_last=False,
        #     output_type=output_type,
        #     temporal_reduction_factor=temporal_reduction_factor
        #     )
        # self.space_nn = DiffConv(
        #                         # in_channels=hidden_size * ((3-1)**2),
        #                         in_channels=hidden_size,
        #                         out_channels=hidden_size,
        #                         k=gnn_kernel)
        # self.time_nn2 = SynapticChain(hidden_size=hidden_size, return_last=True, output_type=output_type)
        # self.graph_processing = nn.Sequential(*blocks)
        self.temporal = nn.ModuleList(temporal)
        self.spatial = nn.ModuleList(spatial)
        self.dense_sconvs = nn.ModuleList(dense_sconvs)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        
        # self.decoder = nn.Linear(hidden_size, input_size * horizon)
        # self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

        self.readout = nn.Sequential(
            nn.ReLU(),
            MLPDecoder(input_size=ff_size,
                       hidden_size=2 * ff_size,
                       output_size=input_size,
                       horizon=horizon,
                       activation='relu'))

    def get_learned_adj(self):
        logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
        adj = torch.softmax(logits, dim=1)
        return adj
        
        
    def forward(self, x, edge_index, edge_weight):
        assert not torch.isnan(x).any()
        # x: [batch time nodes features]
        utils.reset(self.temporal)
        # x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        # x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        # x_temporal = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        # if self.output_type is 'spike':
        #     assert not torch.isnan(s).any()
        #     z = self.space_nn(s, edge_index, edge_weight)  # spatial processing for spikes
        # elif self.output_type is 'synaptic_current':
        #     assert not torch.isnan(n).any()
        #     z = self.space_nn(n, edge_index, edge_weight)  # spatial processing for synaptic current
        # else:
        #     if torch.isnan(m).any():
        #         print(m)
        # assert not torch.isnan(x_temporal).any()
        # z = self.space_nn(x_temporal, edge_index, edge_weight)  # spatial processing for membrane potentials
        # z = self.time_nn2(z)
        x = self.encoder(x)
        adj_z = self.get_learned_adj()
        # learned_edge_index = adj_z.nonzero().t().contiguous()
        for i, (time, space, skip_conn, norm) in enumerate(zip(
            self.temporal,
            self.spatial,
            self.skip_connections,
            self.norms
            )):
            utils.reset(time)
            res = x
            x = checkpoint(time,x)
            assert not torch.isnan(x).any()
            out = checkpoint(skip_conn, x) + out[:, -x.size(1):]
            xs = checkpoint(space, x, edge_index, edge_weight)
            # xs = space(x, learned_edge_index)
            if len(self.dense_sconvs):
                x = xs + self.dense_sconvs[i](x, adj_z)
            # residual connection -> next layer
            x = x = x + res[:, -x.size(1):]
            x = checkpoint(norm, x)
            
        return checkpoint(self.readout, out)
            
        # x_out = self.decoder(x_emb)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        # x_horizon = self.rearrange(x_out)
        # return x_horizon
        
        