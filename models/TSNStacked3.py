import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tsl.nn.layers import NodeEmbedding, DenseGraphConvOrderK, DiffConv, Norm
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks.encoders.mlp import MLP
from einops.layers.torch import Rearrange
from snntorch import utils
from typing_extensions import Literal

from models.layers.SynapticChain import SynapticChain
from models.layers.LearnableWeight import LearnableWeight
from models.layers.MultiParam import MultiParam
from models.layers.DenseGraphConv import DenseBlock

class SkipConnection(nn.Module):
    def forward(self, x, out):
        return x + out[:, -x.size(1):]

class Add(nn.Module):
    def forward(self, x, out):
        return x + out

        

class TSNStacked3(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 256,
                 ff_size: int = 256,
                 gnn_kernel: int = 2,
                 output_type: Literal["spike", "synaptic_current", "membrane_potential"] = "spike",
                 learnable_feature_size = 64,
                 number_of_blocks = 2,
                 number_of_temporal_steps = 3,
                 dropout: float = 0.3,
                 ) -> None:
        super(TSNStacked3, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_type = output_type
        
        self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # assert n_nodes is not None
        # self.source_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        # self.target_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        

        temporal = []
        spatial = []
        dense_sconvs = []
        skip_connections = []
        learnable_weights = []
        # edge_weight_mlp = []
        self.edge_weight_mlp = MLP(1, 32, 1)
        norms = []
        new_hidden_size = 0
        for i in range(number_of_blocks):
            # is_last = i == number_of_blocks - 1
            learnable = LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_feature_size)
            # new_hidden_size = hidden_size + ((i + 1) * learnable_feature_size) 
            # hidden_size = hidden_size + learnable_feature_size
            # time_nn = SynapticChain(
            #     hidden_size=new_hidden_size,
            #     return_last=is_last,
            #     output_type=output_type,
            #     n_layers=number_of_temporal_steps,
            #     )
            # space_nn = DiffConv(
            #                 # in_channels=hidden_size * ((3-1)**2),
            #                 in_channels=hidden_size + learnable_feature_size,
            #                 out_channels=hidden_size,
            #                 k=3)
            space_nn = DenseBlock(
                in_channels=hidden_size + learnable_feature_size,
                out_channels=hidden_size,
                growth_rate=16)
            # dense_sconvs.append(
            #         DenseGraphConvOrderK(input_size=hidden_size,
            #                              output_size=hidden_size,
            #                              support_len=1,
            #                              order=5, # spatial kernel size
            #                              include_self=False,
            #                              channel_last=True))
            learnable_weights.append(learnable)
            skip_connections.append(nn.Linear(hidden_size, hidden_size))
            # temporal.append(time_nn)
            spatial.append(space_nn)
            norms.append(Norm('layer', hidden_size))
            new_hidden_size += hidden_size
        
        self.learnable = nn.ModuleList(learnable_weights)
        self.temporal = nn.ModuleList(temporal)
        self.spatial = nn.ModuleList(spatial)
        # self.dense_sconvs = nn.ModuleList(dense_sconvs)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        # self.time_nn = SynapticChain(hidden_size=new_hidden_size, n_layers=number_of_temporal_steps, output_type=output_type, return_last=True)
        self.time_nn = SynapticChain(hidden_size=hidden_size, n_layers=number_of_temporal_steps, output_type=output_type, return_last=True)
        # _____________________________________________________________________
        
        # self.decoder = nn.Linear(hidden_size, input_size * horizon)
        # self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

        self.readout = nn.Sequential(
            nn.ReLU(),
            MLPDecoder(input_size=hidden_size,
                       hidden_size=4 * hidden_size,
                       output_size=input_size,
                       horizon=horizon,
                       activation='relu'))

    def get_learned_adj(self):
        logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
        adj = torch.softmax(logits, dim=1)
        return adj
        
        
    def forward(self, x, edge_index, edge_weight):
        assert not torch.isnan(x).any()
        utils.reset(self.time_nn)
        x = self.encoder(x) + self.node_embeddings()
        edge_weight = self.edge_weight_mlp(edge_weight.expand(1,-1).T).squeeze()
        # ----------------------------------------------------------------------
        # processed = []
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for i, (add_features, space, skip_connection, norm) in enumerate(zip(
            self.learnable,
            self.spatial,
            # self.dense_sconvs,
            self.skip_connections,
            self.norms,
            )):
            # x = checkpoint(add_features, x)
            out = checkpoint(skip_connection, x) + out[:, -x.size(1):]
            x = checkpoint(add_features, out)
            
            # x = add_features(x)
            # x = space(x, adj_z)
            
            # x = checkpoint(space, out, adj_z)
            x = checkpoint(space, x, edge_index, edge_weight)
            # x = checkpoint(self.dropout, x)
            # print(f'out shape {out.size()} {x.size()}')
            # x = norm(x)
            x = checkpoint(norm, x)
            # processed.append(x)
            
            # res = x
            
            # x = checkpoint(time, x)
            # x = time(x)
            # assert not torch.isnan(x).any()
            # # out = checkpoint(skip_conn, x) + out[:, -x.size(1):]
            # out = skip_conn(x) + out[:, -x.size(1):]
            # # xs = checkpoint(space, x, self.edge_index, self.edge_weight)
            # xs = space(x, self.edge_index, self.edge_weight)
            # # xs = space(x, learned_edge_index)
            # if len(self.dense_sconvs):
            #     # x = xs + checkpoint(self.dense_sconvs[i], x, adj_z)
            #     x = xs + self.dense_sconvs[i](x, adj_z)
            # # residual connection -> next layer
            # x = x + res[:, -x.size(1):]
            # x = norm(x)
        # ----------------------------------------------------------------------
        # out = checkpoint(self.time_nn, torch.cat(processed, dim=-1))
        out = checkpoint(self.time_nn, x)
            
        # return self.readout(out)
        result = checkpoint(self.readout, out)
        assert not torch.isnan(result).any()
        # result = torch.nan_to_num(result, nan=0, posinf=1, neginf=-1)
        return result
            
        # x_out = self.decoder(x_emb)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        # x_horizon = self.rearrange(x_out)
        # return x_horizon
        
        