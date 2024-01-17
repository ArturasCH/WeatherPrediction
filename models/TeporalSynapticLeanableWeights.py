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

from .layers.SynapticChain import SynapticChain
from .layers.LearnableWeight import LearnableWeight
from .layers.MultiParam import MultiParam

class SkipConnection(nn.Module):
    def forward(self, x, out):
        return x + out[:, -x.size(1):]

class Add(nn.Module):
    def forward(self, x, out):
        return x + out

        

class TemporalSynapticLearnableWeights(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 256,
                 ff_size: int = 256,
                 gnn_kernel: int = 2,
                 output_type: Literal["spike", "synaptic_current", "membrane_potential"] = "spike",
                 learnable_feature_size = 64,
                 number_of_blocks = 2,
                 number_of_temporal_steps = 3,
                 dropout: float = 0.3,
                 edge_index=None,
                 edge_weights = None
                 ) -> None:
        super(TemporalSynapticLearnableWeights, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weights

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_type = output_type
        
        self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        assert n_nodes is not None
        self.source_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.target_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        is_last = False
        # lw = []
        # learn_time = []
        # skip_full = []
        # space_dense = []
        # residual_norms = []
        # for i in range(number_of_blocks):
        #     learnable = LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_feature_size)
        #     new_hidden_size = hidden_size + ((i + 1) * learnable_feature_size) 
        #     time_nn = SynapticChain(
        #         hidden_size=new_hidden_size,
        #         return_last=False,
        #         output_type=output_type,
        #         n_layers=number_of_temporal_steps,
        #         )
        #     lw.append(learnable)
        #     learn_time.append(nn.Sequential(learnable, time_nn))
            
        #     skip_con = nn.Linear(new_hidden_size, ff_size)
        #     skip = SkipConnection()
        #     space = DiffConv(
        #         # in_channels=hidden_size * ((3-1)**2),
        #         in_channels=hidden_size,
        #         out_channels=hidden_size,
        #         k=gnn_kernel)
        #     dense = DenseGraphConvOrderK(input_size=new_hidden_size,
        #                     output_size=new_hidden_size,
        #                     support_len=1,
        #                     order=2, # spatial kernel size
        #                     include_self=False,
        #                     channel_last=True)
        #     add = Add()
        #     skip_full.append(nn.Sequential(
        #         MultiParam(skip_con, [0]),
        #         MultiParam(skip, [0, 1]),

        #     ))
        #     space_dense.append(nn.Sequential(MultiParam(space, [0, 1, 2], return_original_x=True),
        #         MultiParam(dense, [-1, 3], return_original_x=True),
        #         MultiParam(add, [0, -1])))
        #     norm = Norm('batch', new_hidden_size)
        #     residual_norms.append(nn.Sequential(
        #       MultiParam(learnable, [1], return_original_x=True),
        #       MultiParam(skip, [0, -1]),
        #       MultiParam(norm, [0])
        #     ))
            
        # self.phase1 = nn.ModuleList(learn_time)
        # self.phase2 = nn.ModuleList(skip_full)
        # self.phase3  = nn.ModuleList(space_dense)
        # self.phase4 = nn.ModuleList(residual_norms)
            
        
        # _____________________________________________________________________
        # start with 1 layer, time then space model lookalike
        # self.time_nn = TemporalSpike(hidden_size=hidden_size, return_last=False)
        temporal = []
        spatial = []
        dense_sconvs = []
        skip_connections = []
        learnable_weights = []
        norms = []
        for i in range(number_of_blocks):
            # is_last = i == number_of_blocks - 1
            is_last = False
            learnable = LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_feature_size)
            new_hidden_size =hidden_size + ((i + 1) * learnable_feature_size) 
            time_nn = SynapticChain(
                hidden_size=new_hidden_size,
                return_last=is_last,
                output_type=output_type,
                n_layers=number_of_temporal_steps,
                )
            space_nn = DiffConv(
                            # in_channels=hidden_size * ((3-1)**2),
                            in_channels=new_hidden_size,
                            out_channels=new_hidden_size,
                            k=gnn_kernel)
            dense_sconvs.append(
                    DenseGraphConvOrderK(input_size=new_hidden_size,
                                         output_size=new_hidden_size,
                                         support_len=1,
                                         order=2, # spatial kernel size
                                         include_self=False,
                                         channel_last=True))
            learnable_weights.append(learnable)
            skip_connections.append(nn.Linear(new_hidden_size, ff_size))
            temporal.append(time_nn)
            spatial.append(space_nn)
            norms.append(Norm('batch', new_hidden_size))
        
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
        self.learnable = nn.ModuleList(learnable_weights)
        self.temporal = nn.ModuleList(temporal)
        self.spatial = nn.ModuleList(spatial)
        self.dense_sconvs = nn.ModuleList(dense_sconvs)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        # _____________________________________________________________________
        
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
        
        
    def forward(self, x):
        assert not torch.isnan(x).any()
        # x: [batch time nodes features]
        # utils.reset(self.temporal)
        # x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        # x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
    
        x = self.encoder(x) + self.node_embeddings()
        adj_z = self.get_learned_adj()
        # for p1, p2, p3, p4 in zip(self.phase1, self.phase2, self.phase3, self.phase4):
        #     resid = x
        #     utils.reset(p1)
        #     x = checkpoint(p1, x, use_reentrant=False)
        #     res = checkpoint(p2, [x, out], use_reentrant=False)
        #     out = res[0]
        #     print(x.size())
        #     res = checkpoint(p3, [x, edge_index, edge_weight, adj_z], use_reentrant=False)
        #     x = res[0]
        #     res = checkpoint(p4, [x, resid], use_reentrant=False)
        #     x = res[0]
        # learned_edge_index = adj_z.nonzero().t().contiguous()
        # ----------------------------------------------------------------------
        for i, (add_features, time, space, skip_conn, norm) in enumerate(zip(
            self.learnable,
            self.temporal,
            self.spatial,
            self.skip_connections,
            self.norms
            )):
            utils.reset(time)
            # x = checkpoint(add_features, x)
            x = add_features(x)
            
            res = x
            
            # x = checkpoint(time, x)
            x = time(x)
            assert not torch.isnan(x).any()
            # out = checkpoint(skip_conn, x) + out[:, -x.size(1):]
            out = skip_conn(x) + out[:, -x.size(1):]
            # xs = checkpoint(space, x, self.edge_index, self.edge_weight)
            xs = space(x, self.edge_index, self.edge_weight)
            # xs = space(x, learned_edge_index)
            if len(self.dense_sconvs):
                # x = xs + checkpoint(self.dense_sconvs[i], x, adj_z)
                x = xs + self.dense_sconvs[i](x, adj_z)
            # residual connection -> next layer
            x = x + res[:, -x.size(1):]
            x = norm(x)
        # ----------------------------------------------------------------------
            
        return self.readout(out)
            
        # x_out = self.decoder(x_emb)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        # x_horizon = self.rearrange(x_out)
        # return x_horizon
        
        