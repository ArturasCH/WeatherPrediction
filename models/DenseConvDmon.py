import torch
from math import ceil, sqrt
import torch.nn as nn
import torch_sparse
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding, DenseGraphConvOrderK
from einops.layers.torch import Rearrange
from tsl.nn.blocks.encoders import TemporalConvNet
from models.layers.BatchedDmonPool import BatchedDmonPool
from models.layers.DenseGraphConv import DenseBlock
from models.layers.DenseTimeConv import DenseTemporalBlock
from tsl.nn.blocks.decoders import MLPDecoder
from torch_geometric.utils import dense_to_sparse

class DenseConvDmon(nn.Module):
    def __init__(self,
                 n_nodes=2048,
                 window=20,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 horizon=4,
                #  n_layers=12,
                n_layers=8
                 ):
        super(DenseConvDmon, self).__init__()
        
        self.n_nodes = n_nodes
        self.window = window
        # self.spatial_adj = None
        
        self.to_time = Rearrange('b t n f -> b f t n')
        self.to_space = Rearrange('b f t n -> b t n f')
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        self.node_embedings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        self.temporal_embedding = NodeEmbedding(n_nodes=window, emb_size=n_nodes)
        
        # for learning temporal adjacency
        # self.temporal_source_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        # self.temporal_target_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        
        # for learning spatial adj
        # self.spatial_source_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        # self.spatial_target_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        clusters = window
        reduction_factor = 0.5
        diffusion_order = 3
        space_convs = []
        time_convs = []
        pool = []
        skip_conn = []
        learned_adj_space = []
        dilation = 2
        time_kernel=2
        receptive_field = 1
        for layer in range(n_layers):
            space_convs.append(
                # DenseGraphConvOrderK(hidden_size, hidden_size, support_len=True, order=diffusion_order, channel_last=True)
                DenseBlock(hidden_size, hidden_size, n_blocks=4, activation=nn.LeakyReLU())
            )
            # time_convs.append(
            #     # DenseGraphConvOrderK(hidden_size, hidden_size, support_len=True, order=diffusion_order, channel_last=True)
            #     DenseBlock(hidden_size, hidden_size, n_blocks=4, activation=nn.ReLU())
            # )
            # time_convs.append(TemporalConvNet(
            #     input_channels=hidden_size,
            #     hidden_channels=hidden_size*2,
            #     kernel_size=3,
            #     dilation=1,
            #     stride=1,
            #     output_channels=hidden_size,
            #     n_layers=2,
            #     dropout=0.3,
            #     activation='leaky_relu',
            #     causal_padding=False
            # ))
            d = dilation**(layer % 2)
            # time_convs.append(TemporalConvNet(input_channels=hidden_size,
            #                     hidden_channels=hidden_size,
            #                     kernel_size=2,
            #                     dilation=d,
            #                     exponential_dilation=False,
            #                     n_layers=1,
            #                     causal_padding=False,
            #                     gated=True))
            time_convs.append(DenseTemporalBlock(
                input_channels=hidden_size,
                hidden_channels=hidden_size,
                output_channels=hidden_size,
                growth_rate=32,
                dilation=d                
                ))
            receptive_field += d * (time_kernel - 1)
            # learned_adj_space.append(DenseGraphConvOrderK(hidden_size, hidden_size, support_len=1, order=diffusion_order, channel_last=True))
            skip_conn.append(nn.Linear(hidden_size, hidden_size))
            # clusters = ceil(clusters * reduction_factor)
            # pool.append(
            #     BatchedDmonPool(hidden_size, clusters)
            # )
            
        self.space_convs = nn.ModuleList(space_convs)
        self.time_convs = nn.ModuleList(time_convs)
        self.skip_conn = nn.ModuleList(skip_conn)
        # self.learned_adj_space = nn.ModuleList(learned_adj_space)
        # self.pool = nn.ModuleList(pool)
        
        print('receptive_field',receptive_field)
        # decoder_receptive_field = clusters #remaining temporal clusters for receptive field
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            receptive_field=2,
            dropout=0.3,
            activation='leaky_relu'
        )
            
    
    def forward(self, x, edge_index):
        x_orig = x
        x = checkpoint(self.encoder, x)
        x = x + self.node_embedings()
        
        time_graph = self.to_time(x)
        time_graph = time_graph + self.temporal_embedding()
        
        x = self.to_space(time_graph)
        
        # space_adj = self.get_spatial_adj()
        # time_adj, time_weights = self.get_temporal_adj()
        # out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for (space, time, skip_conn) in zip(self.space_convs, self.time_convs, self.skip_conn):
            res = x
            x = checkpoint(time, x)
            out = skip_conn(x) + out[:, -x.size(1):]
            x = checkpoint(space, x, edge_index)
            # x = checkpoint(learned_adj_space, x, space_adj)
            x = x + res[:, -x.size(1):]
            
        res = self.readout(F.leaky_relu(x))
        
        return res + x_orig[:, -res.size(1):]
        
        
        
        
        
        
    def get_temporal_adj(self):
        # logits = F.relu(self.temporal_source_emb() @ self.temporal_target_emb().T)
        # adj = torch.softmax(logits, dim=-1)
        # return adj
        return self.time_adj, self.time_weights
    
    def get_spatial_adj(self):
        return self.compute_spatial_adj()
        # if self.spatial_adj == None:
        #     adj = self.compute_spatial_adj()
        #     self.spatial_adj = adj
        # return self.spatial_adj
    
    def compute_spatial_adj(self):
        logits = F.relu(self.spatial_source_emb() @ self.spatial_target_emb().T)
        adj = torch.softmax(logits, dim=-1)
        return adj