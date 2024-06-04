import torch
from math import ceil, sqrt
import torch.nn as nn
import torch_sparse
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding, DenseGraphConvOrderK
from tsl.nn.blocks import ResidualMLP
from tsl.nn.blocks.encoders.conditional import ConditionalBlock, ConditionalTCNBlock
from einops.layers.torch import Rearrange
from tsl.nn.blocks.encoders import TemporalConvNet
from models.layers.DenseGraphConv import DenseBlock
from models.layers.DenseTimeConv import DenseTemporalBlock
from models.layers.VirtualNode import VirtualNode
from models.UnboundTanh import UnbounTanh
from tsl.nn.blocks.decoders import MLPDecoder
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_geometric.utils.convert import from_networkx
import pickle
from models.layers.A3RFCN import A3TGCNBlock


radius_1 = pickle.load(open('./sphere_graph.pickle', 'rb'))
radius_2 = pickle.load(open('./sphere_graph_adj_radius_2.pickle', 'rb'))
radius_3 = pickle.load(open('./sphere_graph_adj_radius_3.pickle', 'rb'))
radius_4 = pickle.load(open('./sphere_graph_adj_radius_4.pickle', 'rb'))
radius_5 = pickle.load(open('./sphere_graph_adj_radius_5.pickle', 'rb'))
radius_6 = pickle.load(open('./sphere_graph_adj_radius_6.pickle', 'rb'))

def get_edge_index_from_graph(graph):
    g = from_networkx(graph)
    return g.edge_index

def get_edge_attributes_from_graph(graph):
    g = from_networkx(graph)
    return torch.stack([torch.sin(g.distance), torch.cos(g.distance)], dim=-1)

class TEdgeConv(nn.Module):
    def __init__(self,
                 n_nodes=2048,
                 window=20,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 horizon=4,
                #  n_layers=12,
                n_layers=6,
                edge_embedding_size=64,
                n_connections=25204
                 ):
        super(TEdgeConv, self).__init__()
        
        self.n_nodes = n_nodes
        self.window = window
        # self.edge_emb = nn.Embedding(n_connections, edge_embedding_size)
        # self.spatial_adj = None
        
        self.to_time = Rearrange('b t n f -> b n t f')
        # self.to_time = Rearrange('b t n f -> b f t n')
        self.to_space = Rearrange('b n t f -> b t n f')
        
        self.edge_indices = [get_edge_index_from_graph(graph) for graph in list(reversed([radius_1, radius_2, radius_3, radius_4, radius_5, radius_6]))]
        self.edge_attrs = [get_edge_attributes_from_graph(graph) for graph in list(reversed([radius_1, radius_2, radius_3, radius_4, radius_5, radius_6]))]
        self.condition_exog = ConditionalBlock(
                 input_size=input_size,
                 exog_size=2, #day of year and land sea mask
                 output_size=hidden_size,
                 dropout=0.3,
                 skip_connection=True,
                 activation='leaky_relu')

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU()
        )
        
        self.node_embedings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        # self.add_virtual_node = VirtualNode(n_nodes, hidden_size)
        self.temporal_embedding = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        self.time_adj = None
        self.edge_index = None
        # temporal_adj = torch.ones((window, window))
        # self.temporal_adj, self.adj_weights = dense_to_sparse(temporal_adj)
        
        
        # for learning temporal adjacency
        # self.temporal_source_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        # self.temporal_target_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        
        # for learning spatial adj
        # self.spatial_source_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        # self.spatial_target_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        self.space_convs = nn.ModuleList([])
        self.time_convs = nn.ModuleList([])
        self.skip_conn = nn.ModuleList([])
        self.edge_transforms = nn.ModuleList([])
        dilation = 2
        time_kernel=2
        receptive_field = 1
        for layer in range(n_layers):
            # self.edge_transforms.append(ResidualMLP(
            #         input_size=2,
            #         hidden_size=hidden_size,
            #         output_size=edge_embedding_size,
            #         n_layers=3,
            #         activation='leaky_relu',
            #         dropout=0.3,
            #         parametrized_skip=True,
            #     ))
            self.space_convs.append(
                # DenseGraphConvOrderK(hidden_size, hidden_size, support_len=True, order=diffusion_order, channel_last=True)
                # most ran with 4 blocks
                DenseBlock(hidden_size, hidden_size, n_blocks=2, edge_dim=None, activation=nn.LeakyReLU(0.1))
            )
            d = dilation**(layer % 2)
            # d = dilation ** min((layer % 3), 3)
            # d = dilation ** (layer % 3)
            self.time_convs.append(DenseTemporalBlock(
                input_channels=hidden_size,
                hidden_channels=hidden_size,
                output_channels=hidden_size,
                growth_rate=32,
                dilation=d                
                ))
            receptive_field += d * (time_kernel - 1)
            self.skip_conn.append(nn.Linear(hidden_size, hidden_size))
            
        self.receptive_field = receptive_field
        print('receptive_field',receptive_field)
    
        if receptive_field > window:
            receptive_field = 1
        else:
            receptive_field=(window - receptive_field) +1
            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            # receptive_field=2,
            # receptive_field=(window - receptive_field) +1,
            receptive_field=receptive_field,
            # receptive_field=1,
            dropout=0.3,
            activation='leaky_relu'
        )    
    
    def forward(self, x, edge_index, exog):
        exog = self.scale_day_of_year(exog)
        # x_orig = x
        x = checkpoint(self.condition_exog,x, exog, use_reentrant=True)
        x = checkpoint(self.encoder, x, use_reentrant=True)
        x = x + self.node_embedings()
        
        if self.receptive_field > x.size(1):
            # pad temporal dimension
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))
        
        # time_graph = self.to_time(x)
        # time_graph = time_graph + self.temporal_embedding()
        
        # x = self.to_space(time_graph)
        # x, edge_index = self.add_virtual_node(x, edge_index)
        
        # edge_attrs = self.edge_emb(edge_index[1])
        
        # space_adj = self.get_spatial_adj()
        # time_adj = self.get_temporal_adj(x)
        # edge_index = self.get_edge_index(edge_index)
        # out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        # for (space, time, skip_conn) in zip(self.space_convs, self.time_convs, self.skip_conn):
        for i, (space, time, skip_conn) in enumerate(zip(self.space_convs, self.time_convs, self.skip_conn)):
            res = x
            # x = self.to_time(x)
            x = checkpoint(time, x, use_reentrant=True)
            # x = self.to_space(x)
            out = checkpoint(skip_conn,x, use_reentrant=True) + out[:, -x.size(1):]
            # edge_attr = checkpoint(edge_mlp, self.edge_attrs[i].to(x.device), use_reentrant=True)
            adj_idx = i % len(self.edge_indices)
            x = checkpoint(space, x, self.edge_indices[adj_idx].to(x.device), use_reentrant=True)
            # x = checkpoint(space, x, edge_index, use_reentrant=False)
            # x = checkpoint(learned_adj_space, x, space_adj)
            x = x + res[:, -x.size(1):]
            
        # res = self.readout(F.leaky_relu(x[:, :, :-1])) # exclude virtual node
        # single_timestep = self.time_squeeze(x, edge_index).unsqueeze(1) + out[:, -1:]#unsqueeze for a3tgcn
        res = self.readout(x + out[:, -x.size(1):])
        return res
    
    def scale_day_of_year(self, exog):
        day_of_year = exog[..., 1:] / 365
        return torch.cat([day_of_year, exog[..., 1:]], dim=-1)
    
    def get_edge_index(self, edge_index):
        if self.edge_index == None:
            self.edge_index = sort_edge_index(edge_index, sort_by_row=False)
            
        return self.edge_index
        
    def filter_adjacency_to_fit_nodes(self, temporal_graph, temporal_adj, temporal_edge_weights):
        remaining_nodes = temporal_graph.size(-2)
        from_node_selector = temporal_adj[0] < remaining_nodes
        to_node_selector = temporal_adj[1] < remaining_nodes
        
        selector = from_node_selector * (from_node_selector == to_node_selector)
        return temporal_adj.T[selector].T, temporal_edge_weights[selector]
        
        
    def get_temporal_adj(self, x):
        # logits = F.relu(self.temporal_source_emb() @ self.temporal_target_emb().T)
        # adj = torch.softmax(logits, dim=-1)
        # return adj
        if self.time_adj == None:
            time_adj, time_weights = dense_to_sparse(torch.ones((self.window, self.window)))
            self.time_adj = sort_edge_index(time_adj, sort_by_row=False).to(x.device)
        return self.time_adj
    
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

class TGatedGCN(nn.Module):
    def __init__(self,
                 n_nodes=2048,
                 window=20,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 horizon=4,
                #  n_layers=12,
                n_layers=6,
                edge_embedding_size=64,
                spatial_block_size=3,
                temporal_block_size=1,
                norm_scale=None,
                lat_lons=None
                 ):
        super(TGatedGCN, self).__init__()
        
        self.n_nodes = n_nodes
        self.window = window
        # self.edge_emb = nn.Embedding(n_connections, edge_embedding_size)
        # self.spatial_adj = None
        
        self.to_time = Rearrange('b t n f -> b n t f')
        # self.to_time = Rearrange('b t n f -> b f t n')
        self.to_space = Rearrange('b n t f -> b t n f')
        
        self.edge_indices = [get_edge_index_from_graph(graph) for graph in list(reversed([radius_1, radius_2, radius_3, radius_4, radius_5, radius_6]))]
        self.edge_attrs = [get_edge_attributes_from_graph(graph) for graph in list(reversed([radius_1, radius_2, radius_3, radius_4, radius_5, radius_6]))]
        self.condition_exog = ConditionalBlock(
                 input_size=input_size,
                 exog_size=2, #day of year and land sea mask
                 output_size=hidden_size,
                 dropout=0.3,
                 skip_connection=True,
                 activation='leaky_relu')

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU()
        )
        
        self.node_embedings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        # self.add_virtual_node = VirtualNode(n_nodes, hidden_size)
        self.temporal_embedding = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        self.time_adj = None
        self.edge_index = None
        # temporal_adj = torch.ones((window, window))
        # self.temporal_adj, self.adj_weights = dense_to_sparse(temporal_adj)
        
        
        # for learning temporal adjacency
        # self.temporal_source_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        # self.temporal_target_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        
        # for learning spatial adj
        # self.spatial_source_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        # self.spatial_target_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        self.space_convs = nn.ModuleList([])
        self.time_convs = nn.ModuleList([])
        self.skip_conn = nn.ModuleList([])
        self.edge_transforms = nn.ModuleList([])
        dilation = 2
        time_kernel=2
        receptive_field = 1
        for layer in range(n_layers):
            # self.edge_transforms.append(ResidualMLP(
            #         input_size=2,
            #         hidden_size=hidden_size,
            #         output_size=edge_embedding_size,
            #         n_layers=3,
            #         activation='leaky_relu',
            #         dropout=0.3,
            #         parametrized_skip=True,
            #     ))
            self.space_convs.append(
                # DenseGraphConvOrderK(hidden_size, hidden_size, support_len=True, order=diffusion_order, channel_last=True)
                # most ran with 4 blocks
                DenseBlock(hidden_size, hidden_size, n_blocks=spatial_block_size, edge_dim=None, activation=nn.LeakyReLU(0.1))
            )
            d = dilation**(layer % 2)
            # d = dilation ** min((layer % 3), 3)
            # d = dilation ** (layer % 3)
            pad = (window - receptive_field) <= 2
            self.time_convs.append(DenseTemporalBlock(
                input_channels=hidden_size,
                hidden_channels=hidden_size,
                output_channels=hidden_size,
                growth_rate=32,
                dilation=d,
                pad=pad,
                n_blocks=temporal_block_size
                ))
            if not pad:
                # receptive_field += d * ((time_kernel - 1))
                receptive_field += 1
            self.skip_conn.append(nn.Linear(hidden_size, hidden_size))
            
        self.receptive_field = receptive_field
        print('receptive_field',receptive_field)
    
        if receptive_field > window:
            receptive_field = 1
        else:
            receptive_field=(window - receptive_field) - 1
            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            # receptive_field=2,
            # receptive_field=(window - receptive_field) +1,
            # receptive_field=receptive_field,
            receptive_field=1,
            dropout=0.3,
            activation='leaky_relu'
        )    
    
    def forward(self, x, edge_index, exog):
        exog = self.scale_day_of_year(exog)
        x_orig = x
        x = checkpoint(self.condition_exog,x, exog, use_reentrant=True)
        x = checkpoint(self.encoder, x, use_reentrant=True)
        x = x + self.node_embedings()
        
        if self.receptive_field > x.size(1):
            # pad temporal dimension
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))
        
        # time_graph = self.to_time(x)
        # time_graph = time_graph + self.temporal_embedding()
        
        # x = self.to_space(time_graph)
        # x, edge_index = self.add_virtual_node(x, edge_index)
        
        # edge_attrs = self.edge_emb(edge_index[1])
        
        # space_adj = self.get_spatial_adj()
        # time_adj = self.get_temporal_adj(x)
        # edge_index = self.get_edge_index(edge_index)
        # out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        # for (space, time, skip_conn) in zip(self.space_convs, self.time_convs, self.skip_conn):
        for i, (space, time, skip_conn) in enumerate(zip(self.space_convs, self.time_convs, self.skip_conn)):
            res = x
            # x = self.to_time(x)
            x = checkpoint(time, x, use_reentrant=True)
            # x = self.to_space(x)
            out = checkpoint(skip_conn,x, use_reentrant=True) + out[:, -x.size(1):]
            # edge_attr = checkpoint(edge_mlp, self.edge_attrs[i].to(x.device), use_reentrant=True)
            # adj_idx = i % len(self.edge_indices)
            x = checkpoint(space, x, edge_index, use_reentrant=True)
            # x = checkpoint(space, x, edge_index, use_reentrant=False)
            # x = checkpoint(learned_adj_space, x, space_adj)
            x = x + res[:, -x.size(1):]
            
        # res = self.readout(F.leaky_relu(x[:, :, :-1])) # exclude virtual node
        # single_timestep = self.time_squeeze(x, edge_index).unsqueeze(1) + out[:, -1:]#unsqueeze for a3tgcn
        res = self.readout(x + out[:, -x.size(1):])
        return res + x_orig[:, -1:]
    
    def scale_day_of_year(self, exog):
        day_of_year = exog[..., 1:] / 365
        return torch.cat([day_of_year, exog[..., 1:]], dim=-1)
    
    def get_edge_index(self, edge_index):
        if self.edge_index == None:
            self.edge_index = sort_edge_index(edge_index, sort_by_row=False)
            
        return self.edge_index
        
    def filter_adjacency_to_fit_nodes(self, temporal_graph, temporal_adj, temporal_edge_weights):
        remaining_nodes = temporal_graph.size(-2)
        from_node_selector = temporal_adj[0] < remaining_nodes
        to_node_selector = temporal_adj[1] < remaining_nodes
        
        selector = from_node_selector * (from_node_selector == to_node_selector)
        return temporal_adj.T[selector].T, temporal_edge_weights[selector]
        
        
    def get_temporal_adj(self, x):
        # logits = F.relu(self.temporal_source_emb() @ self.temporal_target_emb().T)
        # adj = torch.softmax(logits, dim=-1)
        # return adj
        if self.time_adj == None:
            time_adj, time_weights = dense_to_sparse(torch.ones((self.window, self.window)))
            self.time_adj = sort_edge_index(time_adj, sort_by_row=False).to(x.device)
        return self.time_adj
    
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
    
    
# class TGatedFiLMGCN(nn.Module):
#     def __init__(self,
#                  n_nodes=2048,
#                  window=20,
#                  input_size=26,
#                  hidden_size=256,
#                  out_features=26,
#                  horizon=4,
#                 #  n_layers=12,
#                 n_layers=6
#                  ):
#         super(TGatedFiLMGCN, self).__init__()
        
#         self.n_nodes = n_nodes
#         self.window = window
#         # self.spatial_adj = None
        
#         self.to_time = Rearrange('b t n f -> b n t f')
#         # self.to_time = Rearrange('b t n f -> b f t n')
#         self.to_space = Rearrange('b n t f -> b t n f')
        
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.ReLU()
#         )
        
#         self.node_embedings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
#         self.temporal_embedding = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
#         self.time_adj = None
#         self.edge_index = None
#         # temporal_adj = torch.ones((window, window))
#         # self.temporal_adj, self.adj_weights = dense_to_sparse(temporal_adj)
        
        
#         # for learning temporal adjacency
#         # self.temporal_source_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
#         # self.temporal_target_emb = NodeEmbedding(n_nodes=window, emb_size=hidden_size)
        
#         # for learning spatial adj
#         # self.spatial_source_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
#         # self.spatial_target_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
#         clusters = window
#         reduction_factor = 0.5
#         diffusion_order = 3
#         space_convs = []
#         time_convs = []
#         pool = []
#         skip_conn = []
#         learned_adj_space = []
#         dilation = 2
#         time_kernel=2
#         receptive_field = 1
#         for layer in range(n_layers):
#             space_convs.append(
#                 # DenseGraphConvOrderK(hidden_size, hidden_size, support_len=True, order=diffusion_order, channel_last=True)
#                 DenseBlock(hidden_size, hidden_size, n_blocks=4, activation=nn.LeakyReLU())
#             )
#             # time_convs.append(
#             #     # DenseGraphConvOrderK(hidden_size, hidden_size, support_len=True, order=diffusion_order, channel_last=True)
#             #     DenseBlock(hidden_size, hidden_size, n_blocks=4, activation=nn.LeakyReLU())
#             # )
#             # # time_convs.append(TemporalConvNet(
#             # #     input_channels=hidden_size,
#             # #     hidden_channels=hidden_size*2,
#             # #     kernel_size=3,
#             # #     dilation=1,
#             # #     stride=1,
#             # #     output_channels=hidden_size,
#             # #     n_layers=2,
#             # #     dropout=0.3,
#             # #     activation='leaky_relu',
#             # #     causal_padding=False
#             # # ))
#             d = dilation**(layer % 2)
#             # # time_convs.append(TemporalConvNet(input_channels=hidden_size,
#             # #                     hidden_channels=hidden_size,
#             # #                     kernel_size=2,
#             # #                     dilation=d,
#             # #                     exponential_dilation=False,
#             # #                     n_layers=1,
#             # #                     causal_padding=False,
#             # #                     gated=True))
#             time_convs.append(DenseTemporalBlock(
#                 input_channels=hidden_size,
#                 hidden_channels=hidden_size,
#                 output_channels=hidden_size,
#                 growth_rate=32,
#                 dilation=d                
#                 ))
#             receptive_field += d * (time_kernel - 1)
#             # learned_adj_space.append(DenseGraphConvOrderK(hidden_size, hidden_size, support_len=1, order=diffusion_order, channel_last=True))
#             skip_conn.append(nn.Linear(hidden_size, hidden_size))
#             # clusters = ceil(clusters * reduction_factor)
#             # pool.append(
#             #     BatchedDmonPool(hidden_size, clusters)
#             # )
            
#         self.space_convs = nn.ModuleList(space_convs)
#         self.time_convs = nn.ModuleList(time_convs)
#         self.skip_conn = nn.ModuleList(skip_conn)
#         # self.learned_adj_space = nn.ModuleList(learned_adj_space)
#         # self.pool = nn.ModuleList(pool)
        
#         print('receptive_field',receptive_field)
#         # decoder_receptive_field = clusters #remaining temporal clusters for receptive field
#         self.readout = MLPDecoder(
#             input_size=hidden_size,
#             hidden_size=hidden_size*2,
#             output_size=out_features,
#             horizon=horizon,
#             n_layers=4,
#             receptive_field=2,
#             dropout=0.3,
#             activation='leaky_relu'
#         )
            
    
#     def forward(self, x, edge_index):
#         x_orig = x
#         x = checkpoint(self.encoder, x, use_reentrant=False)
#         x = x + self.node_embedings()
        
#         # time_graph = self.to_time(x)
#         # time_graph = time_graph + self.temporal_embedding()
        
#         # x = self.to_space(time_graph)
        
#         # space_adj = self.get_spatial_adj()
#         # time_adj = self.get_temporal_adj(x)
#         # edge_index = self.get_edge_index(edge_index)
#         # out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
#         out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
#         for (space, time, skip_conn) in zip(self.space_convs, self.time_convs, self.skip_conn):
#             res = x
#             # x = self.to_time(x)
#             x = checkpoint(time, x, use_reentrant=False)
#             # x = self.to_space(x)
#             out = skip_conn(x) + out[:, -x.size(1):]
#             x = checkpoint(space, x, edge_index, use_reentrant=False)
#             # x = checkpoint(learned_adj_space, x, space_adj)
#             x = x + res[:, -x.size(1):]
            
#         res = self.readout(F.leaky_relu(x))
        
#         return res + x_orig[:, -res.size(1):]
        
    
#     def get_edge_index(self, edge_index):
#         if self.edge_index == None:
#             self.edge_index = sort_edge_index(edge_index, sort_by_row=False)
            
#         return self.edge_index
        
#     def filter_adjacency_to_fit_nodes(self, temporal_graph, temporal_adj, temporal_edge_weights):
#         remaining_nodes = temporal_graph.size(-2)
#         from_node_selector = temporal_adj[0] < remaining_nodes
#         to_node_selector = temporal_adj[1] < remaining_nodes
        
#         selector = from_node_selector * (from_node_selector == to_node_selector)
#         return temporal_adj.T[selector].T, temporal_edge_weights[selector]
        
        
#     def get_temporal_adj(self, x):
#         # logits = F.relu(self.temporal_source_emb() @ self.temporal_target_emb().T)
#         # adj = torch.softmax(logits, dim=-1)
#         # return adj
#         if self.time_adj == None:
#             time_adj, time_weights = dense_to_sparse(torch.ones((self.window, self.window)))
#             self.time_adj = sort_edge_index(time_adj, sort_by_row=False).to(x.device)
#         return self.time_adj
    
#     def get_spatial_adj(self):
#         return self.compute_spatial_adj()
#         # if self.spatial_adj == None:
#         #     adj = self.compute_spatial_adj()
#         #     self.spatial_adj = adj
#         # return self.spatial_adj
    
#     def compute_spatial_adj(self):
#         logits = F.relu(self.spatial_source_emb() @ self.spatial_target_emb().T)
#         adj = torch.softmax(logits, dim=-1)
#         return adj