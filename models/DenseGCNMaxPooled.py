import torch
from math import ceil
import torch.nn as nn
from torch_geometric.nn import FAConv, PositionalEncoding, GINConv, MLP, SAGPooling
from torch_geometric.utils import dense_to_sparse, sort_edge_index
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from models.layers.TemporalDiffPooling import TemporalDiffPooling
from models.layers.GAT_Batched import GATBatched
from models.layers.aggregations.GenAgg import GenAgg
from models.layers.aggregations.MLPAutoEncoder import MLPAutoencoder
from models.layers.DenseGraphConv import DenseBlock
from models.layers.BatchedFAConv import BatchedFAConv
from models.layers.TemporalMaxPool import TemporalMaxPool
from models.layers.AttentionStack import AttentionStack
from torch_geometric.nn import GraphNorm, MeanSubtractionNorm, DiffGroupNorm, BatchNorm, aggr
from tsl.nn.blocks.encoders.conditional import ConditionalBlock


class DenseGCNMaxPooled(nn.Module):
    def __init__(self, 
                 n_nodes=2048,
                 window=20,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 horizon=4,):
        super(DenseGCNMaxPooled, self).__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.static_arrow_of_time = None
        self.static_weights = None
        
        self.to_time_nodes = Rearrange('b t n f -> b (t n) f')
        self.time_to_space_structure = Rearrange('b (t n) f -> b t n f', n=n_nodes)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU()
        )
        
        self.max_pool = TemporalMaxPool() # divides length to 2, with ceil on
        
        space1_n = 3
        time1_n =3
        
        temporal_lengths = [20, 10, 5, 3, 2, 1]
        
        space_convs = []
        time_convs = []
        for i in temporal_lengths:
            space_convs.append(DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=128, n_blocks=space1_n))
            # time_convs.append(DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=128, n_blocks=space1_n))
            time_convs.append(AttentionStack(time1_n, hidden_size=hidden_size))
        
        self.space_convs = nn.ModuleList(space_convs)
        self.time_convs = nn.ModuleList(time_convs)
        
        readouts = []
        for i in range(horizon):
            readouts.append(
                nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_size * 2, out_features)
                )
            )
        self.readouts = nn.ModuleList(readouts)
        
        # self.readout = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size * 2, out_features)
        #     )
        
    def forward(self, x, edge_index, edge_weight):
        x_orig = x
        
        _x = x = F.leaky_relu(checkpoint(self.encoder, x))
        
        # time_adj, time_weights = self.dense_temporal_adj(x)
        
        # _time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        # x = x.unsqueeze(0)
        for (time, space) in zip(self.time_convs, self.space_convs):
            # time_graph = self.to_time_nodes(x)
            # adj, weights = self.filter_adjacency_to_fit_nodes(time_graph, time_adj, time_weights)
            time_graph = checkpoint(time, x)
            # x = self.time_to_space_structure(time_graph)
            x = checkpoint(space, x, edge_index, edge_weight)
            x = self.max_pool(x)
            
        
        last_timestep = x_orig[:, -1:]
        res = []
        
        for readout in self.readouts:
            res.append(readout(x) + last_timestep) 
        return torch.cat(res, dim=1)
    
    def filter_adjacency_to_fit_nodes(self, temporal_graph, temporal_adj, temporal_edge_weights):
        remaining_nodes = temporal_graph.size(-2)
        from_node_selector = temporal_adj[0] < remaining_nodes
        to_node_selector = temporal_adj[1] < remaining_nodes
        
        selector = from_node_selector * (from_node_selector == to_node_selector)
        return temporal_adj.T[selector].T, temporal_edge_weights[selector]
        
        
    def dense_temporal_adj(self, graph):
        if self.static_arrow_of_time == None:
            b, t,n,f = graph.size()
            # step_size = graph.size(node_dim)
            
            connectivity = torch.sparse.spdiags(torch.ones(t * n), torch.tensor([0]), (t * n, t * n)) #self loops
            # connectivity = 0
            for i in range(n, t * n, n):
                connectivity = connectivity + torch.sparse.spdiags(torch.ones(t * n), torch.tensor([i]), (t * n, t * n))
                
            adj, weights =  dense_to_sparse(connectivity.to_dense())
            self.static_arrow_of_time = adj.to(graph.device)
            self.static_weights = weights.to(graph.device)
            
        return self.static_arrow_of_time, self.static_weights
        
        