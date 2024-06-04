import torch
import torch.nn as nn
from math import ceil

from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from torch_geometric.utils import dense_to_sparse

from models.layers.DenseGraphConv import DenseBlock
from models.layers.AttentionStack import AttentionStack
from models.layers.TemporalDiffPooling import TemporalDiffPooling


class GCNTemporalAttentionPooled(nn.Module):
    def __init__(self,
                 n_nodes=2048,
                 window=20,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 horizon=4):
        super(GCNTemporalAttentionPooled, self).__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.static_arrow_of_time = None
        
        self.to_time_nodes = Rearrange('b t n f -> b (t n) f')
        self.time_to_space_structure = Rearrange('b (t n) f -> b t n f', n=n_nodes)
        
        self.encoder = nn.Linear(input_size, hidden_size)
        # n_stacks = 2
        # stacks_s1 = []
        # stacks_t1 = []
        # for i in range(n_stacks):
        #     stacks_s1.append(DenseBlock(in_channels=hidden_size, out_channels=hidden_size, n_blocks=4, activation='leaky_relu'))
        #     stacks_t1.append(AttentionStack(n_attention_blocks=4, hidden_size=hidden_size))
        
            
        self.time_1 = AttentionStack(n_attention_blocks=4, hidden_size=hidden_size)
        self.space_1 = DenseBlock(
            in_channels=hidden_size,
            out_channels=hidden_size, n_blocks=4, activation='leaky_relu')
        
        first_reduction = 0.5
        self.diff_pool1 = TemporalDiffPooling(
            hidden_size,
            window=window,
            n_nodes=n_nodes,
            n_subgraphs=8,
            # reduction_ratio=.5
            reduction_ratio=first_reduction
            )
        
        # stacks_s2 = []
        # stacks_t2 = []
        # for i in range(n_stacks):
        #     stacks_s2.append(DenseBlock(in_channels=hidden_size, out_channels=hidden_size, n_blocks=4, activation='leaky_relu'))
        #     stacks_t2.append(AttentionStack(n_attention_blocks=4, hidden_size=hidden_size))
        
        self.time_2 = AttentionStack(n_attention_blocks=4, hidden_size=hidden_size)
        self.space_2 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, n_blocks=4, activation='leaky_relu')
        
        second_reduction = 0.4
        self.diff_pool2 = TemporalDiffPooling(
            hidden_size,
            window=ceil(window * first_reduction),
            n_nodes=n_nodes,
            n_subgraphs=4,
            reduction_ratio=second_reduction,
            # cache_subgraphs=False
            reset_subgraph_cache=True,
            subgraph_cache_reset_period=1280
            )
        
        # stacks_s3 = []
        # stacks_t3 = []
        # for i in range(n_stacks):
        #     stacks_s3.append(DenseBlock(in_channels=hidden_size, out_channels=hidden_size, n_blocks=4, activation='leaky_relu'))
        #     stacks_t3.append(AttentionStack(n_attention_blocks=4, hidden_size=hidden_size))
        
        self.time_3 = AttentionStack(n_attention_blocks=4, hidden_size=hidden_size)
        self.space_3 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, n_blocks=4, activation='leaky_relu')
    
        # self.stacks_s1 = nn.ModuleList(stacks_s1)
        # self.stacks_s2 = nn.ModuleList(stacks_s2)
        # self.stacks_s3 = nn.ModuleList(stacks_s3)
        # self.stacks_t1 = nn.ModuleList(stacks_t1)
        # self.stacks_t2 = nn.ModuleList(stacks_t2)
        # self.stacks_t3 = nn.ModuleList(stacks_t3)
        
        self.readout = nn.Linear(hidden_size, out_features)
        
    def forward(self, x, edge_index, edge_weight, exog):
        x_orig = x
        x = checkpoint(self.encoder, x)
        
        x = checkpoint(self.space_1, x.squeeze(), edge_index, edge_weight).unsqueeze(0)
        x = checkpoint(self.time_1, x)
        # for (time, space) in zip(self.stacks_t1, self.stacks_s1):
        #     x = checkpoint(space, x.squeeze(), edge_index, edge_weight).unsqueeze(0)
        #     x = checkpoint(time, x)
        
        temporal_graph = self.to_time_nodes(x)
        temporal_adjacency = self.build_temporal_adjacency(self.n_nodes, self.window, x.device)
        
        pooled_time_graph, pooled_adj, pooled_attrs = checkpoint(self.diff_pool1, temporal_graph, temporal_adjacency)
        
        x = self.time_to_space_structure(pooled_time_graph)
        
        x = checkpoint(self.space_2, x.squeeze(), edge_index, edge_weight).unsqueeze(0)
        x = checkpoint(self.time_2, x)
        # for (time, space) in zip(self.stacks_t2, self.stacks_s2):
        #     x = checkpoint(space, x.squeeze(), edge_index, edge_weight).unsqueeze(0)
        #     x = checkpoint(time, x)
        
        temporal_graph = self.to_time_nodes(x)
        pooled_time_graph, pooled_adj, pooled_attrs = checkpoint(self.diff_pool2, temporal_graph, pooled_adj, pooled_attrs)
        
        x = self.time_to_space_structure(pooled_time_graph)
        
        x = checkpoint(self.space_3, x.squeeze(), edge_index, edge_weight).unsqueeze(0)
        x = checkpoint(self.time_3, x)
        # for (time, space) in zip(self.stacks_t3, self.stacks_s3):
        #     x = checkpoint(space, x.squeeze(), edge_index, edge_weight).unsqueeze(0)
        #     x = checkpoint(time, x)
        
        
        x = self.readout(x)
        
        return x + x_orig[:, -1]
        
        
        
    def build_temporal_adjacency(self, nodes_in_spatial_graph, timesteps, device):
        if self.static_arrow_of_time == None:
            arrow_of_time = []
            step_size = nodes_in_spatial_graph
            for i in range(nodes_in_spatial_graph): # for every node
                source_node = i
                for j in range(timesteps-1): #up to second to last timestep (last timestep is destination node)
                    
                    if j == 0:
                        # self loop?
                        arrow_of_time.append([source_node, source_node])
                    target_node = source_node + step_size
                    arrow_of_time.append([source_node, target_node])
                    source_node = target_node
                    
            self.static_arrow_of_time = torch.tensor(arrow_of_time, device=device).T
        return self.static_arrow_of_time
    
    def dense_temporal_adj(self, spatial_graphs, node_dim=-2):
        b,t,n,f = spatial_graphs.size()
        step_size = spatial_graphs.size(node_dim)
        
        connectivity = torch.sparse.spdiags(torch.ones(t * n), torch.tensor([0]), (t * n, t * n)) #self loops
        # connectivity = 0
        for i in range(n, t * n, step_size):
            connectivity = connectivity + torch.sparse.spdiags(torch.ones(t * n), torch.tensor([i]), (t * n, t * n))
            
        return dense_to_sparse(connectivity.to_dense())
    