import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from models.layers.TemporalDiffPooling import TemporalDiffPooling
from models.layers.GAT_Batched import GATBatched
# from tsl.nn.layers import Norm
from torch_geometric.nn import GraphNorm, MeanSubtractionNorm, DiffGroupNorm, BatchNorm, aggr


class GATDiffPooled(nn.Module):
    def __init__(self,
                 n_nodes,
                 horizon=4,
                 window=56,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 ):
        super(GATDiffPooled, self).__init__()
        
        self.to_time_nodes = Rearrange('b t n f -> b (t n) f')
        self.time_to_space_structure = Rearrange('b (t n) f -> b t n f', n=n_nodes)
        
        self.n_nodes = n_nodes
        self.window = window
        n_heads = 4
        self.fourier = Fourier(input_size, input_size)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.space_gat = GATBatched(hidden_size + (2 * input_size), hidden_size, n_heads, edge_dim=8, aggr=aggr.So)
        self.space_gat_0 = GATBatched(hidden_size * n_heads, hidden_size//n_heads, n_heads, edge_dim=8)
        self.space_gat_1 = GATBatched(hidden_size, hidden_size//n_heads, n_heads, edge_dim=8)
        self.edge_mlp = EdgeMLP(1, 128, 8)
        self.edge_mlp_0 = EdgeMLP(8, 128, 8)
        self.edge_mlp_1 = EdgeMLP(8, 128, 8)
        self.space_norm_1 = BatchNorm(n_nodes)
        
        self.time_gat1_0 = GATBatched(hidden_size, hidden_size//n_heads, n_heads, edge_dim=1)
        self.time_gat1_1 = GATBatched(hidden_size, hidden_size//n_heads, n_heads, edge_dim=1)
        self.time_gat1_2 = GATBatched(hidden_size, hidden_size//n_heads, n_heads, edge_dim=1)
        self.time_gat1_3 = GATBatched(hidden_size, hidden_size//n_heads, n_heads, edge_dim=1)
        
        
        self.norm1 = BatchNorm(hidden_size) # no dim_size==1, maybe iterate over individual samples?
        self.diff_pool1 = TemporalDiffPooling(
            hidden_size,
            window=window,
            n_nodes=n_nodes,
            n_subgraphs=8,
            reduction_ratio=.5
            )
        
        self.norm2 = BatchNorm(hidden_size) # no dim_size==1, maybe iterate over individual samples?
        
        self.space_gat2 = GATBatched(hidden_size, hidden_size//n_heads, n_heads, edge_dim=8)
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(8, 32),
            nn.ELU(),
            nn.Linear(32, 128),
            nn.ELU(),
            nn.Linear(128, 8)
        )
        self.time_gat2 = GATBatched(hidden_size, hidden_size//n_heads, n_heads, edge_dim=1)
        self.diff_pool2 = TemporalDiffPooling(
            hidden_size,
            window=window // 2,
            n_nodes=n_nodes,
            n_subgraphs=4,
            reduction_ratio=.4,
            # cache_subgraphs=False
            reset_subgraph_cache=True,
            subgraph_cache_reset_period=1280
            )
        
        self.norm3 = BatchNorm(hidden_size) # no dim_size==1, maybe iterate over individual samples?
        
        self.gat_out = GATBatched(hidden_size, out_features, 1, edge_dim=8)
        
        self.edge_mlp3 = nn.Sequential(
            nn.Linear(8, 32),
            nn.ELU(),
            nn.Linear(32, 128),
            nn.ELU(),
            nn.Linear(128, 8)
        )
        
        self.static_arrow_of_time = None

        
        
        
    def forward(self, x, edge_index, edge_weight):
        # squeeze in nonlinearities in here, try different norms
        x_in = x
        x_f = checkpoint(self.fourier, x)
        x = checkpoint(self.encoder, x)
        edge_weight = checkpoint(self.edge_mlp, edge_weight.unsqueeze(-1))
        # x = F.elu(checkpoint(self.space_gat, x.squeeze(), edge_index, edge_weight)).unsqueeze(0)
        x = torch.cat([x, x_f], dim=-1)
        x = F.elu(checkpoint(self.space_gat, x.squeeze(), edge_index, edge_weight))
        edge_weight = checkpoint(self.edge_mlp_0, edge_weight)
        space_graph = F.elu(checkpoint(self.space_gat_0, x, edge_index, edge_weight))
        edge_weight = checkpoint(self.edge_mlp_1, edge_weight)
        x = F.elu(checkpoint(self.space_gat_1, space_graph, edge_index, edge_weight))
        x = checkpoint(self.space_norm_1, x)
        # x = x + space_graph
        # x = checkpoint(self.norm1, x).unsqueeze(0)
        
        time_graph = self.to_time_nodes(x.unsqueeze(0))
        time_adj = self.build_temporal_adjacency(self.n_nodes, self.window, x.device)
        _time_graph = F.elu(checkpoint(self.time_gat1_0, time_graph, time_adj))
        _time_graph = F.elu(checkpoint(self.time_gat1_1, _time_graph, time_adj))
        _time_graph = F.elu(checkpoint(self.time_gat1_2, _time_graph, time_adj))
        _time_graph = F.elu(checkpoint(self.time_gat1_3, _time_graph, time_adj))
        time_graph = time_graph + _time_graph # residual
        
        time_graph = checkpoint(self.norm1, time_graph.squeeze()).unsqueeze(0)
        pooled_temporal_graph, pooled_adj, pooled_weights  = self.diff_pool1(time_graph, time_adj)
        
        space_graph = self.time_to_space_structure(pooled_temporal_graph)
        edge_weight = checkpoint(self.edge_mlp2, edge_weight)
        _space_graph = F.elu(checkpoint(self.space_gat2, space_graph.squeeze(), edge_index, edge_weight)).unsqueeze(0)
        space_graph = space_graph + _space_graph # residual
        
        time_graph = self.to_time_nodes(space_graph)
        _time_graph = F.elu(checkpoint(self.time_gat2, time_graph.squeeze(), pooled_adj, pooled_weights)).unsqueeze(0)
        time_graph = time_graph + _time_graph # residual
        
        time_graph = checkpoint(self.norm2, time_graph.squeeze()).unsqueeze(0)
        pooled_temporal_graph, pooled_adj, pooled_weights = self.diff_pool2(time_graph, pooled_adj, pooled_weights)
        
        # pooled_temporal_graph = F.elu(checkpoint(self.time_gat2, pooled_temporal_graph, pooled_adj, pooled_weights))
        # pooled_temporal_graph = checkpoint(self.norm3, pooled_temporal_graph.squeeze()).unsqueeze(0)
        
        space_graph = self.time_to_space_structure(pooled_temporal_graph)
        edge_weight = checkpoint(self.edge_mlp3, edge_weight)
        out = checkpoint(self.gat_out, space_graph.squeeze(), edge_index, edge_weight).unsqueeze(0)
        
        # return out + x_in[:,-1]
        return out
        
        
        
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
    
class DifferentiableGroupNorm(nn.Module):
    def __init__(self, input_channels, n_groups):
        super(DifferentiableGroupNorm, self).__init__()
        self.diff_group_norm = DiffGroupNorm(input_channels, n_groups)
            
    def forward(self, x):
        if x.dim() == 3:
            graphs = []
            for graph in x:
                graphs.append(checkpoint(self.diff_group_norm, graph))
            return torch.stack(graphs)
        
        if x.dim() == 2:
            return checkpoint(self.diff_group_norm, x)
        
        if x.dim() == 4:
            graphs = []
            for graph in x.squeeze():
                graphs.append(checkpoint(self.diff_group_norm, graph))
            return torch.stack(graphs)
        
class Fourier(nn.Module):
	def __init__(self, input_channels=256, output_channels=256, scale=10):
		super(Fourier, self).__init__()
		self.b = torch.randn(input_channels, output_channels)*scale
	def forward(self, v):
		x_proj = torch.matmul(2*torch.pi*v, self.b.to(v.device))
		return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)

class EdgeMLP(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(EdgeMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, out_channels)
        )
        
    def forward(self, edge):
        return self.model(edge)