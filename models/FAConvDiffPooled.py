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
from torch_geometric.nn import GraphNorm, MeanSubtractionNorm, DiffGroupNorm, BatchNorm, aggr
from tsl.nn.blocks.encoders.conditional import ConditionalBlock

class FAConvDiffPooled(nn.Module):
    def __init__(self,
                 n_nodes=2048,
                 window=20,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 horizon=4,
                 ):
        super(FAConvDiffPooled, self).__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.static_arrow_of_time = None
        self.static_weights = None
        
        self.to_time_nodes = Rearrange('b t n f -> b (t n) f')
        self.time_to_space_structure = Rearrange('b (t n) f -> b t n f', n=n_nodes)
        
        self.space_to_batch_norm = Rearrange('t n f -> t f n')
        self.space_from_batch_norm = Rearrange('t f n -> t n f')
        
        
        self.fourier = Fourier(input_size, input_size)
        orders = [-1, 0, 1,2,3,4,5]
        # self.add_fourier_features = FourierFeatures(orders=orders)
        # output_size = input_size + (len(orders) * 2 * input_size)
        # positional_encoding_dim = 16
        # self.positional_encoding = SequencePositionalEncoding(n_nodes, positional_encoding_dim)
        # self.encoder = nn.Linear(input_size + (input_size * 2) + (input_size * positional_encoding_dim), hidden_size)
        # encoding_size = output_size + input_size + (input_size * 2)
        one_hot_encoded_size = 17
        self.condition_exogenous = ConditionalBlock(input_size, one_hot_encoded_size, 256, dropout=0.3, skip_connection=True, activation='leaky_relu')
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ELU()
        )
        # self.encoder = nn.Linear(output_size + input_size + (input_size * 2), hidden_size)
        space1_n = 3
        time1_n = 3
        space_convs1 = []
        # aggrs=['SumAggregation','MeanAggregation','MaxAggregation', 'MinAggregation']
        size = hidden_size
        for i in range(space1_n):
            space_convs1.append(BatchedFAConv(size, size))
            # space_convs1.append(GIN(hidden_size, hidden_size))
            # space_convs1.append(BatchedFAConv(size))
            # size += hidden_size
        # self.space_reduction1 = GATBatched(size, hidden_size, edge_dim=1)
        self.space_conv1 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=32, n_blocks=space1_n)
        
        self.space_norm1 = nn.BatchNorm1d(hidden_size)
        
        self.time_conv1 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=32, n_blocks=time1_n)
        time_convs1 = []
        size = hidden_size
        for i in range(time1_n):
            time_convs1.append(BatchedFAConv(size, size))
            # time_convs1.append(GIN(hidden_size, hidden_size))
            # time_convs1.append(BatchedFAConv(size))
            # size *= 2
        # self.time_reduction1 = GATBatched(size, hidden_size, edge_dim=1)
        self.time_norm1 = nn.BatchNorm1d(hidden_size)
        
        # first_reduction = 0.25
        first_reduction = 0.5
        # self.diff_pool1 = TemporalDiffPooling(
        #     hidden_size,
        #     window=window,
        #     n_nodes=n_nodes,
        #     # n_subgraphs=128,
        #     n_subgraphs=8,
        #     # reduction_ratio=.5
        #     reduction_ratio=first_reduction
        #     )
        self.diff_pool1 = SAGPooling(hidden_size, first_reduction, GNN=BatchedFAConv, nonlinearity='leaky_relu')
        
        
        
        space2_n = 3
        time2_n = 3
        self.space_conv2 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=32, n_blocks=space2_n)
        space_convs2 = []
        # self.dim_reduction1 = BatchedFAConv(size, hidden_size * 2)
        size = hidden_size
        
        for i in range(space2_n):
            space_convs2.append(BatchedFAConv(size, size))
            # space_convs2.append(GIN(hidden_size, hidden_size))
            # space_convs2.append(BatchedFAConv(size))
            # size *= 2
        # self.space_reduction2 = GATBatched(size, hidden_size, edge_dim=1)
        self.space_norm2 = nn.BatchNorm1d(hidden_size)
        
        self.time_conv2 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=32, n_blocks=time2_n)
        time_convs2 = []    
        size = hidden_size
        for i in range(time2_n):
            time_convs2.append(BatchedFAConv(size, size))
            # time_convs2.append(BatchedFAConv(size))
            # size *= 2
        # self.time_reduction2 = GATBatched(size, hidden_size, edge_dim=1)
        self.time_norm2 = nn.BatchNorm1d(hidden_size)
        
        second_reduction = 0.4
        # self.diff_pool2 = TemporalDiffPooling(
        #     hidden_size,
        #     window=ceil(window * first_reduction),
        #     n_nodes=n_nodes,
        #     n_subgraphs=2,
        #     reduction_ratio=second_reduction,
        #     # cache_subgraphs=False
        #     reset_subgraph_cache=True,
        #     subgraph_cache_reset_period=1280
        #     )
        self.diff_pool2 = SAGPooling(hidden_size, second_reduction, GNN=BatchedFAConv, nonlinearity='leaky_relu')
        
        space3_n = 3
        time3_n = 3
        self.time_conv3 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=32, n_blocks=time3_n)
        time_convs3 = []
        # self.dim_reduction2 = BatchedFAConv(size, hidden_size)
        size = hidden_size
        for i in range(time3_n):
            time_convs3.append(BatchedFAConv(hidden_size, hidden_size))
            # time_convs3.append(GIN(hidden_size, hidden_size))
        #     time_convs3.append(BatchedFAConv(size))
        #     size *= 2
        # self.time_reduction3 = GATBatched(size, hidden_size, edge_dim=1)
        self.time_norm3 = nn.BatchNorm1d(hidden_size)
        
        self.space_conv3 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=32, n_blocks=space3_n)
        space_convs3 = []
        size=hidden_size
        for i in range(space3_n):
            space_convs3.append(BatchedFAConv(hidden_size, hidden_size))
            # space_convs3.append(GIN(hidden_size, hidden_size))
        #     space_convs3.append(BatchedFAConv(size))
        #     size *= 2
        # self.space_reduction3 = GATBatched(size, hidden_size, edge_dim=1)
        self.space_norm3 = nn.BatchNorm1d(hidden_size)
        
        # for 1 step predictions
        third_reduction = 0.25
        self.diff_pool3 = SAGPooling(hidden_size, third_reduction, GNN=BatchedFAConv, nonlinearity='leaky_relu')
        # self.diff_pool3 = TemporalDiffPooling(
        #     hidden_size,
        #     window=4,
        #     n_nodes=n_nodes,
        #     n_subgraphs=2,
        #     reduction_ratio=.25,
        #     # cache_subgraphs=False
        #     reset_subgraph_cache=True,
        #     subgraph_cache_reset_period=1280
        #     )
        
        self.space_time_conv4 = DenseBlock(in_channels=hidden_size, out_channels=hidden_size, growth_rate=32, n_blocks=space3_n)
        space4_n = 1
        time4_n = 1
        time_convs4 = []
        for i in range(time4_n):
            time_convs4.append(BatchedFAConv(hidden_size, hidden_size))
        self.time_norm4 = nn.BatchNorm1d(hidden_size)
        
        space_convs4 = []
        for i in range(space4_n):
            space_convs4.append(BatchedFAConv(hidden_size, hidden_size))
        self.space_norm4 = nn.BatchNorm1d(hidden_size)
        
        

        self.space1 = nn.ModuleList(space_convs1)
        self.space2 = nn.ModuleList(space_convs2)
        self.space3 = nn.ModuleList(space_convs3)
        self.space4 = nn.ModuleList(space_convs4)
        self.time1 = nn.ModuleList(time_convs1)
        self.time2 = nn.ModuleList(time_convs2)
        self.time3 = nn.ModuleList(time_convs3)
        self.time4 = nn.ModuleList(time_convs4)
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ELU(),
            nn.Linear(hidden_size * 2, out_features)
            )
        
        self.one_hot_encode_exog = OneHotEncodeVars()
        
    # for dense blocks
    def forward(self, x, edge_index, edge_weight):
        # one_hot_encoded = self.one_hot_encode_exog(exog)
        x_orig = x
        
        # x = checkpoint(self.condition_exogenous, x, one_hot_encoded.unsqueeze(-2).expand(-1, -1, self.n_nodes, -1))
        _x = x = F.leaky_relu(checkpoint(self.encoder, x))
        
        x = checkpoint(self.space_conv1, x, edge_index, edge_weight)
        time_adj, time_weights = self.dense_temporal_adj(x.squeeze())
        
        time_graph = self.to_time_nodes(x) #output is t n f, might need to unsqueeze
        
        time_graph = checkpoint(self.time_conv1, time_graph.unsqueeze(0), time_adj, time_weights) #this one is large, might need to use something else in the beginning
        
        pooled_time_graph, pooled_adj, pooled_attrs, batch_idx, perm, score = checkpoint(self.diff_pool1, time_graph.squeeze(), time_adj, time_weights)
        
        time_graph = checkpoint(self.time_conv2, pooled_time_graph.unsqueeze(0), pooled_adj, pooled_attrs)
        
        x = self.time_to_space_structure(time_graph)
        
        x = checkpoint(self.space_conv2, x, edge_index, edge_weight)
        
        time_graph = self.to_time_nodes(x) #output is t n f, might need to unsqueeze
        
        pooled_time_graph, pooled_adj, pooled_attrs, batch_idx, perm, score = checkpoint(self.diff_pool2, time_graph.squeeze(), pooled_adj, pooled_attrs)
        
        time_graph = checkpoint(self.time_conv3, pooled_time_graph.unsqueeze(0), pooled_adj, pooled_attrs)
        
        x = self.time_to_space_structure(time_graph)
        
        x = checkpoint(self.space_conv3, x, edge_index, edge_weight)
        
        time_graph = self.to_time_nodes(x)
        
        pooled_space_time_graph, pooled_adj, pooled_attrs, batch_idx, perm, score = checkpoint(self.diff_pool3, time_graph.squeeze(), pooled_adj, pooled_attrs)
        
        x = checkpoint(self.space_time_conv4, pooled_space_time_graph.unsqueeze(0), pooled_adj, pooled_attrs)
        out = self.readout(x).unsqueeze(0)
        return out
        
        
    def __forward(self, x, edge_index, edge_weight):
        # one_hot_encoded = self.one_hot_encode_exog(exog)
        x_orig = x
        # x_fourier = self.fourier(x)
        # x_positional_encoding = self.positional_encoding(x)
        # x = torch.cat([x, x_fourier, x_positional_encoding], dim=-1)
        # x_ff = self.add_fourier_features(x)
        # x = torch.cat([x, x_ff, x_fourier],dim=-1)
        # x = checkpoint(self.condition_exogenous, x, one_hot_encoded.unsqueeze(-2).expand(-1, -1, self.n_nodes, -1))
        # _x = x = F.leaky_relu(checkpoint(self.encoder, torch.cat([x, one_hot_encoded.unsqueeze(-2).expand(-1, -1, self.n_nodes, -1)], dim=-1)).squeeze())
        _x = x = F.leaky_relu(checkpoint(self.encoder, x).squeeze())
        
        time_adj, time_weights = self.dense_temporal_adj(x)
        _time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        # 1 step time -> 1 step space
        for (space, time) in zip(self.space1, self.time1):
            time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
            time_graph = F.leaky_relu(checkpoint(time, time_graph, time_adj)) + _time_graph
            
            x = self.time_to_space_structure(time_graph.unsqueeze(0)).squeeze()
            x = F.elu(checkpoint(space, x, edge_index)) + _x
            # x = torch.cat([x, _x], dim=-1)
        # x = self.space_reduction1(x, edge_index, edge_weight)
        # x = x + _x
        x = self.space_from_batch_norm(checkpoint(self.space_norm1, self.space_to_batch_norm(x)))
        
        # time_adj = self.build_temporal_adjacency(self.n_nodes, self.window, x.device)
        
        
        _time_graph = time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        # time_adj = self.build_interconnected_timestep_adj(_time_graph)
        
        # for time in self.time1:
        #     time_graph = F.leaky_relu(checkpoint(time, time_graph, time_adj)) + _time_graph
            # time_graph = torch.cat([time_graph, time_graph_], dim=-1)
            
        # time_graph = self.time_reduction1(time_graph, time_adj)
        # time_graph = time_graph + _time_graph
        # time_graph = checkpoint(self.time_norm1, time_graph)
        
        # pooled_time_graph, pooled_adj, pooled_attrs = checkpoint(self.diff_pool1, time_graph.unsqueeze(0), time_adj, time_weights)
        pooled_time_graph, pooled_adj, pooled_attrs, batch_idx, perm, score = checkpoint(self.diff_pool1, time_graph, time_adj, time_weights)
        
        time_graph = _time_graph = pooled_time_graph.squeeze()
        _x = x = self.time_to_space_structure(pooled_time_graph.unsqueeze(0)).squeeze()
        
        # time_graph = _time_graph = checkpoint(self.dim_reduction1, time_graph, pooled_adj)
        for (space, time) in zip(self.space2 ,self.time2):
            time_graph = F.leaky_relu(checkpoint(time, time_graph, pooled_adj))
            # time_graph = torch.cat([time_graph, time_graph_], dim=-1)
            x = self.time_to_space_structure(time_graph.unsqueeze(0)).squeeze()
            x = F.elu(checkpoint(space, x, edge_index))
            time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        
        # time_graph = self.time_reduction2(time_graph, pooled_adj, pooled_attrs)
        # time_graph = time_graph + pooled_time_graph.squeeze()
        time_graph = checkpoint(self.time_norm2, time_graph)
        
        # _x = x = self.time_to_space_structure(time_graph.unsqueeze(0)).squeeze()
        
        # for space in self.space2:
        #     x = F.leaky_relu(checkpoint(space, x, edge_index)) + _x
            # x = torch.cat([x, x_], dim=-1)
        # x = self.space_reduction2(x, edge_index, edge_weight)
        # x = x + _x
        # x = self.space_from_batch_norm(checkpoint(self.space_norm2, self.space_to_batch_norm(x)))
        
        # _time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        
        # pooled_time_graph, pooled_adj, pooled_attrs = checkpoint(self.diff_pool2, time_graph.unsqueeze(0), pooled_adj, pooled_attrs)
        pooled_time_graph, pooled_adj, pooled_attrs, batch_idx, perm, score = checkpoint(self.diff_pool2, time_graph, pooled_adj, pooled_attrs)
        
        time_graph = _time_graph = pooled_time_graph.squeeze()
        _x = x = self.time_to_space_structure(pooled_time_graph.unsqueeze(0)).squeeze()
        # time_graph = _time_graph = checkpoint(self.dim_reduction2, time_graph, pooled_adj)
        for (space, time) in zip(self.space3 ,self.time3):
            time_graph = F.leaky_relu(checkpoint(time, time_graph, pooled_adj))
            # time_graph = torch.cat([time_graph, time_graph_], dim=-1)
            x = self.time_to_space_structure(time_graph.unsqueeze(0)).squeeze()
            x = F.elu(checkpoint(space, x, edge_index))
            time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        
        # time_graph = self.time_reduction3(time_graph, pooled_adj, pooled_attrs)
        # time_graph = time_graph + pooled_time_graph.squeeze()
        time_graph = checkpoint(self.time_norm3, time_graph)
        
        # x = self.time_to_space_structure(time_graph.unsqueeze(0)).squeeze()
        # for space in self.space3:
        #     x = F.leaky_relu(checkpoint(space, x, edge_index)) +  _x
            # x = torch.cat([x, x_], dim=-1)
        # x = self.space_reduction3(x, edge_index, edge_weight)
        # x = x + _x
        # x = self.space_from_batch_norm(checkpoint(self.space_norm3, self.space_to_batch_norm(x)))
        
        # #----------------- to size 1 ----------------------------
        # _time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        # pooled_time_graph, pooled_adj, pooled_attrs = checkpoint(self.diff_pool3, time_graph.unsqueeze(0), pooled_adj, pooled_attrs)
        
        # time_graph = pooled_time_graph.squeeze()
        # for time in self.time4:
        #     time_graph = checkpoint(time, time_graph, pooled_adj)
        # time_graph = time_graph + pooled_time_graph.squeeze()
        # time_graph = checkpoint(self.time_norm4, time_graph)
        
        # _x = x = self.time_to_space_structure(time_graph.unsqueeze(0)).squeeze()
        # for space in self.space4:
        #     x = F.leaky_relu(checkpoint(space, x, edge_index))
        #     # x = torch.cat([x, x_], dim=-1)
        # # x = self.space_reduction3(x, edge_index, edge_weight)
        # x = x + _x
        # x = self.space_from_batch_norm(checkpoint(self.space_norm4, self.space_to_batch_norm(x.unsqueeze(0))))
        pooled_spacetime_graph, pooled_adj, pooled_attrs, batch_idx, perm, score = checkpoint(self.diff_pool3, time_graph, pooled_adj, pooled_attrs)
        time_graph = pooled_spacetime_graph
        
        # _x = x = self.time_to_space_structure(pooled_time_graph.unsqueeze(0)).squeeze()
        # print(f"time_graph: {time_graph.size()}, x: {x.size()}")
        for i, (space, time) in enumerate(zip(self.space4, self.time4)):
            # print(time_graph.size())
            if torch.isnan(x).any():
                print('iteration',i)
            x = F.elu(checkpoint(time, time_graph, pooled_adj))
            # time_graph = torch.cat([time_graph, time_graph_], dim=-1)
            # x = self.time_to_space_structure(time_graph.unsqueeze(0)).squeeze()
            # print(x.size())
            x = F.elu(checkpoint(space, x, edge_index))
            # print(x.size())
            # time_graph = self.to_time_nodes(x.unsqueeze(0)).squeeze()
        assert not torch.isnan(x).any()
        
        # x = self.time_to_space_structure(time_graph.unsqueeze(0))
        #----------------- to size 1 ----------------------------
        out = self.readout(x.unsqueeze(0)).unsqueeze(0)
        res =  out + x_orig[:, -1]
        if torch.isnan(res).any():
            print(x_orig)
            print(x)
            assert not torch.isnan(res).any()
        return res
        
        
        
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
    
    def build_interconnected_timestep_adj(self, graph):
        if self.static_arrow_of_time == None:
            n = graph.size(-2)
            
            adj_mask = 0
            
            # for i in range(0, n, self.node_group_size):
            for i in range(0, n, n*2):
                adj_mask = adj_mask + torch.diag(torch.ones(n), diagonal=i)[:n,:n]
                
            timeflow = torch.tensor(adj_mask, device=graph.device)
            self.static_arrow_of_time, edge__attrs = dense_to_sparse(timeflow)
            
            
        return self.static_arrow_of_time
    
    def dense_temporal_adj(self, graph):
        if self.static_arrow_of_time == None:
            t,n,f = graph.size()
            # step_size = graph.size(node_dim)
            
            connectivity = torch.sparse.spdiags(torch.ones(t * n), torch.tensor([0]), (t * n, t * n)) #self loops
            # connectivity = 0
            for i in range(n, t * n, n):
                connectivity = connectivity + torch.sparse.spdiags(torch.ones(t * n), torch.tensor([i]), (t * n, t * n))
                
            adj, weights =  dense_to_sparse(connectivity.to_dense())
            self.static_arrow_of_time = adj.to(graph.device)
            self.static_weights = weights.to(graph.device)
            
        return self.static_arrow_of_time, self.static_weights
        


# class BatchedFAConv(nn.Module):
#     def __init__(self, in_channels, out_channels, eps=0.2, dropout=0.3, aggrs=['SumAggregation']) -> None:
#         super(BatchedFAConv, self).__init__()
#         # self.eps = nn.Parameter(torch.randn(1))
#         # fa_convs_for_aggr = [FAConv(in_channels, eps, dropout, cached=False, add_self_loops=True, aggr=ag) for ag in aggrs] 
#         # self.fa_convs_for_aggr = nn.ModuleList(fa_convs_for_aggr)
#         self.fa_conv = FAConv(
#             in_channels,
#             eps,
#             dropout,
#             cached=False,
#             add_self_loops=True,
#             # aggr=aggr.VariancePreservingAggregation()
#             # aggr=GenAgg(MLPAutoencoder, layer_sizes=(1,8,8,16))
#             # aggr=aggr.MultiAggregation(
#             #     aggrs=[
#             #         # aggr.SoftmaxAggregation(t=0.1, learn=True),
#             #         # aggr.SoftmaxAggregation(t=1, learn=True),
#             #         # aggr.SoftmaxAggregation(t=10, learn=True)
#             #         # aggr.PowerMeanAggregation(learn=True, p=1.0),
#             #         aggr.SumAggregation(),
#             #         aggr.MeanAggregation(),
#             #         aggr.StdAggregation(),
#             #         aggr.VarAggregation()
#             #         ],
#             #     mode='proj',
#             #     mode_kwargs={
#             #         # 'num_heads':8,
#             #         'in_channels': in_channels,
#             #         'out_channels': in_channels
#             #     }
#             # )
#         )
#         self.norm = nn.BatchNorm1d(in_channels)
#         self.readout = nn.Sequential(
#             nn.Linear(in_channels, in_channels),
#             nn.ELU(),
#             nn.Linear(in_channels, out_channels)
#             )
        
#     def forward(self, x, edge_index):
#         if x.dim() == 3:
#             res = []
#             for graph in x:
#                 # res.append(self.readout(self._stack_aggrs(graph, edge_index)))
#                 conved = self.fa_conv(graph,graph, edge_index)
#                 res.append(self.readout(self.norm(conved)))
#             return torch.stack(res)
        
#         if x.dim() == 2:
#             # return self.readout(self._stack_aggrs(x,edge_index))
#             return self.readout(self.fa_conv(x,x,edge_index))
        
#     def _stack_aggrs(self, x, edge_index):
#         aggrs = []
#         for fa_conv in self.fa_convs_for_aggr:
#             aggrs.append(fa_conv(x,x,edge_index))
#         return torch.cat(aggrs, dim=-1)
        
#     def reset_parameters(self):
#         # [gcn.reset_parameters() for gcn in self.fa_convs_for_aggr]
#         self.fa_conv.reset_parameters()
        
class GIN(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, hidden_channels=[512, 1024, 512]):
        super(GIN, self).__init__()
        channels = [in_channels] + hidden_channels + [out_channels]
        mlp = MLP(channels, dropout=0.3, act='elu')
        self.gin = GINConv(mlp, eps=0.1, train_eps=True)
        
    def forward(self, x, edge_index):
        if x.dim() == 3:
            res = []
            for graph in x:
                res.append(self.gin(graph, edge_index))
            return torch.stack(res)
        if x.dim() == 2:
            return self.gin(x, edge_index)
    
    def reset_parameters(self):
        self.gin.reset_parameters()
    
        
class Fourier(nn.Module):
	def __init__(self, input_channels=256, output_channels=256, scale=10):
		super(Fourier, self).__init__()
		self.b = torch.randn(input_channels, output_channels)*scale
	def forward(self, v):
		x_proj = torch.matmul(2*torch.pi*v, self.b.to(v.device))
		return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)


class SequencePositionalEncoding(nn.Module):
    def __init__(self, n_nodes, out_channels):
        super(SequencePositionalEncoding, self).__init__()
        self.encoder = PositionalEncoding(out_channels)
        self.positional_encoding_transform = Rearrange('(n p) f -> n (p f)', n=n_nodes)
        
    def forward(self, x):
        batches = []
        for graphs in x:
            _graphs = []
            for graph in graphs:
                _graphs.append(self.positional_encoding_transform(self.encoder(graph)))
            batches.append(torch.stack(_graphs))
            
        return torch.stack(batches)
    
class AttentativeResidual(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(AttentativeResidual,self).__init__()
        self.query_fn = nn.Linear(input_channels, output_channels, bias=False)
        self.key_fn = nn.Linear(input_channels,output_channels, bias=False)
        self.value_fn = nn.Linear(input_channels, output_channels, bias=False)
        
    def forward(self, x, residual_source):
        if residual_source.dim() == 3:
            residual_source = residual_source.unsqueeze(1).expand(-1,x.size(1) ,-1,-1)
        q = self.query_fn(x)
        k = self.key_fn(residual_source)
        v = self.value_fn(residual_source)
        
        affinities = q.squeeze() @ k.squeeze().transpose(-2, -1)
        normalized_affinities = F.softmax(affinities, dim=-1)
        
        residual = normalized_affinities @ v
        
        return x + residual
        
class FourierFeatures(nn.Module):
    def __init__(self, orders=[1,2]) -> None:
        super(FourierFeatures, self).__init__()
        self.orders = orders
            
    def forward(self, x):
        fourier_features = []
        for order in self.orders:
            fourier_features.append(torch.sin(x / 2**order))
            fourier_features.append(torch.cos(x / 2**order))
            
        return torch.cat([x, torch.cat(fourier_features, dim=-1)], dim=-1)

class OneHotEncodeVars(nn.Module):
    def __init__(self) -> None:
        super(OneHotEncodeVars, self).__init__()
        
    def forward(self, exog_data):
        #exog_data - b, t, f
        exog_data = exog_data.to(torch.int64)
        hours = exog_data[:, :, 0]
        days = exog_data[:, :, 1]
        months = exog_data[:, :, 2]
        
        hours_one_hot = torch.nn.functional.one_hot(hours // 6, 4)
        # weekdays_one_hot = torch.nn.functional.one_hot(weekdays, 7)
        days_scaled = days / 31
        months_one_hot = torch.nn.functional.one_hot(months - 1, 12)
        return torch.cat([hours_one_hot, days_scaled.unsqueeze(-1), months_one_hot], dim=-1)