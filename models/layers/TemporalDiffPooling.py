import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch_geometric.nn import dense_diff_pool, dense_mincut_pool, DMoNPooling, SAGPooling
from torch_geometric.utils import dense_to_sparse, sort_edge_index

from models.layers.GAT_Batched import GATBatched
from models.layers.BatchedFAConv import BatchedFAConv
from models.layers.k_hop_subgraph import k_hop_subgraph
from math import ceil

# pool subgraphs
# test splitting and reconstructing subgraphs
class TemporalDiffPooling(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 window,
                 n_nodes,
                #  node_group_size,
                 n_subgraphs,
                 reduction_ratio = 0.5,
                #  cache_subgraphs=True,
                 reset_subgraph_cache=False,
                 subgraph_cache_reset_period=64
                 ) -> None:
        super(TemporalDiffPooling, self).__init__()
        # self.cache_subgraphs = cache_subgraphs
        self.reset_subgraph_cache = reset_subgraph_cache
        self.subgraph_cache_reset_period = subgraph_cache_reset_period
        self.call_counter = 0
        
        self.in_channels = in_channels
        self.window = window
        self.resulting_timesteps = ceil(window * reduction_ratio)
        self.n_nodes = n_nodes
        # self.node_group_size = node_group_size
        self.n_subgraphs = n_subgraphs
        self.space_nodes_in_subgraph = self.n_nodes // n_subgraphs
        clusters = ceil(window * self.space_nodes_in_subgraph * reduction_ratio)
        self.s_conv = BatchedFAConv(in_channels, clusters, heads=1, dropout=0.3)
        # self.s_conv = nn.Sequential(
        #     nn.Linear(in_channels, clusters),
        #     nn.ELU(),
        #     nn.Linear(clusters, clusters * 2),
        #     nn.ELU(),
        #     nn.Linear(clusters * 2, clusters)
        #     )
        self.dmon_pool = DMoNPooling(in_channels, k=clusters, dropout=0.3)
        # self.sag_pool = SAGPooling(in_channels, reduction_ratio, nonlinearity='leaky_relu')
        self.flatten_batches = Rearrange('b n f -> (n b) f')
        self.linear_time_adj = None
        self.timeflow_mask = None
        self.group_split = Rearrange('c b (n s) f -> c b n s f', s = self.space_nodes_in_subgraph) #split cluster nodes into nodes and steps
        self.restack_clusters = Rearrange('c b n s f -> b (n c s) f')
        self.node_selectors = [] 
        self.adj_matrices = [] 
        self.original_adj_indices = []
        self.past_edge_mask = torch.tensor(1)
        
        self.past_edge_map = None
        # self.similarity_check_counter = 0
        self.cross_entropy_loss = 0
        self.loss_1 = 0
        self.loss_2 = 0
        self.cross_entropy_call_counter = 0
        self.accumulate_aux_loss = 128
        # self.register_forward_hook(self.backprop_auxilary_losses)
        
        
    def backprop_auxilary_losses(self, module, grad_input, grad_output):
        # if self.cross_entropy_loss.requires_grad and self.cross_entropy_call_counter == 32 and self.cross_entropy_loss != 0:
        #     (self.cross_entropy_loss / self.cross_entropy_call_counter).backward()
        #     self.cross_entropy_call_counter = 0
        # elif self.cross_entropy_loss.requires_grad == False and self.cross_entropy_loss != 0:
        #     self.cross_entropy_loss = 0
        if self.loss_1.requires_grad:
            if self.loss_1 != 0:
                (self.loss_1 / self.accumulate_aux_loss).backward()
            if self.loss_2 != 0:
                (self.loss_2 / self.accumulate_aux_loss).backward()
            # self.accumulate_aux_loss = 0
        elif self.loss_1.requires_grad == False and (self.loss_1 != 0 or self.loss_2 != 0):
            self.loss_1 = 0
            self.loss_2 = 0
        
    def forward(self, temporal_graph, temporal_adj, edge_weights=None):
        self.call_counter += 1
        # self.cross_entropy_call_counter += 1
        # self.similarity_check_counter += 1
        self.clear_cache_if_needed()

        subgraphs, subgraph_adjs, original_edge_idx, edge_masks = self.split_to_subgraphs(temporal_graph, temporal_adj, self.space_nodes_in_subgraph, self.n_nodes)
        # s = []
        # for (subgraph, adj) in zip(subgraphs, subgraph_adjs):
        #     # s.append(self.s_conv(subgraph, adj))
        #     s.append(self.s_conv(subgraph))
        
        # -------------- SAG pool flow -------------------#
        # pooled_subgraphs = []
        # pooled_adjs = []
        # pooled_attrs = []
        
        # print('Sizes');
        # for (graph, adj, edge_mask) in zip(subgraphs, subgraph_adjs, edge_masks):
        #     print('graph size', graph.size())
        #     if edge_weights != None:
        #         edges = edge_weights[edge_mask]
        #     else:
        #         edges = None
                
        #     pooled_graph, pooled_adj, pooled_attr, batch_idx, perm, score = self.sag_pool(graph.squeeze(), adj, edges)
        #     # pooled_adj, pooled_attr = sort_edge_index(pooled_adj, pooled_attr)
        #     pooled_subgraphs.append(pooled_graph.unsqueeze(0))
        #     pooled_adjs.append(pooled_adj)
        #     pooled_attrs.append(pooled_attr)
        # print('Sizes off')
        
        # pooled_subgraphs = torch.stack(pooled_subgraphs)
        # pooled_adjs = torch.cat(pooled_adjs, dim=-1)
        # pooled_attrs = torch.cat(pooled_attrs, dim=-1)
        
        # temporal_pooled = self.restore_graph_shape(pooled_subgraphs, self.space_nodes_in_subgraph)
        
        # return temporal_pooled, pooled_adjs, pooled_attrs
        # -------------- SAG pool flow -------------------#
        
        # ----------------------------------- dense pooling stuff -----------------------------------
        clustered_graphs = []
        clustered_adjs = []
        # for (subgraph, adj, edge_mask, _s) in zip(subgraphs, subgraph_adjs, edge_masks, s):
        for (subgraph, adj, edge_mask) in zip(subgraphs, subgraph_adjs, edge_masks):
            if edge_weights != None:
                edges = edge_weights[edge_mask]
            else:
                edges = None
            # clustered, clustered_adj, l, e = dense_diff_pool(subgraph, self.to_dense_adj(adj, edges).expand(subgraph.size(0),-1,-1), _s)
            # clustered, clustered_adj, min_cut_loss, ortho_loss = dense_mincut_pool(subgraph, self.to_dense_adj(adj, edges, (subgraph.size(-2), subgraph.size(-2))).expand(subgraph.size(0),-1,-1), _s)
            cluster_assignment, clustered, clustered_adj, spectral_loss, ortho_loss, cluster_loss = self.dmon_pool(subgraph, self.to_dense_adj(adj, edges, (subgraph.size(-2), subgraph.size(-2))).expand(subgraph.size(0),-1,-1))
            # self.cross_entropy_loss += (e / self.n_subgraphs)
            self.loss_1 += (spectral_loss / self.n_subgraphs)
            self.loss_2 += (cluster_loss / self.n_subgraphs)
            clustered_graphs.append(clustered)
            clustered_adjs.append(clustered_adj)
            
        clustered_graphs = torch.stack(clustered_graphs) # list of c,b,n,f
        clustered_adjs = torch.cat(clustered_adjs) # list of b,n,n
        
        
        
        time_flow_mask = self.build_interconnected_timestep_mask(clustered_graphs)
        clustered_adjs = clustered_adjs * time_flow_mask
        
        temporal_pooled = self.restore_graph_shape(clustered_graphs, self.space_nodes_in_subgraph)
        coo_adjacencies = []
        _edge_weights = []
        for adj, original_adj, adj_reindexed in zip(clustered_adjs, original_edge_idx, subgraph_adjs):
            _sparse, attr = dense_to_sparse(adj)
            mapping = torch.stack([adj_reindexed.unique(), original_adj.unique()]).T
            remaped_sparse = torch.zeros_like(_sparse)
            remaped_sparse[0] = self.remap_indices(_sparse[0], mapping)
            remaped_sparse[1] = self.remap_indices(_sparse[1], mapping)
            
            coo_adjacencies.append(remaped_sparse)
            _edge_weights.append(attr)
            
        new_weights = torch.cat(_edge_weights)
        new_adj = torch.cat(coo_adjacencies, dim=-1)
        
        return temporal_pooled, new_adj, new_weights
    # ----------------------------------- dense pooling stuff -----------------------------------
    
    def reset_cached_subgraph_info(self):
        self.node_selectors = [] 
        self.adj_matrices = [] 
        self.original_adj_indices = []
        self.edge_masks = []
        
    def clear_cache_if_needed(self):
        if self.reset_subgraph_cache and self.call_counter % self.subgraph_cache_reset_period == 0 and self.call_counter != 0:
            self.reset_cached_subgraph_info()
            self.call_counter = 0
            
    
    def split_to_subgraphs(self, temporal_graph, temporal_adj, group_size=512, space_graph_size=2048, relabel_nodes=True):
        if len(self.node_selectors) == 0:
            adj_matrices = []
            range_select = torch.arange(0, group_size)
            original_adj_indices = []
            edge_masks = []
            node_selectors = []
            for s in range(0, space_graph_size, group_size):
                if s != 0:
                    range_select = range_select + group_size
                node_selector, selected_adj, mapping, edge_mask, original_adj_index = k_hop_subgraph(
                    range_select,
                    space_graph_size,
                    temporal_adj,
                    flow='target_to_source',
                    directed=True,
                    relabel_nodes=relabel_nodes
                    )
                
                # subgraph = temporal_graph[:, node_selector]
                node_selectors.append(node_selector)
                # subgraphs.append(subgraph)
                adj_matrices.append(selected_adj)
                original_adj_indices.append(original_adj_index)
                edge_masks.append(edge_mask)
                
                # mask_tensor = torch.tensor(edge_masks, dtype=torch.bool)

            # if self.similarity_check_counter % 10 == 0:
                
            #     print(f"Edge mask same: {torch.all(torch.eq(self.past_edge_mask, mask_tensor))}")
            #     self.past_edge_mask = mask_tensor
            #     self.similarity_check_counter = 0                

            self.node_selectors = node_selectors
            self.adj_matrices = adj_matrices
            self.original_adj_indices = original_adj_indices
            self.edge_masks = edge_masks
            
        subgraphs = []
        for selector in self.node_selectors:
            subgraphs.append(temporal_graph[:, selector])
            
        return subgraphs, self.adj_matrices, self.original_adj_indices, self.edge_masks
        
    
    def _get_linear_time_adjacency(self):
        if self.linear_time_adj == None:
            from_node = torch.cat([torch.tensor([0.]),torch.arange(0, (self.window * self.node_group_size)-(self.node_group_size + 1), self.node_group_size)], dim=0)
            to_node = torch.cat([torch.tensor([0.],dtype=torch.float32),torch.arange(self.node_group_size, (self.window * self.node_group_size), self.node_group_size)], dim=0)
            
            
            grouped_time_arrow = torch.stack((from_node, to_node), dim=0)
            time_adjacencies = []
            for i in range(self.node_group_size):
                time_adjacencies.append(grouped_time_arrow + i)
            
            group_time_adjacency = torch.cat(time_adjacencies, dim=-1)
            self.linear_time_adj = group_time_adjacency
        
        return self.linear_time_adj
    
    def build_interconnected_timestep_mask(self, clustered_node_input):
        if self.timeflow_mask == None:
            n = clustered_node_input.size(-2)
            
            adj_mask = 0
            
            # for i in range(0, n, self.node_group_size):
            for i in range(0, n, self.space_nodes_in_subgraph):
                adj_mask = adj_mask + torch.diag(torch.ones(n), diagonal=i)[:n,:n]
                
            self.timeflow_mask = torch.tensor(adj_mask, device=clustered_node_input.device)
            
            
        return self.timeflow_mask
    
    def to_dense_adj(self, coo_adj, edge_weights=None, size=None):
        if edge_weights == None:
            edge_weights = torch.ones(coo_adj.size(1))
            size = (edge_weights.size(0), edge_weights.size(0))
            dense_pseudo_time_arrow =torch.sparse_coo_tensor(coo_adj, edge_weights, size=size, device=coo_adj.device).to_dense()
        elif size != None:
            dense_pseudo_time_arrow =torch.sparse_coo_tensor(coo_adj, edge_weights, size=size).to_dense()
        else:
            dense_pseudo_time_arrow =torch.sparse_coo_tensor(coo_adj, edge_weights).to_dense()

        return torch.tensor(dense_pseudo_time_arrow, dtype=torch.float32, device=coo_adj.device)
    
    def restore_graph_shape(self, subgraphs, spatial_nodes_in_subgraph):
        '''
        temporal graph subgraphs restored
        subgraphs - clusters, batch, nodes, featuers structure
        
        returns batch,nodes,features shaped graph in correct order
        '''
        split_graph = self.group_split(subgraphs)
        stacked = self.restack_clusters(split_graph)
        return stacked
    
    def remap_indices(self, source, mapping):
        # source - 1d tensor n
        # mapping n, 2 tensor of mapping from [:, 0] to [:, 1]
        mask = source == mapping[:, :1]
        mapped_values = (1 - mask.sum(dim=0)) * source + (mask * mapping[:, 1:]).sum(dim=0)
        return mapped_values
    


def to_temporal_graph(spatial_graphs, node_dim=-2):
    # spatial_graphs - b,t,n,f
    # spatial_graphs = batch.input.x
    b,t,n,f = spatial_graphs
    step_size = spatial_graphs.size(node_dim)
    to_time_nodes = Rearrange('b t n f -> b (t n) f')
    time_nodes = to_time_nodes(spatial_graphs)
    
    arrow_of_time = []
    for i in range(n): # for every node
        source_node = i
        for j in range(t-1):
            
            if j == 0:
                # self loop?
                arrow_of_time.append([source_node, source_node])
            target_node = source_node + step_size
            arrow_of_time.append([source_node, target_node])
            source_node = target_node
            
    arrow_of_time = torch.tensor(arrow_of_time).T
    
    return time_nodes, arrow_of_time


def select_group_of_node_flows(temporal_graph, temporal_adjacency, nodes_in_group, n_timesteps, start_node):
    group_start_index = start_node * n_timesteps
    group_end_index = (nodes_in_group * n_timesteps) + group_start_index
    group_of_nodes_time_arrow = temporal_adjacency[:, group_start_index:group_end_index]
    node_group_selector = group_of_nodes_time_arrow.unique()
    
    return temporal_graph[:, node_group_selector]

# restore original shape graph nodes
# def restore_graph_shape(subgraphs, spatial_nodes_in_cluster):
#     '''
#     temporal graph subgraphs restored
#     '''
#     _stacked = torch.stack(subgraphs)
#     group_split = Rearrange('g b (n s) f -> g b n s f', s = spatial_nodes_in_cluster)
#     _split = group_split(_stacked)
#     time_stack = Rearrange('g b n s f -> b (n g s) f')
#     _time_stacked = time_stack(_split)
#     return _time_stacked