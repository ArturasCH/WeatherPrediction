import torch
import torch.nn as nn
import numpy as np
from tsl.nn.blocks import ResidualMLP
from torch.utils.checkpoint import checkpoint

from models.layers.coarsen import CoarseGraphDecoder, FineToCoarseEncoder
from models.layers.MLPGraphEncoder import MLPGraphEncoder


class MLPStepthrough(nn.Module):
    def __init__(self,
                 lat_lons,
                n_nodes=2048,
                window=20,
                input_size=26,
                hidden_size=256,
                out_features=26,
                horizon=4,
                #  n_layers=12,
                n_layers=6,
                edge_embedding_size=10):
        super(MLPStepthrough, self).__init__()
        
        
        # ------------ encoding and preprocessing --------------------
        self.encode_graph = ResidualMLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        self.attribute_encoder = ResidualMLP(
            input_size=2,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        
        self.coarsen_medium = FineToCoarseEncoder(
            lat_lons,
            resolution=1,
            input_dim=input_size,
            output_dim=hidden_size,
            output_edge_dim=hidden_size,
            hidden_dim_processor_node=hidden_size,
            hidden_dim_processor_edge=hidden_size,
            )
        
        self.coarsen_large = FineToCoarseEncoder(
            lat_lons,
            resolution=0,
            input_dim=input_size,
            output_dim=hidden_size,
            output_edge_dim=hidden_size,
            hidden_dim_processor_node=hidden_size,
            hidden_dim_processor_edge=hidden_size,
            )
        
        self.decode_coarse_graph_medium = CoarseGraphDecoder(
            lat_lons,
            resolution=1,
            input_dim=hidden_size
            )
        
        self.decode_coarse_graph_large = CoarseGraphDecoder(
            lat_lons,
            resolution=0,
            input_dim=hidden_size
            )
        
        # ------------ spatial graph convs -------------------------
        self.fine_grained_gcns = nn.ModuleList([])
        self.medium_grained_gcns = nn.ModuleList([])
        self.coarse_grained_gcns = nn.ModuleList([])
        for layer in range(n_layers):
            self.fine_grained_gcns.append(MLPGraphEncoder(
                in_channels=hidden_size,
                hidden_channels=hidden_size,
                out_channels=hidden_size,
                edge_in_channels=hidden_size,
                edge_hidden_channels=hidden_size,
                edge_out_channels=hidden_size
            ))
            self.medium_grained_gcns.append(MLPGraphEncoder(
                in_channels=hidden_size,
                hidden_channels=hidden_size,
                out_channels=hidden_size,
                edge_in_channels=hidden_size,
                edge_hidden_channels=hidden_size,
                edge_out_channels=hidden_size
            ))
            self.coarse_grained_gcns.append(MLPGraphEncoder(
                in_channels=hidden_size,
                hidden_channels=hidden_size,
                out_channels=hidden_size,
                edge_in_channels=hidden_size,
                edge_hidden_channels=hidden_size,
                edge_out_channels=hidden_size
            ))
        # --------------------- step through time ------------------------
        self.s_time_adj, self.s_edge_attrs = self.get_time_transition(n_nodes)
        self.m_time_adj, self.m_edge_attrs = self.get_time_transition(842)
        self.l_time_adj, self.l_edge_attrs = self.get_time_transition(122)
        
        self.st_attribute_encoder = ResidualMLP(
            input_size=2,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        self.mt_attribute_encoder = ResidualMLP(
            input_size=2,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        self.lt_attribute_encoder = ResidualMLP(
            input_size=2,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        
        self.s_temporal_gcn = MLPGraphEncoder(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            edge_in_channels=hidden_size,
            edge_hidden_channels=hidden_size,
            edge_out_channels=hidden_size,
        )
        self.m_temporal_gcn = MLPGraphEncoder(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            edge_in_channels=hidden_size,
            edge_hidden_channels=hidden_size,
            edge_out_channels=hidden_size,
        )
        self.l_temporal_gcn = MLPGraphEncoder(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            edge_in_channels=hidden_size,
            edge_hidden_channels=hidden_size,
            edge_out_channels=hidden_size,
        )
            
            
        self.readout = ResidualMLP(
            input_size=hidden_size * 3,
            hidden_size=hidden_size,
            output_size=out_features,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        
        
    def forward(self, x, edge_index, edge_weight):
        mid_coarse_graph, mid_coarse_adj, mid_coarse_edge_attr = checkpoint(self.coarsen_medium, x)
        large_coarse_graph, large_coarse_adj, large_coarse_edge_attr = checkpoint(self.coarsen_large, x)
        
        encoded_graph = checkpoint(self.encode_graph, x)
        edge_attr = self.get_edge_attr(edge_weight)
        
        self.s_time_adj, s_edge_attrs = self.s_time_adj.to(x.device), self.st_attribute_encoder(self.s_edge_attrs.to(x.device))
        self.m_time_adj, m_edge_attrs = self.m_time_adj.to(x.device), self.mt_attribute_encoder(self.m_edge_attrs.to(x.device))
        self.l_time_adj, l_edge_attrs = self.l_time_adj.to(x.device), self.lt_attribute_encoder(self.l_edge_attrs.to(x.device))
        
        t = x.size(1)
        
        # current = encoded_graph[:,0]
        # next = encoded_graph[:, 1]
        # ------
        # self.s_temporal_gcn()
        
        sg_current, sg_next = encoded_graph[:, 0], encoded_graph[:, 1]
        mg_current, mg_next = mid_coarse_graph[:, 0], mid_coarse_graph[:, 1]
        lg_current, lg_next = large_coarse_graph[:, 0], large_coarse_graph[:, 1]
        
        for step in range(2, t+1, 1):
            # do message passing on all resolutions for timestep t
            # sg = encoded_graph[:,step]
            # mg = mid_coarse_graph[:, step]
            # lg = large_coarse_graph[:, step]
            
            for s_gcn, m_gcn, l_gcn in zip(self.fine_grained_gcns, self.medium_grained_gcns, self.coarse_grained_gcns):
                sg_current, edge_attr = checkpoint(s_gcn, (sg_current, sg_current), edge_index, edge_attr)
                mg_current, mid_coarse_edge_attr = checkpoint(m_gcn, (mg_current, mg_current), mid_coarse_adj, mid_coarse_edge_attr)
                lg_current, large_coarse_edge_attr = checkpoint(l_gcn, (lg_current, lg_current), large_coarse_adj, large_coarse_edge_attr)
                
                
            # message passing into next timestep
            # sg_current, s_edge_attrs = checkpoint(self.s_temporal_gcn, (sg_current, sg_next), self.s_time_adj.to(x.device), s_edge_attrs)
            # mg_current, m_edge_attrs = checkpoint(self.m_temporal_gcn, (mg_current, mg_next), self.m_time_adj.to(x.device), m_edge_attrs)
            # lg_current, l_edge_attrs = checkpoint(self.l_temporal_gcn, (lg_current, lg_next), self.l_time_adj.to(x.device), l_edge_attrs)
            
            sg_current, s_edge_attrs = self.s_temporal_gcn((sg_current, sg_next), self.s_time_adj.to(x.device), s_edge_attrs)
            mg_current, m_edge_attrs = self.m_temporal_gcn((mg_current, mg_next), self.m_time_adj.to(x.device), m_edge_attrs)
            lg_current, l_edge_attrs = self.l_temporal_gcn((lg_current, lg_next), self.l_time_adj.to(x.device), l_edge_attrs)
            
            if step > t:
                sg_next = encoded_graph[:, step]
                mg_next = mid_coarse_graph[:, step]
                lg_next = large_coarse_graph[:, step]
            
        
        mid_to_fine = self.decode_coarse_graph_medium(mg_current)
        coarse_to_fine = self.decode_coarse_graph_large(lg_current)
        # print(f"{sg_current.size()}, {mid_to_fine.size()}, {coarse_to_fine.size()}")
        x_stacked = torch.cat([sg_current.unsqueeze(1), mid_to_fine, coarse_to_fine], dim=-1)
        return self.readout(x_stacked)
                
                
    def get_edge_attr(self, edge_weight):
        edge_attr = torch.stack([
            torch.sin(edge_weight),
            torch.cos(edge_weight)
        ], dim=-1).to(edge_weight.device)
        return checkpoint(self.attribute_encoder, edge_attr)
    
    def get_time_transition(self, n_nodes):
        nodes = torch.arange(n_nodes)
        adj = torch.stack([nodes, nodes], dim=0)
        weights = nn.Parameter(torch.zeros(n_nodes, 2))
        weights = nn.init.xavier_uniform(weights)
        
        return adj, weights