import torch
import torch.nn as nn
from tsl.nn.blocks import ResidualMLP
from torch.utils.checkpoint import checkpoint

from models.layers.coarsen import CoarseGraphDecoder, FineToCoarseEncoder
from models.layers.DenseGraphConv import DenseBlock
from models.layers.A3RFCN import A3TGCNBlock

class MultiResGCN(nn.Module):
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
        super(MultiResGCN, self).__init__()
        
        self.encode_graph = ResidualMLP(
            input_size=input_size,
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
        
        
        self.high_res_rnn = nn.ModuleList([])
        self.mid_res_rnn = nn.ModuleList([])
        self.low_res_rnn = nn.ModuleList([])
        
        for layer in range(n_layers):   
            self.high_res_rnn.append(A3TGCNBlock(
                in_channels=hidden_size,
                out_channels=hidden_size,
                periods=window
            ))
            self.mid_res_rnn.append(A3TGCNBlock(
                in_channels=hidden_size,
                out_channels=hidden_size,
                periods=window
            ))
            self.low_res_rnn.append(A3TGCNBlock(
                in_channels=hidden_size,
                out_channels=hidden_size,
                periods=window
            ))
            
        self.readout = ResidualMLP(
            input_size=hidden_size * 3,
            hidden_size=hidden_size,
            output_size=out_features,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
            
            
    def forward(self, x, edge_index):
        mid_coarse_graph, mid_coarse_adj, mid_coarse_edge_attr = checkpoint(self.coarsen_medium, x)
        large_coarse_graph, large_coarse_adj, large_coarse_edge_attr = checkpoint(self.coarsen_large, x)
        
        encoded_graph = checkpoint(self.encode_graph, x)
        
        h_fine = torch.zeros(*encoded_graph.shape[:-1], encoded_graph.size(-1)).to(x.device)
        h_mid = torch.zeros(*mid_coarse_graph.shape[:-1], mid_coarse_graph.size(-1)).to(x.device)
        h_coarse = torch.zeros(*large_coarse_graph.shape[:-1], large_coarse_graph.size(-1)).to(x.device)
        
        for high_res_rnn, mid_res_rnn, low_res_rnn in zip(self.high_res_rnn, self.mid_res_rnn, self.low_res_rnn):
            res_high, res_mid, res_low = encoded_graph, mid_coarse_graph, large_coarse_graph
            
            h_fine = checkpoint(high_res_rnn, encoded_graph, edge_index, None, h_fine)
            h_mid = checkpoint(mid_res_rnn, mid_coarse_graph, mid_coarse_adj, None,h_mid)
            h_coarse = checkpoint(low_res_rnn, large_coarse_graph, large_coarse_adj, None, h_coarse)
            
            encoded_graph = h_fine + res_high
            mid_coarse_graph = h_mid + res_mid
            large_coarse_graph = h_coarse + res_low
            
            
        mid_to_fine = checkpoint(self.decode_coarse_graph_medium, mid_coarse_graph)
        coarse_to_fine = checkpoint(self.decode_coarse_graph_large, large_coarse_graph)
        
        x_stacked = torch.cat([encoded_graph, mid_to_fine, coarse_to_fine], dim=-1)
        
        out = self.readout(x_stacked[:, -1:])
        
        return out + 0.3 * x[:, -1:]