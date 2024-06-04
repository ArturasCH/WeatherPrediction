import torch
import torch.nn as nn
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks import ResidualMLP
from torch.utils.checkpoint import checkpoint
from models.layers.DenseGraphConv import DenseBlock
from models.layers.DenseTimeConv import DenseTemporalBlock
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from models.layers.coarsen import CoarseGraphDecoder, FineToCoarseEncoder
from torch_geometric.nn.norm import GraphNorm

class CoarseTGatedGCN(nn.Module):
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
                edge_embedding_size=10,
                n_connections=25204
                ):
        super(CoarseTGatedGCN, self).__init__()
        
        self.condition_exog = ConditionalBlock(
                 input_size=input_size,
                 exog_size=2, #day of year and land sea mask
                 output_size=hidden_size,
                 dropout=0.3,
                 skip_connection=True,
                 activation='leaky_relu')
        
        self.encoder = ResidualMLP(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    n_layers=3,
                    activation='leaky_relu',
                    dropout=0.3,
                    parametrized_skip=True,
                )
        
        self.coarsen = FineToCoarseEncoder(
            lat_lons,
            input_dim=hidden_size,
            output_dim=hidden_size,
            output_edge_dim=hidden_size,
            hidden_dim_processor_node=hidden_size,
            hidden_dim_processor_edge=hidden_size,
            )
        
        self.decode_coarse_graph = CoarseGraphDecoder(
            lat_lons,
            input_dim=hidden_size
            )
        
        
        self.space_convs = nn.ModuleList([])
        self.medium_grained_gcns = nn.ModuleList([])
        self.time_convs = nn.ModuleList([])
        self.medium_grained_time_convs = nn.ModuleList([])
        self.skip_conn = nn.ModuleList([])
        self.medium_grained_skip_conn = nn.ModuleList([])
        self.edge_mlps = nn.ModuleList([])
        
        dilation = 2
        time_kernel=2
        receptive_field = 1
        for layer in range(n_layers):
            self.space_convs.append(
                # DenseGraphConvOrderK(hidden_size, hidden_size, support_len=True, order=diffusion_order, channel_last=True)
                # most ran with 4 blocks
                DenseBlock(hidden_size, hidden_size, n_blocks=3, edge_dim=None, activation=nn.LeakyReLU(0.1))
            )
            self.medium_grained_gcns.append(
                DenseBlock(hidden_size, hidden_size, n_blocks=3, edge_dim=hidden_size, activation=nn.LeakyReLU(0.1))
            )
            # d = dilation ** (layer % 3)
            d = dilation**(layer % 2)
            self.time_convs.append(DenseTemporalBlock(
                input_channels=hidden_size,
                hidden_channels=hidden_size,
                output_channels=hidden_size,
                growth_rate=64,
                dilation=d,
                n_blocks=3,
                ))
            self.medium_grained_time_convs.append(
                DenseTemporalBlock(
                input_channels=hidden_size,
                hidden_channels=hidden_size,
                output_channels=hidden_size,
                growth_rate=64,
                dilation=d,
                n_blocks=3,
                )
            )
            receptive_field += d * (time_kernel - 1)
            self.skip_conn.append(nn.Linear(hidden_size, hidden_size))
            self.medium_grained_skip_conn.append(nn.Linear(hidden_size, hidden_size))
            self.edge_mlps.append(ResidualMLP(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                n_layers=2,
                activation='leaky_relu',
                dropout=0.3,
                parametrized_skip=True,
            ))
            
            
        
        self.readout = MLPDecoder(
            input_size=hidden_size*2,
            hidden_size=hidden_size*3,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            # receptive_field=2,
            receptive_field=(window - receptive_field) +1,
            # receptive_field=1,
            dropout=0.3,
            activation='leaky_relu'
        ) 
    
    def forward(self, x, edge_index, exog):
        x = checkpoint(self.condition_exog, x, exog, use_reentrant=True)
        x = checkpoint(self.encoder, x)
        coarse_graph, coarse_adj, coarse_edge_attr = checkpoint(self.coarsen,x)
        
        
        out_m = torch.zeros(1, coarse_graph.size(1), 1, 1, device=x.device)
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for space, m_space, time, m_time, skip_conn, m_skip_conn, edge_mlp in zip(
            self.space_convs,
            self.medium_grained_gcns,
            self.time_convs,
            self.medium_grained_time_convs,
            self.skip_conn,
            self.medium_grained_skip_conn,
            self.edge_mlps):
            
            res = x
            res_m = coarse_graph
            
            coarse_graph = checkpoint(m_time, coarse_graph, use_reentrant=True)
            x = checkpoint(time, x, use_reentrant=True)
            
            out_m  = m_skip_conn(coarse_graph) + out_m[:, -coarse_graph.size(1):]
            out = skip_conn(x) + out[:, -x.size(1):]
            
            coarse_edge_attr = checkpoint(edge_mlp, coarse_edge_attr)
            coarse_graph = checkpoint(m_space, coarse_graph, coarse_adj, coarse_edge_attr, use_reentrant=True)
            x = checkpoint(space, x, edge_index, use_reentrant=True)
            
            coarse_graph = coarse_graph + res_m[:, -coarse_graph.size(1):]
            x = x + res[:, -x.size(1):]
        
        
        fine_grained_graph = self.decode_coarse_graph(coarse_graph + out_m[:, -coarse_graph.size(1):])
        x = torch.cat([x, fine_grained_graph], dim=-1)
        return self.readout(x)