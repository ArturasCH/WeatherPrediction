import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.decoders import MLPDecoder

from models.layers.SGPEncoder import SGPEncoder
from models.layers.SGP import SGPModel
from models.layers.DenseFiLM import DenseFiLMConv


class TemporalDynamics(nn.Module):
    def __init__(
        self,
        lat_lons,
        norm_scale,
        n_nodes=2048,
        window=20,
        input_size=26,
        hidden_size=256,
        out_features=26,
        horizon=4,
        n_layers=6,
        spatial_block_size=3,
        temporal_block_size = 3
        ):
        super(TemporalDynamics, self).__init__()
        
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.lat_lon_positioning = self.encode_lat_lons(lat_lons)
        exog_size = 7
        # self.condition_on_exog = ConditionalBlock(
        #     input_size=input_size,
        #     exog_size=exog_size + hidden_size, #day of year and land sea mask, lat lons
        #     output_size=hidden_size,
        #     dropout=0.0,
        #     skip_connection=True,
        #     activation='tanh')
        
        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           exog_size=exog_size,
                           n_layers=n_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.gcns = nn.ModuleList([])
        for i in range(spatial_block_size):
            self.gcns.append(DenseFiLMConv(
                in_channels=hidden_size,
                out_channels=hidden_size,
                act=nn.SiLU(),
            ))
        # self.sgp_encoder = SGPEncoder(
        #     input_size=input_size,
        #     reservoir_size=hidden_size*5,
        #     reservoir_layers=n_layers,
        #     leaking_rate=0.9,
        #     spectral_radius=0.99,
        #     density=0.6,
        #     input_scaling=1.0,
        #     receptive_field=spatial_block_size,
        #     bidirectional=False,
        #     alpha_decay=True,
        #     global_attr=False,
        #     add_self_loops=False,
        #     undirected=True,
        #     reservoir_activation='tanh')
        
        # order = (1 + spatial_block_size) * n_layers
        # print(order)
        # self.sgp_decoder = SGPModel(
        #     input_size=order*hidden_size * 5,
        #     n_nodes=n_nodes,
        #     hidden_size=hidden_size,
        #     mlp_size=hidden_size*2,
        #     output_size=out_features,
        #     n_layers=temporal_block_size,
        #     positional_encoding=True,
        #     resnet=True,
        #     horizon=horizon,
        #     order=order
        # )
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            receptive_field=1,
            dropout=.0,
            activation='silu'
        )  
        
        
    def forward(self, x, exog):
        x_orig = x
        b,t,n,f = x.size()
        x = self.get_temporal_dynamics(x)
        exog = self.scale_day_of_year(exog)
        lat_lon = self.lat_lon_positioning.to(x.device).unsqueeze(0).unsqueeze(0)
        pos_emb = self.positional_emb().unsqueeze(0).unsqueeze(0)
        full_exog = torch.cat([exog[:, :-1], lat_lon.expand(b,t-1,-1,-1), pos_emb.expand(b,t-1,-1,-1)], dim=-1) #btn6
        
        x = checkpoint(self.time_nn, full_exog, use_reentrant=True)
        x = x.unsqueeze(1)
        x = x + self.positional_emb()
        
        adj = self.compute_spatial_adj()
        for gcn in self.gcns:
            x = checkpoint(gcn, x, adj, use_reentrant=True)
        # x = self.sgp_encoder(x, edge_index, None)
        # res = self.sgp_decoder(x)
        x_dynamics = checkpoint(self.readout, x, use_reentrant=True)
        return x_dynamics + x_orig[:, -1:]
    
    def get_temporal_dynamics(self, x):
        return x.diff(dim=1)
        
        
    def compute_spatial_adj(self):
        logits = F.relu(self.positional_emb() @ self.positional_emb().T)
        adj = torch.softmax(logits, dim=-1)
        # adj = adj + adj.T
        return adj
        
    def encode_lat_lons(self, lat_lons):
        lats = torch.tensor([torch.tensor(ln[0]) for ln in lat_lons])
        lons = torch.tensor(([torch.tensor(ln[1]) for ln in lat_lons]))
        stacked = torch.stack([lats, lons], dim=-1)
        return torch.cat([stacked.sin(), stacked.cos()], dim=-1)
        
    def scale_day_of_year(self, exog):
        # day_of_year = torch.sin(exog[..., 1:] / 365) * 2 * torch.pi
        # place day of year on a circle - day 1 and day 365 is in the same season - basically the same
        # day 180 - half a year away - oposite side of the circle
        xs = torch.cos(exog[..., 1:])
        ys = torch.sin(exog[..., 1:])
        day_of_year = torch.cat([xs, ys], dim=-1)
        return torch.cat([day_of_year, exog[..., 1:]], dim=-1)
        