import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import MLPDecoder, MultiHorizonMLPDecoder
# from tsl.nn.blocks.encoders import TemporalConvNet

from model.layers.DenseGraphConv import DenseBlock
from model.layers.LearnableWeight import LearnableWeight
from model.layers.GatedTenporalConv import GatedTempConv

class DenseFilmWithLearnable(nn.Module):
    def __init__(
        self,
        lat_lons,
        n_nodes=2048,
        window=20,
        input_size=26,
        hidden_size=256,
        out_features=26,
        horizon=4,
        n_layers=6,
        spatial_block_size=3,
        ):
        super(DenseFilmWithLearnable, self).__init__()
        self.out_features = out_features


        self.lat_lon_positioning = self.encode_lat_lons(lat_lons)
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        exog_size = 7
        self.condition_on_exog = ConditionalBlock(
            input_size=input_size,
            exog_size=exog_size + hidden_size, #day of year and land sea mask, lat lons
            output_size=hidden_size,
            dropout=0.0,
            skip_connection=True,
            activation='relu')
        
        self.learnable = nn.ModuleList([])
        self.space_convs = nn.ModuleList([])
        self.time_convs = nn.ModuleList([])
        self.skip_conn = nn.ModuleList([])
        self.dropouts = nn.ModuleList([])
        
        dilation = 2
        time_kernel=2
        receptive_field = 1
        learnable_size = 32
        temporal_kernel_size = 3
        for layer in range(n_layers):
            d = dilation**(layer % 2)
            pad = (window - receptive_field) <= horizon+1
            
            self.time_convs.append(
                GatedTempConv(input_channels=hidden_size,
                                output_channels=hidden_size,
                                kernel_size=temporal_kernel_size,
                                dilation=d,
                                causal_padding=pad
                                )
            )
            if not pad:
                receptive_field += d * (temporal_kernel_size - 1)
            self.skip_conn.append(nn.Linear(hidden_size, hidden_size))
            
            self.learnable.append(LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_size))

            self.space_convs.append(
                DenseBlock(
                    hidden_size + learnable_size,
                    hidden_size,
                    growth_rate=128,
                    n_blocks=spatial_block_size - 1,
                    edge_dim=None,
                    activation=nn.ReLU()
                    )
            )

            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            receptive_field=(window - 1) - (receptive_field-1),
            dropout=.0,
            activation='relu'
        )  
        
        self.film = nn.Linear(input_size, out_features*2)
        
    def forward(self, x, exog):
        b,t,n,f = x.size()
        x_orig = x
        x = self.get_temporal_dynamics(x)
        exog = self.scale_day_of_year(exog)
        lat_lon = self.lat_lon_positioning.to(x.device).unsqueeze(0).unsqueeze(0)
        pos_emb = self.positional_emb().unsqueeze(0).unsqueeze(0)
        full_exog = torch.cat([exog[:, :-1], lat_lon.expand(b,t-1,-1,-1), pos_emb.expand(b,t-1,-1,-1)], dim=-1)
        
        x = checkpoint(self.condition_on_exog, x, full_exog, use_reentrant=True)
        dense_adj = self.compute_spatial_adj()
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        
        for time, space, skip, learnable in zip(
            self.time_convs,
            self.space_convs,
            self.skip_conn,
            self.learnable
            ):
            res = x
            x = checkpoint(time,x, use_reentrant=True)
            out = checkpoint(skip,x) + out[:, -x.size(1):]
            x = learnable(x)
            x = checkpoint(space, x, dense_adj, use_reentrant=True)
            
            x = x + res[:, -x.size(1):]

        res = self.readout(x)
        res = res + x_orig[:, -1:]
        return res
    
    def get_temporal_dynamics(self, x):
        return x.diff(dim=1)
        
    def compute_spatial_adj(self):
        logits = F.relu(self.positional_emb() @ self.positional_emb().T)
        adj = torch.softmax(logits, dim=-1)
        return adj
        
    def encode_lat_lons(self, lat_lons):
        lats = torch.tensor([torch.tensor(ln[0]) for ln in lat_lons])
        lons = torch.tensor(([torch.tensor(ln[1]) for ln in lat_lons]))
        stacked = torch.stack([lats, lons], dim=-1)
        return torch.cat([stacked.sin(), stacked.cos()], dim=-1)
        
    def scale_day_of_year(self, exog):
        xs = torch.cos(exog[..., :1])
        ys = torch.sin(exog[..., :1])
        day_of_year = torch.cat([xs, ys], dim=-1)
        return torch.cat([day_of_year, exog[..., 1:]], dim=-1)