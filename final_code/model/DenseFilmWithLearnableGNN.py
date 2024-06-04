import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import MLPDecoder

from .layers.DenseFiLM import DenseFiLMConv
from .layers.LearnableWeight import LearnableWeight
from .layers.GraphNorm import GraphNorm

class DenseFiLM(nn.Module):
    def __init__(
        self,
        lat_lons,
        n_nodes=2048,
        input_size=26,
        hidden_size=256,
        out_features=26,
        lsm = None,
        n_layers=6,
        ):
        super(DenseFiLM, self).__init__()
        self.out_features = out_features


        self.lat_lon_positioning = self.encode_lat_lons(lat_lons)
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.lsm = lsm
        exog_size = 9 #day*2, hour*2, lat_lons*4, lsm
        self.condition_on_exog = ConditionalBlock(
            input_size=input_size + 1, #radiation to input
            exog_size=exog_size + hidden_size,
            output_size=hidden_size,
            dropout=0.0,
            skip_connection=True,
            activation='elu')
        
        self.learnable = nn.ModuleList([])
        self.space_convs = nn.ModuleList([])
        self.mlps = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        
        receptive_field = 1
        learnable_size = 32
        for layer in range(n_layers):
            self.learnable.append(LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_size))
            self.space_convs.append(
                DenseFiLMConv(
                    hidden_size + learnable_size,
                    hidden_size,
                    act=nn.ELU()
                    )
            )
            
            self.norms.append(
                GraphNorm(hidden_size)
            )
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ELU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, hidden_size)
            ))
            
            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=1,
            n_layers=4,
            receptive_field=1,
            dropout=.0,
            activation='elu'
        )  
        
        self.film = nn.Linear(input_size, out_features*2)
        
    def forward(self, x, edge_index, exog, radiation):
        b,n,f = x.size()
        lsm = self.lsm.T.to(x.device).squeeze().unsqueeze(0).expand(b, -1).unsqueeze(-1)
        x_orig = x
        x = torch.cat([x, radiation], dim=-1)
        exog = self.scale_day_of_year(exog).unsqueeze(1).expand(-1, n, -1)
        lat_lon = self.lat_lon_positioning.to(x.device).unsqueeze(0)
        pos_emb = self.positional_emb().unsqueeze(0)
        full_exog = torch.cat([exog, lat_lon.expand(b,-1,-1), pos_emb.expand(b,-1,-1), lsm], dim=-1)
        
        x = checkpoint(self.condition_on_exog, x, full_exog, use_reentrant=True)
        dense_adj = self.compute_spatial_adj()
        x = x.unsqueeze(1)
        for mlp, space, learnable, norm in zip(
            self.mlps,
            self.space_convs,
            self.learnable,
            self.norms
            ):
            res = x
            x = learnable(x)
            x = checkpoint(space, x, dense_adj, use_reentrant=True)
            x = norm(x)
            x = checkpoint(mlp,x, use_reentrant=True)
            
            x = x + res
        
        res = self.readout(x)
        res = res.squeeze() + x_orig
        
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
        return torch.cat([torch.sin(exog), torch.cos(exog)], dim=-1)