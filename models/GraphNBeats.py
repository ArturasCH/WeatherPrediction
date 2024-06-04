import torch
import torch.nn as nn
from tsl.nn.layers import NodeEmbedding
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.decoders import MLPDecoder

from models.layers.GraphNbeatsBlocks import GenericBasis, GraphNBeatsBlock


class GraphNBeats(nn.Module):
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
        temporal_block_size = 3,
        # blocks: nn.ModuleList = None
        ):
        super(GraphNBeats, self).__init__()
        self.norm_scale = norm_scale
        self.out_features = out_features
        self.encoder = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=2,
            activation='silu'
            )
        basis = GenericBasis(forecast_size=horizon, backcast_size=window)
        self.blocks = nn.ModuleList([GraphNBeatsBlock(
            in_size=hidden_size,
            theta_size=hidden_size,
            basis_function=basis,
            layer_size=hidden_size,
            layers=spatial_block_size
            ) for _ in range(n_layers)])

        # self.time_last = Rearrange('b t n f -> b n f t') #alt variant, smaller attention map
        # self.standard_format = Rearrange('b n f t -> b t n f')
        self.time_last = Rearrange('b t n f -> b n (t f)') #alt variant, smaller attention map
        self.standard_format = Rearrange('b n (t f) -> b t n f', f=out_features)
        # n_nodes=2048
        emb_size = 256
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.readout =  self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=5,
            # receptive_field=(window - receptive_field) +1,
            
            # receptive_field=receptive_field,
            receptive_field=horizon,
            dropout=.0,
            activation='relu'
        )  
        
    def forward(self, x, exog):
        # x - b t n f
        # x = self.normalize(x)
        x = self.encoder(x)
        x = x + self.positional_emb()
        # x = self.time_last(x)
        # residuals - input, but reversed
        residuals = x.flip(dims=(1,))
        # forecast start - last timestep
        # shape b n f 1
        # forecast = x[..., -1:]
        forecast = x[:, -1:]
        adj = self.compute_spatial_adj()
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals, adj)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        # return self.denormalize(self.standard_format(forecast))
        return self.readout(forecast)
        
    def compute_spatial_adj(self):
        logits = F.relu(self.positional_emb() @ self.positional_emb().T)
        adj = torch.softmax(logits, dim=-1)
        adj = adj + adj.T
        return adj
        
    def normalize(self, x):
        # min -1, max 1
        _min = self.norm_scale[0].to(x.device)
        _max = self.norm_scale[1].to(x.device)
        return 2 * ((x - _min) / (_max - _min)) - 1
    
    def denormalize(self, x):
        _min = self.norm_scale[0].to(x.device)
        _max = self.norm_scale[1].to(x.device)
        
        return (x + 1) * ((_max - _min) / 2) + _min