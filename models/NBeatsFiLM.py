import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import einops
from torch.nn import functional as F
from neuralforecast.models import NHITS
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks import ResidualMLP

from models.layers.NBEATS import NBEATS
from models.layers.DenseGraphConv import DenseBlock

class NBEATSFiLM(nn.Module):
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
        temporal_block_size = 3
        ):
        super(NBEATSFiLM, self).__init__()
        
        # self.window = window
        self.horizon = horizon
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.space_convs = nn.ModuleList([])
        for layer in range(n_layers):
            self.space_convs.append(
                    DenseBlock(hidden_size, hidden_size, growth_rate=128, n_blocks=spatial_block_size - 1, edge_dim=None, activation=nn.LeakyReLU(0.1))
                )
        self.encoder = ResidualMLP(
                    input_size=out_features,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    n_layers=3,
                    activation='silu',
                    dropout=0.0,
                    parametrized_skip=True,
                )
        self.readout = MLPDecoder(
                input_size=hidden_size,
                hidden_size=hidden_size*2,
                output_size=out_features,
                horizon=horizon,
                n_layers=5,
                # receptive_field=(window - receptive_field) +1,
                
                # receptive_field=receptive_field,
                receptive_field=horizon,
                dropout=.0,
                activation='silu'
            )  
        
    def forward(self, x, exog):
        b = x.size(0)
        x_orig = x
        x = self.encoder(x)
        x = x + self.positional_emb()
        # x = self.stack(x)
        # windows_batch = {
        #     'insample_y': x,
        #     'insample_mask': torch.ones(x.size()).to(x.device),
        #     'futr_exog': None,
        #     'hist_exog': None,
        #     'stat_exog': None
        # }
        # x = self.nhits(windows_batch)
        # x = x.squeeze()
        # x = einops.rearrange(x, '(b n) (t f) -> b t n f', b=b, f=self.hidden_size, t=self.horizon)
        
        adj = self.compute_spatial_adj()
        
        # 
        for space in self.space_convs:
          x = space(x, adj)  
        
        x = self.readout(x)
        
        return x + x_orig[:, -1:]
        
    def compute_spatial_adj(self):
        logits = F.relu(self.positional_emb() @ self.positional_emb().T)
        adj = torch.softmax(logits, dim=-1)
        adj = adj + adj.T
        return adj