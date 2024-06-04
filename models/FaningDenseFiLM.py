import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import MLPDecoder, MultiHorizonMLPDecoder
# from tsl.nn.blocks.encoders import TemporalConvNet
from tsl.nn.blocks import ResidualMLP

from models.layers.DenseGraphConv import DenseBlock
from models.layers.DenseTimeConv import DenseTemporalBlock
from models.layers.Memory import MemoryStore, MeanUpdate, IdentityReadout, GRUMemoryUpdate, GRUReadout
from models.layers.LearnableWeight import LearnableWeight
from models.layers.GLUTenporalConv import GatedTempConv
from models.layers.FiLM_Conv import FiLMConv
from models.layers.EdgeFilm import EdgeFiLMConv
from models.layers.ReGLUMLP import ReGluMLP
from models.layers.ReGLU import ReGLU

class FaningDenseFiLM(nn.Module):
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
        learnable_size = 32,
        ):
        super(FaningDenseFiLM, self).__init__()
        self.out_features = out_features
        # update_fn = MeanUpdate()
        # readout = IdentityReadout()
        # memory_size = (10, n_nodes, input_size)
        # update_fn = GRUMemoryUpdate(input_size, input_size)
        # readout = GRUReadout(input_size, input_size)
        # memory_size = (10, input_size)
        # self.memory = MemoryStore(update_fn=update_fn, readout=readout, memory_size=memory_size, key_sequence_size=window)
        # self.memory_readout = nn.Linear(input_size, hidden_size)


        self.lat_lon_positioning = self.encode_lat_lons(lat_lons)
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        exog_size = 7
        self.condition_on_exog = ConditionalBlock(
            input_size=input_size,
            exog_size=exog_size + hidden_size, #day of year and land sea mask, lat lons
            output_size=hidden_size,
            dropout=0.0,
            skip_connection=True,
            activation='elu')
        
        self.learnable = nn.ModuleList([])
        self.space_convs = nn.ModuleList([])
        self.time_convs = nn.ModuleList([])
        self.dropouts = nn.ModuleList([])
        
        dilation = 2
        receptive_field = 1
        temporal_kernel_size = 3
        for layer in range(n_layers):

            d = dilation**(layer % 2)
            pad = (window - receptive_field) <= temporal_kernel_size
            self.learnable.append(LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_size))
            
            time = ResidualMLP(input_size=hidden_size+learnable_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=4,
            activation='elu') if pad else GatedTempConv(input_channels=hidden_size + learnable_size,
                                output_channels=hidden_size,
                                kernel_size=temporal_kernel_size,
                                dilation=d,
                                causal_padding=True
                                )
            
            self.time_convs.append(
                # TemporalConvNet(input_channels=hidden_size,
                #                 hidden_channels=hidden_size,
                #                 kernel_size=temporal_kernel_size,
                #                 dilation=d,
                #                 # exponential_dilation=False,
                #                 n_layers=1,
                #                 gated=True,
                #                 causal_padding=pad
                #                 )
                # GatedTempConv(input_channels=hidden_size + learnable_size,
                #                 output_channels=hidden_size,
                #                 kernel_size=temporal_kernel_size,
                #                 dilation=d,
                #                 # exponential_dilation=False,
                #                 causal_padding=True
                #                 )
                time
            )
            # if not pad:
            receptive_field += d * (temporal_kernel_size - 1)
                # receptive_field += 1
            # self.skip_conn.append(nn.Linear(hidden_size, hidden_size))
            
            # self.shape_match2.append(nn.Linear(hidden_size, hidden_size + learnable_size))

            # hidden_size = hidden_size + learnable_size
            
            self.space_convs.append(
                # DenseBlock(
                #     hidden_size + learnable_size,
                #     hidden_size,
                #     growth_rate=128,
                #     n_blocks=spatial_block_size,
                #     edge_dim=None,
                #     activation=nn.ELU()
                #     )
                # FiLMConv(
                #     in_channels=hidden_size,
                #     out_channels=hidden_size,
                #     )
                EdgeFiLMConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    act=nn.ELU()
                )
            )
            
            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            receptive_field=temporal_kernel_size,
            dropout=.0,
            activation='elu'
        )  
        
        self.film = nn.Linear(input_size, out_features*2)
        
    def forward(self, x, edge_index, exog):
        b,t,n,f = x.size()
        x_orig = x
        x = self.get_temporal_dynamics(x)
        
        
        exog = self.scale_day_of_year(exog)

        lat_lon = self.lat_lon_positioning.to(x.device).unsqueeze(0).unsqueeze(0)
        pos_emb = self.positional_emb().unsqueeze(0).unsqueeze(0)
        full_exog = torch.cat([exog[:, :-1], lat_lon.expand(b,t-1,-1,-1), pos_emb.expand(b,t-1,-1,-1)], dim=-1) #btn6
        # full_exog = torch.cat([exog, lat_lon.expand(b,t,-1,-1), pos_emb.expand(b,t,-1,-1)], dim=-1) #btn7
        
        x = checkpoint(self.condition_on_exog, x, full_exog, use_reentrant=True)        
        for time, space,  learnable in zip(
            self.time_convs,
            self.space_convs,
            self.learnable
            ):
            res = x
            x = learnable(x)
            x = checkpoint(time,x, use_reentrant=True)
            x = checkpoint(space, x, edge_index, use_reentrant=True)
            
            x = x + res[:, -x.size(1):]
            
        res = self.readout(x)
        res = res + x_orig[:, -1:]
        
        return res
    
    def get_temporal_dynamics(self, x):
        return x.diff(dim=1)
        
    # def normalize(self, x):
    #     # min 0, max 1
    #     _min = self.norm_scale[0].to(x.device)
    #     _max = self.norm_scale[1].to(x.device)
    #     return 2 * ((x - _min) / (_max - _min)) - 1
    
    def denormalize(self, x):
        _min = self.norm_scale[0].to(x.device)
        _max = self.norm_scale[1].to(x.device)
        
        return (x + 1) * ((_max - _min) / 2) + _min
        
    def encode_lat_lons(self, lat_lons):
        lats = torch.tensor([torch.tensor(ln[0]) for ln in lat_lons])
        lons = torch.tensor(([torch.tensor(ln[1]) for ln in lat_lons]))
        stacked = torch.stack([lats, lons], dim=-1)
        return torch.cat([stacked.sin(), stacked.cos()], dim=-1)
        
    def scale_day_of_year(self, exog):
        # place day of year on a circle - day 1 and day 365 is in the same season - basically the same
        # day 180 - half a year away - oposite side of the circle

        xs = torch.cos(exog[..., :1])
        ys = torch.sin(exog[..., :1])
        day_of_year = torch.cat([xs, ys], dim=-1)
        return torch.cat([day_of_year, exog[..., 1:]], dim=-1)
    