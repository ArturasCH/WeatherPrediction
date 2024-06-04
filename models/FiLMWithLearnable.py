import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import MLPDecoder, MultiHorizonMLPDecoder
# from tsl.nn.blocks.encoders import TemporalConvNet

from models.layers.DenseGraphConv import DenseBlock
from models.layers.DenseTimeConv import DenseTemporalBlock
from models.layers.Memory import MemoryStore, MeanUpdate, IdentityReadout, GRUMemoryUpdate, GRUReadout
from models.layers.LearnableWeight import LearnableWeight
from models.layers.GLUTenporalConv import GatedTempConv
from models.layers.DenseFiLM import DenseFiLMConv

class FiLMWithLearnable(nn.Module):
    def __init__(
        self,
        lat_lons,
        lsm,
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
        super(FiLMWithLearnable, self).__init__()
        # self.norm_scale = norm_scale
        self.out_features = out_features
        # update_fn = MeanUpdate()
        # readout = IdentityReadout()
        # memory_size = (10, n_nodes, input_size)
        # update_fn = GRUMemoryUpdate(input_size, input_size)
        # readout = GRUReadout(input_size, input_size)
        # memory_size = (10, input_size)
        # self.memory = MemoryStore(update_fn=update_fn, readout=readout, memory_size=memory_size, key_sequence_size=window)
        # self.memory_readout = nn.Linear(input_size, hidden_size)

        self.lsm = lsm
        self.lat_lon_positioning = self.encode_lat_lons(lat_lons)
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        exog_size = 9
        self.condition_on_exog = ConditionalBlock(
            input_size=input_size + 1,
            exog_size=exog_size + hidden_size, #day of year and land sea mask, lat lons
            output_size=hidden_size,
            dropout=0.0,
            skip_connection=True,
            activation='relu')
        
        self.learnable = nn.ModuleList([])
        # self.shape_match = nn.ModuleList([])
        # self.shape_match2 = nn.ModuleList([])
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

            # if layer == 0:
            #     self.shape_match.append(nn.Identity())
            # else:
            #     self.shape_match.append(nn.Linear(hidden_size, hidden_size + learnable_size))
            d = dilation**(layer % 2)
            # d = dilation ** min((layer % 3), 3)
            # d = dilation ** (layer % 3)
            pad = (window - receptive_field) <= horizon+1
            
            # self.time_convs.append(DenseTemporalBlock(
            #     input_channels=hidden_size + learnable_size,
            #     hidden_channels=hidden_size,
            #     output_channels=hidden_size,
            #     growth_rate=32,
            #     dilation=d,
            #     pad=pad,
            #     n_blocks=temporal_block_size
            #     ))
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
                GatedTempConv(input_channels=hidden_size,
                                output_channels=hidden_size,
                                kernel_size=temporal_kernel_size,
                                dilation=d,
                                # exponential_dilation=False,
                                causal_padding=pad
                                )
            )
            if not pad:
                receptive_field += d * (temporal_kernel_size - 1)
                # receptive_field += 1
            self.skip_conn.append(nn.Linear(hidden_size, hidden_size))
            
            # self.shape_match2.append(nn.Linear(hidden_size, hidden_size + learnable_size))
            self.learnable.append(LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_size))

            # hidden_size = hidden_size + learnable_size
            self.space_convs.append(
                # DenseBlock(
                #     hidden_size + learnable_size,
                #     hidden_size,
                #     growth_rate=128,
                #     n_blocks=spatial_block_size - 1,
                #     edge_dim=None,
                #     activation=nn.ReLU()
                #     )
                DenseFiLMConv(
                    hidden_size + learnable_size,
                    hidden_size,
                    # edge_dim=None,
                    act=nn.ELU()
                    )
            )
 
            
            
        # self.readout = MultiHorizonMLPDecoder(
        #     input_size=hidden_size,
        #     exog_size=exog_size,
        #     hidden_size=hidden_size*2,
        #     context_size=hidden_size*2,
        #     output_size=out_features,
        #     horizon=horizon,
        #     n_layers=3,
        #     activation='leaky_relu',
        #     dropout=0.0,
        # )
        # if receptive_field > window:
        #     receptive_field = 1
        # else:
        #     receptive_field=(window - receptive_field) - 1
            
        print("readout receptive field", receptive_field)
        # self.readouts = nn.ModuleList([])
        # self.readout_films = nn.ModuleList([])
        # for readout in range(horizon):
        #     self.readouts.append(MLPDecoder(
        #         input_size=hidden_size,
        #         hidden_size=hidden_size*2,
        #         output_size=out_features,
        #         horizon=1,
        #         n_layers=4,
        #         # receptive_field=(window - receptive_field) +1,
                
        #         # receptive_field=1,
        #         receptive_field=receptive_field,
        #         dropout=.0,
        #         activation='leaky_relu'
        #     ))
            
        #     self.readout_films.append(nn.Linear(input_size, out_features*2))
            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            # receptive_field=(window - receptive_field) +1,
            
            # receptive_field=receptive_field,
            receptive_field=(window - 1) - (receptive_field-1),
            dropout=.0,
            activation='relu'
        )  
        
        self.film = nn.Linear(input_size, out_features*2)
        
    def forward(self, x, exog, radiation):
        # x = self.normalize(x)
        b,t,n,f = x.size()
        x_orig = x
        # x = self.get_temporal_dynamics(x)
        x = torch.cat([x, radiation], dim=-1)
        lsm = self.lsm.T.to(x.device).squeeze().unsqueeze(0).unsqueeze(0).expand(b,t, -1).unsqueeze(-1)
        # b n 1
        exog = self.scale_day_of_year(exog).unsqueeze(-2).expand(-1, -1, n, -1)
        # memory_key = exog[:, :, 0,0] #use encoded day of year representation for key
        # self.memory.update_state(memory_key, x)
        # w_memory = self.memory(memory_key, x).unsqueeze(1) #integrate information with memory
        lat_lon = self.lat_lon_positioning.to(x.device).unsqueeze(0).unsqueeze(0)
        pos_emb = self.positional_emb().unsqueeze(0).unsqueeze(0)
        full_exog = torch.cat([exog, lat_lon.expand(b,t,-1,-1), pos_emb.expand(b,t,-1,-1), lsm], dim=-1) #btn6
        # full_exog = torch.cat([exog, lat_lon.expand(b,t,-1,-1), pos_emb.expand(b,t,-1,-1)], dim=-1) #btn7
        
        x = checkpoint(self.condition_on_exog, x, full_exog, use_reentrant=True)
        # memory = self.memory(memory_key)
        # x = x+memory.unsqueeze(1)
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
            
        # print(x.size(), torch.cat([exog[:, -x.size(1):], lat_lon.expand(b, x.size(1), -1,-1)], dim=-1).size())
        # res = self.readout(x, torch.cat([exog[:, -1:], lat_lon.expand(b, 1, -1,-1)], dim=-1))
        # memory = self.memory(memory_key)
        # res = self.readout(x + out[:, -x.size(1):] + self.memory_readout(memory))
        # memory = self.memory(memory_key)
        # res = self.readout(x + out[:, -x.size(1):]) + w_memory
        
        # standard readout
        res = self.readout(x)
        # # self.memory.update_state(memory_key, x_orig)
        # beta, gamma = self.film(res).split(self.out_features, dim=-1)
        # return res + (gamma * x_orig[:, -1:] + beta)
        res = res + x_orig[:, -1:]
        # res = self.denormalize(res)
        # standard readout
        # res = []
        # for readout, film in zip(self.readouts, self.readout_films):
        #     r = readout(x + out[:, -x.size(-1):])
        #     beta, gamma = film(r).split(self.out_features, dim=-1)
        #     res.append(r + (gamma * x_orig[:, -1:] + beta))
            
        # res = torch.cat(res, dim=1)
        return res
    
    def get_temporal_dynamics(self, x):
        return x.diff(dim=1)
        
    def normalize(self, x):
        # min 0, max 1
        _min = self.norm_scale[0].to(x.device)
        _max = self.norm_scale[1].to(x.device)
        return 2 * ((x - _min) / (_max - _min)) - 1
    
    def denormalize(self, x):
        _min = self.norm_scale[0].to(x.device)
        _max = self.norm_scale[1].to(x.device)
        
        return (x + 1) * ((_max - _min) / 2) + _min
        
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
        # xs = torch.cos(exog[..., 1:])
        # ys = torch.sin(exog[..., 1:])
        
        # xs = torch.cos(exog[..., :1])
        # ys = torch.sin(exog[..., :1])
        # day_of_year = torch.cat([xs, ys], dim=-1)
        # return torch.cat([day_of_year, exog[..., 1:]], dim=-1)
        
        return torch.cat([torch.sin(exog), torch.cos(exog)], dim=-1)