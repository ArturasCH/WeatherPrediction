import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import MLPDecoder, MultiHorizonMLPDecoder
from tsl.nn.blocks.encoders import TemporalConvNet

from models.layers.DenseGraphConv import DenseBlock
from models.layers.DenseFiLM import DenseFiLMConv
from models.layers.FiLM_Conv import FiLMConv
from models.layers.Memory import MemoryStore, MeanUpdate, IdentityReadout, GRUMemoryUpdate, GRUReadout
from models.layers.LearnableWeight import LearnableWeight
from models.layers.GraphNorm import GraphNorm

class DenseFiLM(nn.Module):
    def __init__(
        self,
        lat_lons,
        # norm_scale,
        n_nodes=2048,
        input_size=26,
        hidden_size=256,
        out_features=26,
        # horizon=4,
        lsm = None,
        n_layers=6,
        ):
        super(DenseFiLM, self).__init__()
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


        self.lat_lon_positioning = self.encode_lat_lons(lat_lons)
        self.positional_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.lsm = lsm
        exog_size = 9 #day*2, hour*2, lat_lons*4, lsm, radiation
        self.condition_on_exog = ConditionalBlock(
            input_size=input_size + 1, #radiation to input
            exog_size=exog_size + hidden_size, #day of year and land sea mask, lat lons
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
                # FiLMConv(
                #     hidden_size,
                #     hidden_size,
                #     act=nn.ELU(),
                # )
                DenseFiLMConv(
                    hidden_size + learnable_size,
                    hidden_size,
                    # edge_dim=None,
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
            # self.skip_conn.append(nn.Linear(hidden_size, hidden_size))
            
        
            
        print("readout receptive field", receptive_field)
            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=1,
            n_layers=4,
            # receptive_field=(window - receptive_field) +1,
            
            receptive_field=1,
            dropout=.0,
            activation='elu'
        )  
        
        self.film = nn.Linear(input_size, out_features*2)
        
    def forward(self, x, edge_index, exog, radiation):
        b,n,f = x.size()
        # print(exog.size())
        lsm = self.lsm.T.to(x.device).squeeze().unsqueeze(0).expand(b, -1).unsqueeze(-1)
        x_orig = x
        x = torch.cat([x, radiation], dim=-1)
        exog = self.scale_day_of_year(exog).unsqueeze(1).expand(-1, n, -1)
        # memory_key = exog[:, :, 0,0] #use encoded day of year representation for key
        # self.memory.update_state(memory_key, x)
        # w_memory = self.memory(memory_key, x).unsqueeze(1) #integrate information with memory
        lat_lon = self.lat_lon_positioning.to(x.device).unsqueeze(0)
        pos_emb = self.positional_emb().unsqueeze(0)
        # print( exog.size(), lat_lon.size(), pos_emb.size(), self.lsm.size())
        full_exog = torch.cat([exog, lat_lon.expand(b,-1,-1), pos_emb.expand(b,-1,-1), lsm], dim=-1)
        
        x = checkpoint(self.condition_on_exog, x, full_exog, use_reentrant=True)
        # memory = self.memory(memory_key)
        # x = x+memory.unsqueeze(1)
        dense_adj = self.compute_spatial_adj()
        # out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        x = x.unsqueeze(1)
        for mlp, space, learnable, norm in zip(
            self.mlps,
            self.space_convs,
            # self.skip_conn,
            self.learnable,
            self.norms
            ):
            res = x
            x = learnable(x)
            x = checkpoint(space, x, dense_adj, use_reentrant=True)
            x = norm(x)
            x = checkpoint(mlp,x, use_reentrant=True)
            # out = checkpoint(skip,x) + out[:, -x.size(1):]
            
            x = x + res
        
        
        # standard readout
        res = self.readout(x)
        res = res.squeeze() + x_orig
        
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
        # day_of_year = torch.cat([xs, ys], dim=-1)
        # return torch.cat([day_of_year, exog[..., 1:]], dim=-1)
        
        return torch.cat([torch.sin(exog), torch.cos(exog)], dim=-1)