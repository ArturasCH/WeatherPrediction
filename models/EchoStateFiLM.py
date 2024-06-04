import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import MLPDecoder, MultiHorizonMLPDecoder

from models.layers.DenseGraphConv import DenseBlock
from models.layers.DenseTimeConv import DenseTemporalBlock
from models.layers.Memory import MemoryStore, MeanUpdate, IdentityReadout, GRUMemoryUpdate, GRUReadout
from models.layers.LearnableWeight import LearnableWeight
from models.layers.Reservoir import Reservoir
from tsl.nn.blocks.decoders import LinearReadout
from models.layers.SGP import SGPModel
from models.layers.FiLM_Conv import FiLMConv

# PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

class EchoStateFiLM(nn.Module):
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
        super(EchoStateFiLM, self).__init__()
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
        # self.condition_on_exog = ConditionalBlock(
        #     input_size=input_size,
        #     exog_size=exog_size + hidden_size, #day of year and land sea mask, lat lons
        #     output_size=hidden_size,
        #     dropout=0.0,
        #     skip_connection=True,
        #     activation='leaky_relu')
        
        self.reservoir = Reservoir(
            input_size=input_size + exog_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            leaking_rate=0.9,
            spectral_radius=0.99,
            density=0.9,
        )
        
        self.reservoir_out_size = hidden_size * n_layers
        
        self.learnable = nn.ModuleList([])
        # self.shape_match = nn.ModuleList([])
        # self.shape_match2 = nn.ModuleList([])
        self.space_convs = nn.ModuleList([])
        # self.time_convs = nn.ModuleList([])
        # self.skip_conn = nn.ModuleList([])
        learnable_size = 32
        for i in range(spatial_block_size):
            if i == 0:
                in_size = self.reservoir_out_size
            else:
                in_size = hidden_size
                # in_size = hidden_size
            self.learnable.append(LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_size))
            self.space_convs.append(DenseBlock(
                in_size,
                hidden_size,
                growth_rate=128,
                n_blocks=spatial_block_size - 1,
                edge_dim=None,
                activation=nn.SiLU())
                )
            # self.space_convs.append(FiLMConv(
            #     in_channels=in_size,
            #     out_channels=hidden_size,
            #     act=nn.SiLU()
            # ))
        
        # self.readout = MLPDecoder(
        #     input_size=hidden_size,
        #     hidden_size=hidden_size*2,
        #     output_size=out_features,
        #     horizon=horizon,
        #     n_layers=5,
        #     # receptive_field=(window - receptive_field) +1,
            
        #     receptive_field=1,
        #     # receptive_field=window - (receptive_field-1),
        #     dropout=.0,
        #     activation='tanh'
        # )
        
        self.sgp_readout = SGPModel(
            input_size=hidden_size,
            order=n_layers,
            n_nodes=n_nodes,
            hidden_size=hidden_size,
            mlp_size=hidden_size*2,
            output_size=out_features,
            positional_encoding=True,
            fully_connected=True,
            n_layers=4,
            horizon=horizon
        )
        
        self.film = nn.Linear(input_size, out_features*2)
        
    def forward(self, x, exog):
        b,t,n,f = x.size()
        x_orig = x
        exog = self.scale_day_of_year(exog)
        # memory_key = exog[:, :, 0,0] #use encoded day of year representation for key
        # self.memory.update_state(memory_key, x)
        # w_memory = self.memory(memory_key, x).unsqueeze(1) #integrate information with memory
        lat_lon = self.lat_lon_positioning.to(x.device).unsqueeze(0).unsqueeze(0)
        pos_emb = self.positional_emb().unsqueeze(0).unsqueeze(0)
        # full_exog = torch.cat([exog, lat_lon.expand(b,t,-1,-1), pos_emb.expand(b,t,-1,-1)], dim=-1) #btn7
        full_exog = torch.cat([exog, lat_lon.expand(b,t,-1,-1)], dim=-1) #btn7
        
        # x = checkpoint(self.condition_on_exog,x, full_exog, use_reentrant=True)
        # memory = self.memory(memory_key)
        # x = x+memory.unsqueeze(1)
        dense_adj = self.compute_spatial_adj()
        # out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        
        x = torch.cat([x, full_exog], dim=-1)
        x = self.reservoir(x, return_last_state=True)
        # x = self.reservoir(x)
        
        x = x.unsqueeze(1)
        for space, learnable in zip(self.space_convs,self.learnable):
            
            # x = learnable(x)
            
            # x = checkpoint(space, x, dense_adj, use_reentrant=True)
            x = checkpoint(space, x, dense_adj, use_reentrant=True)
        #     out = x + out[:, -x.size(1):]
            
        res = self.sgp_readout(x)
        assert not torch.isnan(res).any()
        return res + x_orig[:, -res.size(1):]
        
        
    def compute_spatial_adj(self):
        logits = F.relu(self.positional_emb() @ self.positional_emb().T)
        adj = torch.softmax(logits, dim=-1)
        # undirected graph, add backwards connection
        adj = adj + adj.T
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