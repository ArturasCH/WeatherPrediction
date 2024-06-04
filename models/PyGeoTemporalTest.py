import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange

from snntorch import utils

# from torch_geometric_temporal.nn.recurrent import A3TGCN2

from tsl.nn.layers.base import TemporalSelfAttention
from tsl.nn.layers import Norm
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.decoders import AttPool
from tsl.nn.blocks.encoders import EvolveGCN
from tsl.nn.blocks.decoders import MultiHorizonMLPDecoder

from models.layers.DenseGraphConv import DenseBlock
from models.layers.LearnableWeight import LearnableWeight
from models.layers.SynapticChain import SynapticChain
from models.layers.A3RFCN import A3TGCNBlock
from models.layers.AttentionStack import AttentionStack

class A3TGCN(torch.nn.Module):
    def __init__(self,
                 n_nodes,
                 horizon=4,
                 window=56,
                 n_batches=2,
                 number_of_blocks=3,
                 number_of_temporal_steps=3,
                 hidden_size=256,
                 input_size=26,
                 out_features=26,
                 growth_rate=32,
                 learnable_feature_size=32,
                 sin_lat_lons=None,
                 cos_lat_lons=None
                 ):
        
        super(A3TGCN, self).__init__()
        self.lat_lons = torch.cat([torch.tensor(sin_lat_lons, dtype=torch.float32), torch.tensor(cos_lat_lons, dtype=torch.float32)], dim=-1)
        self.positional_encoder = nn.Sequential(nn.Linear(4, input_size), nn.Hardswish())
        
        self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        
        learnable = []
        time = []
        # time_non_linearities = []
        space = []
        skip_connections = []
        norms = []
        # edge_mpls = []
        shape_match = []
        for i in range(number_of_blocks):
            # learnable.append(LearnableWeight(n_nodes=n_nodes, learnable_weight_dim=learnable_feature_size))
            # new_hidden_size = hidden_size + learnable
            # if i == 0:
            #     shape_match.append(nn.Identity())
            # else:
            # shape_match.append(nn.Linear(hidden_size, hidden_size))
            # hidden_size = hidden_size + learnable_feature_size
            
            # time.append(TemporalSelfAttention(hidden_size, 16, dropout=0.3, causal=True))
            time.append(AttentionStack(n_attention_blocks=1, hidden_size=hidden_size))
            # time.append(SynapticChain(
            #     hidden_size=hidden_size,
            #     return_last=False,
            #     output_type='membrane_potential',
            #     n_layers=number_of_temporal_steps,
            #     ))
            # time_non_linearities.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Hardswish()))
            space.append(DenseBlock(in_channels=hidden_size, out_channels=hidden_size, n_blocks=5))
            skip_connections.append(nn.Linear(hidden_size, hidden_size))
            norms.append(Norm('batch', hidden_size))
            # edge_mpls.append(MLP(1,32,1))
            
            
        # self.readout = MultiHorizonMLPDecoder(
        #     input_size=hidden_size,
        #     exog_size=input_size,
        #     hidden_size=hidden_size*2,
        #     context_size=hidden_size*2,
        #     output_size=out_features,
        #     n_layers=3,
        #     horizon=horizon,
        #     dropout=0.3,
        #     activation='silu'
        #     )
        # self.rearrange_for_readout = Rearrange('b t n f -> b n f t')
        # self.rearrange_post_readout = Rearrange('b n f t -> b t n f')
        
        output_A3TGCN = []
        output_nonlinearities = []
        # pool_output_results = []
        for i in range(horizon):
            output_A3TGCN.append(
                A3TGCNBlock(in_channels=hidden_size, out_channels=out_features, periods=window),
                )
        #     output_A3TGCN.append(EvolveGCN(
        #         input_size=hidden_size,
        #         hidden_size=hidden_size,
        #         n_layers=1,
        #         norm='asym',
        #         activation='silu'
        #         ))
        #     # output_A3TGCN.append(
        #     #     TemporalSelfAttention(hidden_size, 8, dropout=0.3, causal=True)
        #     #     )
        #     # pool_output_results.append(AttPool(hidden_size, dim=1))
            # output_nonlinearities.append(nn.Sequential
            #     (nn.ELU(),
            #     nn.Linear(out_features, out_features)))
        #     # output_A3TGCN.append(SynapticChain(
        #     #     hidden_size=hidden_size,
        #     #     return_last=True,
        #     #     output_type='membrane_potential',
        #     #     n_layers=number_of_temporal_steps,
        #     #     output_size=out_features,
        #     #     ))
        #     # output_edge_mlps.append(MLP(1,32,1))
            
        # self.learnable = nn.ModuleList(learnable)
        # self.shape_match = nn.ModuleList(shape_match)
        self.time = nn.ModuleList(time)
        # self.time_non_linearities = nn.ModuleList(time_non_linearities)
        self.space = nn.ModuleList(space)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        # self.edge_mpls = nn.ModuleList(edge_mpls)
        self.output_A3TGCN = nn.ModuleList(output_A3TGCN)
        self.output_nonlinearities = nn.ModuleList(output_nonlinearities)
        # self.pool_output_results = nn.ModuleList(pool_output_results)
        # self.output_skip = nn.Linear(hidden_size, out_features)
        
        
        
    def forward(self, x, edge_index, edge_weight):
        # x - batch, timesteps, nodes, features
        # assert not torch.isnan(x).any()
        x_orig = x
        positional_encoding = checkpoint(self.positional_encoder, self.lat_lons.to(x.device))
        
        x = x + positional_encoding.expand(x.size(0), x.size(1), -1, -1)
        x = checkpoint(self.encoder, x)
        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        
        for ( time, space, skip, norm) in zip(
            # self.learnable,
            self.time,
            # self.time_non_linearities,
            self.space,
            self.skip_connections,
            self.norms,
            # self.edge_mpls,
            # self.shape_match
            ):
            # utils.reset(time)
            
            # standard loop
            # x = checkpoint(learnable, x)
            # res = x
            # x = checkpoint(space, x, edge_index, edge_weight)
            # out = checkpoint(skip, x) + checkpoint(shape_match,out)[:, -x.size(1):]
            # x, _ = checkpoint(time, x)
            # x = checkpoint(tine_nonlinearity, x)
            # x = x + res[:, -x.size(1):]
            # x = norm(x)
            # standard loop
            
            # dense residual loop
            # x = checkpoint(learnable, x)
            res = x
            time_processed = checkpoint(time, x)
            # nonlinear_time = checkpoint(tine_nonlinearity, time_processed)
            out = checkpoint(skip, time_processed)
            space_processed = checkpoint(space, out, edge_index, edge_weight)
            
            with_long_residual = space_processed + res[:, -x.size(1):]
            x = norm(with_long_residual)
            
            
        # ----------------- head per step readout ---------------
        output_steps = []
        # x = self.rearrange_for_readout(x)
        # assert not torch.isnan(x).any()
        # out_skip = checkpoint(self.output_skip, x)
        # x = x + x_orig
        # print("pre loop")
        
        for output_head in self.output_A3TGCN:
            # print("in loop")
            # edge_features = checkpoint(edge_mlp, edge_weight.expand(1,-1).T).squeeze()
            step_output = checkpoint(output_head, x, edge_index, edge_weight)
            # weighted, scores = checkpoint(output_head, x)
            # pooled = checkpoint(temporal_pooling, weighted)
            # nonlinear = checkpoint(nonlinearity, pooled)
            # output = self.rearrange_post_readout(step_output)
            # print(f"step_output: {step_output.size()}")
            output_steps.append(step_output[:, -1:])
            
            # output_steps.append(nonlinear)
            
        # print(output_steps)
        res = torch.cat(output_steps, dim=1) # concat across time producing {horizon} outputs
        # ----------------- head per step readout ---------------
        
        # res = self.readout(x, x_orig[:, -4:])
        # print(res.size(), x_orig[:, -4:].size())
        # assert not torch.isnan(res).any()
        return res + x_orig[:, -4:]
        # if torch.isnan(x).any():
        #     print(f"still NaN at result: {torch.isnan(res.view(-1)).sum()}")
        # #     res = torch.nan_to_num(res) 
        # return res + x_orig[:, -4:]