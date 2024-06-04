import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tsl.nn.layers import NodeEmbedding, DenseGraphConvOrderK, DiffConv, Norm
from torch_geometric.nn import ClusterGCNConv, EdgeConv, MixHopConv
from tsl.nn.layers import Norm
from tsl.nn.blocks import ResidualMLP, MLP
from torch_geometric.nn.norm import GraphNorm
import einops

from models.layers.GATEConv import GATEConv
from models.layers.BatchedFAConv import BatchedFAConv, BatchedFiLMConv, BatchedResGatedGraphConv, BatchedResGatedFiLMConv
from models.layers.GAT_Batched import GATBatched
from models.UnboundTanh import UnbounTanh
from models.layers.VirtualNode import VirtualNode
from models.layers.FiLM_Conv import FiLMConv
from models.layers.MultihopFilm import MultiHopFiLM
from models.layers.DenseFiLM import DenseFiLMConv
from models.layers.GraphNorm import GraphNorm

# CompiledModule = torch.compile(model, mode='reduce-overhead')
class DenseGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=8, k=3, activation=nn.ReLU()):
        super(DenseGraphConv, self).__init__()
        # self.diag_lambda = torch.nn.Parameter(torch.tensor(0.1))
        # self.graph_conv = BatchedFAConv(in_channels=in_channels, out_channels=out_channels)
        # self.graph_conv = DiffConv(in_channels=in_channels, out_channels=out_channels, k=k)
        # self.graph_conv = GATEConv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim)
        # self.edge_attrs_mlp = nn.Sequential(
        #     nn.Linear(1, 32),
        #     nn.ELU(),
        #     nn.Linear(32, 128),
        #     nn.ELU(),
        #     nn.Linear(128, edge_dim)
        # )
        
        # self.graph_conv = BatchedFiLMConv(
        #     in_channels,
        #     out_channels,
        #     act=activation)
        
        # self.graph_conv = BatchedResGatedGraphConv(
        #     in_channels,
        #     out_channels,
        #     act=torch.nn.LeakyReLU(),
        #     edge_dim=edge_dim
        #     )
        # nn = ResidualMLP(
        #     input_size=in_channels * 2,
        #     hidden_size=in_channels,
        #     output_size=out_channels,
        #     n_layers=3,
        #     activation='leaky_relu',
        #     dropout=0.3,
        #     parametrized_skip=True,
        # )
        # self.graph_conv = EdgeConv(nn)
        # self.graph_conv = DenseFiLMConv(in_channels=in_channels, out_channels=out_channels)
        
        self.graph_conv = DenseFiLMConv(in_channels, out_channels, act=activation)
        # self.graph_conv = DenseGraphConvOrderK(in_channels, out_channels, support_len=1, order=k, channel_last=True)
        # self.nonlinearity = nn.Sequential(
        #     nn.Linear(out_channels, out_channels),
        #     activation
        # )
        self.norm = GraphNorm(out_channels)
        
        
    def forward(self, x, edge_index, edge_attr=None,):
        # processed_edge_attrs = checkpoint(self.edge_attrs_mlp, edge_attr.view(-1, 1))
        # processed_edge_attrs = checkpoint(self.edge_attrs_mlp, edge_attr.unsqueeze(-1))
        # y = checkpoint(self.graph_conv,x, edge_index, edge_attr, cache_support) # diff conv
        # y = self.graph_conv(x, edge_index)
        y = self.norm(checkpoint(self.graph_conv, x, edge_index, use_reentrant=True))
        # y = checkpoint(self.nonlinearity, conved)
        # y = self.nonlinearity(conved)
        
        # normed = checkpoint(self.norm,y)
        # stacked = torch.cat([F.elu(y), x], dim=-1)
        return torch.cat([y, x], dim=-1)
        # return stacked
    

class DenseBlock(nn.Module):
    # current training 32/2
    # try out 64/3
    def __init__(self,
                 in_channels,
                 out_channels,
                 growth_rate=64,
                 edge_dim=8,
                 num_relations=3,
                 n_blocks=3,
                 activation=nn.ReLU()
                 ):
        super(DenseBlock, self).__init__()
        # self.diag_lambda = torch.nn.Parameter(torch.tensor(0.1))
        # dense_convs = []
        # norms = []
        # self.add_virtual_node = VirtualNode(n_nodes=2048, n_channels=in_channels)
        
        self.dense_convs = nn.ModuleList([
            DenseFiLMConv(in_channels, out_channels, act=activation)
            # DenseGraphConv(
            #     in_channels=in_channels,
            #     out_channels=in_channels,
            #     edge_dim=edge_dim,
            #     activation=activation
            #     )
            ])
        
        # self.gn = GraphNorm(out_channels, 1e-7)
        # for i in range(n_blocks):
            # self.dense_convs.append(DenseGraphConv(
            #     in_channels=in_channels+growth_rate*i,
            #     out_channels=growth_rate,
            #     edge_dim=edge_dim,
            #     activation=activation))

    
        # size = in_channels
        size = out_channels
  
        # size = in_channels+growth_rate*(i+1)
        # size=out_channels
        # self.reduction = ResidualMLP(
        #         input_size=size,
        #         hidden_size=size,
        #         output_size=in_channels,
        #         n_layers=3,
        #         activation='leaky_relu',
        #         dropout=0.3,
        #         parametrized_skip=True,
        #     )
        
        # self.conv = FiLMConv(in_channels=in_channels, out_channels=out_channels, num_relations=num_relations, act=activation)
        # self.conv = MixHopConv(in_channels, out_channels, torch.arange(num_relations))
        self.norm = GraphNorm(out_channels)
        # self.reduction = nn.Sequential(nn.Linear(size, out_channels))
        self.reduction = MLP(
            input_size=size,
            hidden_size=size*2,
            output_size=out_channels,
            exog_size=None,
            n_layers=3,
            activation='leaky_relu',
            dropout=0.0
            
        )
        # self.reduction = DenseFiLMConv(size, out_channels)
        
        self.reset_parameters()
        
        
        
    def forward(self, x, edge_index, edge_attr=None):
        # x, edge_index = self.add_virtual_node(x, edge_index)
        
        for conv in self.dense_convs:
            # print(x.size())
            x=checkpoint(conv,x,edge_index, use_reentrant=True)
        # edge attr here is rel_designation
        # x = checkpoint(self.conv,x, edge_index, edge_attr, use_reentrant=True)
        # x = self.norm(x)
        # out = (checkpoint(self.reduction, x, edge_index, use_reentrant=True))
        out = (checkpoint(self.reduction, x, use_reentrant=True))
        return self.norm(out)
    
    def reset_parameters(self):
        # self.reset_mlp(self.reduction.mlp)
        # self.reset_mlp(self.reduction.skip_connections)
        # self.reset_weights(self.reduction.readout)
        self.reduction.reset_parameters()

    def reset_weights(self, mlp):
        for layer in mlp:
            if hasattr(layer, 'weight'):
                torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                
    def reset_mlp(self, mlp):
        for layer in mlp:
            if isinstance(layer, (nn.Sequential)):
                self.reset_mlp(layer)
            if isinstance(layer, Dense):
                self.reset_weights(layer.affinity)
            if hasattr(layer, 'weight'):
                self.reset_weights(layer)