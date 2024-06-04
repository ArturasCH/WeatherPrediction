import torch
import torch.nn as nn
from torch_geometric.nn import FAConv
from einops import rearrange
from torch_geometric.nn import aggr
from torch_geometric.utils import sort_edge_index

from models.UnboundTanh import UnbounTanh
from models.layers.FiLM_Conv import FiLMConv
from models.layers.ResGatedGraphConv import ResGatedGraphConv
from models.layers.ResGatedFiLMConv import ResGatedFiLMConv


class BatchedFAConv(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0.1, dropout=0.3, aggrs=['SumAggregation']) -> None:
        super(BatchedFAConv, self).__init__()
        # self.eps = nn.Parameter(torch.randn(1))
        # fa_convs_for_aggr = [FAConv(in_channels, eps, dropout, cached=False, add_self_loops=True, aggr=ag) for ag in aggrs] 
        # self.fa_convs_for_aggr = nn.ModuleList(fa_convs_for_aggr)
        self.fa_conv = FAConv(
            in_channels,
            eps,
            dropout,
            cached=False,
            add_self_loops=True,
            # aggr=aggr.VariancePreservingAggregation()
            # aggr=GenAgg(MLPAutoencoder, layer_sizes=(1,8,8,16))
            # aggr=aggr.MultiAggregation(
            #     aggrs=[
            #         # aggr.SoftmaxAggregation(t=0.1, learn=True),
            #         # aggr.SoftmaxAggregation(t=1, learn=True),
            #         # aggr.SoftmaxAggregation(t=10, learn=True)
            #         # aggr.PowerMeanAggregation(learn=True, p=1.0),
            #         aggr.SumAggregation(),
            #         aggr.MeanAggregation(),
            #         aggr.StdAggregation(),
            #         aggr.VarAggregation()
            #         ],
            #     mode='proj',
            #     mode_kwargs={
            #         # 'num_heads':8,
            #         'in_channels': in_channels,
            #         'out_channels': in_channels
            #     }
            # )
        )
        self.norm = nn.BatchNorm1d(in_channels)
        self.readout = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ELU(),
            nn.Linear(in_channels, out_channels)
            )
        
    def forward(self, x, edge_index):
        if x.dim() == 4:
            return self.forward_btnf_graph(x, edge_index)
        
        if x.dim() == 3:
            return self.forward_tnf_graph(x, edge_index)
        
        if x.dim() == 2:
            return self.forward_nf_graph(x,edge_index)
        
    def forward_nf_graph(self, graph, edge_index):
        # return self.readout(self.norm(self.fa_conv(graph,graph,edge_index)))
        return self.readout(self.fa_conv(graph,graph,edge_index))
    
    def forward_tnf_graph(self, x, edge_index):
        res = []
        for graph in x:
            # res.append(self.readout(self._stack_aggrs(graph, edge_index)))
            # conved = self.fa_conv(graph,graph, edge_index)
            # res.append(self.readout(self.norm(conved)))
            res.append(self.forward_nf_graph(graph, edge_index))
        return torch.stack(res)
    
    def forward_btnf_graph(self,x,edge_index):
        batches = []
        for tnf_graph in x:
            batches.append(self.forward_tnf_graph(tnf_graph, edge_index))
        
        return torch.stack(batches)
        
    def _stack_aggrs(self, x, edge_index):
        aggrs = []
        for fa_conv in self.fa_convs_for_aggr:
            aggrs.append(fa_conv(x,x,edge_index))
        return torch.cat(aggrs, dim=-1)
        
    def reset_parameters(self):
        # [gcn.reset_parameters() for gcn in self.fa_convs_for_aggr]
        self.fa_conv.reset_parameters()

class BatchedGCN(nn.Module):
    def __init__(self, gcn):
        super(BatchedGCN, self).__init__()
        self.gcn = gcn
        
    def forward(self, x, edge_index, edge_attr=None):
        if x.dim() == 4:
            return self.forward_btnf_graph(x, edge_index, edge_attr)
        
        if x.dim() == 3:
            return self.forward_tnf_graph(x, edge_index, edge_attr)
        
        if x.dim() == 2:
            return self.forward_nf_graph(x,edge_index, edge_attr)
        
        
    def forward_nf_graph(self,x, edge_index, edge_attr=None):
        # return self.readout(self.norm(self.fa_conv(graph,graph,edge_index)))
        return self.gcn(x, edge_index, edge_attr)
    
    def forward_tnf_graph(self, x, edge_index, edge_attr=None):
        res = []
        for graph in x:
            # res.append(self.readout(self._stack_aggrs(graph, edge_index)))
            # conved = self.fa_conv(graph,graph, edge_index)
            # res.append(self.readout(self.norm(conved)))
            res.append(self.forward_nf_graph(graph, edge_index, edge_attr))
        return torch.stack(res)
    
    def forward_btnf_graph(self,x,edge_index, edge_attr=None):
        batches = []
        for tnf_graph in x:
            batches.append(self.forward_tnf_graph(tnf_graph, edge_index, edge_attr))
        
        return torch.stack(batches)
        
    def reset_parameters(self):
        # [gcn.reset_parameters() for gcn in self.fa_convs_for_aggr]
        self.gcn.reset_parameters()
        
        
class BatchedFiLMConv(nn.Module):
    def __init__(self, in_channels,
            out_channels,
            act=torch.nn.LeakyReLU()):
            # act=UnbounTanh()):
        super(BatchedFiLMConv, self).__init__()
        self.gcn = FiLMConv(in_channels, out_channels, act=act)
        # self.batching = BatchedGCN(self.gcn)
        
    def forward(self, x, edge_index):
        # b, t, n, f = x.size()
        # stacked = rearrange(x, 'b t n f -> (b t) n f')
        conved = self.gcn(x, edge_index)
        # return self.batching(x, edge_index)
        # return rearrange(conved, '(b t) n f -> b t n f', b=b)
        return conved
    
class BatchedResGatedGraphConv(nn.Module):
    def __init__(self, in_channels,
            out_channels,
            act=torch.nn.LeakyReLU(),
            edge_dim=None
            ):
        super(BatchedResGatedGraphConv, self).__init__()
        self.gcn = ResGatedGraphConv(in_channels=in_channels, out_channels=out_channels, act=act, edge_dim=edge_dim)
        # self.batching = BatchedGCN(self.gcn)
        
    def forward(self, x, edge_index, edge_attr=None):
        b = x.size(0)
        stacked = rearrange(x, 'b t n f -> (b t) n f')
        conved = self.gcn(stacked, edge_index, edge_attr)
        # return self.batching(x, edge_index, edge_attr)
        return rearrange(conved, '(b t) n f -> b t n f', b=b)
    
    
class BatchedResGatedFiLMConv(nn.Module):
    def __init__(self, in_channels,
            out_channels,
            act=torch.nn.LeakyReLU()):
        super(BatchedResGatedFiLMConv, self).__init__()
        self.gcn = ResGatedFiLMConv(in_channels=in_channels, out_channels=out_channels, act=act)
        
    def forward(self, x, edge_index):
        b, t, n, f = x.size()
        stacked = rearrange(x, 'b t n f -> (b t) n f')
        conved = self.gcn(stacked, edge_index)
        # return self.batching(x, edge_index)
        return rearrange(conved, '(b t) n f -> b t n f', b=b)