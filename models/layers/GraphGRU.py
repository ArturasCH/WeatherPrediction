import torch
import torch.nn
from tsl.nn.layers.recurrent.base import GraphGRUCellBase
from models.layers.BatchedFAConv import BatchedResGatedGraphConv
    
    
    
class GraphGRU(GraphGRUCellBase):
    """The Adaptive Graph Convolutional cell from the paper `"Adaptive Graph
    Convolutional Recurrent Network for Traffic Forecasting"
    <https://arxiv.org/abs/2007.02842>`_ (Bai et al., NeurIPS 2020).

    Args:
        input_size: Size of the input.
        emb_size: Size of the input node embeddings.
        hidden_size: Output size.
        num_nodes: Number of nodes in the input graph.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 edge_dim = None,
                 bias: bool = True):
        self.input_size = input_size
        # instantiate gates
        forget_gate = BatchedResGatedGraphConv(input_size + hidden_size,
                                        out_channels=hidden_size,
                                        edge_dim=edge_dim)
        update_gate = BatchedResGatedGraphConv(input_size + hidden_size,
                                        out_channels=hidden_size,
                                        edge_dim=edge_dim)
        candidate_gate = BatchedResGatedGraphConv(input_size + hidden_size,
                                           out_channels=hidden_size,
                                           edge_dim=edge_dim)
        super(GraphGRU, self).__init__(hidden_size=hidden_size,
                                        forget_gate=forget_gate,
                                        update_gate=update_gate,
                                        candidate_gate=candidate_gate)
