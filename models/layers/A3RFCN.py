# import torch
# from torch_geometric.nn import ClusterGCNConv
# from einops.layers.torch import Rearrange
# from torch_geometric.typing import Adj, OptTensor


# class TGCN2(torch.nn.Module):
#     r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
#     For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
#     Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

#     Args:
#         in_channels (int): Number of input features.
#         out_channels (int): Number of output features.
#         batch_size (int): Size of the batch.
#         improved (bool): Stronger self loops. Default is False.
#         cached (bool): Caching the message weights. Default is False.
#         add_self_loops (bool): Adding self-loops for smoothing. Default is True.
#     """

#     def __init__(self, in_channels: int, out_channels: int):
#         super(TGCN2, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self._create_parameters_and_layers()

#     def _create_update_gate_parameters_and_layers(self):
#         self.conv_z = ClusterGCNConv(in_channels=self.in_channels,  out_channels=self.out_channels)
#         self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

#     def _create_reset_gate_parameters_and_layers(self):
#         self.conv_r = ClusterGCNConv(in_channels=self.in_channels, out_channels=self.out_channels)
#         self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

#     def _create_candidate_state_parameters_and_layers(self):
#         self.conv_h = ClusterGCNConv(in_channels=self.in_channels, out_channels=self.out_channels)
#         self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

#     def _create_parameters_and_layers(self):
#         self._create_update_gate_parameters_and_layers()
#         self._create_reset_gate_parameters_and_layers()
#         self._create_candidate_state_parameters_and_layers()

#     def _set_hidden_state(self, X, H):
#         # if H is None:
#         #     # can infer batch_size from X.shape, because X is [B, N, F]
#         #     H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
#         if not isinstance(H, torch.Tensor):
#             H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
#         return H

#     def _calculate_update_gate(self, X, edge_index, H):
#         Z = torch.cat([self.conv_z(X, edge_index), H], axis=-1) # (b, 207, 64)
#         Z = self.linear_z(Z) # (b, 207, 32)
#         # Z = torch.sigmoid(Z)
#         Z = torch.tanh(Z)

#         return Z

#     def _calculate_reset_gate(self, X, edge_index, H):
#         R = torch.cat([self.conv_r(X, edge_index), H], axis=-1) # (b, 207, 64)
#         R = self.linear_r(R) # (b, 207, 32)
#         # R = torch.sigmoid(R)
#         R = torch.tanh(R)

#         return R

#     def _calculate_candidate_state(self, X, edge_index, H, R):
#         H_tilde = torch.cat([self.conv_h(X, edge_index), H * R], axis=-1) # (b, 207, 64)
#         H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
#         H_tilde = torch.tanh(H_tilde)

#         return H_tilde

#     def _calculate_hidden_state(self, Z, H, H_tilde):
#         H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
#         return H

#     def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor,
#                 H: torch.FloatTensor = None ) -> torch.FloatTensor:
#         """
#         Making a forward pass. If edge weights are not present the forward pass
#         defaults to an unweighted graph. If the hidden state matrix is not present
#         when the forward pass is called it is initialized with zeros.

#         Arg types:
#             * **X** *(PyTorch Float Tensor)* - Node features.
#             * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
#             * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
#             * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

#         Return types:
#             * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
#         """
#         H = self._set_hidden_state(X, H)
#         Z = self._calculate_update_gate(X, edge_index, H)
#         R = self._calculate_reset_gate(X, edge_index, H)
#         H_tilde = self._calculate_candidate_state(X, edge_index, H, R)
#         H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
#         return H

# class A3TGCNBlock(torch.nn.Module):
#     r"""An implementation THAT SUPPORTS BATCHES of the Attention Temporal Graph Convolutional Cell.
#     For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
#     Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

#     Args:
#         in_channels (int): Number of input features.
#         out_channels (int): Number of output features.
#         periods (int): Number of time periods.
#         improved (bool): Stronger self loops (default :obj:`False`).
#         cached (bool): Caching the message weights (default :obj:`False`).
#         add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
#     """

#     def __init__(
#         self,
#         in_channels: int = 26, 
#         out_channels: int = 26,  
#         periods: int = 56):
#         super(A3TGCNBlock, self).__init__()

#         self.in_channels = in_channels  # 2
#         self.out_channels = out_channels # 32
#         self.periods = periods # 12
        
#         self.rearrange = Rearrange('b t n f -> b n f t')
#         self._setup_layers()

#     def _setup_layers(self):
#         self._base_tgcn = TGCN2(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels)

#         # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self._attention = torch.nn.Parameter(torch.empty(self.periods))
#         torch.nn.init.uniform_(self._attention)

#     def forward( 
#         self, 
#         X: torch.FloatTensor,
#         edge_index: Adj, 
#         H: torch.FloatTensor = None
#     ) -> torch.FloatTensor:
#         """
#         Making a forward pass. If edge weights are not present the forward pass
#         defaults to an unweighted graph. If the hidden state matrix is not present
#         when the forward pass is called it is initialized with zeros.

#         Arg types:
#             * **X** (PyTorch Float Tensor): Node features for T time periods.
#             * **edge_index** (PyTorch Long Tensor): Graph edge indices.
#             * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
#             * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

#         Return types:
#             * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
#         """
#         X = self.rearrange(X)
#         H_accum = 0
#         probs = torch.nn.functional.softmax(self._attention, dim=0)
#         for period in range(self.periods):

#             H_accum = H_accum + probs[period] * self._base_tgcn( X[:, :, :, period], edge_index, H_accum) #([32, 207, 32]
#         # self._base_tgcn(X, edge_index, H)

#         return H_accum


import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import ClusterGCNConv
from models.layers.GATEConv import GATEConv2 as GATEConv
from einops.layers.torch import Rearrange
from torch_geometric.typing import Adj, OptTensor
from models.layers.ResGatedGraphConv import ResGatedGraphConv
from models.layers.BatchedFAConv import BatchedFiLMConv


class TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        # self.conv_z = GATEConv(in_channels=self.in_channels,  out_channels=self.out_channels, edge_dim=1)
        self.conv_z = BatchedFiLMConv(in_channels=self.in_channels,  out_channels=self.out_channels, act=nn.LeakyReLU())
        self.linear_z = nn.Sequential(nn.Linear(2 * self.out_channels, self.out_channels), nn.Tanh())

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = BatchedFiLMConv(in_channels=self.in_channels,  out_channels=self.out_channels, act=nn.LeakyReLU())
        self.linear_r = nn.Sequential(nn.Linear(2 * self.out_channels, self.out_channels), nn.Tanh())

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = BatchedFiLMConv(in_channels=self.in_channels,  out_channels=self.out_channels, act=nn.LeakyReLU())
        self.linear_h = nn.Sequential(nn.Linear(2 * self.out_channels, self.out_channels), nn.LeakyReLU())

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        # if H is None:
        #     # can infer batch_size from X.shape, because X is [B, N, F]
        #     H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        if not isinstance(H, torch.Tensor):
            # H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
            H = torch.zeros(*X.shape[:-1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([checkpoint(self.conv_z,X, edge_index), H], axis=-1) # (b, 207, 64)
        # Z = self.linear_z(Z) # (b, 207, 32)
        Z = checkpoint(self.linear_z,Z) # (b, 207, 32)
        # Z = torch.sigmoid(Z)
        # nn.sequential added
        # Z = torch.tanh(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([checkpoint(self.conv_r,X, edge_index), H], axis=-1) # (b, 207, 64)
        R = checkpoint(self.linear_r,R) # (b, 207, 32)
        # R = torch.sigmoid(R)
        # nn.sequential added
        # R = torch.tanh(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([checkpoint(self.conv_h,X, edge_index), H * R], axis=-1) # (b, 207, 64)
        H_tilde = checkpoint(self.linear_h,H_tilde) # (b, 207, 32)
        # H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H

class A3TGCNBlock(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int = 26, 
        out_channels: int = 26,  
        periods: int = 56,
        return_emb=False):
        super(A3TGCNBlock, self).__init__()

        self.in_channels = in_channels  # 2
        self.out_channels = out_channels # 32
        self.periods = periods # 12
        self.return_emb = return_emb
        
        self.rearrange = Rearrange('b t n f -> b n f t')
        self.arrange_back = Rearrange('b n f t -> b t n f')
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods))
        torch.nn.init.uniform_(self._attention)

    def forward( 
        self, 
        X: torch.FloatTensor,
        edge_index: Adj, 
        edge_weight = None,
        H: torch.FloatTensor = 0
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        # X = self.rearrange(X)
        H_accum = H
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        # out = self._base_tgcn(X, edge_index, H)
        # print("out", out.size())
        # rearranged = self.rearrange(out)
        # H_accum = probs * rearranged
        # print(f"H_accum {H_accum.size()}")
        H_accum = 0
        embs = []
        for period in range(self.periods):
            # H = self._base_tgcn( X[:, :, :, period], edge_index, edge_weight, H_accum) #([32, 207, 32]
            # H = self._base_tgcn( X[:, period], edge_index, edge_weight, H_accum) #([32, 207, 32]
            
            H = checkpoint(self._base_tgcn, X[:, period], edge_index, edge_weight, H_accum) #([32, 207, 32]
            
            # H_accum = H_accum + probs[period] * self._base_tgcn( X[:, period], edge_index, edge_weight, H) #([32, 207, 32]
            H_accum = H_accum + probs[period] * H
            embs.append(H_accum)
        # self._base_tgcn(X, edge_index, H)

        if self.return_emb:
            return H_accum
        
        return torch.stack(embs, dim=1)
        # return H_accum.sum(dim=-1)