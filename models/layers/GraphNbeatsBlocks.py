import torch.nn as nn
from models.layers.DenseFiLM import DenseFiLMConv, DenseFiLMConvTemporal
# use spatial attention after rearranging to time_last. Will apply attention on shape f (b n) t from (b t n f)
from tsl.nn.layers.base import TemporalSelfAttention, SpatialSelfAttention
from einops.layers.torch import Rearrange
from tsl.nn.blocks import ResidualMLP
from torch.utils.checkpoint import checkpoint

class GraphBlock(nn.Module):
    def __init__(
        self,
        in_size=20,
        layer_size=256,
        ):
        super(GraphBlock, self).__init__()
        # self.time_last = Rearrange('b t n f -> b f n t')
        # self.standard_format = Rearrange('b f n t -> b t n f')
        # another alt variant - merge time and features (b n (tf)) for sequential multivariate support
        # self.time_last = Rearrange('b t n f -> b n f t') #alt variant, smaller attention map
        # self.standard_format = Rearrange('b n f t -> b t n f')
        # time then space
        self.time = TemporalSelfAttention(embed_dim=layer_size, num_heads=4, in_channels=in_size)
        self.time_mlp = ResidualMLP(
            input_size=layer_size,
            hidden_size=layer_size,
            output_size=layer_size,
            n_layers=2,
            activation='silu',
            parametrized_skip=True
        )
        self.gcn = DenseFiLMConv(in_channels=layer_size, out_channels=layer_size, act=nn.SiLU())
        
    def forward(self, x, adj):
        # x expected shape b n f t
        # x = self.time_last(x)
        
        # x, _ = checkpoint(self.time, x, use_reentrant=True)
        x = checkpoint(self.time_mlp, x, use_reentrant=True)
        x = checkpoint(self.gcn, x, adj, use_reentrant=True)
        
        # x = self.standard_format(x)
        
        return x

class GenericBasis(nn.Module):
    def __init__(
        self,
        forecast_size,
        backcast_size,
        ):
        super(GenericBasis, self).__init__()
        self.forecast_size = forecast_size
        self.backacast_size = backcast_size
        
    def forward(self, theta):
        # theta shape - b n f t
        # return theta[..., :self.backacast_size], theta[..., -self.forecast_size:]
        return theta[:, :self.backacast_size], theta[:, -self.forecast_size:]
    
class GraphNBeatsBlock(nn.Module):
    def __init__(
        self,
        in_size=20,
        theta_size=256,
        basis_function: nn.Module = None,
        layers = 5,
        layer_size = 256
        ):
        super(GraphNBeatsBlock, self).__init__()
        self.layers = nn.ModuleList([GraphBlock(in_size=in_size, layer_size=layer_size)] +
            [GraphBlock(in_size=layer_size, layer_size=layer_size) for _ in range(layers - 1)])
        self.basis_params = nn.Linear(layer_size, theta_size)
        self.basis_fn = basis_function
        
    def forward(self, x, adj):
        block_input = x
        for layer in self.layers:
            block_input = layer(block_input, adj)
        basis_params = self.basis_params(block_input)
        return self.basis_fn(basis_params)
        
