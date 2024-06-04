from einops import rearrange
from torch import nn

from models.layers.SGPSpatialEncoding import SGPSpatialEncoder
from models.layers.Reservoir import Reservoir


class SGPEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 reservoir_size,
                 reservoir_layers,
                 leaking_rate,
                 spectral_radius,
                 density,
                 input_scaling,
                 receptive_field,
                 bidirectional,
                 alpha_decay,
                 global_attr,
                 add_self_loops=False,
                 undirected=False,
                 reservoir_activation='tanh'
                 ):
        super(SGPEncoder, self).__init__()
        self.reservoir = Reservoir(input_size=input_size,
                                   hidden_size=reservoir_size,
                                   input_scaling=input_scaling,
                                   num_layers=reservoir_layers,
                                   leaking_rate=leaking_rate,
                                   spectral_radius=spectral_radius,
                                   density=density,
                                   activation=reservoir_activation,
                                   alpha_decay=alpha_decay)

        self.sgp_encoder = SGPSpatialEncoder(
            receptive_field=receptive_field,
            bidirectional=bidirectional,
            undirected=undirected,
            add_self_loops=add_self_loops,
            global_attr=global_attr
        )
        

    def forward(self, x, edge_index, edge_weight):
        # x : [t n f], actually [b t n f]
        # x = rearrange(x, 't n f -> 1 t n f')
        x = self.reservoir(x)
        # x = x[0]
        x = self.sgp_encoder(x, edge_index, edge_weight)
        return x