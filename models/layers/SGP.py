import torch
from einops.layers.torch import Rearrange
from torch import nn
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import MLP, ResidualMLP
from tsl.nn.functional import expand_then_cat
from tsl.nn.utils import get_layer_activation
# from tsl.utils.parser_utils import ArgParser, str_to_bool

from models.layers.sgp_spatial_embedding import sgp_spatial_embedding


class SGPModel(nn.Module):
    def __init__(self,
                 input_size,
                 order,
                 n_nodes,
                 hidden_size,
                 mlp_size,
                 output_size,
                 n_layers,
                 horizon,
                 positional_encoding,
                 emb_size=32,
                 exog_size=None,
                 resnet=False,
                 fully_connected=False,
                 dropout=0.,
                 activation='silu'):
        super(SGPModel, self).__init__()

        if fully_connected:
            out_channels = hidden_size
            self.input_encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )
        else:
            out_channels = hidden_size - hidden_size % order
            self.input_encoder = nn.Sequential(
                # [b n f] -> [b 1 n f]
                Rearrange('b n f -> b f n '),
                nn.Conv1d(in_channels=input_size,
                          out_channels=out_channels,
                          kernel_size=1,
                          groups=order),
                Rearrange('b f n -> b n f'),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )

        if resnet:
            self.mlp = ResidualMLP(
                input_size=out_channels,
                hidden_size=mlp_size,
                exog_size=exog_size,
                n_layers=n_layers,
                activation=activation,
                dropout=dropout,
                parametrized_skip=True
            )
        else:
            self.mlp = MLP(
                input_size=out_channels,
                n_layers=n_layers,
                hidden_size=mlp_size,
                exog_size=exog_size,
                activation=activation,
                dropout=dropout
            )

        if positional_encoding:
            self.node_emb = NodeEmbedding(
                n_nodes=n_nodes,
                emb_size=emb_size
            )
            self.lin_emb = nn.Linear(emb_size, out_channels)

        else:
            self.register_parameter('node_emb', None)
            self.register_parameter('lin_emb', None)

        self.readout = LinearReadout(
            input_size=mlp_size,
            output_size=output_size,
            horizon=horizon,
        )

    def forward(self, x, u=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        x = x[:, -1] if x.ndim == 4 else x
        x = self.input_encoder(x)
        if self.node_emb is not None:
            x = x + self.lin_emb(self.node_emb())
        if u is not None:
            u = u[:, -1] if u.ndim == 4 else u
            x = expand_then_cat([x, u], dim=-1)
        x = self.mlp(x)

        return self.readout(x)
