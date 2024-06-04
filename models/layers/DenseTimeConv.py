import torch
import torch.nn as nn
from tsl.nn.blocks.encoders import TemporalConvNet
from tsl.nn.layers import Norm
from torch.utils.checkpoint import checkpoint
from models.layers.GraphNorm import GraphNorm
from models.layers.FiLM import FiLM

class DenseTemporalConvNet(nn.Module):
    def __init__(self, 
                 input_channels,
                 hidden_channels,
                 growth_rate,
                 dilation
                 ) -> None:
        super(DenseTemporalConvNet,self).__init__()
        self.tconv = TemporalConvNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=growth_rate,
            kernel_size=2,
            dilation=dilation,
            exponential_dilation=False,
            n_layers=1,
            causal_padding=True,
            gated=True,
            activation='silu'
        )
        self.norm = Norm('batch', input_channels + growth_rate)
        
    def forward(self,x):
        y = self.tconv(x)
        
        # return torch.cat([x, y], dim=-1)
        res = torch.cat([x, y], dim=-1)
        return self.norm(res)

class DenseTemporalBlock(nn.Module):
    def __init__(self, input_channels,
                 hidden_channels,
                 output_channels,
                 growth_rate,
                 dilation,
                 n_blocks=4,
                 pad=False,
                 ) -> None:
        super(DenseTemporalBlock, self).__init__()
        # self.dense_blocks = nn.ModuleList([])
        # _dilation=2
        # for i in range(n_blocks):
        #     # d = _dilation**(i % 2)
        #     d = _dilation**i
        #     self.dense_blocks.append(
        #         DenseTemporalConvNet(
        #             input_channels=input_channels+growth_rate*i,
        #             hidden_channels=hidden_channels,
        #             growth_rate=growth_rate,
        #             dilation=d
        #         )
        #     )
        self.pad = pad
        # if not pad:
        self.tconv = TemporalConvNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=2,
            dilation=1,
            exponential_dilation=False,
            n_layers=n_blocks,
            causal_padding=True,
            gated=True,
            activation='silu'
        )
        self.tconv_gnorm = GraphNorm(output_channels)
        self.dilated_tconv = TemporalConvNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=2,
            dilation=2,
            exponential_dilation=True,
            n_layers=n_blocks,
            causal_padding=True,
            gated=True,
            activation='silu',
            channel_last=True,
        )
        self.dilated_gnorm = GraphNorm(output_channels)
        # self.norm = Norm('batch', input_channels+growth_rate*(i+1))
        # self.norm = Norm('batch', output_channels * 2)
        self.readout = TemporalConvNet(input_channels=output_channels * 2,
                                hidden_channels=hidden_channels,
                                output_channels=output_channels,
                                kernel_size=2,
                                dilation=1,
                                exponential_dilation=False,
                                n_layers=1,
                                causal_padding=pad,
                                gated=True,
                                activation='leaky_relu',
                                channel_last=True
                                )
        self.out_gnorm = GraphNorm(output_channels)
        # else:
            # self.out_gnorm = FiLM(input_channels, output_channels)
            # self.out_gnorm = nn.Linear(input_channels, output_channels)
        
    def forward(self,x):
        # for tconv in self.dense_blocks:
        #     x = checkpoint(tconv, x, use_reentrant=True)
            # x = tconv(x)
        # if not self.pad:
        x_local = self.tconv_gnorm(checkpoint(self.tconv, x, use_reentrant=True))
        x_global = self.dilated_gnorm(checkpoint(self.dilated_tconv, x, use_reentrant=True))
        
        x = torch.cat([x_local, x_global], dim=-1)
        # return checkpoint(self.readout, x)
        # x = self.norm(x)
        x = self.readout(x)
        return self.out_gnorm(x)