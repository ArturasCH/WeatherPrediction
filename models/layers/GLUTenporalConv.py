import torch.nn as nn
from einops import rearrange
from tsl.nn.functional import gated_tanh

class GatedTempConv(nn.Module):
    def __init__(
        self,
        input_channels,
        kernel_size,
        dilation=1,
        stride=1,
        output_channels=None,
        # n_layers=1,
        # gated=False,
        # dropout=0.,
        # activation='relu',
        causal_padding=True,
        bias=True,
        # channel_last=True,
        padding=0,
        ):
        super(GatedTempConv, self).__init__()
        if causal_padding:
            self.padding = ((kernel_size - 1) * dilation, 0)
        else:
            self.padding = padding
        self.pad_layer = nn.ZeroPad1d(self.padding)
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=2 * output_channels, #gated - generate 2x output kernels for gate + output
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias
        )
        
    def forward(self, x):
        # x - b t n f
        b = x.size(0)
        
        # for 1d convs
        x = rearrange(x, 'b t n f -> (b n) f t')
        x = self.pad_layer(x)
        x = self.conv(x)
        
        # reshape back
        x = rearrange(x, '(b n) f t -> b t n f', b=b)
            
        
        return gated_tanh(x, dim=-1)