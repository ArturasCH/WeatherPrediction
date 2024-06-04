import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from tsl.nn.layers.base import TemporalSelfAttention
from tsl.nn.layers import Norm


class AttentionStack(nn.Module):
    def __init__(self, n_attention_blocks=1, hidden_size=256, n_heads=4, dropout=0.3):
        super(AttentionStack, self).__init__();
        
        attention_blocks = []
        nonlinearities = []
        for i in range(n_attention_blocks):
            attention_blocks.append(TemporalSelfAttention(hidden_size, n_heads, dropout=dropout, causal=True))
            nonlinearities.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU()))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.nonlinearities = nn.ModuleList(nonlinearities)
        self.norm = Norm('batch', hidden_size)
        
        
    def forward(self, x):
        
        x_orig = x
        
        for (attention, nonlinearity) in zip(self.attention_blocks, self.nonlinearities):
            attn, _ = checkpoint(attention, x)
            x = checkpoint(nonlinearity, attn)
            
            
        return checkpoint(self.norm, x) + x_orig