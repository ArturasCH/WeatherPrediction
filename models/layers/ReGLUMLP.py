import torch
import torch.nn as nn

from models.layers.ReGLU import ReGLU

class ReGluMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        exog_size=None,
        n_layers=1,
        ):
        super(ReGluMLP, self).__init__()
        
        self.layers = nn.ModuleList([
                ReGLU(input_size if i == 0 else hidden_size, hidden_size) for i in range(n_layers)
        ])
        
        
        self.readout = ReGLU(hidden_size, output_size)
        
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            
        return x
    
class ConditionalReGluMLP(nn.Module):
    def __init__(
        self,
        input_size,
        exog_size,
        output_size,
        ):
        super(ConditionalReGluMLP, self).__init__()
        
        self.input_encoder = ReGLU(input_size, output_size)
        self.exog_encoder = ReGLU(exog_size, output_size)
        
        self.out_inputs_encoder = ReGLU(output_size, output_size)
        self.out_exog_encoder = ReGLU(output_size, output_size)
        
    def forward(self, x, exog):
        
        x = self.input_encoder(x)
        exog = self.exog_encoder(exog)
        
        
        return self.out_inputs_encoder(x) + self.out_exog_encoder(exog)
        