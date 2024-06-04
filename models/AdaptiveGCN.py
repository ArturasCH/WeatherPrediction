import torch
import torch.nn as nn

from tsl.nn.blocks import ResidualMLP
from torch.utils.checkpoint import checkpoint
from tsl.nn.blocks.encoders.recurrent import AGCRN

class AdaptiveGRUGCN(nn.Module):
    def __init__(self,
                 lat_lons,
                n_nodes=2048,
                window=20,
                input_size=26,
                hidden_size=256,
                out_features=26,
                horizon=4,
                #  n_layers=12,
                n_layers=6,
                edge_embedding_size=10):
        super(AdaptiveGRUGCN, self).__init__()
        
        self.encode_graph = ResidualMLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        
        # self.adaptive_gru = nn.ModuleList([])
        # for layer in range(n_layers):
        #     self.adaptive_gru.append(
                
        #     )
        
        self.adaptive_gru = AGCRN(
            input_size= hidden_size,
            emb_size= hidden_size,
            hidden_size= hidden_size,
            num_nodes= n_nodes,
            n_layers= 2,
            cat_states_layers= False,
            return_only_last_state=True,
            bias= True
        )
        
        self.readout = ResidualMLP(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=out_features,
            n_layers=2,
            activation='leaky_relu',
            dropout=0.3,
            parametrized_skip=True,
        )
        
        
    def forward(self, x):
        x = checkpoint(self.encode_graph,x)
        x = checkpoint(self.adaptive_gru, x)
        x = self.readout(x)
        
        return x.unsqueeze(1)
    