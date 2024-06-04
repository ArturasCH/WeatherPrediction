import torch
import torch.nn as nn
import einops
from torch_geometric.nn import PNA
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import MLPDecoder
from torch.utils.checkpoint import checkpoint

class TPNA(nn.Module):
    def __init__(self,
        deg=None,
        n_nodes=2048,
        window=20,
        input_size=26,
        hidden_size=256,
        out_features=26,
        horizon=4,
        #  n_layers=12,
        n_layers=6,
        edge_embedding_size=64,
        n_relations=3    
    ):
        super(TPNA,self).__init__()
        
        self.condition_exog = ConditionalBlock(
                 input_size=input_size,
                 exog_size=2, #day of year and land sea mask
                 output_size=hidden_size,
                 dropout=0.3,
                 skip_connection=True,
                 activation='leaky_relu')
        
        self.node_embedings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU()
        )
        self.rnn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers,
                           cell='gru',
                           return_only_last_state=True)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.pna = PNA(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            num_layers=n_layers,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg
        )
        
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            # receptive_field=2,
            # receptive_field=(window - receptive_field) +1,
            receptive_field=1,
            # receptive_field=1,
            dropout=0.3,
            activation='leaky_relu'
        )
        
        
    def forward(self, x, edge_index, exog):
        b, t = x.size(0), x.size(1)
        exog = self.scale_day_of_year(exog)
        x = checkpoint(self.condition_exog,x, exog, use_reentrant=True)
        x = torch.cat([x, self.node_embedings().unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)], dim=-1)
        x = checkpoint(self.encoder,x)
        
        x_time = checkpoint(self.rnn, x)
        x = checkpoint(self.pna, einops.rearrange(x_time, 'b n f -> (b n) f'), edge_index)
        x = einops.rearrange(x, '(b t n) f -> b t n f', b=b, t=1)
        
        return self.readout(x + x_time.unsqueeze(1))
        
        
        
        
    def scale_day_of_year(self, exog):
        day_of_year = exog[..., 1:] / 365
        return torch.cat([day_of_year, exog[..., 1:]], dim=-1)
        