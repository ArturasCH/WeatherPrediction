import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tsl.nn.layers import NodeEmbedding, DenseGraphConvOrderK
from tsl.nn.blocks.decoders import MLPDecoder

class AdjPretrain(nn.Module):
    def __init__(self,
                 n_nodes=2048,
                 input_size=26,
                 hidden_size=256,
                 out_features=26,
                 horizon=1,
                 n_layers=4
                 ):
        super(AdjPretrain, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU()
        )
        
        self.node_embedings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.spatial_source_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.spatial_target_emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        
        convs = []
        nonlinearities = []
        for i in range(n_layers):
            convs.append(
                DenseGraphConvOrderK(hidden_size, hidden_size, support_len=1, order=3, channel_last=True)
            )
            nonlinearities.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU()
            ))
            
        self.convs = nn.ModuleList(convs)
        self.nonlinearities = nn.ModuleList(nonlinearities)
            
        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            output_size=out_features,
            horizon=horizon,
            n_layers=4,
            receptive_field=5,
            dropout=0.3,
            activation='leaky_relu'
        )
        
        
    def forward(self, x):
        x = checkpoint(self.encoder, x)
        x = x + self.node_embedings()
        
        space_adj = self.compute_spatial_adj()
        
        for conv, nn in zip(self.convs, self.nonlinearities):
            x = checkpoint(conv, x, space_adj)
            x = checkpoint(nn, x)
            
        return self.readout(x)
        
    def compute_spatial_adj(self):
        logits = F.relu(self.spatial_source_emb() @ self.spatial_target_emb().T)
        adj = torch.softmax(logits, dim=-1)
        return adj