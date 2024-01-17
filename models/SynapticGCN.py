import torch
from lightning import LightningModule

from .TemporalSynapticGraphConvNet import TemporalSynapticGraphConvNet

class SynapticGCN(LightningModule):
    def __init__(self, input_size: int,
                 n_nodes: int,
                 horizon: int,
                 hidden_size: int = 32,
                 gnn_kernel: int = 2,
                 output_type = 'spike',
                 temporal_reduction_factor = 2,
                 number_of_blocks = 2,
                 prediction_horizon = 40
                 ) -> None:
        super(SynapticGCN, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.sequence_horizon = horizon
        self.n_rollouts = self.sequence_horizon / self.prediction_horizon
        self.model = TemporalSynapticGraphConvNet(
            input_size=input_size,
            n_nodes=n_nodes,
            horizon=prediction_horizon,
            hidden_size=hidden_size,
            gnn_kernel=gnn_kernel,
            output_type=output_type,
            temporal_reduction_factor=temporal_reduction_factor,
            number_of_blocks=number_of_blocks
            )
        
        self.automatic_optimization = False
        
    def forward(self,  x, edge_index, edge_weight):
        return self.model(x, edge_index, edge_weight)
        
        
    def training_step(self, batch, batch_idx):
        x = batch.input.x
        edge_index = batch.input.edge_index
        edge_weight = batch.input.edge_weight
        output = batch.output.y
        print(self.loss)
        
        prediction = self(x, edge_index, edge_weight)
        loss = self.loss(prediction, output[:, :self.prediction_horizon, :, :])
        
        for rollout in range(1, self.n_rollouts):
            input_start_index = rollout * self.prediction_horizon
            output_start_index = rollout * self.prediction_horizon
            output_end_index = output_start_index + self.prediction_horizon
            
            prediction = self(torch.cat((x[:,input_start_index:, :, :], prediction[:,:, :, :]), dim=1), edge_index, edge_weight)
            loss += self.loss(prediction, output[:, output_end_index:output_end_index, :, :])
            
        return loss
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.output_type = output_type
        # self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        # self.node_embeddings = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        
        # # start with 1 layer, time then space model lookalike
        # # self.time_nn = TemporalSpike(hidden_size=hidden_size, return_last=False)
        # temporal = []
        # spatial = []
        # for i in range(number_of_blocks):
        #     is_last = i == number_of_blocks - 1
        #     time_nn = SynapticChain(
        #         hidden_size=hidden_size,
        #         return_last=is_last,
        #         output_type=output_type,
        #         temporal_reduction_factor=temporal_reduction_factor
        #         )
        #     space_nn = DiffConv(
        #                     # in_channels=hidden_size * ((3-1)**2),
        #                     in_channels=hidden_size,
        #                     out_channels=hidden_size,
        #                     k=gnn_kernel)
        #     temporal.append(time_nn)
        #     spatial.append(space_nn)
        
        # self.time_nn = SynapticChain(
        #     hidden_size=hidden_size,
        #     return_last=False,
        #     output_type=output_type,
        #     temporal_reduction_factor=temporal_reduction_factor
        #     )
        # self.space_nn = DiffConv(
        #                         # in_channels=hidden_size * ((3-1)**2),
        #                         in_channels=hidden_size,
        #                         out_channels=hidden_size,
        #                         k=gnn_kernel)
        # self.time_nn2 = SynapticChain(hidden_size=hidden_size, return_last=True, output_type=output_type)
        # self.graph_processing = nn.Sequential(*blocks)
        # self.temporal = nn.ModuleList(temporal)
        # self.spatial = nn.ModuleList(spatial)
        
        # self.decoder = nn.Linear(hidden_size, input_size * horizon)
        # self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
        
    # def forward(self, x, edge_index, edge_weight):
    #     # x: [batch time nodes features]
    #     utils.reset(self.temporal)
    #     x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
    #     x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
    
    #     # x_temporal = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
    #     # if self.output_type is 'spike':
    #     #     assert not torch.isnan(s).any()
    #     #     z = self.space_nn(s, edge_index, edge_weight)  # spatial processing for spikes
    #     # elif self.output_type is 'synaptic_current':
    #     #     assert not torch.isnan(n).any()
    #     #     z = self.space_nn(n, edge_index, edge_weight)  # spatial processing for synaptic current
    #     # else:
    #     #     if torch.isnan(m).any():
    #     #         print(m)
    #     # assert not torch.isnan(x_temporal).any()
    #     # z = self.space_nn(x_temporal, edge_index, edge_weight)  # spatial processing for membrane potentials
    #     # z = self.time_nn2(z)
    #     for time, space in zip(self.temporal, self.spatial):
    #         utils.reset(time)
    #         x_emb = time(x_emb)
    #         assert not torch.isnan(x_emb).any()
    #         x_emb = space(x_emb, edge_index, edge_weight)
    #     x_out = self.decoder(x_emb)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
    #     x_horizon = self.rearrange(x_out)
    #     return x_horizon
        
        