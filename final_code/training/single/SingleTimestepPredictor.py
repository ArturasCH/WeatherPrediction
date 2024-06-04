import inspect
import lightning as pl
import torch
from tsl.utils import foo_signature
from tsl.metrics.torch import MaskedMetric
from torchmetrics import MetricCollection

import torch
import xarray as xr

from model.DenseFilmWithLearnableGNN import DenseFiLM

hidden_size = 256   #@param
constants = xr.open_dataset('./data/data_files/constants_5.625deg.nc').stack(node=['lat', 'lon'])
lsm = torch.tensor(constants['lsm'].data)
lat_lons = constants.node.values
input_size = 65
model = DenseFiLM(
        lsm=lsm,
        lat_lons=lat_lons,
        n_nodes=2048,
        input_size=input_size,
        hidden_size=hidden_size,
        out_features=input_size,
        n_layers=18,
    )

class GNNMultistepPredictor(pl.LightningModule):
    def __init__(self,
                #  model=None,
                 loss_fn=None,
                 prediction_horizon=1,
                 sequence_horizon=20,
                 accumulate_grad_batches=1,
                 initial_rollouts = 1,
                 rollout_epoch_breakpoint=3) -> None:
        super(GNNMultistepPredictor, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.sequence_horizon = sequence_horizon
        self.max_n_rollouts = self.sequence_horizon // self.prediction_horizon
        self.loss_fn = loss_fn
        self.automatic_optimization = False
        self.rollout_epoch_breakpoint = rollout_epoch_breakpoint
        self.n_rollouts = initial_rollouts
        self.accumulate_grad_batches = accumulate_grad_batches
        
        self.model = model
    
    def forward(self, x, radiation, exog, edge_index):
        return self.model(x, edge_index, exog, radiation)
  
        
    def predict_step(self, batch, batch_idx, run_n_rollouts=1):
        with torch.no_grad():
            x = batch.input.x
            edge_index = batch.input.edge_index
            radiation = batch.radiation
            exog = batch.exog
            
            predictions = []
            _input = x[:, 0]
            
            for rollout in range(0, run_n_rollouts):
                
                rad = radiation[:, rollout]
                ex = exog[:, rollout]
                
                y = self(_input, rad, ex, edge_index)
                predictions.append(y)
                _input = y
                
            return torch.stack(predictions, dim=1)
        
            