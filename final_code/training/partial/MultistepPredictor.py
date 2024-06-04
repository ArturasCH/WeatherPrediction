import inspect
import lightning as pl
import torch
from typing import Any
from tsl.metrics.torch import MaskedMetric
from torchmetrics import Metric, MetricCollection

import torch
import xarray as xr

from model.DenseFiLMWithLearnable import DenseFilmWithLearnable

prediction_horizon = 1
n_layers = 20
hidden_size = 256   #@param
edge_embedding_size = 64
spatial_block_size = 3
temporal_block_size = 1
constants = xr.open_dataset('data/data_files/constants_5.625deg.nc').stack(node=['lat', 'lon'])
lat_lons = constants.node.values
input_size = 26   # n channel
n_nodes = 2048         # n nodes
horizon = 20         # n prediction time steps
model = DenseFilmWithLearnable(
    lat_lons=lat_lons,
    n_nodes=2048,
    horizon=prediction_horizon,
    window=20,
    input_size=input_size,
    hidden_size=hidden_size,
    out_features=input_size,
    n_layers=n_layers,
    spatial_block_size=spatial_block_size,
)

class MultistepPredictorPartial(pl.LightningModule):
    def __init__(self,
                 loss_fn=None,
                 metrics=None,
                 prediction_horizon=1,
                 sequence_horizon=20,
                 accumulate_grad_batches=1,
                 initial_rollouts = 1,
                 rollout_epoch_breakpoint=3) -> None:
        super(MultistepPredictorPartial, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.sequence_horizon = sequence_horizon
        self.max_n_rollouts = self.sequence_horizon // self.prediction_horizon
        self.loss_fn = loss_fn
        self.automatic_optimization = False
        self.rollout_epoch_breakpoint = rollout_epoch_breakpoint
        self.n_rollouts = initial_rollouts
        self.accumulate_grad_batches = accumulate_grad_batches
        
        self.model = model
        if metrics != None:
            self._set_metrics(metrics=metrics)
    
    def forward(self, x, exog):
        return self.model(x, exog)
  
    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch.input.x
            edge_index = batch.input.edge_index
            edge_weight = batch.input.edge_weight
            exog = batch.exog
            output = batch.target.y
            
            prediction = self(x, exog=exog)
            run_n_rollouts = self.sequence_horizon // self.prediction_horizon
            
            predictions = prediction
            next_exog = exog
            
            for rollout in range(1, run_n_rollouts):
                input_start_index = rollout * self.prediction_horizon
                output_start_index = rollout * self.prediction_horizon
                output_end_index = output_start_index + self.prediction_horizon
                
                # if has exog
                next_exog = self.advance_days_by_n_steps(next_exog[:, input_start_index:], input_start_index)
                next_input = torch.cat((x[:,input_start_index:], predictions), dim=1)
                prediction = self(next_input, exog=next_exog)
                
                predictions = torch.cat([predictions, prediction], dim=1)
                
            return predictions     
        
        
    def advance_days_by_n_steps(self, exog, n_steps):
        new_exog = torch.clone(exog)
        for step in range(n_steps):
            next_step = self.generate_next_step(new_exog)
            new_exog = torch.cat([new_exog, next_step], dim=1)
        
        return new_exog
    
    def generate_next_step(self, exog):
        #check last {timesteps per day}
        day_window = exog[:, -4:,0, 0]
        # last day values
        last_day = exog[:, -1,0, 0] 
        last_day_mask = day_window == last_day.unsqueeze(-1)
        counts_per_batch = last_day_mask.sum(dim=1)
        remaining_same_steps = torch.ones_like(counts_per_batch) * 4 - counts_per_batch
        # advance to next day where needed rolling over 365 days a year
        next_day = torch.where(remaining_same_steps != 0, last_day, (last_day+1) % 356) 
        #expand to accomodate exog dims
        next_step_for_batch = next_day.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, exog.size(-2), -1) 
        #concat static land sea mask to have proper dims for time concat
        next_step_for_batch = torch.cat([next_step_for_batch, exog[:, :1, :, -1:]],dim=-1) 
        return next_step_for_batch 

    
    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            metrics={k: self._check_metric(m, on_step=True)
                     for k, m in metrics.items()},
            prefix='train_')
        self.val_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='val_')
        self.test_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='test_')
    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = dict()
            return MaskedMetric(metric,
                                compute_on_step=on_step,
                                metric_fn_kwargs=metric_kwargs)
        metric = metric.clone()
        metric.reset()
        return metric

    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(metrics,
                      on_step=True,
                      on_epoch=True,
                      logger=True,
                      prog_bar=True,
                      **kwargs)

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(name + '_loss',
                 loss.detach(),
                 on_step=True,
                 on_epoch=True,
                 logger=True,
                 prog_bar=True,
                 **kwargs)
    

            