import lightning as pl    
import torch
from torchmetrics import MetricCollection
from tsl.metrics.torch import MaskedMetric
import xarray as xr

from model.FiLMWithLearnableTGNN import FiLMWithLearnable

n_layers = 20
hidden_size = 256
edge_embedding_size = 64
spatial_block_size = 3
temporal_block_size = 1
constants = xr.open_dataset('./data/data_files/constants_5.625deg.nc').stack(node=['lat', 'lon'])
lat_lons = constants.node.values
input_size = 65
n_nodes = 2048
lsm = torch.tensor(constants['lsm'].data)
model = FiLMWithLearnable(
    lsm=lsm,
    lat_lons=lat_lons,
    n_nodes=n_nodes,
    input_size=input_size,
    hidden_size=hidden_size,
    out_features=input_size,
    n_layers=n_layers,
    window=20,
    horizon=1,
    spatial_block_size=spatial_block_size,
)


class Predictor(pl.LightningModule):
    def __init__(self,
                 loss_fn=None,
                 metrics=None,
                 prediction_horizon=1,
                 sequence_horizon=20,
                 window = 20,
                 accumulate_grad_batches=1,
                 initial_rollouts = 1,
                 rollout_epoch_breakpoint=3) -> None:
        super(Predictor, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.sequence_horizon = sequence_horizon
        self.max_n_rollouts = self.sequence_horizon // self.prediction_horizon
        self.loss_fn = loss_fn
        self.window = window
        self.automatic_optimization = False
        self.rollout_epoch_breakpoint = rollout_epoch_breakpoint
        self.n_rollouts = initial_rollouts
        self.accumulate_grad_batches = accumulate_grad_batches
        
        self.model = model
        if metrics != None:
            self._set_metrics(metrics=metrics)
    
    def forward(self, x, radiation, exog):
        return self.model(x, exog, radiation)
    
    def predict_step(self, batch, batch_idx, run_n_rollouts=1):
        with torch.no_grad():
            predictions = self._calculate_multirollout_loss(batch, batch_idx, run_n_rollouts)
            
            return predictions
        
    
    def _calculate_multirollout_loss(self, batch, batch_idx = 0, run_n_rollouts=1):
        x = batch.input.x
        radiation = batch.radiation
        exog = batch.exog
        
        y = x[:, self.window:]
        x = x[:, :self.window]
        
        rad = radiation[:, :self.window]
        ex = exog[:, :self.window]
        
        
        prediction = self(x, radiation=rad, exog=ex)
        sequence_horizon = y.size(1)
        run_n_rollouts = min(run_n_rollouts, sequence_horizon // self.prediction_horizon)
        predictions = prediction.detach()
        
        sequence_length = x.size(1)
        for rollout in range(1, run_n_rollouts):
            input_start_index = rollout * self.prediction_horizon
            input_end_index = input_start_index+sequence_length
            output_start_index = rollout * self.prediction_horizon
            output_end_index = output_start_index + self.prediction_horizon
            
            next_input = torch.cat((x[:,input_start_index:], predictions), dim=1)
            
            rad = radiation[:, input_start_index:input_end_index]
            ex = exog[:, input_start_index:input_end_index]
            prediction = self(next_input, radiation=rad, exog=ex)
            
            predictions = torch.cat([predictions, prediction.detach()], dim=1)
            
        return predictions
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        return optimizer
    
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

            