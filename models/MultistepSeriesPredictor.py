import inspect
import lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR
from typing import Any
from tsl.utils import foo_signature
from tsl.metrics.torch import MaskedMetric
from torchmetrics import Metric, MetricCollection

class MultistepPredictor(pl.LightningModule):
    def __init__(self,
                 model=None,
                 loss_fn=None,
                 metrics=None,
                 prediction_horizon=4,
                 sequence_horizon=20,
                 window=20,
                 accumulate_grad_batches=1,
                 initial_rollouts = 1,
                 rollout_epoch_breakpoint=3) -> None:
        super(MultistepPredictor, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.sequence_horizon = sequence_horizon
        self.window = window
        self.max_n_rollouts = self.sequence_horizon // self.prediction_horizon
        self.loss_fn = loss_fn
        self.automatic_optimization = False
        self.rollout_epoch_breakpoint = rollout_epoch_breakpoint
        self.n_rollouts = initial_rollouts
        self.accumulate_grad_batches = accumulate_grad_batches
        
        self.model = model
        self._model_fwd_signature = foo_signature(self.model.forward)
        self._set_metrics(metrics=metrics)
    
    def _filter_forward_kwargs(self, kwargs: dict) -> dict:
        """"""
        # if self._check_kwargs:
        model_args = self._model_fwd_signature['signature']
        filtered = set(kwargs).difference(model_args)
        forwarded = set(kwargs).intersection(model_args)
        msg = f"Only args {list(forwarded)} are forwarded to the model " \
                f"({self.model.__class__.__name__}). "
        if len(filtered):
            msg = f"Arguments {list(filtered)} are filtered out. " + msg
        # logger.warning(msg)
        self._check_kwargs = False
        return {
            k: v
            for k, v in kwargs.items()
            if k in self._model_fwd_signature['signature']
        }

    def forward(self, x, radiation, exog):
        return self.model(x, exog, radiation)
  
        
    def training_step(self, batch, batch_idx):
        if self.current_epoch != 0 and self.current_epoch % self.rollout_epoch_breakpoint == 0 and batch_idx == 0 and self.n_rollouts < self.max_n_rollouts:
            print(f"batch: {batch_idx}, epoch: {self.current_epoch}, n_rollouts: {self.n_rollouts}")
            self.n_rollouts += 1
            
        opts = self.optimizers()
        loss, predictions = self._calculate_multirollout_loss(batch, batch_idx, self.n_rollouts)
        mask = batch.get('mask')
        # output = batch.target.y
        x = batch.input.x
        y = x[:, self.window:]
        predicted_steps = predictions.size(1)
            
        self.train_metrics.update(predictions, y[:, :predicted_steps], mask)
        if batch_idx % self.accumulate_grad_batches == 0:
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', loss, batch_size=batch.batch_size)
            
        # loss = self.compute_loss(batch) / self.accumulate_grad_batches
        # self.manual_backward(loss / self.accumulate_grad_batches, retain_graph=True)
        self.manual_backward(loss)
        # self.manual_backward(loss)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            # self.clip_gradients(opts, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opts.step()
            opts.zero_grad()
            
        
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, predictions = self._calculate_multirollout_loss(batch, batch_idx, self.n_rollouts + 3)
            mask = batch.get('mask')
            x = batch.input.x
            y = x[:, self.window:]
            predicted_stes = predictions.size(1)
                
            self.val_metrics.update(predictions, y[:, :predicted_stes], mask)
            if batch_idx % 10 == 0 or batch_idx == 0:
                self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
                self.log_loss('val', loss, batch_size=batch.batch_size)
                
            return loss
    
    # def predict_step(self, batch, batch_idx):
    #     x = batch.input.x
    #     edge_index = batch.input.edge_index
    #     edge_weight = batch.input.edge_weight
    #     exog = batch.exog
    #     output = batch.target.y
        
    #     prediction = self(x, edge_index=edge_index, edge_weight=edge_weight, exog=exog)
    #     run_n_rollouts = self.sequence_horizon // self.prediction_horizon
        
    #     predictions = prediction
    #     next_exog = exog
        
    #     for rollout in range(1, run_n_rollouts):
    #         input_start_index = rollout * self.prediction_horizon
    #         output_start_index = rollout * self.prediction_horizon
    #         output_end_index = output_start_index + self.prediction_horizon
            
    #         # if has exog
    #         next_exog = self.advance_days_by_n_steps(next_exog[:, input_start_index:], input_start_index)
    #         next_input = torch.cat((x[:,input_start_index:], predictions), dim=1)
    #         prediction = self(next_input, edge_index=edge_index, edge_weight=edge_weight, exog=next_exog)
            
    #         predictions = torch.cat([predictions, prediction], dim=1)
            
    #     return predictions
        
    
    def _calculate_multirollout_loss(self, batch, batch_idx = 0, run_n_rollouts=1):
        x = batch.input.x
        edge_index = batch.input.edge_index
        # edge_weight = batch.input.edge_weight
        radiation = batch.radiation
        # lsm = batch.lsm
        exog = batch.exog
        
        # y = x[:, -self.sequence_horizon:]
        y = x[:, self.window:]
        x = x[:, :self.window]
        
        rad = radiation[:, :self.window]
        ex = exog[:, :self.window]
        
        
        prediction = self(x, radiation=rad, exog=ex)
        sequence_horizon = y.size(1)
        run_n_rollouts = min(run_n_rollouts, sequence_horizon // self.prediction_horizon)
        steps = torch.arange(run_n_rollouts + 1)
        decay_factor = 0.3
        rollout_weights = (1 * (1 - decay_factor) ** steps).to(x.device)
        
        loss = self.loss_fn(prediction, y[:, :self.prediction_horizon]) / run_n_rollouts
        # loss = self.loss_fn(prediction, y[:, :self.prediction_horizon]) + rollout_weights[1]
        predictions = prediction.detach()
        
        sequence_length = x.size(1)
        for rollout in range(1, run_n_rollouts):
            input_start_index = rollout * self.prediction_horizon
            input_end_index = input_start_index+sequence_length
            output_start_index = rollout * self.prediction_horizon
            output_end_index = output_start_index + self.prediction_horizon
            
            # if has exog
            # next_exog = self.advance_days_by_n_steps(next_exog[:, input_start_index:], input_start_index)
            next_input = torch.cat((x[:,input_start_index:], predictions), dim=1)
            
            rad = radiation[:, input_start_index:input_end_index]
            ex = exog[:, input_start_index:input_end_index]
            prediction = self(next_input, radiation=rad, exog=ex)
            
            loss += (self.loss_fn(prediction, y[:, output_start_index:output_end_index]) / run_n_rollouts)
            # loss += (self.loss_fn(prediction, y[:, output_start_index:output_end_index]) * rollout_weights[rollout + 1])
            predictions = torch.cat([predictions, prediction.detach()], dim=1)
            
        return loss, predictions
    
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
    
    
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # return {'optimizer':optimizer, 'lr_scheduler':scheduler}
        
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
    # def load_from_checkpoint(self, checkpoint_path):
    #     return super().load_from_checkpoint(checkpoint_path)
    

from load_data import get_data
from DataLoader import WeatherDL2
import pickle
from models.FiLMWithLearnable import FiLMWithLearnable
import xarray as xr

data = get_data()
mean_std = pickle.load(open('./mean_std_stacked.pickle', 'rb'))
normalization_range = {'min': -1, 'max': 1}
batch_size = 8
num_workers = 6
prefetch_factor = 1
persistent_workers=True
pin_memory = False
temporal_resolution = 6
window = 24
sequence_horizon = 1
horizon = 0

# train = WeatherDL2(
#         data,
#         # time="2003-06",
#         # time=slice('1992', '2003'),
#         # time=slice('2012', '2017'),
#         time=slice('2000', '2017'),
#         temporal_resolution=temporal_resolution,
#         mean=mean_std['mean'],
#         std=mean_std['std'],
#         batch_size=batch_size,
#         num_workers=num_workers,
#         window=window,
#         horizon=horizon,
#         persistent_workers=persistent_workers,
#         prefetch_factor=prefetch_factor,
#         pin_memory=pin_memory,
#         shuffle=True,
#         normalization_range=normalization_range,
#         # multiprocessing_context='fork'
#         )
prediction_horizon = 1
n_layers = 20
hidden_size = 256   #@param
edge_embedding_size = 64
spatial_block_size = 3
temporal_block_size = 1
# lat_lons = train.data_wrapper.get_data().node.values
input_size = 65   # n channel
n_nodes = 2048         # n nodes
constants = xr.open_dataset('./data/all_5.625deg/constants/constants_5.625deg.nc').stack(node=['lat', 'lon'])
lat_lons = constants.node.values
lsm = torch.tensor(constants['lsm'].data)
model = FiLMWithLearnable(
    lsm=lsm,
    # norm_scale=norm_scale,
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
                #  model=None,
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
  
        
    def training_step(self, batch, batch_idx):
        if self.current_epoch != 0 and self.current_epoch % self.rollout_epoch_breakpoint == 0 and batch_idx == 0 and self.n_rollouts < self.max_n_rollouts:
            print(f"batch: {batch_idx}, epoch: {self.current_epoch}, n_rollouts: {self.n_rollouts}")
            self.n_rollouts += 1
            
        opts = self.optimizers()
        loss, predictions = self._calculate_multirollout_loss(batch, batch_idx, self.n_rollouts)
        mask = batch.get('mask')
        # output = batch.target.y
        x = batch.input.x
        y = x[:, self.window:]
        predicted_steps = predictions.size(1)
            
        self.train_metrics.update(predictions, y[:, :predicted_steps], mask)
        if batch_idx % self.accumulate_grad_batches == 0:
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', loss, batch_size=batch.batch_size)
            
        # loss = self.compute_loss(batch) / self.accumulate_grad_batches
        # self.manual_backward(loss / self.accumulate_grad_batches, retain_graph=True)
        self.manual_backward(loss / self.accumulate_grad_batches)
        # self.manual_backward(loss)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            # self.clip_gradients(opts, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opts.step()
            opts.zero_grad()
            
        
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, predictions = self._calculate_multirollout_loss(batch, batch_idx, self.n_rollouts + 3)
            mask = batch.get('mask')
            x = batch.input.x
            y = x[:, self.window:]
            predicted_steps = predictions.size(1)
                
            self.val_metrics.update(predictions, y[:, :predicted_steps], mask)
            if batch_idx % 10 == 0 or batch_idx == 0:
                self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
                self.log_loss('val', loss, batch_size=batch.batch_size)
                
            return loss
    
    def predict_step(self, batch, batch_idx, run_n_rollouts=1):
        with torch.no_grad():
            predictions = self._calculate_multirollout_loss(batch, batch_idx, run_n_rollouts)
            
            return predictions
            # x = batch.input.x
            # edge_index = batch.input.edge_index
            # # edge_weight = batch.input.edge_weight
            # radiation = batch.radiation
            # # lsm = batch.lsm
            # exog = batch.exog
            
            # # y = x[:, -self.sequence_horizon:]
            # # y = x[:, self.window:]
            # x = x[:, :self.window]
            
            # rad = radiation[:, :self.window]
            # ex = exog[:, :self.window]
            
            
            # prediction = self(x, radiation=rad, exog=ex)
            # # sequence_horizon = y.size(1)
            # # run_n_rollouts = min(run_n_rollouts, sequence_horizon // self.prediction_horizon)
            # # steps = torch.arange(run_n_rollouts + 1)
            # # decay_factor = 0.3
            # # rollout_weights = (1 * (1 - decay_factor) ** steps).to(x.device)
            
            # # loss = self.loss_fn(prediction, y[:, :self.prediction_horizon]) / run_n_rollouts
            # # loss = self.loss_fn(prediction, y[:, :self.prediction_horizon]) + rollout_weights[1]
            # predictions = prediction.detach()
            
            # sequence_length = x.size(1)
            # for rollout in range(1, run_n_rollouts):
            #     input_start_index = rollout * self.prediction_horizon
            #     input_end_index = input_start_index+self.window
            #     # output_start_index = rollout * self.prediction_horizon
            #     # output_end_index = output_start_index + self.prediction_horizon
                
            #     # if has exog
            #     # next_exog = self.advance_days_by_n_steps(next_exog[:, input_start_index:], input_start_index)
            #     next_input = torch.cat((x[:,input_start_index:], predictions), dim=1)
                
            #     rad = radiation[:, input_start_index:input_end_index]
            #     ex = exog[:, input_start_index:input_end_index]
            #     prediction = self(next_input, radiation=rad, exog=ex)
                
            #     # loss += (self.loss_fn(prediction, y[:, output_start_index:output_end_index]) / run_n_rollouts)
            #     # loss += (self.loss_fn(prediction, y[:, output_start_index:output_end_index]) * rollout_weights[rollout + 1])
            #     predictions = torch.cat([predictions, prediction.detach()], dim=1)
                
            # return predictions
        
    
    def _calculate_multirollout_loss(self, batch, batch_idx = 0, run_n_rollouts=1):
        x = batch.input.x
        edge_index = batch.input.edge_index
        # edge_weight = batch.input.edge_weight
        radiation = batch.radiation
        # lsm = batch.lsm
        exog = batch.exog
        
        # y = x[:, -self.sequence_horizon:]
        y = x[:, self.window:]
        x = x[:, :self.window]
        
        rad = radiation[:, :self.window]
        ex = exog[:, :self.window]
        
        
        prediction = self(x, radiation=rad, exog=ex)
        sequence_horizon = y.size(1)
        run_n_rollouts = min(run_n_rollouts, sequence_horizon // self.prediction_horizon)
        steps = torch.arange(run_n_rollouts + 1)
        decay_factor = 0.3
        rollout_weights = (1 * (1 - decay_factor) ** steps).to(x.device)
        
        # loss = self.loss_fn(prediction, y[:, :self.prediction_horizon]) / run_n_rollouts
        # loss = self.loss_fn(prediction, y[:, :self.prediction_horizon]) + rollout_weights[1]
        predictions = prediction.detach()
        
        sequence_length = x.size(1)
        for rollout in range(1, run_n_rollouts):
            input_start_index = rollout * self.prediction_horizon
            input_end_index = input_start_index+sequence_length
            output_start_index = rollout * self.prediction_horizon
            output_end_index = output_start_index + self.prediction_horizon
            
            # if has exog
            # next_exog = self.advance_days_by_n_steps(next_exog[:, input_start_index:], input_start_index)
            next_input = torch.cat((x[:,input_start_index:], predictions), dim=1)
            
            rad = radiation[:, input_start_index:input_end_index]
            ex = exog[:, input_start_index:input_end_index]
            prediction = self(next_input, radiation=rad, exog=ex)
            
            # loss += (self.loss_fn(prediction, y[:, output_start_index:output_end_index]) / run_n_rollouts)
            # loss += (self.loss_fn(prediction, y[:, output_start_index:output_end_index]) * rollout_weights[rollout + 1])
            predictions = torch.cat([predictions, prediction.detach()], dim=1)
            
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
    
    
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # return {'optimizer':optimizer, 'lr_scheduler':scheduler}
        
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
    # def load_from_checkpoint(self, checkpoint_path):
    #     return super().load_from_checkpoint(checkpoint_path)
    

            