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
                 accumulate_grad_batches=1,
                 initial_rollouts = 1,
                 rollout_epoch_breakpoint=3) -> None:
        super(MultistepPredictor, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.sequence_horizon = sequence_horizon
        self.max_n_rollouts = self.sequence_horizon // self.prediction_horizon
        self.loss_fn = loss_fn
        self.automatic_optimization = False
        self.rollout_epoch_breakpoint = rollout_epoch_breakpoint
        self.n_rollouts = initial_rollouts
        self.accumulate_grad_batches = accumulate_grad_batches
        
        self.model = model
        self._set_metrics(metrics=metrics)
    
    def forward(self, x, exog):
        return self.model(x, exog)
  
        
    def training_step(self, batch, batch_idx):
        if self.current_epoch != 0 and self.current_epoch % self.rollout_epoch_breakpoint == 0 and batch_idx == 0 and self.n_rollouts < self.max_n_rollouts:
            print(f"batch: {batch_idx}, epoch: {self.current_epoch}, n_rollouts: {self.n_rollouts}")
            self.n_rollouts += 1
            
        opts = self.optimizers()
        loss, predictions = self._calculate_multirollout_loss(batch, batch_idx, self.n_rollouts)
        mask = batch.get('mask')
        output = batch.target.y
        predicted_steps = predictions.size(1)
            
        self.train_metrics.update(predictions, output[:, :predicted_steps], mask)
        if batch_idx % self.accumulate_grad_batches == 0:
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', loss, batch_size=batch.batch_size)
            
        self.manual_backward(loss)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            # self.clip_gradients(opts, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opts.step()
            opts.zero_grad()
            
        
    
    def validation_step(self, batch, batch_idx):
        loss, predictions = self._calculate_multirollout_loss(batch, batch_idx, self.n_rollouts + 3)
        mask = batch.get('mask')
        output = batch.target.y
        predicted_stes = predictions.size(1)
            
        self.val_metrics.update(predictions, output[:, :predicted_stes], mask)
        if batch_idx % 10 == 0 or batch_idx == 0:
            self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
            self.log_loss('val', loss, batch_size=batch.batch_size)
            
        return loss
    
    # for convenient checkpoint loading predictor is separate
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
        edge_weight = batch.input.edge_weight
        exog = batch.exog
        output = batch.target.y
        
        prediction = self(x, edge_index=edge_index, edge_weight=edge_weight, exog=exog)
        run_n_rollouts = min(run_n_rollouts, self.sequence_horizon // self.prediction_horizon)
        
        loss = self.loss_fn(prediction, output[:, :self.prediction_horizon]) / run_n_rollouts
        predictions = prediction.detach()
        next_exog = exog
        for rollout in range(1, run_n_rollouts):
            input_start_index = rollout * self.prediction_horizon
            output_start_index = rollout * self.prediction_horizon
            output_end_index = output_start_index + self.prediction_horizon
            
            next_exog = self.advance_days_by_n_steps(next_exog[:, input_start_index:], input_start_index)
            next_input = torch.cat((x[:,input_start_index:], predictions), dim=1)
            prediction = self(next_input, edge_index=edge_index, edge_weight=edge_weight, exog=next_exog)
            
            loss += (self.loss_fn(prediction, output[:, output_start_index:output_end_index]) / run_n_rollouts)
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
    

            