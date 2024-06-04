import inspect
import lightning as pl
import torch
from tsl.metrics.torch import MaskedMetric
from torchmetrics import MetricCollection




class MultistepPredictor(pl.LightningModule):
    def __init__(self,
                 model=None,
                 loss_fn=None,
                 metrics=None,
                 prediction_horizon=1,
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
        if metrics != None:
            self._set_metrics(metrics=metrics)

    def forward(self, x, radiation, exog, edge_index):
        return self.model(x, edge_index, exog, radiation)
  
        
    def training_step(self, batch, batch_idx):
        if self.current_epoch != 0 and self.current_epoch % self.rollout_epoch_breakpoint == 0 and batch_idx == 0 and self.n_rollouts < self.max_n_rollouts:
            print(f"batch: {batch_idx}, epoch: {self.current_epoch}, n_rollouts: {self.n_rollouts}")
            self.n_rollouts += 1
            
        opts = self.optimizers()
        loss, predictions = self._calculate_multirollout_loss(batch, batch_idx)
        mask = batch.get('mask')
        predicted_steps = predictions.size(1)
        x = batch.input.x
        output = x[:, 1:predicted_steps+1]
        self.train_metrics.update(predictions, output, mask)
        if batch_idx % self.accumulate_grad_batches == 0:
            self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
            self.log_loss('train', loss, batch_size=batch.batch_size)
            
        self.manual_backward(loss)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(opts, gradient_clip_val=.5, gradient_clip_algorithm="norm")
            opts.step()
            opts.zero_grad()
            
        
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, predictions = self._calculate_multirollout_loss(batch, batch_idx)
            mask = batch.get('mask')
            predicted_steps = predictions.size(1)
            x = batch.input.x
            output = x[:, 1:predicted_steps+1]
                
            self.val_metrics.update(predictions, output, mask)
            if batch_idx % 10 == 0 or batch_idx == 0:
                self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
                self.log_loss('val', loss, batch_size=batch.batch_size)
                
            return loss
        
    
    def _calculate_multirollout_loss(self, batch, batch_idx = 0):
        x = batch.input.x
        edge_index = batch.input.edge_index
        radiation = batch.radiation
        exog = batch.exog
        
        sequence_horizon = x.size(1) - 1
        
        run_n_rollouts = sequence_horizon
        
        loss = 0
        predictions = []
        _input = x[:, 0]
        
        # exponentially decaying weighting of future rollouts
        steps = torch.arange(run_n_rollouts + 1)
        decay_factor = 0.3
        rollout_weights = (1 * (1 - decay_factor) ** steps).to(x.device)
        for rollout in range(0, run_n_rollouts):
            _output = x[:, rollout+1]
            
            rad = radiation[:, rollout]
            ex = exog[:, rollout]
            
            y = self(_input, rad, ex, edge_index)
            predictions.append(y)
            # loss += self.loss_fn(y, _output) / run_n_rollouts
            loss += self.loss_fn(y, _output) * rollout_weights[rollout+1]
            _input = y.detach()
                
        
            
        return loss, torch.stack(predictions, dim=1)
    
    
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
    

            