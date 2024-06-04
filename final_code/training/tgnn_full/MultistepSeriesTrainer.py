import inspect
import lightning as pl
import torch
from tsl.metrics.torch import MaskedMetric
from torchmetrics import MetricCollection

class MultistepTrainer(pl.LightningModule):
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
        super(MultistepTrainer, self).__init__()
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
        x = batch.input.x
        y = x[:, self.window:]
        predicted_steps = predictions.size(1)
            
        self.train_metrics.update(predictions, y[:, :predicted_steps], mask)
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
        
    
    def _calculate_multirollout_loss(self, batch, batch_idx = 0, run_n_rollouts=1):
        x = batch.input.x
        edge_index = batch.input.edge_index
        radiation = batch.radiation
        exog = batch.exog
        
        y = x[:, self.window:]
        x = x[:, :self.window]
        
        rad = radiation[:, :self.window]
        ex = exog[:, :self.window]
        
        
        prediction = self(x, radiation=rad, exog=ex)
        sequence_horizon = y.size(1)
        run_n_rollouts = min(run_n_rollouts, sequence_horizon // self.prediction_horizon)
        
        loss = self.loss_fn(prediction, y[:, :self.prediction_horizon]) / run_n_rollouts
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
            
            loss += (self.loss_fn(prediction, y[:, output_start_index:output_end_index]) / run_n_rollouts)
            predictions = torch.cat([predictions, prediction.detach()], dim=1)
            
        return loss, predictions
    
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
