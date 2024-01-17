from tsl.engines import Predictor

import inspect
from typing import Callable, Mapping, Optional, Type

import pytorch_lightning as pl
import torch
from torchmetrics import Metric, MetricCollection

from tsl import logger
from tsl.data import Data
from tsl.metrics.torch import MaskedMetric
from tsl.nn.models import BaseModel
from tsl.utils import foo_signature



class MultistepPredictor(Predictor):
    def _set_multistep_attrs(self, prediction_horizon, sequence_horizon):
        self.prediction_horizon = prediction_horizon
        self.sequence_horizon = sequence_horizon
        self.n_rollouts = self.sequence_horizon // self.prediction_horizon
        print(self.n_rollouts)
        
    def forward(self, *args, **kwargs):
        """"""
        if self.filter_forward_kwargs:
            kwargs = self._filter_forward_kwargs(kwargs)
        
        x, edge_index, edge_weight = kwargs['x'], kwargs['edge_index'], kwargs['edge_weight']
        
        predictions = [self.model(x, edge_index, edge_weight)]
        
        for rollout in range(1, self.n_rollouts):
            input_start_index = rollout * self.prediction_horizon
            output_start_index = rollout * self.prediction_horizon
            output_end_index = output_start_index + self.prediction_horizon
            
            predictions.append(self.model(torch.cat((x[:,input_start_index:, :, :], predictions[rollout-1]), dim=1), edge_index, edge_weight))
            print(f"rollout {rollout} running")
            # loss += self.loss_fn(prediction, output[:, output_start_index:output_end_index, :, :])
        
        # return self.model(*args, **kwargs)
        return torch.cat(predictions, dim=1)
        
    # def training_step(self, batch, batch_idx):
    #     x = batch.input.x
    #     edge_index = batch.input.edge_index
    #     edge_weight = batch.input.edge_weight
    #     output = batch.target.y
        
    #     # print("loss", self.loss)
    #     print("loss_fn", self.loss_fn)
        
    #     prediction = self(x, edge_index, edge_weight)
    #     print(f"shapes - prediction: {prediction.size()}, output: {output[:, :self.prediction_horizon, :, :].size()}")
    #     loss = self.loss_fn(prediction, output[:, :self.prediction_horizon, :, :])
    #     print(f"loss shape - {loss}")
    #     for rollout in range(1, self.n_rollouts):
    #         input_start_index = rollout * self.prediction_horizon
    #         output_start_index = rollout * self.prediction_horizon
    #         output_end_index = output_start_index + self.prediction_horizon
            
    #         prediction = self(torch.cat((x[:,input_start_index:, :, :], prediction[:,:, :, :]), dim=1), edge_index, edge_weight)
    #         print(f"rollout {rollout} loss accumulation")
    #         loss += self.loss_fn(prediction, output[:, output_start_index:output_end_index, :, :])
            
    #     return loss