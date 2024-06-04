import torch
import pickle
import numpy as np
from einops.layers.torch import Rearrange
from training.metrics.metric_utils import WeatherVariable, select_variable, select_variable_zt, _spatial_average_l2_norm, _spatial_average_l2_norm_no_sqrt
from tsl.metrics.torch.metric_base import MaskedMetric
from typing import Union
from typing_extensions import Literal

zt_standartization = pickle.load(open('./data/data_files/zt_standardization.pkl', 'rb'))

z_slice = slice(0,13)
t_slice = slice(13,26)

z_mean = torch.tensor([torch.tensor(zt_standartization['mean'].isel(level=z_slice).mean().data)]).expand(13)
t_mean = torch.tensor([torch.tensor(zt_standartization['mean'].isel(level=t_slice).mean().data)]).expand(13)
mean = torch.cat([z_mean, t_mean])

z_std = torch.tensor([torch.tensor(zt_standartization['std'].isel(level=z_slice).mean().data)]).expand(13)
t_std = torch.tensor([torch.tensor(zt_standartization['std'].isel(level=t_slice).mean().data)]).expand(13)
std = torch.cat([z_std, t_std])

z_start = 0
z_end = 12
t_start = 26
t_end = 38
zt_indices = np.arange(t_start,t_end + 1)
class _WeightedRMSE(torch.nn.Module):
    def __init__(self,
                 weights,
                 variables: Union[list[WeatherVariable], Literal['all']],
                 at=None,
                 denormalize=False,
                 no_sqrt = False,
                 lat_size=32,
                 lon_size=64,
                 ) -> None:
        super(_WeightedRMSE, self).__init__()
        self.weights = weights
        self.at = at
        self._loss_fn = _spatial_average_l2_norm_no_sqrt if no_sqrt else _spatial_average_l2_norm
        
        if variables == 'all':
            self.variables = variables
        else:
            self.variables = [select_variable_zt(variable) for variable in variables]

        self.mean = mean
        self.std = std
        self.denormalize = denormalize
        self.restore_shape = Rearrange('b t f (lat lon) -> b t f lat lon', lat=lat_size, lon=lon_size)
        
    def forward(self, *args, **kw_args):
        predictions = args[0]
        ground_truth = args[1]
        
        
        predicted_steps = predictions.size(1)
        
        if self.denormalize:
            predictions, ground_truth = self._inverse_transform(predictions), self._inverse_transform(ground_truth)
        else:
            predictions, ground_truth = predictions, ground_truth
        
        if self.at != None:
            if self.at >= predicted_steps:
                predictions, ground_truth = predictions[:, -1], ground_truth[:, predicted_steps-1]
            else:    
                predictions, ground_truth = predictions[:, self.at], ground_truth[:, self.at]
            
        self.weights = self.weights.to(predictions.device)
        
        predictions, ground_truth = self.to_original_shape(predictions), self.to_original_shape(ground_truth)
        
        if self.variables == 'all':
            diff = predictions - ground_truth[:, :predicted_steps]
        else:
            diff = (predictions[:, :, self.variables] - ground_truth[:, :predicted_steps,self.variables])
        diff = self.restore_shape(diff).refine_names('batch', 'time', 'variables', 'lat', 'lon')
        return self._loss_fn(diff, self.weights.refine_names('lat')).mean()
    
    def _inverse_transform(self, x):
        return (x * self.std.to(x.device)) + self.mean.to(x.device)
        
    def to_original_shape(self, x):
        return torch.movedim(x, (0,1,2,3), (0,1,3,2))
    
    def __call__(self, *args, **kw_args):
        return self.forward(*args, **kw_args)
        
class WeightedRMSE(MaskedMetric):
    def __init__(self,
                weights,
                variables: list[WeatherVariable],
                denormalize=False,
                at=None,
                lat_size=32,
                lon_size=64,
                mask_nans=True,
                mask_inf=True,
                no_sqrt=False,
                **kwargs
                 ):
        
        is_differentiable: bool = True
        higher_is_better: bool = False
        full_state_update: bool = False
        super(WeightedRMSE, self).__init__(metric_fn=_WeightedRMSE(
                                            weights=weights,
                                            variables=variables,
                                            lat_size=lat_size,
                                            lon_size=lon_size,
                                            denormalize=denormalize,
                                            no_sqrt=no_sqrt,
                                            ),
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        **kwargs)
        