import torch
from einops.layers.torch import Rearrange
from metrics.metric_utils import WeatherVariable, select_variable, _spatial_average_l2_norm
from tsl.metrics.torch.metric_base import MaskedMetric


class _WeightedRMSE(torch.nn.Module):
    def __init__(self, weights, variables: list[WeatherVariable], at=None, lat_size=32, lon_size=64) -> None:
        super(_WeightedRMSE, self).__init__()
        self.weights = weights
        self.at = at
        self.variables = [select_variable(variable) for variable in variables]
        self.restore_shape = Rearrange('b t (lat lon) f -> b t lat lon f', lat=lat_size, lon=lon_size)
        
    # def forward(self, predictions:Any, ground_truth: Any):
    def forward(self, *args, **kw_args):
        predictions = args[0]
        ground_truth = args[1]
        
        if self.at != None:
            predictions, ground_truth = predictions[:, self.at], ground_truth[:, self.at]
            
        self.weights = self.weights.to(predictions.device)
        diff = (predictions[..., self.variables] - ground_truth[...,self.variables])
        diff = self.restore_shape(diff).refine_names('batch', 'time', 'lat', 'lon', 'variables')
        return _spatial_average_l2_norm(diff, self.weights.refine_names('lat')).mean()
    
    def __call__(self, *args, **kw_args):
        return self.forward(*args, **kw_args)
        
class WeightedRMSE(MaskedMetric):
    def __init__(self,
                weights,
                variables: list[WeatherVariable],
                at=None,
                lat_size=32,
                lon_size=64,
                mask_nans=True,
                mask_inf=True,
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
                                            ),
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        **kwargs)
        