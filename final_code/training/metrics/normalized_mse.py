import torch
import torch.nn as nn
from tsl.metrics.torch.metric_base import MaskedMetric
from dataclasses import dataclass

@dataclass
class WeatherVariable:
    variable: str
    level: int

variable_set_idx = {v: k+1 for k,v in dict(enumerate(['z', 'q', 't', 'u', 'v'])).items()}
variable_set_idx = {v: k+1 for k,v in dict(enumerate(['z', 't'])).items()}
levels = [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
            1000]
level_idx = {v: k for k,v in dict(enumerate(levels)).items()}

def select_variable(definition: WeatherVariable):
    level_skips = variable_set_idx[definition.variable]
    level_index = level_idx[definition.level]
    
    return ((len(levels)) * (level_skips - 1)) + level_index


class NormalizedMSELoss(nn.Module):
    def __init__(self, variables='all', feature_variance=None, lat_lons=None, at=None, denormalize=False):
        super(NormalizedMSELoss,self).__init__()
        self.feature_variance = torch.tensor(feature_variance)
        self.denormalize = denormalize
        
        if variables == 'all':
            self.variables = variables
        else:
            self.variables = [select_variable(variable) for variable in variables]
            
        self.at = at
        weights = []
        for lat, lon in lat_lons:
            weights.append(torch.cos(torch.tensor(lat) * torch.pi / 180.0))
        self.weights = torch.tensor(weights, dtype=torch.float)
        
    def forward(self, *args, **kw_args):
        predictions = args[0]
        ground_truth = args[1]
        if predictions.dim() == 3:
            predictions, ground_truth = predictions.unsqueeze(1), ground_truth.unsqueeze(1)
        
        predicted_steps = predictions.size(1)
        
        if self.at != None:
            if self.at >= predicted_steps:
                predictions, ground_truth = predictions[:, -1], ground_truth[:, predicted_steps-1]
            else:    
                predictions, ground_truth = predictions[:, self.at], ground_truth[:, self.at]
                
        self.weights = self.weights.to(predictions.device)
        self.feature_variance = self.feature_variance.to(predictions.device)
        predictions = predictions / self.feature_variance
        ground_truth = ground_truth / self.feature_variance
        if self.variables == 'all':
            diff = predictions - ground_truth[:, :predicted_steps]
            weights = self.weights
        else:
            diff = (predictions[:, :, self.variables] - ground_truth[:, :predicted_steps,self.variables])
            weights = self.weights[self.variables]
            
        out = diff ** 2
        
        out = out.mean(-1)
        out = out * weights.expand_as(out)
        # mean across time
        return out.mean()
    
    
class NormalizedMSE(MaskedMetric):
    def __init__(self,
                lat_lons = None,
                feature_variance = None,
                variables: list[WeatherVariable] = None,
                denormalize=False,
                at=None,
                mask_nans=True,
                mask_inf=True,
                **kwargs
                 ):
        
        is_differentiable: bool = True
        higher_is_better: bool = False
        full_state_update: bool = False
        super(NormalizedMSE, self).__init__(metric_fn=NormalizedMSELoss(
                                            variables=variables,
                                            feature_variance=feature_variance,
                                            lat_lons=lat_lons,
                                            at=at,
                                            denormalize=denormalize,
                                            ),
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        **kwargs)
        