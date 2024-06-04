import torch
from dataclasses import dataclass

@dataclass
class WeatherVariable:
    variable: str
    level: int

variable_set_idx = {v: k+1 for k,v in dict(enumerate(['z', 'q', 't', 'u', 'v'])).items()}
variable_set_idx_zt = {v: k+1 for k,v in dict(enumerate(['z', 't'])).items()}
levels = [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
            1000]
# levels = [850]
level_idx = {v: k for k,v in dict(enumerate(levels)).items()}

def select_variable(definition: WeatherVariable):
    level_skips = variable_set_idx[definition.variable]
    level_index = level_idx[definition.level]
    
    return ((len(levels)) * (level_skips - 1)) + level_index

def select_variable_zt(definition: WeatherVariable):
    level_skips = variable_set_idx_zt[definition.variable]
    level_index = level_idx[definition.level]
    
    return ((len(levels)) * (level_skips - 1)) + level_index
    
from collections import Counter

def multi_dot(*arrays, dims = None):

    # arrays = [named_z500, weights]
    all_dims = []
    for tens in arrays:
        all_dims += [d for d in tens.names if d not in all_dims]
    all_dims

    einsum_axes = "abcdefghijklmnopqrstuvwxyz"
    dim_map = {d: einsum_axes[i] for i, d in enumerate(all_dims)}
    dim_map
    common_dims = set.intersection(*(set(arr.names) for arr in arrays))
    common_dims

    # dims = ['lat', 'lon']
    if dims is ...:
            dims = all_dims
    elif isinstance(dims, str):
        dims = (dims,)
    elif dims is None:
        # find dimensions that occur more than one times
        dim_counts: Counter = Counter()
        for arr in arrays:
            dim_counts.update(arr.dims)
        dims = tuple(d for d, c in dim_counts.items() if c > 1)

    dot_dims = set(dims)
    broadcast_dims = common_dims - dot_dims
    broadcast_dims

    input_core_dims = [
        [d for d in arr.names if d not in broadcast_dims] for arr in arrays
    ]
    output_core_dims = [
        [d for d in all_dims if d not in dot_dims and d not in broadcast_dims]
    ]
    subscripts_list = [
        "..." + "".join(dim_map[d] for d in ds) for ds in input_core_dims
    ]
    subscripts = ",".join(subscripts_list)
    subscripts += "->..." + "".join(dim_map[d] for d in output_core_dims[0])

    return torch.einsum(subscripts, *[a.rename(None) for a in arrays])

def weighted_sum(data, weights, dims=None):
    return multi_dot(data, weights, dims=dims)

def sum_of_weights(data, weights, dims=None):
    summed_weights = multi_dot((~torch.isnan(data)).int().float().to(data.device), weights, dims=dims)
    valid_weights = summed_weights != 0.0
    
    return summed_weights.where(valid_weights, 1e-15)

def weighted_mean(data, weights, dims=None):
    weighted_sum_ = weighted_sum(data, weights, dims=dims)
    sum_o_w = sum_of_weights(data, weights, dims=dims)

    return weighted_sum_ / sum_o_w

def _spatial_average_l2_norm(diff, weights):
    return torch.sqrt(weighted_mean(diff**2, weights, dims=['lat', 'lon']))

def _spatial_average_l2_norm_no_sqrt(diff, weights):
    return weighted_mean(diff**2, weights, dims=['lat', 'lon'])