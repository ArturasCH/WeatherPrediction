import xarray as xr
import numpy as np
import pickle
import networkx as nx
import torch
import torch_sparse
import warnings
warnings.filterwarnings('ignore')


from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import contains_isolated_nodes, dense_to_sparse

sphere_graph = pickle.load(open('./sphere_graph.pickle', 'rb'))

def _assert_increasing(x: np.ndarray):
  if not (np.diff(x) > 0).all():
    raise ValueError(f"array is not increasing: {x}")

def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
  pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
  return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
  """Calculate the area overlap as a function of latitude."""
  bounds = _latitude_cell_bounds(points)
  _assert_increasing(bounds)
  upper = bounds[1:]
  lower = bounds[:-1]
  # normalized cell area: integral from lower to upper of cos(latitude)
  return np.sin(upper) - np.sin(lower)

constants = xr.open_dataset('./data/all_5.625deg/constants/constants_5.625deg.nc').stack(node=['lat', 'lon'])
zt_standartization = pickle.load(open('./zt_standardization.pkl', 'rb'))

z_slice = slice(0,13)
t_slice = slice(13,26)

class WeatherDataLoader():
    def __init__(self,
                 data: xr.core.dataarray.DataArray,
                 time: str | slice | list,
                 temporal_resolution: int = 3,
                 min = None,
                 max = None,
                 mean = None,
                 std = None,
                 normalization_range={'min': -1, 'max': 1 },
                 ) -> None:
        """_summary_

        Args:
            data (xarray handle):
                data reference for weatherbench
            lags (int, optional): 
                Input timesteps. Defaults to 112.
            target_length (int, optional): 
                Output timestep length. Defaults to 40.
            temporal_resolution (int, optional): 
                Defines what timesteps to take for both input and output. 1 - every timestep (hour), 3 - every 3rd hour. Defaults to 3.
        """
        super(WeatherDataLoader).__init__()
        self.timeframe = time
        self.temporal_resolution = temporal_resolution
        self.normalization_range = normalization_range
        self.mean, self.std = mean, std
        self.data = self.select_data_in_timerange(data, time)
        lat_lons = np.array(np.meshgrid(data.lat.values, data.lon.values)).T.reshape((-1, 2))
        self.sin_lat_lons = np.sin(lat_lons)
        self.cos_lat_lons = np.cos(lat_lons)
        graph_data = from_networkx(sphere_graph)
        self.edge_index = graph_data.edge_index
        self.edge_weight = graph_data.distance
        
        
        self.node_weights = self._calculate_weights(self.data)
        self.resampled = self.temporal_resampling(self.data, temporal_resolution)
        self.stacked = self.stack_data(self.resampled)
        print('WeatherDataLoader done')
        
        
    def select_data_in_timerange(self, data, time):
        if isinstance(time, list):
            return self._selectTimeRanges(time, data)
        else:
            return data.sel(time=time)
        
    def _calculate_weights(self, data):
        cell_weights = _cell_area_from_latitude(np.deg2rad(data.lat.data))
        cell_weights /= cell_weights.mean()
        weights = data.lat.copy(data=cell_weights)
        weights_tensor = torch.tensor(weights.data, dtype=torch.float32)
        return weights_tensor
    
    def _selectTimeRanges(self, timeframes, data):
        data_selections = []
        for timeframe in timeframes:
            data_selections.append(data.sel(time=timeframe))
            
        return xr.concat(data_selections,dim='time')
        
    def get_data(self):
        if self.timeframe:
            return self.standardize(self.stacked)
    
        return self.standardize(self.stacked)
        
    def temporal_resampling(self, data: xr.core.dataarray.Dataset, temporal_resolution: int) -> xr.core.dataarray.Dataset:
        if temporal_resolution:
            return data.resample(time=f"{temporal_resolution}h").nearest(tolerance="1h")
        return data
        
    def stack_data(self, data: xr.core.dataarray.DataArray) -> xr.core.dataarray.DataArray:
        grouped = data.stack(node=['lat', 'lon'])

        levels = [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
            1000]
        var_dict = {
            'z': levels,
            't': levels,
        }
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])

        data_list = []
        def process_levels(d, data_ref):
            _data_list = []    
            for var, levels in d.items():
                try:
                    if levels is None:
                        _data_list.append(data_ref[var].expand_dims({'level': generic_level}, 1))
                    else:    
                        _data_list.append(data_ref[var].sel(level=levels))
                except ValueError:
                    _data_list.append(data_ref[var].expand_dims({'level': generic_level}, 1))
                except KeyError:
                    _data_list.append(data_ref[var])
                    
            return _data_list
        data_list = process_levels(var_dict, grouped)
        return xr.concat(data_list, 'level').transpose('time', 'node', 'level')
        
    def get_connectivity(self):
        return self.edge_index, self.edge_weight
        
    def standardize(self, data: xr.core.dataarray.DataArray) -> xr.core.dataarray.Dataset:
        z_standard = (data.isel(level=z_slice) - zt_standartization['mean'].isel(level=z_slice).mean()) / zt_standartization['std'].isel(level=z_slice).mean()
        t_standard = (data.isel(level=t_slice) - zt_standartization['mean'].isel(level=t_slice).mean()) / zt_standartization['std'].isel(level=t_slice).mean()
        return xr.concat([z_standard,t_standard], 'level')
        
    def get_exogenous(self) -> torch.Tensor:
        d = self.resampled.assign({
            'day': self.resampled.time.dt.dayofyear,
            'lsm': constants.lsm
        })
        return xr.concat([d['day'], d['lsm']], 'level').transpose('time', 'node', 'level')