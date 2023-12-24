import xarray as xr
import numpy as np
import pickle
import networkx as nx
import torch
import warnings
warnings.filterwarnings('ignore')
# # chunks = {'time': 10000}
# # temperature_2m = xr.open_mfdataset('./data/all_5.625deg/2m_temperature/2m_temperature_*.nc', chunks=chunks)
# # u_wind_10m = xr.open_mfdataset('./data/all_5.625deg/10m_u_component_of_wind/10m_u_component_of_wind_*.nc', chunks=chunks)
# # v_wind_10m = xr.open_mfdataset('./data/all_5.625deg/10m_v_component_of_wind/10m_v_component_of_wind_*.nc', chunks=chunks)
# geopotential = xr.open_mfdataset('./data/all_5.625deg/geopotential/geopotential_*.nc')
# temperature = xr.open_mfdataset('./data/all_5.625deg/temperature/temperature_*.nc')
# specific_humidity = xr.open_mfdataset('./data/all_5.625deg/specific_humidity/specific_humidity_*.nc')
# u_wind = xr.open_mfdataset('./data/all_5.625deg/u_component_of_wind/u_component_of_wind_*.nc')
# v_wind = xr.open_mfdataset('./data/all_5.625deg/v_component_of_wind/v_component_of_wind_*.nc')

# # geopotential_500 = xr.open_mfdataset('./data/all_5.625deg/geopotential_500/geopotential_*.nc', chunks=chunks) #single level, vs 13 levels
# # vorticity_potential = xr.open_mfdataset('./data/all_5.625deg/potential_vorticity/potential_vorticity_*.nc')
# # vorticity = xr.open_mfdataset('./data/all_5.625deg/vorticity/vorticity_*.nc')
# # relative_humidity = xr.open_mfdataset('./data/all_5.625deg/relative_humidity/relative_humidity_*.nc')


# # temperature_850 = xr.open_mfdataset('./data/all_5.625deg/temperature_850/temperature_850hPa_*.nc') #single level, vs 13 levels
# # solar_radiation = xr.open_mfdataset('./data/all_5.625deg/toa_incident_solar_radiation/toa_incident_solar_radiation_*.nc')
# # cloud_cover = xr.open_mfdataset('./data/all_5.625deg/total_cloud_cover/total_cloud_cover_*.nc')
# # precipitation_total = xr.open_mfdataset('./data/all_5.625deg/total_precipitation/total_precipitation_*.nc')


# data = xr.merge([
#     # temperature_2m,
#     # u_wind_10m,
#     # v_wind_10m,
#     geopotential,
#     temperature,
#     specific_humidity,
#     u_wind,
#     v_wind,
#     # geopotential_500,
#     # vorticity_potential,
#     # vorticity,
#     # relative_humidity,
  
    
#     # temperature_850,
#     # solar_radiation,
#     # cloud_cover,
#     # precipitation_total,

#     ])

from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

sphere_graph = pickle.load(open('./sphere_graph.pickle', 'rb'))
sphere_graph_weighted = pickle.load(open('./sphere_graph_weighted.pickle', 'rb'))
class WeatherDataLoader():
    def __init__(self,data: xr.core.dataarray.DataArray, time: str | slice, temporal_resolution: int = 3, min = None, max = None) -> None:
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
        self.min, self.max = min, max
        graph_data = from_networkx(sphere_graph)
        weighted_graph_data = from_networkx(sphere_graph_weighted)
        self.edge_index = graph_data.edge_index
        self.edge_weight = weighted_graph_data.distance
        self.adjacency = nx.adjacency_matrix(sphere_graph_weighted).todense()
        self.adjacency_weighted =nx.adjacency_matrix(sphere_graph_weighted, weight='distance').todense()
        self.timeframe = time
        
        self.stacked = self.stack_data(data)
        self.resampled = self.temporal_resampling(self.stacked, temporal_resolution)
        self.min, self.max = self.getMinMaxValues()
        
    def getMinMaxValues(self):
        if self.min is None or self.max is None:
            self.min, self.max = self.resampled.min(dim="time").compute(), self.resampled.max(dim="time").compute()
            
        return self.min, self.max
            
        
    def get_data(self):
        if self.timeframe:
            return self.normalize(self.resampled.sel(time=self.timeframe))
    
        return self.normalize(self.resampled)
        
    def temporal_resampling(self, data: xr.core.dataarray.DataArray, temporal_resolution: int) -> xr.core.dataarray.DataArray:
        return data.resample(time=f"{temporal_resolution}h").nearest(tolerance="1h")
        
    def stack_data(self, data: xr.core.dataarray.DataArray) -> xr.core.dataarray.DataArray:
        grouped = data.stack(node=['lat', 'lon'])

        levels = [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
            1000]
        var_dict = {
            'z': levels,
            # 'pv': levels,
            # 'vo': levels,
            'q': levels,
            't': levels,
            'u': levels,
            'v': levels,
            # 'tisr': None,
            # 'tp': None, 
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
    
    def normalize(self, data: xr.core.dataarray.DataArray) -> xr.core.dataarray.DataArray:
        # min, max = data.min(dim='time'), data.max(dim='time')
        return (data - self.min) / (self.max - self.min)
        

