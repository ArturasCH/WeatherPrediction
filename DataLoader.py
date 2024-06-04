import xarray as xr
from tsl.data.loader import StaticGraphLoader
from WeatherDataLoader import WeatherDataLoader
from WeatherDataLoader2 import WeatherDataLoader as WeatherDataLoader2
# from WeatherDataLoader_nr import WeatherDataLoader
from SpatioTemporalDataset import SpatioTemporalDataset


class  WeatherDL:
    def __init__(self,
                 data: xr.core.dataarray.DataArray,
                 time: str | slice,
                 temporal_resolution: int = 3,
                 window = 112,
                 horizon = 40,
                 stride = 1,
                 min = None,
                 max = None,
                 mean=None,
                 std=None,
                 **kwargs) -> None:
        print('weather loader init')
        self.data_wrapper = WeatherDataLoader(data,
                                              time=time,
                                              temporal_resolution=temporal_resolution, 
                                              min=min,
                                              max=max,
                                              mean=mean,
                                              std=std
                                              )
        print('weather loader exists')
        print('SpatioTemporalDataset init')
        self.spatio_temporal_dataset = SpatioTemporalDataset(target=self.data_wrapper.get_data(),
                                      connectivity=self.data_wrapper.get_connectivity(),
                                    #   index=time_index,
                                      horizon=horizon,
                                      window=window,
                                      stride=stride)
        self.spatio_temporal_dataset.add_exogenous('exog', self.data_wrapper.get_exogenous(), node_level=True, add_to_input_map=True)
        print('SpatioTemporalDataset exists')
        print('StaticGraphLoader init')
        self.data_loader = StaticGraphLoader(self.spatio_temporal_dataset, drop_last=True, **kwargs)
        print('StaticGraphLoader exists')
        
        
class  WeatherDL2:
  def __init__(self,
                data: xr.core.dataarray.DataArray,
                time: str | slice,
                temporal_resolution: int = 3,
                window = 112,
                horizon = 40,
                stride = 1,
                min = None,
                max = None,
                mean=None,
                std=None,
                **kwargs) -> None:
      print('weather loader init')
      self.data_wrapper = WeatherDataLoader2(data,
                                            time=time,
                                            temporal_resolution=temporal_resolution, 
                                            mean=mean,
                                            std=std
                                            )
      print('weather loader exists')
      print('SpatioTemporalDataset init')
      self.spatio_temporal_dataset = SpatioTemporalDataset(target=self.data_wrapper.get_data(),
                                    connectivity=self.data_wrapper.get_connectivity(),
                                  #   index=time_index,
                                    horizon=horizon,
                                    window=window,
                                    stride=stride)
      self.spatio_temporal_dataset.add_exogenous('global_exog', self.data_wrapper.get_exogenous())
      self.spatio_temporal_dataset.add_exogenous('radiation', self.data_wrapper.get_solar_radiation())
      print('SpatioTemporalDataset exists')
      print('StaticGraphLoader init')
      self.data_loader = StaticGraphLoader(self.spatio_temporal_dataset, drop_last=True, **kwargs)
      print('StaticGraphLoader exists')