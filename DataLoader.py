import xarray as xr
from tsl.data.loader import StaticGraphLoader
from WeatherDataLoader import WeatherDataLoader
from SpatioTemporalDataset import SpatioTemporalDataset


class WeatherDL:
    def __init__(self, data: xr.core.dataarray.DataArray, time: str | slice, temporal_resolution: int = 3, window = 112, horizon = 40, stride = 1, min = None, max = None, **kwargs) -> None:
        print('weather loader init')
        self.data_wrapper = WeatherDataLoader(data, time=time, temporal_resolution=temporal_resolution, min=min, max=max)
        print('weather loader exists')
        print('SpatioTemporalDataset init')
        self.spatio_temporal_dataset = SpatioTemporalDataset(target=self.data_wrapper.get_data(),
                                      connectivity=self.data_wrapper.get_connectivity(),
                                    #   index=time_index,
                                      horizon=horizon,
                                      window=window,
                                      stride=stride)
        print('SpatioTemporalDataset exists')
        print('StaticGraphLoader init')
        self.data_loader = StaticGraphLoader(self.spatio_temporal_dataset, **kwargs)
        print('StaticGraphLoader exists')