import xarray as xr
from tsl.data.loader import StaticGraphLoader
from WeatherDataLoader import WeatherDataLoader
from SpatioTemporalDataset import SpatioTemporalDataset


class WeatherDL:
    def __init__(self, data: xr.core.dataarray.DataArray, time: str | slice, temporal_resolution: int = 3, min = None, max = None, **kwargs) -> None:
        self.data_wrapper = WeatherDataLoader(data, time=time, temporal_resolution=temporal_resolution, min=min, max=max)
        self.spatio_temporal_dataset = SpatioTemporalDataset(target=self.data_wrapper.get_data(),
                                      connectivity=self.data_wrapper.get_connectivity(),
                                    #   index=time_index,
                                      horizon=40,
                                      window=112,
                                      stride=1)
        self.data_loader = StaticGraphLoader(self.spatio_temporal_dataset, **kwargs)