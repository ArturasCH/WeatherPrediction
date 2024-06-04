import xarray as xr
import numpy as np
import pickle
import networkx as nx
import warnings
import dask

dask.config.set(scheduler='synchronous')
warnings.filterwarnings('ignore')
# chunks = {'time': 10000}
# temperature_2m = xr.open_mfdataset('./data/all_5.625deg/2m_temperature/2m_temperature_*.nc', chunks=chunks)
# u_wind_10m = xr.open_mfdataset('./data/all_5.625deg/10m_u_component_of_wind/10m_u_component_of_wind_*.nc', chunks=chunks)
# v_wind_10m = xr.open_mfdataset('./data/all_5.625deg/10m_v_component_of_wind/10m_v_component_of_wind_*.nc', chunks=chunks)


def get_data():
    # geopotential = xr.open_mfdataset('./data/all_5.625deg/geopotential/geopotential_*.nc', engine="h5netcdf")
    # temperature = xr.open_mfdataset('./data/all_5.625deg/temperature/temperature_*.nc', engine="h5netcdf")
    # specific_humidity = xr.open_mfdataset('./data/all_5.625deg/specific_humidity/specific_humidity_*.nc', engine="h5netcdf")
    # u_wind = xr.open_mfdataset('./data/all_5.625deg/u_component_of_wind/u_component_of_wind_*.nc', engine="h5netcdf")
    # v_wind = xr.open_mfdataset('./data/all_5.625deg/v_component_of_wind/v_component_of_wind_*.nc', engine="h5netcdf")

# geopotential_500 = xr.open_mfdataset('./data/all_5.625deg/geopotential_500/geopotential_*.nc', chunks=chunks) #single level, vs 13 levels
# vorticity_potential = xr.open_mfdataset('./data/all_5.625deg/potential_vorticity/potential_vorticity_*.nc')
# vorticity = xr.open_mfdataset('./data/all_5.625deg/vorticity/vorticity_*.nc')
# relative_humidity = xr.open_mfdataset('./data/all_5.625deg/relative_humidity/relative_humidity_*.nc')


# temperature_850 = xr.open_mfdataset('./data/all_5.625deg/temperature_850/temperature_850hPa_*.nc') #single level, vs 13 levels
# solar_radiation = xr.open_mfdataset('./data/all_5.625deg/toa_incident_solar_radiation/toa_incident_solar_radiation_*.nc')
# cloud_cover = xr.open_mfdataset('./data/all_5.625deg/total_cloud_cover/total_cloud_cover_*.nc')
# precipitation_total = xr.open_mfdataset('./data/all_5.625deg/total_precipitation/total_precipitation_*.nc')


    # data = xr.merge([
    #     # temperature_2m,
    #     # u_wind_10m,
    #     # v_wind_10m,
        # geopotential,
        # temperature,
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

        # ])
    data = xr.open_dataset('./data/resampled/3h_backfilled_resampling.nc', engine="h5netcdf", chunks={'time': 112})
    return data