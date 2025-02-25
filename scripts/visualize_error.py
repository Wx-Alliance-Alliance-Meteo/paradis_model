import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Input the paths associated with the forecasts/reference datasets in weatherbench format
paradis_path = "/home/siw001/hall6/paradis_model_rk4/results/fe_t6hr_f6hr_2018_1_1/forecast_result.zarr"
base_path = "/home/cap003/hall6/weatherbench_raw/weatherbench_5.625deg_13level/"

# Generate plots with forecast data
ds_paradis = xr.open_dataset(paradis_path, engine="zarr")
ds_base = xr.open_mfdataset(base_path, engine="zarr")

date = "2018-01-01"
end_date = "2018-01-01"
forecast_ind = 0

ds_paradis = (
    ds_paradis.sel(time=date).isel(prediction_timedelta=forecast_ind).sel(level=1000)
)
ds_base = (
    ds_base.sel(time=slice(date, end_date)).isel(time=forecast_ind + 1).sel(level=1000)
)

# Get the latitude and longitude grid
longitude, latitude = np.meshgrid(ds_base.longitude.values, ds_base.latitude.values)

# Extract the values to plot
temp_paradis = ds_paradis["2m_temperature"].values
temp_base = ds_base["2m_temperature"].values.transpose()

geop_paradis =  ds_paradis["geopotential"].values 
geop_base = ds_base["geopotential"].values.transpose()

# Compute the error
temp_error_paradis = temp_paradis - temp_base
geop_error_paradis = geop_paradis - geop_base

print("2m_temperature_error", np.mean(temp_error_paradis**2))
print("geopotenetial_error", np.mean(geop_error_paradis**2))

fig, ax = plt.subplots(ncols=3, figsize=(18, 5))

vmax = np.max(temp_base)
vmin = np.min(temp_base)

ax[0].contourf(
    longitude, latitude, temp_base, 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin
)
ax[1].contourf(
    longitude, latitude, temp_paradis, 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin
)
contours = ax[2].contourf(longitude, latitude, temp_error_paradis, 100, cmap="hot_r")
plt.colorbar(contours)

ax[0].set_title("ERA5")
ax[1].set_title("PARADIS")
ax[2].set_title("Error")

plt.tight_layout()

fig.savefig("../testoutput-vicky/forecast_error_plot/2m_temperature_"+date+"_ind_"+ str(forecast_ind) +".png")

# plot geopotential error
fig, ax = plt.subplots(ncols=3, figsize=(18, 5))

vmax = np.max(geop_base)
vmin = np.min(geop_base)

ax[0].contourf(
    longitude, latitude, geop_base, 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin
)
ax[1].contourf(
    longitude, latitude, geop_paradis, 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin
)
contours = ax[2].contourf(longitude, latitude, geop_error_paradis, 100, cmap="hot_r")
plt.colorbar(contours)

ax[0].set_title("ERA5")
ax[1].set_title("PARADIS")
ax[2].set_title("Error")

plt.tight_layout()

fig.savefig("../testoutput-vicky/forecast_error_plot/geopotential_500hPa_"+date+"_ind_"+ str(forecast_ind) +".png")
