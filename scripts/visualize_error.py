import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Input the paths associated with the forecasts/reference datasets in weatherbench format
paradis_path = "forecast_result.zarr"
base_path = "weatherbench_5.625deg_13level/"

# Generate plots with forecast data
ds_paradis = xr.open_dataset(paradis_path, engine="zarr")
ds_base = xr.open_mfdataset(base_path, engine="zarr")

date = "2020-01-01T00:00:00"
end_date = "2020-01-02"
forecast_ind = 0

ds_base = ds_base.sel(time=slice(date, end_date)).isel(time=forecast_ind + 1)
ds_paradis = ds_paradis.sel(time=date).isel(prediction_timedelta=forecast_ind)

# Get the latitude and longitude grid
longitude, latitude = np.meshgrid(ds_base.longitude.values, ds_base.latitude.values)

g_base = ds_base.sel(level=500)["geopotential"].values.transpose()
g_para = ds_paradis.sel(level=500)["geopotential"].values

# Get a simple pointwise error
error_paradis = g_para - g_base

# Reduce value using RMSE adn MAE
rmse_paradis = np.sqrt(np.mean(((error_paradis) / 10) ** 2))
mae_paradis = np.mean(np.abs((error_paradis) / 10))

fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
figs = [plt.figure(figsize=(8, 4)) for i in range(3)]
ax = [fig.add_subplot() for fig in figs]

vmax = np.max(g_base)
vmin = np.min(g_base)

# Plot base results
ax[0].contourf(longitude, latitude, g_base, 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin)

# Plot PARADIS results
ax[1].contourf(longitude, latitude, g_para, 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin)

# Generate a pointwise error contour
contours = ax[2].imshow(error_paradis, cmap="hot_r")
plt.colorbar(contours)

ax[0].set_title("GZ500 ERA5")
ax[1].set_title("GZ500 PARADIS")
ax[2].set_title(f"GZ500 error")

plt.tight_layout()

for i, fig in enumerate(figs):
    fig.savefig(f"gz500_{i}.png")

# Show result to screen
print("GZ500-RMSE-PARADIS", f"{rmse_paradis:.2f}m")
print("GZ500-MAE-PARADIS", f"{mae_paradis:.2f}m")