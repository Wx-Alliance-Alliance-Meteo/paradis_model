import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Input the paths associated with the forecasts/reference datasets in weatherbench format
paradis_path = "/home/siw001/hall6/paradis_model_os/results/forecast_result.zarr"
base_path = "/home/cap003/hall6/weatherbench_raw/weatherbench_5.625deg_13level/"

# Generate plots with forecast data
ds_paradis = xr.open_dataset(paradis_path, engine="zarr")
ds_base = xr.open_mfdataset(base_path, engine="zarr")

# Period of forecast input (forecasts start 6h after)
date = "2022-01-01"
end_date = "2022-01-31"

forecast_ind=0
ds_paradis = ds_paradis.sel(time=slice(date, end_date)).isel(
    prediction_timedelta=forecast_ind
)

# Transform PARADIS's time dimension to be that of the forecast at that time
ds_paradis["time"] = ds_paradis.time + ds_paradis.prediction_timedelta

# Extract the appropriate time instances from the base
ds_base = ds_base.sel(time=ds_paradis.time)

# Clean up xarrays to match
ds_paradis = ds_paradis.drop_vars("prediction_timedelta")
ds_paradis = ds_paradis.reindex_like(ds_base)

# Align arrays
ds_paradis, ds_base = xr.align(ds_paradis, ds_base, join="inner")

# Get the latitude and longitude grid
longitude, latitude = np.meshgrid(ds_base.longitude.values, ds_base.latitude.values)

g_base = ds_base.sel(level=500).geopotential.compute()
g_para = ds_paradis.sel(level=500).geopotential.compute()

# Get a simple pointwise error
diff = g_para - g_base

rmse_paradis = ((diff / 10) ** 2).mean(dim=["time", "latitude", "longitude"]) ** 0.5
mae_paradis = (np.abs(diff / 10)).mean(dim=["time", "latitude", "longitude"])

fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
figs = [plt.figure(figsize=(8, 4)) for i in range(3)]
ax = [fig.add_subplot() for fig in figs]

vmax = np.max(g_base)
vmin = np.min(g_base)

# Plot base results
ax[0].contourf(longitude, latitude, g_base[0].T, 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin)

# Plot PARADIS results
ax[1].contourf(longitude, latitude, g_para[0], 100, cmap="RdYlBu_r", vmax=vmax, vmin=vmin)

# Generate a pointwise error contour
contours = ax[2].imshow(diff[0], cmap="hot_r")
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
