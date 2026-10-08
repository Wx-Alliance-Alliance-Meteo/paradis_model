"""WeatherBench-style ERA5 dataset handling for PARADIS."""

import json
import re
from pathlib import Path

import numpy
import torch
import xarray
from omegaconf import DictConfig

from data.forcings import time_forcings, toa_radiation


def _standardize_weatherbench_coords(ds: xarray.Dataset) -> xarray.Dataset:
    """Rename common coordinate aliases and enforce increasing lat/lon order."""
    rename = {}
    aliases = {
        "lat": "latitude",
        "lon": "longitude",
        "valid_time": "time",
    }

    for old, new in aliases.items():
        if old in ds.dims or old in ds.coords:
            if new not in ds.dims and new not in ds.coords:
                rename[old] = new

    if rename:
        ds = ds.rename(rename)

    required = {"time", "latitude", "longitude"}
    available = set(ds.dims) | set(ds.coords)
    missing = required - available
    if missing:
        raise ValueError(
            "WeatherBench dataset is missing required coordinates/dimensions: "
            f"{sorted(missing)}"
        )

    ds = ds.assign_coords(
        latitude=ds.latitude.astype("float64").round(6),
        longitude=ds.longitude.astype("float64").round(6),
    )

    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")

    if ds.longitude.values[0] > ds.longitude.values[-1]:
        ds = ds.sortby("longitude")

    return ds


def _is_zarr_store(path: Path) -> bool:
    return (
        path.is_dir()
        and (
            (path / ".zgroup").exists()
            or (path / ".zmetadata").exists()
            or (path / "zarr.json").exists()
        )
    )


def _open_weatherbench_dataset(root_dir: str, cfg: DictConfig) -> xarray.Dataset:
    """Open a WeatherBench-style Zarr store or NetCDF collection lazily."""
    configured_path = cfg.dataset.get("weatherbench_path", root_dir)
    path = Path(str(configured_path)).expanduser()

    if path.is_file():
        return xarray.open_dataset(path, chunks=None, cache=False)

    if _is_zarr_store(path):
        return xarray.open_zarr(path, chunks=None)

    if not path.is_dir():
        raise FileNotFoundError(f"WeatherBench path does not exist: {path}")

    # One Zarr store per variable is also supported.
    zarr_paths = sorted(p for p in path.rglob("*.zarr") if _is_zarr_store(p))
    if zarr_paths:
        datasets = [xarray.open_zarr(p, chunks=None) for p in zarr_paths]
        return xarray.combine_by_coords(
            datasets,
            compat="override",
            join="exact",
            coords="minimal",
            data_vars="all",
            combine_attrs="override",
        )

    # Standard WeatherBench layouts often contain NetCDF files split by
    # variable and/or year.
    nc_paths = sorted(
        [
            *path.rglob("*.nc"),
            *path.rglob("*.nc4"),
            *path.rglob("*.netcdf"),
        ]
    )
    if nc_paths:
        return xarray.open_mfdataset(
            [str(p) for p in nc_paths],
            chunks=None,
            combine="by_coords",
            join="exact",
            compat="override",
            coords="minimal",
            data_vars="all",
            combine_attrs="override",
        )

    raise FileNotFoundError(
        f"No Zarr store or NetCDF files were found under {path}"
    )


def _load_stats(root_dir: str, cfg: DictConfig) -> dict:
    """Load mean/std statistics from JSON."""
    root_path = Path(root_dir).expanduser()
    default_path = (
        root_path / "stats.json"
        if root_path.is_dir()
        else root_path.parent / "stats.json"
    )
    stats_path = Path(str(cfg.dataset.get("stats_path", default_path))).expanduser()

    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics JSON does not exist: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    if not isinstance(stats, dict):
        raise TypeError("Statistics JSON must contain a JSON object at its root")

    return stats


def _find_stats(stats: dict, feature: str, required: bool = True) -> dict | None:
    """Find statistics in a flat or sectioned JSON layout."""
    for section in ("features", "forcings"):
        section_data = stats.get(section, {})
        if isinstance(section_data, dict) and feature in section_data:
            entry = section_data[feature]
            if not isinstance(entry, dict):
                raise TypeError(f"Statistics entry for '{feature}' must be an object")
            return entry

    if feature in stats:
        entry = stats[feature]
        if not isinstance(entry, dict):
            raise TypeError(f"Statistics entry for '{feature}' must be an object")
        return entry

    if required:
        raise KeyError(
            f"No statistics found for '{feature}'. Expected it in "
            "stats['features'], stats['forcings'], or at the JSON root."
        )

    return None


def _mean_std(stats: dict, feature: str) -> tuple[float, float]:
    entry = _find_stats(stats, feature)

    try:
        mean = float(entry["mean"])
        std = float(entry["std"])
    except KeyError as exc:
        raise KeyError(
            f"Statistics for '{feature}' must contain 'mean' and 'std'"
        ) from exc

    if not numpy.isfinite(mean):
        raise ValueError(f"Non-finite mean for '{feature}': {mean}")

    if not numpy.isfinite(std) or std <= 0.0:
        raise ValueError(
            f"Standard deviation for '{feature}' must be > 0, got {std}"
        )

    return mean, std


def _split_feature_name(feature: str) -> tuple[str, int | None]:
    """Split canonical PARADIS names such as temperature_h500."""
    match = re.fullmatch(r"(.+)_h(-?\d+)", feature)
    if match is None:
        return feature, None

    return match.group(1), int(match.group(2))


def _level_position(
    da: xarray.DataArray,
    feature: str,
    requested_level: int,
) -> int:
    """Return the integer position of a pressure level without indexing data."""
    if "level" not in da.dims:
        raise ValueError(
            f"Feature '{feature}' requests pressure level {requested_level}, but "
            f"variable '{da.name}' has dimensions {da.dims} and no 'level' dimension."
        )

    level_values = numpy.asarray(da["level"].values)
    matches = numpy.flatnonzero(level_values == requested_level)
    if matches.size == 0:
        raise KeyError(
            f"Level {requested_level} for feature '{feature}' was not found in "
            f"'{da.name}'. Available values: {level_values.tolist()}"
        )
    return int(matches[0])


def _first_existing_variable_name(
    ds: xarray.Dataset,
    candidates: tuple[str, ...],
    feature: str,
) -> str:
    """Return the first available WeatherBench variable name."""
    for name in candidates:
        if name in ds.data_vars:
            return name

    raise KeyError(
        f"None of {list(candidates)} required to construct feature "
        f"'{feature}' are present in the WeatherBench dataset."
    )


def _load_weatherbench_field(
    ds: xarray.Dataset,
    variable_name: str,
    feature: str,
    level: int | None = None,
) -> numpy.ndarray:
    """Read one field as float32 [time, latitude, longitude].

    Pressure-level reads deliberately retain the level dimension while data are
    fetched from the backend.  Some Zarr/Xarray combinations fail when a scalar
    level selection is pushed down to a Zarr array whose physical chunk still
    has shape [time, 1, latitude, longitude].  Squeezing only after ``load``
    avoids that backend shape mismatch.
    """
    if variable_name not in ds.data_vars:
        raise KeyError(
            f"Variable '{variable_name}' required by feature '{feature}' is missing"
        )

    da = ds[variable_name]

    if level is not None:
        level_pos = _level_position(da, feature, level)
        # Keep level length 1 during backend I/O.  Do not use scalar .sel/.isel.
        da = da.isel(level=slice(level_pos, level_pos + 1))

    # At this point ``ds`` has already been sliced to only the sample times.
    # Materialize the small requested field before any dimension is squeezed.
    da = da.load()

    if level is not None and "level" in da.dims:
        da = da.isel(level=0, drop=True)

    squeeze_dims = [
        dim
        for dim in da.dims
        if dim not in {"time", "latitude", "longitude"} and da.sizes[dim] == 1
    ]
    if squeeze_dims:
        da = da.squeeze(squeeze_dims, drop=True)

    expected_dims = {"time", "latitude", "longitude"}
    extra_dims = set(da.dims) - expected_dims
    missing_dims = expected_dims - set(da.dims)
    if extra_dims or missing_dims:
        raise ValueError(
            f"Feature '{feature}' must resolve to dimensions "
            "('time', 'latitude', 'longitude'). "
            f"Got {da.dims}; extra={sorted(extra_dims)}, "
            f"missing={sorted(missing_dims)}."
        )

    array = numpy.asarray(
        da.transpose("time", "latitude", "longitude").data,
        dtype=numpy.float32,
    )
    return array


def _weatherbench_feature_available(ds: xarray.Dataset, feature: str) -> bool:
    """Check whether a configured PARADIS feature can be built."""
    variable, level = _split_feature_name(feature)

    def has_level(name: str) -> bool:
        if name not in ds.data_vars:
            return False
        da = ds[name]
        if level is None:
            return True
        return "level" in da.dims and bool(numpy.any(da.level.values == level))

    if variable in ds.data_vars:
        return has_level(variable)

    if variable in {"wind_x", "wind_y", "wind_z"}:
        if level is None:
            return False
        u_name = next(
            (name for name in ("u_component_of_wind", "u") if name in ds.data_vars),
            None,
        )
        v_name = next(
            (name for name in ("v_component_of_wind", "v") if name in ds.data_vars),
            None,
        )
        if u_name is None or v_name is None:
            return False
        return all(
            "level" in ds[name].dims
            and bool(numpy.any(ds[name].level.values == level))
            for name in (u_name, v_name)
        )

    if variable in {"wind_x_10m", "wind_y_10m", "wind_z_10m"}:
        has_u = any(
            name in ds.data_vars
            for name in ("10m_u_component_of_wind", "u10", "10m_u_wind")
        )
        has_v = any(
            name in ds.data_vars
            for name in ("10m_v_component_of_wind", "v10", "10m_v_wind")
        )
        return has_u and has_v

    aliases = {
        "mslp": ("mean_sea_level_pressure", "msl"),
    }
    return any(has_level(name) for name in aliases.get(variable, ()))


def _cartesian_wind_arrays(
    ds: xarray.Dataset,
    feature: str,
    level: int | None,
    ten_meter: bool,
    components: set[str],
) -> dict[str, numpy.ndarray]:
    """Compute only the requested Cartesian wind components for one level."""
    g = numpy.float32(9.80616)
    gas_constant = numpy.float32(287.05)

    if ten_meter:
        u_name = _first_existing_variable_name(
            ds,
            ("10m_u_component_of_wind", "u10", "10m_u_wind"),
            feature,
        )
        v_name = _first_existing_variable_name(
            ds,
            ("10m_v_component_of_wind", "v10", "10m_v_wind"),
            feature,
        )
        u = _load_weatherbench_field(ds, u_name, feature)
        v = _load_weatherbench_field(ds, v_name, feature)
        dr_dt: float | numpy.ndarray = 0.0
    else:
        if level is None:
            raise ValueError(
                f"Atmospheric Cartesian wind feature '{feature}' requires a pressure level."
            )

        u_name = _first_existing_variable_name(
            ds,
            ("u_component_of_wind", "u"),
            feature,
        )
        v_name = _first_existing_variable_name(
            ds,
            ("v_component_of_wind", "v"),
            feature,
        )
        u = _load_weatherbench_field(ds, u_name, feature, level)
        v = _load_weatherbench_field(ds, v_name, feature, level)

        omega_name = None
        if "vertical_velocity" in ds.data_vars:
            omega_name = "vertical_velocity"
        elif "w" in ds.data_vars:
            omega_name = "w"

        if omega_name is None:
            dr_dt = 0.0
        else:
            temperature_name = _first_existing_variable_name(
                ds,
                ("temperature", "t"),
                feature,
            )
            temperature = _load_weatherbench_field(
                ds,
                temperature_name,
                feature,
                level,
            )
            omega = _load_weatherbench_field(
                ds,
                omega_name,
                feature,
                level,
            )
            scale = gas_constant / numpy.float32(level * 100.0) / g
            dr_dt = -omega * temperature * scale

    # Float32 geometry avoids promoting every full-grid temporary to float64.
    lat = numpy.deg2rad(
        numpy.asarray(ds.latitude.values, dtype=numpy.float32)
    )[:, None]
    lon = numpy.deg2rad(
        numpy.asarray(ds.longitude.values, dtype=numpy.float32)
    )[None, :]

    sin_lat = numpy.sin(lat)
    cos_lat = numpy.cos(lat)
    sin_lon = numpy.sin(lon)
    cos_lon = numpy.cos(lon)

    results: dict[str, numpy.ndarray] = {}

    if "x" in components:
        value = (
            dr_dt * cos_lat * cos_lon
            - sin_lat * cos_lon * v
            - sin_lon * u
        )
        results["x"] = numpy.asarray(value, dtype=numpy.float32)

    if "y" in components:
        value = (
            dr_dt * cos_lat * sin_lon
            - sin_lat * sin_lon * v
            + cos_lon * u
        )
        results["y"] = numpy.asarray(value, dtype=numpy.float32)

    if "z" in components:
        value = dr_dt * sin_lat + cos_lat * v
        results["z"] = numpy.asarray(value, dtype=numpy.float32)

    return results


def _stack_weatherbench_features(
    ds: xarray.Dataset,
    features: list[str],
    allow_missing: bool,
    zero_template: xarray.DataArray | None = None,
) -> tuple[numpy.ndarray, list[str]]:
    """Build [time, latitude, longitude, features] without Xarray concat.

    The output buffer is allocated channel-first internally so that each loaded
    field is copied contiguously.  A transpose view exposes the legacy
    [time, latitude, longitude, features] order without another full copy.
    """
    del zero_template  # Retained in the signature for compatibility.

    n_time = ds.sizes["time"]
    n_lat = ds.sizes["latitude"]
    n_lon = ds.sizes["longitude"]
    n_features = len(features)

    # [features, time, latitude, longitude] gives contiguous writes per channel.
    stacked = numpy.empty(
        (n_features, n_time, n_lat, n_lon),
        dtype=numpy.float32,
    )
    missing: list[str] = []
    filled = numpy.zeros(n_features, dtype=bool)

    aliases = {
        "mslp": ("mean_sea_level_pressure", "msl"),
    }
    atmospheric_winds = {"wind_x": "x", "wind_y": "y", "wind_z": "z"}
    surface_winds = {
        "wind_x_10m": "x",
        "wind_y_10m": "y",
        "wind_z_10m": "z",
    }

    for index, feature in enumerate(features):
        if filled[index]:
            continue

        variable, level = _split_feature_name(feature)

        try:
            # Stored PARADIS-form variables take precedence over derived aliases.
            if variable in ds.data_vars:
                stacked[index] = _load_weatherbench_field(
                    ds,
                    variable,
                    feature,
                    level,
                )
                filled[index] = True
                continue

            if variable in atmospheric_winds:
                group: list[tuple[int, str]] = []
                for other_index, other_feature in enumerate(features):
                    if filled[other_index]:
                        continue
                    other_variable, other_level = _split_feature_name(other_feature)
                    if (
                        other_level == level
                        and other_variable in atmospheric_winds
                        and other_variable not in ds.data_vars
                    ):
                        group.append(
                            (other_index, atmospheric_winds[other_variable])
                        )

                requested = {component for _, component in group}
                values = _cartesian_wind_arrays(
                    ds,
                    feature=feature,
                    level=level,
                    ten_meter=False,
                    components=requested,
                )
                for other_index, component in group:
                    stacked[other_index] = values[component]
                    filled[other_index] = True
                continue

            if variable in surface_winds:
                group = []
                for other_index, other_feature in enumerate(features):
                    if filled[other_index]:
                        continue
                    other_variable, other_level = _split_feature_name(other_feature)
                    if (
                        other_level is None
                        and other_variable in surface_winds
                        and other_variable not in ds.data_vars
                    ):
                        group.append((other_index, surface_winds[other_variable]))

                requested = {component for _, component in group}
                values = _cartesian_wind_arrays(
                    ds,
                    feature=feature,
                    level=None,
                    ten_meter=True,
                    components=requested,
                )
                for other_index, component in group:
                    stacked[other_index] = values[component]
                    filled[other_index] = True
                continue

            candidates = aliases.get(variable, ())
            if not candidates:
                raise KeyError(
                    f"Variable '{variable}' required by feature '{feature}' is missing"
                )
            variable_name = _first_existing_variable_name(ds, candidates, feature)
            stacked[index] = _load_weatherbench_field(
                ds,
                variable_name,
                feature,
                level,
            )
            filled[index] = True

        except KeyError:
            if not allow_missing:
                raise
            stacked[index].fill(0.0)
            filled[index] = True
            missing.append(feature)

    if not bool(numpy.all(filled)):
        unresolved = [
            feature for feature, is_filled in zip(features, filled) if not is_filled
        ]
        raise RuntimeError(f"Failed to construct WeatherBench features: {unresolved}")

    # Legacy order expected by the rest of the loader. This is a zero-copy view.
    return stacked.transpose(1, 2, 3, 0), missing


class ERA5Dataset(torch.utils.data.Dataset):
    """Prepare WeatherBench-style ERA5 data for PARADIS.

    Dynamic variables are expected in WeatherBench-style form:

      * atmospheric: [time, level, latitude, longitude]
      * surface:     [time, latitude, longitude]

    Dynamic variables are read from the WeatherBench-style dataset. Static
    fields are read from ``root_dir/constants`` using the same Zarr format as
    the previous loader. Geometric constants are computed internally.

    The public constructor and returned sample tuples are kept compatible with
    the previous stacked-feature ERA5Dataset.
    """

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        forecast_steps: int,
        dtype: torch.dtype = torch.float32,
        preload: bool = False,
        cfg: DictConfig = DictConfig({}),
        time_interval: str = None,
        prediction_stage: bool = False,
    ) -> None:
        self.cfg = cfg
        features_cfg = cfg.features
        self.preload = preload
        self.root_dir = root_dir
        self.forecast_steps = forecast_steps
        self.dtype = dtype
        self.forcing_inputs = list(features_cfg.input.forcings)
        self.concat_input = cfg.dataset.n_time_inputs > 1
        self.n_time_inputs = cfg.dataset.n_time_inputs if self.concat_input else 1
        self.prediction_stage = prediction_stage

        # Kept for compatibility with postprocessing.
        # WeatherBench loader always uses standard z-score normalization.
        self.custom_normalization = False

        ds = _standardize_weatherbench_coords(
            _open_weatherbench_dataset(root_dir, cfg)
        )
        stats = _load_stats(root_dir, cfg)
        self.stats = stats

        if "T" not in start_date:
            start_date += "T00:00:00"

        time_resolution = int(cfg.dataset.time_resolution[:-1])

        if time_interval is None:
            time_interval_hours = time_resolution
        else:
            time_interval_hours = int(time_interval[:-1])

        if time_interval_hours % time_resolution != 0:
            raise ValueError(
                "time_interval must be an integer multiple of dataset.time_resolution"
            )

        self.interval_steps = time_interval_hours // time_resolution
        self.prediction_shift = (
            int(cfg.dataset.prediction_delta[:-1]) // time_resolution - 1
        ) * self.interval_steps

        start_date_dt = numpy.datetime64(start_date, "s")
        step = numpy.timedelta64(time_resolution, "h")
        adjusted_start_date = start_date_dt - (self.n_time_inputs - 1) * step

        if end_date is not None and "T" not in end_date:
            end_date += "T23:59:59"

        # Keep the WeatherBench backend untouched.  Store absolute integer time
        # indices instead of constructing a multi-decade Xarray slice.
        all_times = numpy.asarray(ds.time.values)
        if all_times.ndim != 1 or all_times.size == 0:
            raise ValueError("WeatherBench time coordinate must be a non-empty 1D array")
        if numpy.any(all_times[1:] < all_times[:-1]):
            raise ValueError("WeatherBench time coordinate must be monotonically increasing")

        start_index = int(numpy.searchsorted(all_times, start_date_dt, side="left"))
        if end_date is None:
            end_index = all_times.size
        else:
            end_index = int(
                numpy.searchsorted(
                    all_times,
                    numpy.datetime64(end_date, "s"),
                    side="right",
                )
            )

        self.init_time_indices = numpy.arange(
            start_index,
            end_index,
            self.interval_steps,
            dtype=numpy.int64,
        )
        self.time = all_times[self.init_time_indices]
        self.length = int(self.init_time_indices.size)

        history_start_index = int(
            numpy.searchsorted(all_times, adjusted_start_date, side="left")
        )
        if self.length > 0:
            required_history_index = int(self.init_time_indices[0]) - (self.n_time_inputs - 1)
            if required_history_index < 0 or history_start_index > required_history_index:
                raise ValueError(
                    "Dataset does not contain enough history before the requested start_date "
                    f"for n_time_inputs={self.n_time_inputs}."
                )

        self.lat = torch.from_numpy(ds.latitude.values.copy())
        self.lon = torch.from_numpy(ds.longitude.values.copy())
        self.lat_size = len(self.lat)
        self.lon_size = len(self.lon)
        self.pressure_levels = features_cfg.pressure_levels
        self.grid_size = self.lat_size * self.lon_size

        input_atmospheric = [
            variable + f"_h{level}"
            for variable in features_cfg.input.atmospheric
            for level in features_cfg.pressure_levels
        ]

        output_atmospheric = [
            variable + f"_h{level}"
            for variable in features_cfg.output.atmospheric
            for level in features_cfg.pressure_levels
        ]

        input_features = input_atmospheric + list(features_cfg.input.surface)
        output_features = output_atmospheric + list(features_cfg.output.surface)

        common_features = [feature for feature in output_features if feature in input_features]
        output_only_features = [
            feature for feature in output_features if feature not in input_features
        ]
        input_only_features = [
            feature for feature in input_features if feature not in output_features
        ]

        self.num_common_features = len(common_features)
        self.dyn_input_features = common_features + input_only_features
        self.dyn_output_features = common_features + output_only_features
        self.num_dyn_inputs_single = len(self.dyn_input_features)

        # Keep only the raw WeatherBench dataset here.  Building a stacked
        # [time, lat, lon, features] object for the full multi-decade archive
        # creates a very large Dask graph.  Feature stacking is done only after
        # slicing the few time steps required by each sample in __getitem__.
        self.ds = ds

        self.missing_output_features = [
            feature
            for feature in self.dyn_output_features
            if not _weatherbench_feature_available(ds, feature)
        ]

        missing_output_set = set(self.missing_output_features)
        self.missing_output_indices = torch.tensor(
            [
                index
                for index, feature in enumerate(self.dyn_output_features)
                if feature in missing_output_set
            ],
            dtype=torch.long,
        )

        if self.missing_output_features:
            print(
                "Warning: dataset is missing configured output features "
                f"{self.missing_output_features}. Their target channels will "
                "be filled with normalized zeros."
            )

        if self.preload:
            print(
                "Warning: preload=True is ignored for the WeatherBench loader to "
                "avoid materializing the multi-decade dataset in CPU memory."
            )

        # Keep the exposed initialization-time view without constructing a
        # feature stack.  Only coordinates/metadata are retained here.
        self.ds_loader = xarray.Dataset(
            coords={"time": self.time}
        )

        self._prepare_constants(ds)
        self._prepare_normalization(stats)

        # Preserve the old public list behavior when input history is concatenated.
        if self.concat_input:
            self.dyn_input_features *= self.n_time_inputs

        self.num_in_dyn_features = (
            len(self.dyn_input_features)
            + len(self.forcing_inputs) * self.n_time_inputs
        )
        self.num_in_static_features = self.constant_data.shape[-1]
        self.num_in_features = self.num_in_dyn_features + self.num_in_static_features
        self.num_out_features = len(self.dyn_output_features)

        self._run_dataset_checks()

        if not cfg.forecast.enable and cfg.training.reports.enable:
            report_features = list(cfg.training.reports.features)
            self.report_stats = {
                "mean": numpy.asarray(
                    [_mean_std(stats, feature)[0] for feature in report_features]
                ),
                "std": numpy.asarray(
                    [_mean_std(stats, feature)[1] for feature in report_features]
                ),
            }

        # Backward compatibility. To be removed
        self.custom_normalization = False

        self.norm_precip_in = torch.empty(0, dtype=torch.long)
        self.norm_precip_out = torch.empty(0, dtype=torch.long)
        self.norm_humidity_in = torch.empty(0, dtype=torch.long)
        self.norm_humidity_out = torch.empty(0, dtype=torch.long)

        self.norm_zscore_in = torch.arange(
            self.num_dyn_inputs_single * self.n_time_inputs,
            dtype=torch.long,
        )

        self.norm_zscore_out = torch.arange(
            len(self.dyn_output_features),
            dtype=torch.long,
        )

    def __len__(self):
        return self.length

    def _getitem_standard(self, ind: int):
        steps = self.forecast_steps
        init_index = int(self.init_time_indices[ind])

        # Dynamic history ends at initialization time.
        input_ini = init_index - (self.n_time_inputs - 1)
        input_end = init_index + 1

        # First target time.
        output_ini = init_index + 1 + self.prediction_shift
        output_end = output_ini + steps

        # Forcing history ends at each corresponding target time.
        #
        # n_time_inputs = 2:
        #
        # dynamic:  [t-6, t]
        # forcing:  [t, t+6]
        # target:         t+6
        forcing_ini = output_ini - (self.n_time_inputs - 1)

        if input_ini < 0 or output_end > self.ds.sizes["time"]:
            raise IndexError(
                f"Sample {ind} requires time indices "
                f"[{input_ini}, {output_end}), "
                f"but dataset has {self.ds.sizes['time']} times."
            )

        input_ds = self.ds.isel(
            time=slice(input_ini, input_end)
        )

        forcing_ds = self.ds.isel(
            time=slice(forcing_ini, output_end)
        )

        output_ds = self.ds.isel(
            time=slice(output_ini, output_end)
        )

        input_data, _ = _stack_weatherbench_features(
            input_ds,
            self.dyn_input_features[: self.num_dyn_inputs_single],
            allow_missing=False,
        )

        true_data, _ = _stack_weatherbench_features(
            output_ds,
            self.dyn_output_features,
            allow_missing=True,
        )

        x = torch.from_numpy(input_data)
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if self.n_time_inputs > 1:
            x = torch.cat(
                [x[j] for j in range(self.n_time_inputs)],
                dim=-1,
            ).unsqueeze(0)
        else:
            x = x[0].unsqueeze(0)

        y = torch.from_numpy(true_data)
        if y.dtype != self.dtype:
            y = y.to(self.dtype)

        self._apply_normalization(x, y)
        forcings = self._compute_forcings(forcing_ds, steps)

        x_grid = x.permute(0, 3, 1, 2)
        y_grid = y.permute(0, 3, 1, 2)

        return (
            x_grid.float(),
            y_grid.float(),
            forcings,
            self.constant_data[:1],
        )


    def _getitem_prediction(self, ind: int):
        ind0 = ind
        steps = self.forecast_steps
        init_index = int(self.init_time_indices[ind])

        input_ini = init_index - (self.n_time_inputs - 1)
        input_end = init_index + 1

        # Prediction has no target data, but we still need the
        # target-time indices to construct future forcings.
        output_ini = init_index + 1 + self.prediction_shift
        output_end = output_ini + steps

        forcing_ini = output_ini - (self.n_time_inputs - 1)

        if input_ini < 0 or output_end > self.ds.sizes["time"]:
            raise IndexError(
                f"Prediction sample {ind} requires time indices "
                f"[{input_ini}, {output_end}), "
                f"but dataset has {self.ds.sizes['time']} times."
            )

        input_ds = self.ds.isel(
            time=slice(input_ini, input_end)
        )

        forcing_ds = self.ds.isel(
            time=slice(forcing_ini, output_end)
        )

        input_data, _ = _stack_weatherbench_features(
            input_ds,
            self.dyn_input_features[: self.num_dyn_inputs_single],
            allow_missing=False,
        )

        x = torch.from_numpy(input_data)
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if self.n_time_inputs > 1:
            x = torch.cat(
                [x[j] for j in range(self.n_time_inputs)],
                dim=-1,
            ).unsqueeze(0)
        else:
            x = x[0].unsqueeze(0)

        self._apply_normalization(x, None)
        forcings = self._compute_forcings(forcing_ds, steps)

        x_grid = x.permute(0, 3, 1, 2)

        return (
            ind0,
            x_grid.float(),
            forcings,
            self.constant_data[:1],
        )

    def __getitem__(self, ind: int):
        if self.prediction_stage:
            return self._getitem_prediction(ind)

        return self._getitem_standard(ind)

    def _prepare_constants(self, ds: xarray.Dataset) -> None:
        """Load legacy constants Zarr and compute geometric constant channels."""
        # constants_path = Path(self.root_dir).expanduser() / "constants"
        # if not constants_path.exists():
        #     raise FileNotFoundError(
        #         f"Constants Zarr store does not exist: {constants_path}"
        #     )

        constants_path = Path(
            "/home/cap003/site7/datasets/era5_0.25deg_13level_lq/constants"
        )
        ds_constants = xarray.open_dataset(constants_path, engine="zarr").compute()

        if "latitude" in ds_constants.coords:
            ds_constants = ds_constants.assign_coords(
                latitude=ds_constants.latitude.astype("float64").round(6)
            )
        if "longitude" in ds_constants.coords:
            ds_constants = ds_constants.assign_coords(
                longitude=ds_constants.longitude.astype("float64").round(6)
            )

        if ds_constants.latitude.values[0] > ds_constants.latitude.values[-1]:
            ds_constants = ds_constants.sortby("latitude")

        if ds_constants.longitude.values[0] > ds_constants.longitude.values[-1]:
            ds_constants = ds_constants.sortby("longitude")

        # Align the legacy constant grid with the WeatherBench dynamic grid.
        if (
            len(ds_constants.latitude) != self.lat_size
            or len(ds_constants.longitude) != self.lon_size
            or not numpy.allclose(ds_constants.latitude.values, ds.latitude.values)
            or not numpy.allclose(ds_constants.longitude.values, ds.longitude.values)
        ):
            ds_constants = ds_constants.sel(
                latitude=ds.latitude,
                longitude=ds.longitude,
            )

        lat_rad = torch.deg2rad(self.lat).to(self.dtype)
        lon_rad = torch.deg2rad(self.lon).to(self.dtype)
        self.lat_rad_grid, self.lon_rad_grid = torch.meshgrid(
            lat_rad,
            lon_rad,
            indexing="ij",
        )

        # These are the same file-backed constants normalized by the previous
        # loader using mean/std attributes stored in the constants Zarr.
        normalize_const_vars = {
            "geopotential_at_surface",
            "slope_of_sub_gridscale_orography",
            "standard_deviation_of_orography",
        }

        pre_constants = []
        for var in self.cfg.features.input.constants:
            if var in normalize_const_vars:
                if var not in ds_constants:
                    raise KeyError(
                        f"Configured constant '{var}' is missing from {constants_path}"
                    )
                if "mean" not in ds_constants[var].attrs or "std" not in ds_constants[var].attrs:
                    raise KeyError(
                        f"Constant '{var}' must contain 'mean' and 'std' attributes "
                        f"in {constants_path}"
                    )

                array = (
                    torch.from_numpy(ds_constants[var].data).to(self.dtype)
                    - float(ds_constants[var].attrs["mean"])
                ) / float(ds_constants[var].attrs["std"])
                pre_constants.append(array)

        if "land_sea_mask" in self.cfg.features.input.constants:
            if "land_sea_mask" not in ds_constants:
                raise KeyError(
                    f"Configured constant 'land_sea_mask' is missing from {constants_path}"
                )
            pre_constants.append(
                torch.from_numpy(ds_constants["land_sea_mask"].data).to(self.dtype)
            )

        self._compute_geometric_constants()

        post_constants = []
        for feature in [
            "lon_spacing",
            "cos_latitude",
            "cos_longitude",
            "sin_longitude",
            "latitude",
            "longitude",
        ]:
            if feature in self.cfg.features.input.constants:
                if feature == "lon_spacing":
                    post_constants.append(self.d_lon_inv)
                elif feature == "latitude":
                    post_constants.append(self.lat_rad_grid)
                elif feature == "longitude":
                    post_constants.append(self.lon_rad_grid)
                elif feature == "cos_latitude":
                    post_constants.append(torch.cos(self.lat_rad_grid))
                elif feature == "cos_longitude":
                    post_constants.append(torch.cos(self.lon_rad_grid))
                elif feature == "sin_longitude":
                    post_constants.append(torch.sin(self.lon_rad_grid))

        actual_total = len(pre_constants) + len(post_constants)
        expected_total = len(self.cfg.features.input.constants)
        if actual_total != expected_total:
            raise ValueError(
                f"Constant count mismatch: expected {expected_total} constants from "
                f"configuration, but built {actual_total} ({len(pre_constants)} file-backed, "
                f"{len(post_constants)} computed)."
            )

        self.constant_data = (
            torch.stack([*pre_constants, *post_constants])
            .permute(1, 2, 0)
            .reshape(self.lat_size, self.lon_size, -1)
            .unsqueeze(0)
            .expand(self.forecast_steps, -1, -1, -1)
        )

        self.ds_constants = ds_constants

    def _prepare_normalization(self, stats: dict) -> None:
        """Prepare standard z-score statistics for all dynamic channels."""
        input_pairs = [_mean_std(stats, feature) for feature in self.dyn_input_features]
        output_pairs = [_mean_std(stats, feature) for feature in self.dyn_output_features]

        input_mean_single = torch.tensor(
            [pair[0] for pair in input_pairs],
            dtype=self.dtype,
        )
        input_std_single = torch.tensor(
            [pair[1] for pair in input_pairs],
            dtype=self.dtype,
        )

        self.input_mean = input_mean_single.repeat(self.n_time_inputs)
        self.input_std = input_std_single.repeat(self.n_time_inputs)
        self.output_mean = torch.tensor(
            [pair[0] for pair in output_pairs],
            dtype=self.dtype,
        )
        self.output_std = torch.tensor(
            [pair[1] for pair in output_pairs],
            dtype=self.dtype,
        )

        all_features = set(self.dyn_input_features + self.dyn_output_features)
        self.var_stats = {
            feature: {
                "mean": _mean_std(stats, feature)[0],
                "std": _mean_std(stats, feature)[1],
            }
            for feature in all_features
        }

        if "toa_incident_solar_radiation" in self.forcing_inputs:
            toa_entry = _find_stats(
                stats,
                "toa_incident_solar_radiation",
                required=False,
            )
            if toa_entry is not None:
                self.toa_rad_mean, self.toa_rad_std = _mean_std(
                    stats,
                    "toa_incident_solar_radiation",
                )
            elif (
                "toa_radiation_mean" in stats
                and "toa_radiation_std" in stats
            ):
                self.toa_rad_mean = float(stats["toa_radiation_mean"])
                self.toa_rad_std = float(stats["toa_radiation_std"])
                if not numpy.isfinite(self.toa_rad_std) or self.toa_rad_std <= 0.0:
                    raise ValueError(
                        "toa_radiation_std in statistics JSON must be positive"
                    )
            else:
                raise KeyError(
                    "Statistics JSON must contain mean/std for "
                    "'toa_incident_solar_radiation' (for example under "
                    "'forcings') or top-level toa_radiation_mean/toa_radiation_std."
                )
        else:
            self.toa_rad_mean = 0.0
            self.toa_rad_std = 1.0

        # Keep tendency statistics in the same legacy Zarr store as before.
        tendency_stats_path = Path(self.root_dir).expanduser() / "tendency_stats_6h"
        if tendency_stats_path.exists():
            ds_tendency = xarray.open_dataset(tendency_stats_path, engine="zarr")
            tendency_lookup = {
                name: float(value)
                for name, value in zip(
                    ds_tendency.features.values,
                    ds_tendency["tendency_std"].values,
                )
            }

            tendency_values = []
            missing_tendency = []
            for feature in self.dyn_output_features:
                if feature in tendency_lookup:
                    tendency_values.append(tendency_lookup[feature])
                else:
                    tendency_values.append(1.0)
                    missing_tendency.append(feature)

            if missing_tendency:
                print(
                    "Warning: missing tendency stds for output features "
                    f"{missing_tendency}; using 1.0 for those channels."
                )

            self.output_tendency_std = torch.tensor(
                tendency_values,
                dtype=self.dtype,
            )
        else:
            self.output_tendency_std = None
            print(
                f"Warning: no tendency stats found at {tendency_stats_path}. "
                "Variance-normalized loss will not be available."
            )

    def get_raw_input_data(self, times) -> numpy.ndarray:
        """
        Return unnormalized dynamic input fields at initialization times.

        Shape:
            [time, latitude, longitude, features]

        Only the current-time state is returned, not concatenated history.
        """
        times = numpy.asarray(times)

        all_times = numpy.asarray(self.ds.time.values)
        indices = numpy.searchsorted(all_times, times)

        if numpy.any(indices >= len(all_times)):
            raise KeyError("Requested initialization time is outside the dataset")

        if not numpy.all(all_times[indices] == times):
            missing = times[all_times[indices] != times]
            raise KeyError(
                f"Initialization times not found in dataset: {missing}"
            )

        ds = self.ds.isel(time=indices)

        data, _ = _stack_weatherbench_features(
            ds,
            self.dyn_input_features[: self.num_dyn_inputs_single],
            allow_missing=False,
        )

        return data

    def _apply_normalization(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor = None,
    ) -> None:
        """Apply standard z-score normalization to every dynamic channel."""
        input_data.sub_(self.input_mean).div_(self.input_std)

        if output_data is not None:
            output_data.sub_(self.output_mean).div_(self.output_std)

            if self.missing_output_indices.numel() > 0:
                output_data[..., self.missing_output_indices] = 0.0

    def _compute_forcings(
        self,
        input_data: xarray.Dataset | xarray.DataArray,
        steps: int,
    ) -> torch.Tensor | None:
        """Compute forcing parameters from the sample times."""
        forcings_time_ds = time_forcings(input_data["time"].values)

        forcings = []

        for var in self.forcing_inputs:
            if var == "toa_incident_solar_radiation":
                toa_rad = toa_radiation(
                    input_data["time"].values,
                    self.lat.cpu().numpy(),
                    self.lon.cpu().numpy(),
                )

                toa_rad = torch.tensor(
                    (toa_rad - self.toa_rad_mean) / self.toa_rad_std,
                    dtype=self.dtype,
                )
                toa_rad = toa_rad.unfold(0, self.n_time_inputs, 1)
                forcings.append(toa_rad)

            elif var in forcings_time_ds:
                var_ds = forcings_time_ds[var]
                var_forcing = torch.tensor(var_ds.data, dtype=self.dtype)
                var_forcing = (
                    var_forcing.unfold(0, self.n_time_inputs, 1)
                    .view(steps, 1, 1, self.n_time_inputs)
                    .expand(
                        steps,
                        self.lat_size,
                        self.lon_size,
                        self.n_time_inputs,
                    )
                )
                forcings.append(var_forcing)

        if forcings:
            return torch.cat(forcings, dim=-1)

        return None

    def _compute_geometric_constants(self) -> None:
        """Compute the normalized inverse zonal grid spacing."""
        earth_radius_km = 6371
        dlon = torch.diff(torch.deg2rad(self.lon))[0]

        distance_lon_inv = 1.0 / (
            2
            * torch.arcsin(
                torch.cos(self.lat_rad_grid) ** 2 * torch.sin(dlon / 2)
            )
            * earth_radius_km
        )

        distance_lon_mean = torch.mean(distance_lon_inv)
        distance_lon_std = torch.std(distance_lon_inv)
        self.d_lon_inv = (
            distance_lon_inv - distance_lon_mean
        ) / distance_lon_std

    def _run_dataset_checks(self) -> None:
        """Check assumptions required by the model."""
        has_poles = torch.any(
            torch.isclose(
                torch.abs(self.lat_rad_grid),
                torch.tensor(
                    torch.pi / 2,
                    dtype=self.lat_rad_grid.dtype,
                    device=self.lat_rad_grid.device,
                ),
            )
        )
        # assert not has_poles, "Grid with poles unsupported!"

        assert (
            self.cfg.features.input.constants[-2] == "latitude"
        ), "Latitude must be the second-to-last feature in constants!"

        assert (
            self.cfg.features.input.constants[-1] == "longitude"
        ), "Longitude must be the last feature in constants!"
