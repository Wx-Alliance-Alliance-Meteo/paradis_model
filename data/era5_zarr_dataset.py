"""Direct Zarr reads for training and benchmarking, without Dask arrays/graphs.

Xarray retains coordinate/CF decoding and the existing feature calculations.
All disk reads use Zarr orthogonal indexing; only the requested sample's fields
are materialized, once per variable, before assembling pressure-level channels.
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing
import zarr

from data.era5_dataset import (
    ERA5Dataset,
    _is_zarr_store,
    _split_feature_name,
    _stack_weatherbench_features,
)


class _Store:
    """Share metadata/array handles within a worker, reopen after fork or spawn."""

    def __init__(self, path):
        self.path = str(path)
        self._pid = None
        self._group = None
        self._arrays = {}

    def group(self):
        if self._pid != os.getpid():
            if (Path(self.path) / ".zmetadata").exists():
                self._group = zarr.open_consolidated(self.path, mode="r")
            else:
                self._group = zarr.open_group(self.path, mode="r")
            self._pid = os.getpid()
            self._arrays = {}
        return self._group

    def array(self, name):
        group = self.group()
        if name not in self._arrays:
            self._arrays[name] = group[name]
        return self._arrays[name]

    def __getstate__(self):
        return dict(path=self.path, _pid=None, _group=None, _arrays={})


class _ZarrArray(BackendArray):
    def __init__(self, store, name, array):
        self.store = store
        self.name = name
        self.shape = array.shape
        self.dtype = array.dtype

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._read
        )

    def _read(self, key):
        # Zarr does not support negative slice steps. Coordinate sorting may
        # introduce them, so express those axes as orthogonal integer indices.
        key = tuple(
            (
                np.arange(size)[item]
                if isinstance(item, slice) and item.step is not None and item.step < 0
                else item
            )
            for size, item in zip(self.shape, key)
        )
        return self.store.array(self.name).oindex[key]


def open_zarr_direct(path):
    """Read a local, Xarray-compatible Zarr v2 group without its Xarray engine.

    Preserve CF time, fill values, scale/offset, and named dimensions. Coordinate
    vectors are eager; dynamic fields stay lazy until the sample is selected.
    """
    path = Path(path).expanduser()
    if not (path / ".zgroup").exists():
        raise ValueError(f"Direct backend requires a local Zarr v2 group: {path}")
    store = _Store(path)
    group = store.group()
    variables = {}
    for name, array in group.arrays():
        attrs = dict(array.attrs)
        dims = attrs.pop("_ARRAY_DIMENSIONS", None)
        if dims is None:
            raise ValueError(f"{path}/{name} is missing _ARRAY_DIMENSIONS")
        if array.fill_value is not None:
            attrs.setdefault("_FillValue", array.fill_value)
        data = indexing.LazilyIndexedArray(_ZarrArray(store, name, array))
        variables[name] = xr.Variable(dims, data, attrs)
    return xr.decode_cf(xr.Dataset(variables, attrs=dict(group.attrs)))


class ERA5ZarrDataset(ERA5Dataset):
    """Same samples as ERA5Dataset, using direct reads grouped by raw variable.

    Supports one local Zarr v2 group or one group per variable with identical
    coordinates. Time-sharded stores and NetCDF are intentionally unsupported.
    """

    def _open_dynamic_dataset(self, root_dir, cfg):
        path = Path(str(cfg.dataset.get("weatherbench_path", root_dir))).expanduser()
        if _is_zarr_store(path):
            return open_zarr_direct(path)
        paths = sorted(p for p in path.rglob("*.zarr") if _is_zarr_store(p))
        if not paths:
            raise FileNotFoundError(f"No local Zarr stores found at {path}")
        datasets = [open_zarr_direct(p) for p in paths]
        seen = set()
        for ds in datasets:
            overlap = seen.intersection(ds.data_vars)
            if overlap:
                raise ValueError(
                    "Direct backend requires one store per variable; repeated "
                    f"variables (possibly time shards): {sorted(overlap)}"
                )
            seen.update(ds.data_vars)
        return xr.merge(datasets, join="exact", compat="equals")

    def _open_auxiliary_dataset(self, path):
        return open_zarr_direct(path)

    def _stack_features(self, ds, features, allow_missing):
        # Determine raw dependencies without fetching data. Keeping only these
        # fields avoids reading unrelated variables in the WeatherBench store.
        required = {}

        def need(candidates, level):
            name = next((name for name in candidates if name in ds.data_vars), None)
            if name is not None:
                required.setdefault(name, set()).add(level)

        for feature in features:
            variable, level = _split_feature_name(feature)
            if variable in ds.data_vars:
                need((variable,), level)
            elif variable in {"wind_x", "wind_y", "wind_z"}:
                need(("u_component_of_wind", "u"), level)
                need(("v_component_of_wind", "v"), level)
                if "vertical_velocity" in ds or "w" in ds:
                    need(("vertical_velocity", "w"), level)
                    need(("temperature", "t"), level)
            elif variable in {"wind_x_10m", "wind_y_10m", "wind_z_10m"}:
                need(("10m_u_component_of_wind", "u10", "10m_u_wind"), None)
                need(("10m_v_component_of_wind", "v10", "10m_v_wind"), None)
            elif variable == "mslp":
                need(("mean_sea_level_pressure", "msl"), level)

        # Each variable is loaded once for this input/target window. The shared
        # feature assembler below now indexes NumPy buffers, so pressure-level
        # chunks and wind dependencies are not decompressed repeatedly.
        fields = {}
        for name, levels in required.items():
            field = ds[name]
            if "level" in field.dims and None not in levels:
                positions = np.flatnonzero(np.isin(field.level.values, list(levels)))
                field = field.isel(level=positions)
            fields[name] = field.load()
        sample = xr.Dataset(
            fields,
            coords={
                name: coord for name, coord in ds.coords.items() if name != "level"
            },
        )
        return _stack_weatherbench_features(sample, features, allow_missing)
