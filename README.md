**PARADIS**: **P**hysically-informed **A**dvection, **R**eaction **A**nd **DI**ffusion on the **S**phere

# TODO

1. setup evaluatins | understand the format of the data (more)
2. optional wandb support that removes lots of the cmd line output
3. figure out what is causing the dataloading bottleneck during training time (profile)
4. based on eval see what needs work
5. model edits
    1. test different module architecutes -> ConvNext, Transformer (if too large Swin)
    2. output the direct result after vector (need to understand where it would land on sphere)
        

#### Key Features

- **Physically-informed neural architecture:** Combines convolutional neural networks with operators akin to the advection, diffusion, and reaction (ADR) equation to handle complex physical phenomena.
- **Spherical Semi-Lagrangian advection operator:** Enables the non-local transformation of information in spherical coordinates.
- **Geocyclic Padding:** Addresses the challenge of representing the Earth’s spherical geometry

#### Dependencies
Necessary python packages can be installed using **pip**:

```
pip install -r requirements.txt --break-system-packages
```

#### Usage
##### Training
For training, the `paradis_settings.yaml` file in the `config/` directory is used to manage inputs
```
python train.py [override_args]
```
where `[override_args]` can override the inputs in the config file.


#### Dataset
Download the original dataset from WeatherBench 2:

```
gsutil -m cp -r   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/.zattrs"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/.zgroup"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/.zmetadata"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/10m_u_component_of_wind"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/10m_v_component_of_wind"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/10m_wind_speed"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/2m_temperature"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/geopotential"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/geopotential_at_surface"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/land_sea_mask"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/latitude"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/level"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/longitude"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/mean_sea_level_pressure"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/specific_humidity"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/temperature"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/time"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/toa_incident_solar_radiation"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/total_precipitation_6hr"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/u_component_of_wind"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/v_component_of_wind"   "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/vertical_velocity"  /path/to/ERA5/5.625deg_wb2
```
and then preprocess it

```
python scripts/preprocess_weatherbench_data.py -i /path/to/ERA5/5.625deg_wb2 -o /path/to/ERA5/5.65deg
```

### Evaluations

*Todo*

### Wandb (support)

*Todo*

#### Acknowledgements

This project draws significant inspiration from the paper ["Advection Augmented Convolutional Neural Networks"](https://arxiv.org/abs/2406.19253) by Niloufar Zakariaei, Siddharth Rout, Eldad Haber and Moshe Eliasof.

The Geocyclic Padding method is inspired from the paper ["Karina: An efficient deep learning model for global weather forecast"](https://arxiv.org/abs/2403.10555) by
Minjong Cheon, Yo-Hwan Choi, Seon-Yu Kang, Yumi Choi, Jeong-Gil Lee and Daehyun Kang.
