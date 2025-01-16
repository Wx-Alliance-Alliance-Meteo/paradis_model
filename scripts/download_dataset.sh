BASE_PATH="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

OUTPUT_PATH="ERA5/5.625deg_wb3"

gsutil -m cp -r \
    "${BASE_PATH}/.zattrs" \
    "${BASE_PATH}/.zgroup" \
    "${BASE_PATH}/.zmetadata" \
    "${BASE_PATH}/10m_u_component_of_wind" \
    "${BASE_PATH}/10m_v_component_of_wind" \
    "${BASE_PATH}/2m_temperature" \
    "${BASE_PATH}/mean_sea_level_pressure" \
    "${BASE_PATH}/surface_pressure" \
    "${BASE_PATH}/temperature" \
    "${BASE_PATH}/land_sea_mask" \
    "${BASE_PATH}/time" \
    "${BASE_PATH}/u_component_of_wind" \
    "${BASE_PATH}/v_component_of_wind" \
    "${BASE_PATH}/vertical_velocity" \
    "${BASE_PATH}/level" \
    "${BASE_PATH}/specific_humidity" \
    "${BASE_PATH}/geopotential" \
    "${BASE_PATH}/latitude" \
    "${BASE_PATH}/longitude" \
    "${BASE_PATH}/geopotential_at_surface" \
    "${BASE_PATH}/total_precipitation_6hr" \
    "${BASE_PATH}/total_column_water" \
    "${BASE_PATH}/standard_deviation_of_orography" \
    "${BASE_PATH}/slope_of_sub_gridscale_orography" \
    $OUTPUT_PATH