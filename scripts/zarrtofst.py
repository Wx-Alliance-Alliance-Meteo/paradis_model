import xarray as xr
import numpy as np
import rpnpy.librmn.all as rmn
import os
import pandas as pd
import sys
import yaml  # Import for reading the YAML file

# Define the path to the YAML file
yaml_file_path = "variable_mappings.yaml"

# Load variable mappings from the YAML file
try:
    with open(yaml_file_path, "r") as yaml_file:
        variable_mappings = yaml.safe_load(yaml_file).get("variable_mappings", {})
    print("Successfully loaded variable mappings.")
except Exception as e:
    raise Exception(f"Failed to load variable mappings. Error: {str(e)}")

# Define the path to the Zarr dataset
zarr_path = "/home/cap003/hall5/to_share/mohammad/forecast_result2.zarr"

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Open the Zarr dataset
try:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    print("Successfully loaded Zarr dataset.")
except Exception as e:
    raise Exception(f"Failed to load Zarr dataset. Error: {str(e)}")

# Extract dimensions
latitudes = ds.latitude.values
longitudes = ds.longitude.values
levels = (
    ds.level.values if "level" in ds.coords else [0]
)  # Handle datasets without 'level'
time_steps = ds.time.values if "time" in ds.coords else [0]

# Extract prediction_timedelta and calculate the time step interval (deet)
if "prediction_timedelta" in ds.coords:
    prediction_deltas = ds["prediction_timedelta"].values
    forecast_hours = [
        int(delta / np.timedelta64(1, "h")) for delta in prediction_deltas
    ]
    deet = int(
        prediction_deltas[0] / np.timedelta64(1, "s")
    )  # Use the first timedelta value
else:
    forecast_hours = [0]  # No forecast hours
    deet = 3600  # Default to 3600 seconds (1 hour) if not available


try:
    ig1234 = rmn.cxgaig(
        "L",
        latitudes[0],
        longitudes[0],
        latitudes[1] - latitudes[0],
        longitudes[1] - longitudes[0],
    )
except rmn.RMNBaseError:
    sys.stderr.write("There was a problem getting encoded grid values.")


# Iterate over time_steps and prediction_timedelta to create individual FST files
try:
    for t_idx, t_val in enumerate(time_steps):
        for fh_idx, forecast_hh in enumerate(forecast_hours):
            # Derive the FST file name
            init_time = pd.to_datetime(t_val)
            yyyymmddhh = init_time.strftime("%Y%m%d%H")
            hhh = f"{forecast_hh:03d}"
            # fst_file_name = f"{yyyymmddhh}_{hhh}.fst"
            fst_file_name = os.path.join(output_dir, f"{yyyymmddhh}_{hhh}.fst")

            # Remove existing FST file if present
            if os.path.exists(fst_file_name):
                os.remove(fst_file_name)
                print(f"Removed existing file: {fst_file_name}")

            # Open the FST file for writing
            try:
                file_id = rmn.fstopenall(fst_file_name, rmn.FST_RW)
                # print(f"Successfully created and opened the FST file: {fst_file_name}")
            except Exception as e:
                raise rmn.FSTDError(
                    f"Failed to create/open the FST file: {fst_file_name}. Error: {str(e)}"
                )

            # Write data to the FST file
            for var_name in ds.data_vars:
                zarr_var = ds[var_name]

                # Get variable mapping details from the YAML file
                var_mapping = variable_mappings.get(var_name, {})
                new_var_name = var_mapping.get(
                    "new_name", var_name
                )  # Default to the original name if no mapping exists
                ip1_value = var_mapping.get(
                    "ip1", 0
                )  # Default to 0 if no `ip1` value is defined
                conversion_formula = var_mapping.get(
                    "conversion"
                )  # Retrieve conversion formula

                if "time" in zarr_var.dims and "level" in zarr_var.dims:
                    for level_idx, level_val in enumerate(levels):
                        # Extract the data slice for the current time, level, and forecast hour
                        data_slice = np.asfortranarray(
                            zarr_var.sel(
                                time=t_val,
                                prediction_timedelta=prediction_deltas[fh_idx],
                                level=level_val,
                                method="nearest",
                            )
                            .values.astype(np.float32)
                            .T
                        )

                        # Apply conversion formula if it exists
                        if conversion_formula:
                            data_slice = eval(conversion_formula)

                        # Encode the date into CMC timestamp
                        yyyymmdd = int(init_time.strftime("%Y%m%d"))
                        hhmmsshh = int(
                            init_time.strftime("%H%M%S") + "00"
                        )  # Append two zeros
                        dateo = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmsshh)

                        # Define metadata for the record
                        metadata = {
                            "nomvar": new_var_name,  # Use the actual variable name
                            "typvar": "p",
                            "ni": data_slice.shape[0],  # Latitude dimension
                            "nj": data_slice.shape[1],  # Longitude dimension
                            "ig1": ig1234[0],  # Grid info
                            "ig2": ig1234[1],
                            "ig3": ig1234[2],
                            "ig4": ig1234[3],
                            "grtyp": "L",
                            "dateo": dateo,  # Corrected date-time format
                            "deet": deet,  # Time step interval (seconds)
                            "ip1": int(level_val),  # Level value in pressure or index
                            "ip2": forecast_hh,  # Forecast hour
                            "ip3": 0,  # Additional metadata
                        }

                        # Write the data slice to the FST file
                        rmn.fstecr(file_id, data_slice, metadata)
                        # print(f"Successfully wrote record for time={t_val}, forecast_hour={forecast_hh}, level={level_val}, variable={var_name}")

                elif "time" in zarr_var.dims:
                    # Handle variables with time but no levels
                    data_slice = np.asfortranarray(
                        zarr_var.sel(
                            time=t_val,
                            prediction_timedelta=prediction_deltas[fh_idx],
                            method="nearest",
                        )
                        .values.astype(np.float32)
                        .T
                    )

                    # Apply conversion formula if it exists
                    if conversion_formula:
                        data_slice = eval(conversion_formula)

                    # Encode the date into CMC timestamp
                    yyyymmdd = int(init_time.strftime("%Y%m%d"))
                    hhmmsshh = int(
                        init_time.strftime("%H%M%S") + "00"
                    )  # Append two zeros
                    dateo = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmsshh)

                    # Define metadata for the record
                    metadata = {
                        "nomvar": new_var_name,  # Use the actual variable name
                        "typvar": "p",
                        "ni": data_slice.shape[0],  # Latitude dimension
                        "nj": data_slice.shape[1],  # Longitude dimension
                        "ig1": ig1234[0],  # Grid info
                        "ig2": ig1234[1],
                        "ig3": ig1234[2],
                        "ig4": ig1234[3],
                        "grtyp": "L",
                        "dateo": dateo,  # Corrected date-time format
                        "deet": deet,  # Time step interval (seconds)
                        "ip1": ip1_value,  # No level info
                        "ip2": forecast_hh,  # Forecast hour
                        "ip3": 0,  # Additional metadata
                    }

                    # Write the data slice to the FST file
                    rmn.fstecr(file_id, data_slice, metadata)
                    # print(f"Successfully wrote record for time={t_val}, forecast_hour={forecast_hh}, variable={var_name}")

                elif "level" in zarr_var.dims:
                    # Handle variables with levels but no time
                    for level_idx, level_val in enumerate(levels):
                        data_slice = np.asfortranarray(
                            zarr_var.sel(level=level_val).values.astype(np.float32).T
                        )

                        # Apply conversion formula if it exists
                        if conversion_formula:
                            data_slice = eval(conversion_formula)

                        # Define metadata for the record
                        metadata = {
                            "nomvar": new_var_name,  # Use the actual variable name
                            "typvar": "p",
                            "ni": data_slice.shape[0],  # Latitude dimension
                            "nj": data_slice.shape[1],  # Longitude dimension
                            "ig1": ig1234[0],  # Grid info
                            "ig2": ig1234[1],
                            "ig3": ig1234[2],
                            "ig4": ig1234[3],
                            "grtyp": "L",
                            "dateo": 0,  # No time info
                            "deet": deet,  # Time step interval (seconds)
                            "ip1": int(level_val),  # Level value in pressure or index
                            "ip2": 0,  # No time index
                            "ip3": 0,  # Additional metadata
                        }

                        # Write the data slice to the FST file
                        rmn.fstecr(file_id, data_slice, metadata)
                        # print(f"Successfully wrote record for level={level_val}, variable={var_name}")

                else:
                    # Handle variables without time and level dimensions
                    data_slice = np.asfortranarray(zarr_var.values.astype(np.float32).T)

                    # Apply conversion formula if it exists
                    if conversion_formula:
                        data_slice = eval(conversion_formula)

                    # Define metadata for the record
                    metadata = {
                        "nomvar": new_var_name,  # Use the actual variable name
                        "typvar": "p",
                        "ni": data_slice.shape[0],  # Latitude dimension
                        "nj": data_slice.shape[1],  # Longitude dimension
                        "ig1": ig1234[0],  # Grid info
                        "ig2": ig1234[1],
                        "ig3": ig1234[2],
                        "ig4": ig1234[3],
                        "grtyp": "L",
                        "dateo": 0,  # No time info
                        "deet": deet,  # Time step interval (seconds)
                        "ip1": ip1_value,  # No level info
                        "ip2": 0,  # No time index
                        "ip3": 0,  # Additional metadata
                    }

                    # Write the data slice to the FST file
                    rmn.fstecr(file_id, data_slice, metadata)
                    # print(f"Successfully wrote record for variable={var_name}")

            # Close the FST file
            rmn.fstcloseall(file_id)
            print(f"Successfully closed the FST file: {fst_file_name}")

except Exception as e:
    print(f"Failed to write data to the FST file. Error: {e}")
