import xarray as xr
import numpy as np
import rpnpy.librmn.all as rmn
import os
import pandas as pd
import sys
import yaml
import multiprocessing
import argparse


def build_metadata(new_var_name, ni, nj, ig1234, dateo, deet, npas, ip1, ip2, ip3=0):
    """Create the metadata dict used by rmn.fstecr."""
    return {
        "nomvar": new_var_name,
        "typvar": "p",
        "ni": ni,
        "nj": nj,
        "ig1": ig1234[0],
        "ig2": ig1234[1],
        "ig3": ig1234[2],
        "ig4": ig1234[3],
        "grtyp": "L",
        "dateo": dateo,
        "deet": deet,
        "ip1": ip1,
        "ip2": ip2,
        "ip3": ip3,
    }


def get_source_var(var_name, use_base):
    """Return the variable from the appropriate dataset."""
    if use_base:
        if var_name not in ds_base:
            # If a variable isn’t in the base dataset, just skip it at step 0
            raise KeyError(f"{var_name} not found in base dataset.")
        return ds_base[var_name]
    else:
        return ds[var_name]


def write_fst_for_forecast(npas, t_val, init_time, forecast_hh, use_base, pred_idx=None):
    """
    Write one FST file for a given (init_time, forecast_hh).
    - use_base=True  -> use ds_base (no prediction_timedelta)
    - use_base=False -> use ds and prediction_deltas[pred_idx]
    """
    yyyymmddhh = init_time.strftime("%Y%m%d%H")
    hhh = f"{forecast_hh:03d}"
    fst_file_name = os.path.join(output_dir, f"{yyyymmddhh}_{hhh}.fst")

    # Remove existing file if present
    if os.path.exists(fst_file_name):
        os.remove(fst_file_name)
        print(f"Removed existing file: {fst_file_name}")

    # Open FST for writing
    try:
        file_id = rmn.fstopenall(fst_file_name, rmn.FST_RW)
        print(f"Successfully created and opened the FST file: {fst_file_name}")
    except Exception as e:
        raise rmn.FSTDError(
            f"Failed to create/open the FST file: {fst_file_name}. Error: {str(e)}"
        )

    # Encode analysis date into CMC timestamp (same for all vars)
    yyyymmdd = int(init_time.strftime("%Y%m%d"))
    hhmmsshh = int(init_time.strftime("%H%M%S") + "00")
    dateo = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmsshh)

    # Loop over variables
    for var_name in ds.data_vars:
        var_mapping = variable_mappings.get(var_name, {})
        new_var_name = var_mapping.get("new_name", var_name)
        ip1_value = var_mapping.get("ip1", 0)
        conversion_formula = var_mapping.get("conversion")

        try:
            src_var = get_source_var(var_name, use_base)
        except KeyError as e:
            # Skip variables missing from base dataset at step 0
            if use_base:
                print(f"Skipping {var_name} at step 0: {e}")
                continue
            else:
                raise

        # Extract slice depending on dims and whether we use base or forecast
        if ("time" in src_var.dims) and ("level" in src_var.dims):
            for level_val in levels:
                if use_base:
                    sel = src_var.sel(
                        time=t_val,
                        level=level_val,
                        method="nearest",
                    )
                    data_slice = np.asfortranarray(sel.values.astype(np.float32))
                else:
                    sel = src_var.sel(
                        time=t_val,
                        prediction_timedelta=prediction_deltas[pred_idx],
                        level=level_val,
                        method="nearest",
                    )
                    data_slice = np.asfortranarray(sel.values.astype(np.float32).T)

                # Apply conversion if needed
                if conversion_formula:
                    data_slice = eval(conversion_formula)

                ni, nj = data_slice.shape
                metadata = build_metadata(
                    new_var_name=new_var_name,
                    ni=ni,
                    nj=nj,
                    ig1234=ig1234,
                    dateo=dateo,
                    deet=deet,
                    npas=npas,
                    ip1=int(level_val),
                    ip2=forecast_hh,
                    ip3=0,
                )
                rmn.fstecr(file_id, data_slice, metadata)

        elif "time" in src_var.dims:
            if use_base:
                sel = src_var.sel(
                    time=t_val,
                    method="nearest",
                )
                data_slice = np.asfortranarray(sel.values.astype(np.float32))

            else:
                sel = src_var.sel(
                    time=t_val,
                    prediction_timedelta=prediction_deltas[pred_idx],
                    method="nearest",
                )
                data_slice = np.asfortranarray(sel.values.astype(np.float32).T)

            if conversion_formula:
                data_slice = eval(conversion_formula)

            ni, nj = data_slice.shape

            metadata = build_metadata(
                new_var_name=new_var_name,
                ni=ni,
                nj=nj,
                ig1234=ig1234,
                dateo=dateo,
                deet=deet,
                npas=npas,
                ip1=ip1_value,
                ip2=forecast_hh,
                ip3=0,
            )
            rmn.fstecr(file_id, data_slice, metadata)

        elif "level" in src_var.dims:
            # No time dim; same for all forecast_hh, but we still write it
            for level_val in levels:
                sel = src_var.sel(level=level_val)
                data_slice = np.asfortranarray(sel.values.astype(np.float32).T)

                if conversion_formula:
                    data_slice = eval(conversion_formula)

                ni, nj = data_slice.shape
                metadata = build_metadata(
                    new_var_name=new_var_name,
                    ni=ni,
                    nj=nj,
                    ig1234=ig1234,
                    dateo=0,  # no time info
                    deet=deet,
                    npas=npas,
                    ip1=int(level_val),
                    ip2=0,
                    ip3=0,
                )
                rmn.fstecr(file_id, data_slice, metadata)

        else:
            # No time, no level
            if use_base:
                data_slice = np.asfortranarray(src_var.values.astype(np.float32))
            else:
                data_slice = np.asfortranarray(src_var.values.astype(np.float32).T)

            if conversion_formula:
                data_slice = eval(conversion_formula)

            ni, nj = data_slice.shape
            metadata = build_metadata(
                new_var_name=new_var_name,
                ni=ni,
                nj=nj,
                ig1234=ig1234,
                dateo=0,
                deet=deet,
                npas=npas,
                ip1=ip1_value,
                ip2=0,
                ip3=0,
            )
            rmn.fstecr(file_id, data_slice, metadata)

    # Close FST file
    rmn.fstcloseall(file_id)
    print(f"Successfully closed the FST file: {fst_file_name}")


# convert all forecast files corresponding to a particular date.
def convert_one_date(t_val, init_time, forecast_hours, deet):

        output = f"t_val: {t_val}, init_time: {init_time}, forecast_hours: {forecast_hours}, deet: {deet}"
    
        # 1) Forecast step 0: copy data from base ERA5 dataset (ds_base)
        npas = 0
        write_fst_for_forecast(
            npas,
            t_val=t_val,
            init_time=init_time,
            forecast_hh=0,
            use_base=True,
            pred_idx=None,
        )

        # 2) All positive forecast steps: use model output (ds) with prediction_timedelta
        for fh_idx, forecast_hh in enumerate(forecast_hours):

            #print('\nfh_idx, forecast_hh = ',fh_idx, forecast_hh)
            
            npas = forecast_hh * (3600 // deet)
            write_fst_for_forecast(
                npas,
                t_val=t_val,
                init_time=init_time,
                forecast_hh=forecast_hh,
                use_base=False,
                pred_idx=fh_idx,
            )

        return output  

if __name__=="__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,prog='zarrtofst_deet_mproc.py',description='Converts the output from PARADIS which is in zarr format to fst format. Several forecast files exist corresponding to any given initial date.')
    parser.add_argument('time_step_start',type=int,help='Index of the first initial file to be processed.')
    parser.add_argument('time_step_end',type=int,help='Index of the last initial file to be processed.')
    parser.add_argument('outdir',type=str,help='A writable path where the converted FST files are written.')
    
    args = parser.parse_args()

    t1 = args.time_step_start
    t2 = args.time_step_end
    outdir = args.outdir
    
    print('\n\tt1, t2 = ',t1,t2)

    print('\n\toutdir = ',outdir)    

    assert t2>t1, "t2 should be higher than t1."
    
    yaml_file_path = "variable_mappings.yaml"

    # Load variable mappings from the YAML file
    try:
        with open(yaml_file_path, "r") as yaml_file:
            variable_mappings = yaml.safe_load(yaml_file).get("variable_mappings", {})
            print("Successfully loaded variable mappings.")
    except Exception as e:
        raise Exception(f"Failed to load variable mappings. Error: {str(e)}")

    # Paths
    zarr_path = "/home/saz001/ss6/paradis_results/U3/logs/lightning_logs/latcorrectionv21e5fs12/summer2022_last.zarr"
    #zarr_path =  "/fs/homeu2/eccc/cmd/cmda/vmk001/data/ppp5/PARADIS/five_degree/summer_vk_hydros_with_td0_from_base.zarr"
    base_path = (
        "/home/cap003/hall6/weatherbench_raw/weatherbench_1deg_13level_conservative/"
    )
    output_dir = (
        #"/home/cap003/hall6/paradis-logs/verification/latcorrectionv21e5fs12/summer_fst"
       outdir
    )
    os.makedirs(output_dir, exist_ok=True)

    # Datasets
    ds_base = xr.open_zarr(base_path, consolidated=True)

    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        print("Successfully loaded Zarr dataset.")
    except Exception as e:
        raise Exception(f"Failed to load Zarr dataset. Error: {str(e)}")

    # Extract dimensions
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    levels = ds.level.values if "level" in ds.coords else [0]
    time_steps = ds.time.values if "time" in ds.coords else [0]

    print('time_steps = ',time_steps)

    # Forecast hours from prediction_timedelta
    if "prediction_timedelta" in ds.coords:
        prediction_deltas = ds["prediction_timedelta"].values
        forecast_hours = [
            int(delta / np.timedelta64(1, "h")) for delta in prediction_deltas
        ]
    else:
        prediction_deltas = None
        forecast_hours = []

    deet = 450

    # Grid encoding
    try:
        ig1234 = rmn.cxgaig(
        "L",
        latitudes[0],
        longitudes[0],
        latitudes[1] - latitudes[0],
        longitudes[1] - longitudes[0],
        )
    except rmn.RMNBaseError:
        sys.stderr.write("There was a problem getting encoded grid values.\n")
        raise


    print('\nExecuting main loop now. len(time_steps[t1:t2]) = ',len(time_steps[t1:t2]))



    arg_list = [
        (
            t_val,
            pd.to_datetime(t_val),
            forecast_hours,
            deet
        )
        for t_val in time_steps[t1:t2]
    ]

    print('\n\n\n\targ_list = ',arg_list,'\n\n\n')

    # Parallelize with starmap
    with multiprocessing.Pool() as pool:
        results = pool.starmap(convert_one_date, arg_list)

    print('\n\tMake sure that each worker is converting a different file :')    
    for i, output in enumerate(results):
        print(f"Worker {i}: {output}\n{'-'*40}")    


