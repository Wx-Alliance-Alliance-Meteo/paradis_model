#!/bin/bash


. r.load.dot /fs/ssm/eccc/cmd/cmds/fstpy/bundle/20240400

. r.load.dot /fs/ssm/eccc/cmd/cmds/env/python/py310_2023.07.28_all

cd /fs/homeu2/eccc/cmd/cmda/vmk001/AI/PARADIS/paradis_model/arcad_verdict_verif/

echo 'Executing zarrtofst.py for hydros'

# The input zarr files and output directory paths and time_steps counter need to be edited inside the python script
python zarrtofst.py > /fs/homeu2/eccc/cmd/cmda/vmk001/data/ppp5/PARADIS/five_degree/for_48hr_init/listings/zarrtofst_100_hydros.out

source /fs/ssm/eccc/cmd/cmds/env/python/py310_2023.07.28_all/py310/bin/deactivate
