#!/bin/bash

set +x

. r.load.dot /fs/ssm/eccc/cmd/cmds/fstpy/bundle/20240400

. r.load.dot /fs/ssm/eccc/cmd/cmds/env/python/py310_2023.07.28_all

cd /fs/homeu2/eccc/cmd/cmda/vmk001/AI/PARADIS/paradis_model/arcad_verdict_verif/

time_step_start=0
time_step_end=94
outdir=/fs/homeu2/eccc/cmd/cmda/vmk001/data/ppp5/PARADIS/five_degree/fst_mproc/
#outdir=/fs/homeu2/eccc/cmd/cmda/vmk001/data/ppp5/PARADIS/five_degree/fst_sequential/


echo 'Executing zarrtofst_deet_mproc.py'

startdate=`date`

# The input zarr files and output directory paths and time_steps counter need to be edited inside the python script
#python zarrtofst.py > /fs/homeu2/eccc/cmd/cmda/vmk001/data/ppp5/PARADIS/five_degree/for_48hr_init/listings/zarrtofst_100_hydros.out
python zarrtofst_deet_mproc.py ${time_step_start} ${time_step_end} ${outdir} > /fs/homeu2/eccc/cmd/cmda/vmk001/data/ppp5/PARADIS/five_degree/for_48hr_init/listings/zarrtofst_mproc.out

echo -e '\nstartdate time = '$startdate
echo -e '\n  enddate time = '`date`

source /fs/ssm/eccc/cmd/cmds/env/python/py310_2023.07.28_all/py310/bin/deactivate

