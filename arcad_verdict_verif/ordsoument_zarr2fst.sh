#!/bin/bash


listingdir=/fs/homeu2/eccc/cmd/cmda/vmk001/data/ppp5/PARADIS/five_degree/for_48hr_init/listings/

ord_soumet drive_zarr2fst.sh -mach ppp5 -cpus 40 -m 50G -jn drive_zarr2fst -listing ${listingdir} -w 360

