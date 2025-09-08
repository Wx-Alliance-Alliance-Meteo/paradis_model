

The PARADIS output files have to be converted to the FST format.

In `zarrtofst.py` change the `zarr_path` and `output_dir` variables to point to the input and output
directory paths respectively. Then execute the following.

`ordsoument_zarr2fst.sh -> drive_zarr2fst.sh -> zarrtofst.py`

The steps involved in generating the ARCAD scores are as follows. The ARCAD scores typically compare two experiments
called the `control` and the `experiment`.

**(1)** Generate the structure files for the `control` and the `experiment`.

`cd` to `/home/username/.verifications_pour_arcad_rc/`

Copy `ua_paradis_48.cfg` from my directory to your directory. You could change the name of the script to
any other name of the format ua_*.cfg. In this script the important variables which should be changed are,

`exp` This token will appear in the name of the structure files.

`datedebut` and `datefin` are the start and end date of the experiment.

`typeverif` and `typesortie` should be set to the maximum hour of the forecast for which the verification is desired.
This token appears in the name of the structure files. 

`arcin` should point to the location of the input FST files.

`desti` should point to the location of the output directory where the structure files will be generated.


Execute the script by,

`ssh ppp5`

`cd` to your `.verifications_pour_arcad_rc` directory. 

`. ssmuse-sh -d eccc/cmd/cmda/arcad/22.04`

`verifications_pour_arcad  ua_paradis_48`

Note that the .cfg is not present in the script name. Then the structure files are created which have the names
s_ua*.omm2020.

The script has to be executed twice (`control` and `experiment`) with appropriate changes to generate structure files
for the two experiments. The variables `exp`, `arcin` and `desti` have to be changed. 


Documentation about ARCAD,

https://goc-dx.science.gc.ca/~smon400/ARCAD_V3/page4.html