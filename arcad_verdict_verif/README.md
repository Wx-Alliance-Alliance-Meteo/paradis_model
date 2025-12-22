
The PARADIS output files have to be first converted to the FST format.

`zarrtofst_deet_mproc.py` converts the zarr files to FST format files. I have modified the code originally written by Carlos.
I parallelized the conversion using python's `multiprocessing` library.

This code which uses 40 cpus on ppp5 with 50G memory converts 3760 files
(94 initial dates x 40) in about 5.5 minutes. The sequential version (which does not use multiprocessing) takes about 10
minutes to convert 400 files (10 initial dates x 40) using the same ord_soumet.

Change the `zarr_path` to point to the input
directory paths in the python script. In `drive_zarr2ft.sh` modify the start and end indices of the initiation files and also
the name of the output directory.
Then execute the following.

`ordsoument_zarr2fst.sh -> drive_zarr2fst.sh -> zarrtofst_deet_mproc.py`


The creation and display of ARCAD scrores is explained with the help of inference files provided by Shoyon Pandey
for the month of June, 2022. He trained two versions of PARADIS models. He modified the basic version of
PARADIS to include a hydrostatic constrain. Please get in touch with him for more informaiton. 
He provided zarr inferences files corresponding to these experiments. `nohydros` is the control and `hydros` is the experiment to be compared against the control..


The steps involved in generating the ARCAD scores are as follows. The ARCAD scores typically compare two experiments
called the `control` and the `experiment`.

**(1)** Generate the structure files for the `control` and the `experiment`.

`cd` to `/home/username/.verifications_pour_arcad_rc/`

Copy `ua_paradis_120.cfg` from my directory to your directory. You could change the name of the script to
any other name of the format ua_*.cfg. In this script the important variables which should be changed are,

`exp` This token will appear in the name of the structure files.

`datedebut` and `datefin` are the start and end date of the experiment.

`typeverif` and `typesortie` should be set to the maximum hour of the forecast for which the verification is desired.
This token will appear in the name of the structure files. 

`arcin` should point to the location of the input FST files.

`desti` should point to the location of the output directory where the structure files will be generated.

`obsin` points to the location of the observations. This has to be change according to the period of the experiment.

For example, if period is in 2020, the directory is `/fs/site5/eccc/cmd/s/sanl444/arcad/structures/observations.banco.postalt.g2/omm2020/`.

If period is in 2022, the directory is `/fs/site5/eccc/cmd/s/sanl444/cycledata/ete2022/observations.banco.postalt.g2/omm2020`.

Execute the script by,

`ssh ppp5`

`cd` to your `.verifications_pour_arcad_rc` directory. 

`. ssmuse-sh -d eccc/cmd/cmda/arcad/22.04`

`verifications_pour_arcad  ua_paradis_120`

Note that the .cfg is not present in the script name. Then the structure files are created which have the names
s_ua*.omm2020.

The script has to be executed twice (`control` and `experiment`) with appropriate changes to generate structure files
for the two experiments. The variables `exp`, `arcin` and `desti` have to be changed. 

**(2)** Generate the `m_ua` files.

cd to directory `/home/username/.moyenne_temporelle_rc/`

I used the configuration script, `ua_paradis-5degre_120_nohydros_hydros.cfg`. (You could make a copy of this and modify).
The format of the name of this script is `ua_control_experiment.cfg`. In this script, `exp` and `expcmp` variables
should point to the control and experiment tokens respectively. 
The `strucin` and `strucin_cmp` should point to the
location of the `s\*` files under the respective `arcad/` directory for the control and the experiment respectively. `datedebut`
and `datefin` should be set to the start and end date respectively. ext and extcmp should be set to the format
of the filenames without the date token. Note the format of the values - there is a space in the place of the date.
For example, `ext=s_ua _120_paradis-5degre-nohydros.omm2020`. Set `typeverif` to the lead time. The output directory of the script is controlled by desti.

This .cfg is executed by,

```
ssh -X ppp5
. ssmuse-sh -d eccc/cmd/cmda/arcad/22.04
moyenne_temporelle hist
moyenne_temporelle -+
```

Pres Ctrl-X to execute the script. This executes the last modified configuration file in the directory. The `hist` option
shows the list of all the configuration files with the latest being the last. The output are two files m_ua* which
appear at `/home/vmk001/data_maestro/ppp5/arcad/moyenne_temporelle/`. Then names of this files have tokens corresponding
to control and experiment. The first file has the token `control_experiment`  (`m_ua220601_220630_120_coloc_ua_paradis-5degre-nohydros.ua_paradis-5degre-hydros`)  and the second one has the token `experiment_control`.

(3) Generate the images.

cd to your directory `/home/username/.faire_images_arcad_rc/`

Create a script whose name should have the format  `ua_control_experiment.cfg`. You could copy my script
`ua_nohydros_hydros.cfg` and modify it. The `titre` variable will appear in the title of the webpage which will display ARCAD scores.
The `struc1` and `struc2` should be set to the path of the `m\*` files produced in step (2) above. `struc1` should be set to 
the `m\*` file with the token `control_experiment`.

The output of the execution of this
script goes to a sub-directory pointed to by `exp` under the directory pointed to by desti_images. Execute this script by,

````
. ssmuse-sh -d eccc/cmd/cmda/arcad/22.04
/home/username/scripts/faire_images_arcad -+
```

Pres Ctrl-X to execute the script. The .png images are generated and (in my case) saved in,
`/home/vmk001/data_maestro/ppp5/arcad/structures/faire_images_arcad/ua_hydros/`

In the directory `/home/username/data_maestro/ppp5/arcad/structures/faire_images_arcad/` create a soft link
to the `exp` directory containing the images. Name this soft link `ua_control_experiment`.


(4) Create the ARCAD webpage.

cd to directory `/home/username/public_html/ARCAD/.` In this directory create a softlink named `figures`
to `/home/username/data_maestro/ppp5/arcad/structures/faire_images_arcad/`.

Create a script whose name should have the format  `arcad_profils_ua_control_experiment.html`.
You may modify my script,
`/home/vmk001/public_html/ARCAD/arcad_profils_ua_paradis-5degre-nohydros_paradis-5degre-hydros.html`

In the arcad_profils_ua_control_experiment.cfg the following changes are needed,

(i)  On line 109 replace vmk001 by your username. (file="http://goc-dx.science.gc.ca/~vmk001/ARCAD/...)

(ii) On line 158 replace `paradis-5degre-nohydros` by the name of your control.

(iii) On line 169 replace  `paradis-5degre-hydros` by the name of your experiment.

(iv) Starting on line 195 there are options in a dropdown list. Comment off lead times not present in your files.

(v) Between line 254 and 272 replace the names and dates of control and experiment. Also under Information add
description pertinent to your control and experiment.

You can see a link appear at,
https://goc-dx.science.gc.ca/~username/ARCAD/
which displays the ARCAD scores.

In case, the graphs do not appear on the webpage, click on `File Name` button on the left. It pop-up box shows a 
path and filename of the image. Most probably the path does not exist. Change the name of the softlink create in step (3)
above to match that shown in the pop-up box.

In my case, the page is,

https://goc-dx.science.gc.ca/~vmk001/ARCAD/arcad_profils_ua_paradis-5degre-nohydros_paradis-5degre-hydros.html

The blue and red curves are of the control and the experiment respectively. The solid curves are standard deviations and dotted curves are
biases. Smaller the standard deviations the better. Closer the biases are to zero, the better. The red and blue boxes on the left
and right side of each panel show the confidence level of difference between the control and experiment. The left side shows the confidence
level of the biases and the right side shows that of the standard deviation. If at a particular level the confidence box is absent it means
that the difference in not significant. If the box is red it means that the experiment is better than the control at that level.

Inspecting the figures, the experiment has a better bias overall than the control.

Though the above steps are done with PARADIS 5 degree inference files, these remain the same for other resolutions.

`zarrtofst.py` could be improved (using python multi-processing, for example) to make the conversion from zarr to FST faster.

Documentation about ARCAD,

https://goc-dx.science.gc.ca/~smon400/ARCAD_V3/page4.html