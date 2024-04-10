# Chiari sub-section

Important packages:


os
socket
numpy
nibabel
pandas
subprocess
sys
shutil
glob
re
fnmatch
dipy
scipy
itertools
json
collections
seaborn
networkx
statsmodels

DTC(found at https://github.com/JacquesStout/DTC_private/tree/main/DTC)

create_dataset_description.py
creates a basic description file for the fmri prep BIDS setup

Chiari_to_BIDS.py
a basic reorganizer to extract the anatomical and functional information and begin formatting it to BIDS. Might require specific manual checks using:
https://bids-standard.github.io/bids-validator/

launch fmriprep:
fmriprep was launched using the munin cluster and singularity specifically. The command used was:
singularity run --cleanenv /mnt/munin2/Badea/Lab/human/fmriprep.simg /mnt/munin2/Jasien/ADSB.01/Analysis/fmriprepped/Chiari /mnt/munin2/Jasien/ADSB.01/Analysis/fmri_output/ participant --participant-label 01277 01402 01501 01516 01541 04086 04129 04300 04472 04602 -w /mnt/munin2/Jasien/ADSB.01/Analysis/work_dir --fs-license-file /mnt/munin2/Badea/Lab/human/license.txt --output-spaces T1w --nthreads 20

/mnt/munin2/Badea/Lab/human/fmriprep.simg => clean fmriprep installation
/mnt/munin2/Jasien/ADSB.01/Analysis/fmriprepped/Chiari => the input in BIDS format
/mnt/munin2/Jasien/ADSB.01/Analysis/fmri_output/ => the output folder
--participant-label => The subject labels to go through
-w  /mnt/munin2/Jasien/ADSB.01/Analysis/work_dir => the work directory (useful when needing to rerun the process with different parameters)
--fs-license-file /mnt/munin2/Badea/Lab/human/license.txt => the address of the license file
--output-spaces T1w => specify this to output the functional results in the same space as the anatomical images, so that labels can be applied to them for connectomes
--nthreads 20 => the number of threads used


Chiari_diff_preproc.py
This file is used to preprocess the diffusion files associated with the BIDS project and outputs the tracts at: 
Analysis/mrtrix_pipeline/perm_files

Chiari_tract_connectomes.py
This file uses the created tracts and other outputs from 'Chiari_diff_preproc.py' and generates the connectomes at:
Analysis/connectomes/tract_conn

Chiari_fmri_connectome.py
Goes through the results of the fmri_prep pipeline, as well as the labels created via SAMBA pipeline, and creates the connectomes for fmri data at:
Analysis/connectomes/func_conn

SAMBA_stats_ANOVAS.py
Uses the stat files created via SAMBA for ANOVA comparisons for genotype groups (created Figure 1)

display_connectomes.py
Uses the files created by Chiari_tract_connectomes.py and Chiari_fmri_connectome.py to generate the connectome and average connectome figures (created Figure 2)

spearman_cog_regcompare.py
does a spearman correlation between the main (averaged) cognitive results and the SAMBA calculated metrics such as volume and FA (created Figure 3)


