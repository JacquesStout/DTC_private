#! /bin/bash

from DTC.file_manager.file_tools import mkcdir
import os

"""
inpath = '/Users/${T_USER}/bruker/raw_pcasl_data'
outpath = '/Users/${T_USER}/bruker/bruker_processing_plant'
#outpath = ~ / bruker_processing_plant /
#inpath = ~ / raw_pcasl_data / bruker /
ndir =os.path.join(outpath,'niftis')
mkcdir([outpath,ndir])

# Need to make sure nanconvert will work properly:
ncb_path = '/Users/rja20/Applications/nanconvert - build /;'
export
PATH =${ncb_path}:$PATH

# Blindly convert all data we can find in our local Bruker data repository:

for data_dir in $(ls ${inpath});do
cmd = "/Users/rja20/Applications/nanconvert/Scripts/nanbruker -z -v -l -o ${ndir} ${inpath}/${data_dir}/";

# We're going to assume that if the output folder exists, then it was properly processed and all relevant outputs are there.
if [[ ! -d "${ndir}/${data_dir}"]];then
echo
"No results detected for ${data_dir}; attempting to generate Niftis with nanconvert:";
echo
"Command:";
echo
"$cmd"
$cmd;
else
echo
"Result folder for ${data_dir} already detected; skipping."
fi
done

# We will next deal with interleaving and maybe masking?
"""