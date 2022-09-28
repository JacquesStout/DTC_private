#! /bin/bash
pdir=~/bruker_processing_plant/
raw_dir=~/raw_pcasl_data/bruker/
ndir=${pdir}/niftis/

if [[ ! -d $pdir ]];then
	mkdir -m 775 $pdir;
fi

if [[ ! -d $ndir ]];then
	mkdir -m 775 $ndir;
fi

# Need to make sure nanconvert will work properly:
ncb_path=/Users/rja20/Applications/nanconvert-build/;
export PATH=${ncb_path}:$PATH


# Blindly convert all data we can find in our local Bruker data repository:

for data_dir in $(ls ${raw_dir});do
	cmd="/Users/rja20/Applications/nanconvert/Scripts/nanbruker -z -v -l -o ${ndir} ${raw_dir}/${data_dir}/";
	
	# We're going to assume that if the output folder exists, then it was properly processed and all relevant outputs are there.
	if [[ ! -d "${ndir}/${data_dir}" ]];then
		echo "No results detected for ${data_dir}; attempting to generate Niftis with nanconvert:";
		echo "Command:";
		echo "$cmd"
		$cmd;
	else
		echo "Result folder for ${data_dir} already detected; skipping."
	fi
done

# We will next deal with interleaving and maybe masking?