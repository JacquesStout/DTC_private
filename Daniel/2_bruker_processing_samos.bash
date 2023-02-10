#! /bin/bash
raw_dir=/Volumes/documents/paros_DB/BRUKER/data/
#ndir=${pdir}/niftis/
ndir=/Volumes/documents/paros_DB/BRUKER/niftis/

if [[ ! -d $ndir ]];then
	mkdir -m 775 $ndir;
fi

# Need to make sure nanconvert will work properly:
ncb_path=/Volumes/Data/Badea/Lab/jacques/Downloads/nanconvert-macos/;
export PATH=${ncb_path}:$PATH


# Blindly convert all data we can find in our local Bruker data repository:

date=20221214

#;do
for data_dir in $(ls ${raw_dir});do
    if [[ "$data_dir" == *"$date"* ]]; then

        cmd="/Users/jas/bass/gitfolder/nanconvert/Scripts/nanbruker -z -v -l -o ${ndir} ${raw_dir}/${data_dir}/";

        # We're going to assume that if the output folder exists, then it was properly processed and all relevant outputs are there.
        if [[ ! -d "${ndir}/${data_dir}" ]];then
            echo "No results detected for ${data_dir}; attempting to generate Niftis with nanconvert:";
            echo "Command:";
            echo "$cmd"
            $cmd;
        else
            echo "Result folder for ${data_dir} already detected; skipping."
        fi
    fi
done

# We will next deal with interleaving and maybe masking?