import os
import glob
import datetime
import re

folder_path = '/Volumes/Data/Badea/Lab/mouse/ADRC_jacques_pipeline/mrtrix_sbatch/'
time_threshold = datetime.datetime.now() - datetime.timedelta(hours=24)

recent_slurm_files = [file for file in glob.glob(os.path.join(folder_path, '*.out'))
                     if datetime.datetime.fromtimestamp(os.path.getmtime(file)) > time_threshold]

error_subj = []
for text_file in recent_slurm_files:
    with open(text_file, 'r') as file:
        content = file.read()

        ADRC_match = re.search(r'ADRC(\d{4})', content)
        if ADRC_match:
            subj = ADRC_match.group(0)

        if 'Error' in content:
            #print(f"Failure found for subj: {subj}")
            error_subj.append(subj)
        #else:
            #print(f'Successful run for subj: {subj}')

print(error_subj)