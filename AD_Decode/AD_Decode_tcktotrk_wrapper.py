import glob
import subprocess, re, os
import numpy as np
job_descrp = "tck_to_trk"

BD = '/mnt/munin2/Badea/Lab/mouse/'
pipefolder = os.path.join(BD,'tcktotrk_pipeline')
if not os.path.exists(pipefolder): os.mkdir(pipefolder)
sbatch_folder_path = BD +"/tcktotrk_pipeline/" +job_descrp + '_sbatch/'
if not os.path.exists(sbatch_folder_path): os.mkdir(sbatch_folder_path)

tck_files = glob.glob(os.path.join(BD,'mrtrix_ad_decode/perm_files/*.tck'))
trk_folder = '/mnt/munin2/Badea/Lab/human/AD_Decode_trk_transfer/TRK'
reference_folder = '/mnt/munin2/Badea/ADdecode.01/Analysis/DWI/'
trk_MDT_folder = '/mnt/munin2/Badea/Lab/human/AD_Decode_trk_transfer/TRK_MDT'
GD = '~/gunnies/'

overwrite=False
test_mode = False

for tck_file in tck_files:
    found=0
    subj = os.path.basename(tck_file)[:6]
    reference_file = os.path.join(reference_folder,f'{subj}_subjspace_fa.nii.gz')
    trk_file = os.path.join(trk_folder,os.path.basename(tck_file).replace('.tck','.trk'))
    trk_MDT_file = os.path.join(trk_MDT_folder,f'{subj}*')
    if np.size(glob.glob(trk_MDT_file))>0:
        found=1
    if found:
        #print(f'found {subj}')
        continue
    if not os.path.exists(trk_file) or overwrite:
        python_command = f"python3 ~/DTC_private/AD_Decode/AD_Decode_tcktotrk.py {tck_file} {trk_file} {reference_file} 2000000"
        # python_command = "python3 ~/DTC_private/AMD//AMD_subj_to_MDT_clustered.py " + subj
        job_name = job_descrp + "_" + subj
        command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        if test_mode:
            print(command)
            #command=1
        else:
            os.system(command)
