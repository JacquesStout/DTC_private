import glob
import subprocess, re, os

job_descrp = "tck_to_trk"

BD = '/mnt/munin2/Badea/Lab/mouse/'
sbatch_folder_path = BD +"/tracttoMDT_ADDecode_pipeline/" +job_descrp + '_sbatch/'

tck_files = glob.glob(os.path.join(BD,'mrtrix_ad_decode/perm_files/*.tck'))
trk_folder = '/mnt/munin2/Badea/Lab/human/AD_Decode_trk_transfer/TRK'
reference_folder = '/mnt/munin2/Badea/ADdecode.01/Analysis/DWI/'

GD = '~/gunnies/'

overwrite=False
test_mode = True

for tck_file in tck_files:
    subj = os.path.basename(tck_file)[:6]
    reference_file = os.path.join(reference_folder,f'{subj}_subjspace_fa.nii.gz')
    trk_file = os.path.join(trk_folder,os.path.basename(tck_file).replace('.tck','.trk'))
    if not os.path.exists(trk_file) or overwrite:
        python_command = f"python3 ~/DTC_private/AD_Decode/AD_Decode_tcktotrk.py {tck_file} {trk_file} {reference_file}"
        # python_command = "python3 ~/DTC_private/AMD//AMD_subj_to_MDT_clustered.py " + subj
        job_name = job_descrp + "_" + subj
        command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        if test_mode:
            print(command)
        else:
            os.system(command)