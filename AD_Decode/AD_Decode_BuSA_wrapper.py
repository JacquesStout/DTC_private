import os, glob
import sys, subprocess
from DTC.file_manager.file_tools import mkcdir, getfromfile
from DTC.file_manager.computer_nav import read_parameters_from_ini
# import nibabel as nib
import numpy as np
import socket

try:
    BD = os.environ['BIGGUS_DISKUS']
# os.environ['GIT_PAGER']
except KeyError:
    print('BD not found locally')
    BD = '/mnt/munin2/Badea/Lab/mouse'
    # BD ='/Volumes/Data/Badea/Lab/mouse'
else:
    print("BD is found locally.")
# create sbatch folder

job_descrp = "BuSA"
sbatch_folder_path = BD + "/ADRC_preprocessing_pipeline/" + job_descrp + '_sbatch/'

#kcdir([sbatch_folder_path])

GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'

if 'santorini' in socket.gethostname().split('.')[0]:
    project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/BuSA_headfiles'
if 'blade' in socket.gethostname().split('.')[0]:
    project_headfile_folder = '/mnt/munin2/Badea/Lab/jacques/BuSA_headfiles'

project_run_identifier = '202311_10template_test01'
project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')

parts = ['5']

subjects = []
project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')
params = read_parameters_from_ini(project_summary_file)

template_subjects = params['template_subjects']
added_subjects = params['added_subjects']
num_bundles = int(params['num_bundles'])

full_subjects_list = template_subjects + added_subjects

sides = ['left', 'right']
bundle_ids = np.arange(num_bundles)

overwrite = False

if '5' in parts:
    for subj in full_subjects_list:
        for side in sides:
            for bundle_id in bundle_ids:
                python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/05_stats_splitbundles.py {project_summary_file} {subj} {side} {bundle_id}"
                job_name = job_descrp + "_" + subj
                command = os.path.join(GD ,"submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
                os.system(command)
