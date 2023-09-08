#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:30:59 2023

@author: ali
"""

import os, glob, shutil
import sys, subprocess, copy

# import nibabel as nib

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
job_descrp = "mrtrix"
sbatch_folder_path = BD + "/ADRC_jacques_pipeline/" + job_descrp + '_sbatch/'

if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}")
    # os.makedirs(sbatch_folder_path)
GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'
# GD = '/mnt/munin2/Badea/Lab/mouse/mrtrix_pipeline/'


data_folder_path = '/mnt/munin2/Badea/Lab/ADRC-20230511/'
# list_folders_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
list_folders_path = os.listdir(data_folder_path)
directories = [item for item in list_folders_path if os.path.isdir(os.path.join(data_folder_path, item))]
list_of_subjs_long = [i for i in directories if 'ADRC' in i]
list_of_subjs = list_of_subjs_long

completion_checker = True
cleanup = False
test = True

list_of_subjs.sort()
list_of_subjs_true = copy.deepcopy(list_of_subjs)


if completion_checker:
    conn_folder = os.path.join(BD, 'ADRC_jacques_pipeline', 'connectomes')
    for subj in list_of_subjs:
        conn_folder_subj = os.path.join(conn_folder,subj)
        distances_csv = os.path.join(conn_folder_subj, subj + '_distances.csv')
        mean_FA_connectome = os.path.join(conn_folder_subj, subj + '_mean_FA_connectome.csv')
        if os.path.exists(distances_csv) and os.path.exists(mean_FA_connectome):
            list_of_subjs_true.remove(subj)

if test:
    print(list_of_subjs_true)

# list_of_subjs[90]
# list_of_subjs = [i.partition('_subjspace_dwi.nii.gz')[0] for i in list_of_subjs_long]

# list_of_subjs_true = ['ADRC0017', 'ADRC0025', 'ADRC0026', 'ADRC0028', 'ADRC0033', 'ADRC0040', 'ADRC0043', 'ADRC0047', 'ADRC0084', 'ADRC0085', 'ADRC0101', 'ADRC0102', 'ADRC0111']

# list_of_subjs_true = ['ADRC0028']
#list_of_subjs_true = ['ADRC0091', 'ADRC0080', 'ADRC0086', 'ADRC0082', 'ADRC0070', 'ADRC0064', 'ADRC0065', 'ADRC0063',
#                      'ADRC0079', 'ADRC0048', 'ADRC0066', 'ADRC0095', 'ADRC0071', 'ADRC0076', 'ADRC0050', 'ADRC0087',
#                      'ADRC0099', 'ADRC0100', 'ADRC0078', 'ADRC0083', 'ADRC0062', 'ADRC0096', 'ADRC0097']

if not test:
    for subj in list_of_subjs_true:
        # print(subj)
        # fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz"
        # nib.load(fmri_file)
        python_command = "python /mnt/munin2/Badea/Lab/mouse/ADRC_jacques_pipeline/ADRC_mrtrix_connectomes.py " + subj
        # python_command = "python /mnt/munin2/Badea/Lab/mouse/mrtrix_pipeline/main_trc_conn.py "+subj
        job_name = job_descrp + "_" + subj
        command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        os.system(command)
    #    subprocess.call(command, shell=True)
    #    os.system('qsub -S '+python_command  )

'''


subj = "ADRC0001"

 #print(subj)
python_command = "python /mnt/munin2/Badea/Lab/mouse/ADRC_jacques_pipeline/ADRC_preprocessing_pipeline.py "+subj
 #python_command = "python /mnt/munin2/Badea/Lab/mouse/mrtrix_pipeline/main_trc_conn.py "+subj
job_name = job_descrp + "_"+ subj
command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " "+ job_name + " 0 0 '"+ python_command+"'"   
os.system(command)

'''

