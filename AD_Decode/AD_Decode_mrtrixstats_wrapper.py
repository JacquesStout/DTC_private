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
sbatch_folder_path = os.path.join(BD, "..", "human", "AD_Decode", job_descrp + '_sbatch/')

if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}")
    # os.makedirs(sbatch_folder_path)
GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'
# GD = '/mnt/munin2/Badea/Lab/mouse/mrtrix_pipeline/'


data_folder_path = os.path.join(BD,'..',"human", "AD_Decode",'diffusion_prep_locale')
# list_folders_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
list_folders_path = os.listdir(data_folder_path)
directories = [item for item in list_folders_path if os.path.isdir(os.path.join(data_folder_path, item))]
list_of_subjs_long = [i for i in directories if 'diffusion_prep' in i]
list_of_subjs = [subj.split('prep_')[1] for subj in list_of_subjs_long]

completion_checker = True
cleanup = False
test = False

list_of_subjs.sort()
list_of_subjs_true = copy.deepcopy(list_of_subjs)

print(list_of_subjs_true)
proc_name ="diffusion_prep_"

if completion_checker:
    perm_folder = os.path.join(BD, 'ADRC_jacques_pipeline', 'perm_files')
    for subj in list_of_subjs:
        subj_path = os.path.join(data_folder_path, proc_name + subj)

        fa_nii = os.path.join(subj_path, subj + '_mrtrixfa.nii.gz')

        if os.path.exists(fa_nii):
            list_of_subjs_true.remove(subj)

if test:
    print(list_of_subjs_true)

if not test:
    for subj in list_of_subjs_true:
        # print(subj)
        # fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz"
        # nib.load(fmri_file)
        #python_command = "python /mnt/munin2/Badea/Lab/mouse/ADRC_jacques_pipeline/ADRC_preprocessing_pipeline.py " + subj
        # python_command = "python /mnt/munin2/Badea/Lab/mouse/mrtrix_pipeline/main_trc_conn.py "+subj
        python_command = "python ~/DTC_private/AD_Decode/AD_Decode_makemrtrixstats.py " + subj
        job_name = job_descrp + "_" + subj
        command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        os.system(command)
    #    subprocess.call(command, shell=True)
    #    os.system('qsub -S '+python_command  )
