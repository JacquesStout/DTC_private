
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:30:59 2023
@author: ali
"""

import os , glob
import sys

# import nibabel as nib

try :
    BD = os.environ['BIGGUS_DISKUS']
# os.environ['GIT_PAGER']
except KeyError:
    print('BD not found locally')
    BD = '***/mouse'
    # BD ='***/example'
else:
    print("BD is found locally.")
# create sbatch folder
job_descrp =  "trk_toMDT_reg"

mrtrix=False
if mrtrix:
    sbatch_folder_path = BD +"/streamline_prep_mrtrix_MDT_pipeline/" +job_descrp + '_sbatch/'
else:
    sbatch_folder_path = BD +"/streamline_prep_MDT_pipeline/" +job_descrp + '_sbatch/'


if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}" )
    # os.makedirs(sbatch_folder_path)
GD = '~/gunnies/'

list_of_subjs = ['H22825', 'H21850', 'H29225', 'H29304', 'H29060', 'H23210', 'H21836', 'H29618', 'H22644', 'H22574',
            'H22369', 'H29627', 'H29056', 'H22536', 'H23143', 'H22320', 'H22898', 'H22864', 'H29264', 'H22683',
            'H29403', 'H22102', 'H29502', 'H22276', 'H29878', 'H29410', 'H22331', 'H22368', 'H21729', 'H29556',
            'H21956', 'H22140', 'H23309', 'H22101', 'H23157', 'H21593', 'H21990', 'H22228', 'H23028', 'H21915',
            'H27852', 'H28029', 'H26966', 'H27126', 'H29161', 'H28955', 'H26862', 'H27842', 'H27999', 'H28325',
            'H26841', 'H27719', 'H27100', 'H27682', 'H29002', 'H27488', 'H27841', 'H28820', 'H28208', 'H27686',
            'H29020', 'H26637', 'H26765', 'H28308', 'H28433', 'H26660', 'H28182', 'H27111', 'H27391', 'H28748',
            'H28662', 'H26578', 'H28698', 'H27495', 'H28861', 'H28115', 'H28377', 'H26890', 'H28373', 'H27164']
#list_of_subjs = ['H22825']
#list_of_subjs = ['H21850', 'H29225', 'H29304', 'H29060', 'H23210', 'H21836', 'H29618', 'H22644', 'H22574',
#            'H22369', 'H29627', 'H29056', 'H22536', 'H23143', 'H22320', 'H22898', 'H22864', 'H29264', 'H22683',
#            'H29403', 'H22102', 'H29502', 'H22276', 'H29878', 'H29410', 'H22331', 'H22368', 'H21729', 'H29556',
#            'H21956', 'H22140', 'H23309', 'H22101', 'H23157', 'H21593', 'H21990', 'H22228', 'H23028', 'H21915',
#            'H27852', 'H28029', 'H26966', 'H27126', 'H29161', 'H28955', 'H26862', 'H27842', 'H27999', 'H28325',
#            'H26841', 'H27719', 'H27100', 'H27682', 'H29002', 'H27488', 'H27841', 'H28820', 'H28208', 'H27686',
#            'H29020', 'H26637', 'H26765', 'H28308', 'H28433', 'H26660', 'H28182', 'H27111', 'H27391', 'H28748',
#            'H28662', 'H26578', 'H28698', 'H27495', 'H28861', 'H28115', 'H28377', 'H26890', 'H28373', 'H27164']


for subj in list_of_subjs:
    # print(subj)
    # fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz"
    # nib.load(fmri_file)
    if mrtrix:
        python_command = "python3 ~/DTC_private/AMD/streamline_average_prep_AMD_clustered.py " + subj + ' 1'
    else:
        python_command = "python3 ~/DTC_private/AMD/streamline_average_prep_AMD_clustered.py " + subj + ' 0'
    job_name = job_descrp + "_" + subj
    command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    os.system(command)
    #$print(command)
