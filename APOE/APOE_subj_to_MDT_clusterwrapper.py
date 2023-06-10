
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
    sbatch_folder_path = BD +"/tracttoMDT_pipeline/" +job_descrp + '_sbatch/'
else:
    sbatch_folder_path = BD +"/tracttoMDT_notmrtrix_pipeline/" +job_descrp + '_sbatch/'


if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}" )
    # os.makedirs(sbatch_folder_path)
GD = '~/gunnies/'

list_of_subjs = subjects = ['N57437', 'N57446','N57447','N57449','N57451','N57496','N57498','N57500','N57502','N57504','N57513','N57515','N57518','N57520','N57522','N57546','N57548','N57550','N57552','N57554','N57559','N57580','N57582','N57584','N57587','N57590','N57692','N57694','N57700','N57702','N57709','N58214','N58215','N58216','N58217' ,'N58218','N58219','N58221','N58222','N58223' ,'N58224','N58225','N58226','N58228','N58229','N58230','N58231','N58232','N58610','N58612','N58633','N58634','N58635','N58636','N58649','N58650','N58651','N58653','N58654','N58889','N59066','N59109']




for subj in list_of_subjs:
    # print(subj)
    # fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz"
    # nib.load(fmri_file)
    if mrtrix:
        python_command = "python3 ~/DTC_private/AMD//AMD_trix_subj_to_MDT_clustered.py " + subj
    else:
        python_command = "python3 ~/DTC_private/AMD//AMD_subj_to_MDT_clustered.py " + subj
    job_name = job_descrp + "_" + subj
    command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    os.system(command)
    #print(command)
