#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:30:59 2023

@author: ali
"""

import os , glob
import sys, subprocess
from DTC.file_manager.qstat_tools import limit_jobs, check_job_name
#import nibabel as nib

try :
    BD = os.environ['BIGGUS_DISKUS']
#os.environ['GIT_PAGER']
except KeyError:  
    print('BD not found locally')
    BD = '/mnt/munin2/Badea/Lab/mouse'    
    BD ='/Volumes/Data/Badea/Lab/mouse'
else:
    print("BD is found locally.")
#create sbatch folder
job_descrp =  "mrtrix"
sbatch_folder_path = BD+"/mrtrix_ad_decode/"+job_descrp + '_sbatch/'

if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}" )
    #os.makedirs(sbatch_folder_path)
GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'
#GD = '/mnt/munin2/Badea/Lab/mouse/mrtrix_pipeline/'


act = True
if act:
    contrast = 'subjspace_fa'
    act_string = '_act'
else:
    contrast = 'dwi'
    act_string = ''

#list_folders_path ='/mnt/munin2/Badea/ADdecode.01/Analysis/DWI/'
inputfiles_path = os.path.join(BD,'../../ADdecode.01/Analysis/DWI/')
list_folders_path = os.listdir(inputfiles_path)
list_of_subjs_long = [i for i in list_folders_path if f'{contrast}.nii.gz' in i]

list_of_subjs = [i.partition(f'_{contrast}.nii.gz')[0] for i in list_of_subjs_long]

#print(list_of_subjs)
for subj in list_of_subjs:
    #print('hi')
    coreg_path = f'{subj}_subjspace_coreg.nii.gz'
    if coreg_path not in list_folders_path:
        print(subj)
        list_of_subjs.remove(subj)

#list_of_coregs = [i.partition(f'_subjspace_coreg.nii.gz')[0] for i in list_of_subjs_long]

print('total list')
print(list_of_subjs)
#print(list_of_coregs)

#conn_path = f'/mnt/munin2/Badea/Lab/mouse/mrtrix_ad_decode/connectome{act_string}/'
conn_path = os.path.join(BD, f'mrtrix_ad_decode/connectome{act_string}/')
check_con = True

if check_con and os.path.isdir(conn_path):
    done_subj = os.listdir(conn_path)
    done_subj = [i for i in done_subj if 'conn_plain' in i]
    done_subj = [i.partition('_conn_plain.csv')[0] for i in done_subj]
    list_of_subjs = set(list_of_subjs) - set(done_subj)
#list_fmri_folders.remove(".DS_Store")

test_mode = True 
limit = 20

#list_of_subjs = ['S01912']
print('remaining list')
print(list_of_subjs)
#list_of_subjs.remove('S02765')
for subj in list_of_subjs:
    #print(subj)
    #fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz" 
    #nib.load(fmri_file)
    if act:
        python_command = f"python ~/DTC_private/AD_Decode/AD_Decode_trc_generation.py {subj} True"
    else:
        python_command = f"python ~/DTC_private/AD_Decode/AD_Decode_trc_generation.py {subj}"

    job_name = job_descrp + "_"+ subj
    command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " "+ job_name + " 0 0 '"+ python_command+"'"

    job_name_running = check_job_name(job_name)
    if job_name_running:
        print(f'{job_name} is already running')
    if test_mode:
        print(command)
    else:
        if limit is not None:
            limit_jobs(limit=limit)
        if not job_name_running:
            os.system(command)
