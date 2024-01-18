
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
job_descrp = "trk_toMDT_reg"

sbatch_folder_path = BD +"/tracttoMDT_ADDecode_pipeline/" +job_descrp + '_sbatch/'

if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}" )
    # os.makedirs(sbatch_folder_path)
GD = '~/gunnies/'

list_of_subjs = subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320',
                            'S02361', 'S02363','S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424',
                            'S02446', 'S02451', 'S02469', 'S02473','S02485', 'S02491', 'S02490', 'S02506', 'S02523',
                            'S02524', 'S02535', 'S02654', 'S02666', 'S02670', 'S02686', 'S02690', 'S02695', 'S02715',
                            'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771', 'S02781', 'S02802', 'S02804',
                            'S02813', 'S02812', 'S02817', 'S02840', 'S02842', 'S02871', 'S02877', 'S02898', 'S02926',
                            'S02938', 'S02939', 'S02954', 'S02967', 'S02987', 'S03010', 'S03017', 'S03028', 'S03033',
                            'S03034', 'S03045', 'S03048', 'S03069', 'S03225', 'S03265', 'S03293', 'S03308', 'S03321',
                            'S03343', 'S03350', 'S03378', 'S03391', 'S03394', 'S03847', 'S03866', 'S03867', 'S03889',
                            'S03890', 'S03896']

subj_files = glob.glob('/mnt/munin2/Badea/Lab/human/AD_Decode_trk_transfer/TRK/*.trk')
output_folder = ('/mnt/munin2/Badea/Lab/human/AD_Decode_trk_transfer/TRK_MDT')
list_of_subjs = []

pre_erase_unfinished = True
test_mode = False

#print(subj_files)
for subj_file in subj_files:
    subj = os.path.basename(subj_file)[:6]
    list_of_subjs.append(subj)

if pre_erase_unfinished:
    for subj in list_of_subjs:
        list_files_unfinished = glob.glob(os.path.join(output_folder,f'{subj}*'))
        for file in list_files_unfinished:
            if test_mode:
                print(f'meant to erase file {file}')
            else:
                os.remove(file)
    
print(list_of_subjs)

for subj_trk in subj_files:
    # print(subj)
    # fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz"
    # nib.load(fmri_file)
    python_command = "python3 ~/DTC_private/AD_Decode/AD_Decode_subj_to_MDT_clustered.py " + subj_trk
    #python_command = "python3 ~/DTC_private/AMD//AMD_subj_to_MDT_clustered.py " + subj
    subj = os.path.basename(subj_trk).split('_')[0]
    job_name = job_descrp + "_" + subj
    command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    if test_mode:
        print(command)
    else:
        os.system(command)
    #print(command)

"""
for subj in list_of_subjs:
    # print(subj)
    # fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz"
    # nib.load(fmri_file)
    python_command = "python3 ~/DTC_private/AD_Decode/AD_Decode_subj_to_MDT_clustered.py " + subj
    #python_command = "python3 ~/DTC_private/AMD//AMD_subj_to_MDT_clustered.py " + subj
    job_name = job_descrp + "_" + subj
    command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    if test_mode:
        print(command)
    else:
        os.system(command)
    #print(command)
"""
