import os, glob, shutil, copy
from DTC.file_manager.file_tools import mkcdir, getfromfile
import pandas as pd
from DTC.nifti_handlers.atlas_handlers.mask_handler import applymask_samespace
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote, copy_loctoremote, checkfile_exists_remote, save_nifti_remote
import numpy as np
import nibabel as nib

remote=False
overwrite=True

BRUKER_init = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS/'
RARE_mask_folder = '/Users/jas/jacques/Daniel_test/RARE_mask_binary/'

project='Daniel'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
if remote:
    _, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
    BRUKER_f = '/mnt/paros_WORK/daniel/project/BRUKER_organized_JS_masked/'
else:
    BRUKER_f = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_masked/'
    sftp=None

mkcdir(BRUKER_f,sftp)

folders = glob.glob(os.path.join(BRUKER_init,'*/'))
folders.sort()

subj_list = []
for folder in folders:
    subj_list.append(folder.split('/')[-2])
subj_list.sort()
print(subj_list)

for folder in folders:
    #folder = os.path.join(BRUKER_init,key)
    key = folder.split('/')[-2]
    if 'sub' not in key:
        print('bug')
    folder_anat = os.path.join(folder,'ses-1','anat')
    folder_func = os.path.join(folder,'ses-1','func')

    raremask_paths = glob.glob(os.path.join(RARE_mask_folder,'*'+key.replace('-','')+'_mask.nii.gz'))
    if np.size(raremask_paths)==1:
        raremask_path = raremask_paths[0]

    rare_prepath = os.path.join(folder_anat, f'{key}_ses-1_T1w.nii.gz')
    func_prepath = os.path.join(folder_func, f'{key}_ses-1_bold.nii.gz')

    folder_f = os.path.join(BRUKER_f,key)
    folder_anat = os.path.join(folder_f,'ses-1','anat')
    folder_func = os.path.join(folder_f,'ses-1','func')
    mkcdir([folder_f,os.path.join(folder_f,'ses-1'),folder_anat,folder_func], sftp)
    rare_newpath = os.path.join(folder_anat, f'{key}_ses-1_T1w.nii.gz')
    func_newpath = os.path.join(folder_func, f'{key}_ses-1_bold.nii.gz')

    if not os.path.exists(rare_newpath) or overwrite:
        applymask_samespace(rare_prepath,raremask_path,outpath = rare_newpath)
        print(f'Saved {rare_newpath}')
    if not os.path.exists(func_newpath) or overwrite:
        shutil.copy(func_prepath,func_newpath)
        print(f'Saved {func_newpath}')