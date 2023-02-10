import os, glob, shutil
from DTC.file_manager.file_tools import mkcdir, getfromfile
import pandas as pd
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote, copy_loctoremote, checkfile_exists_remote, load_nifti_remote
import nibabel as nib
import time
import tempfile
import numpy as np


def is_subpath(path, of_paths):
    if isinstance(of_paths, str): of_paths = [of_paths]
    abs_of_paths = [os.path.abspath(of_path) for of_path in of_paths]
    return any(os.path.abspath(path).startswith(subpath) for subpath in abs_of_paths)

BRUKER_orig = ''
csv_summary_path = '/Users/jas/jacques/Daniel_test/FMRI_mastersheet.xlsx'


BRUKER_processed = '/Volumes/documents/paros_DB/BRUKER/niftis/'
BRUKER_f = '/Volumes/documents/paros_WORK/daniel/project/BRUKER_organized_JS'

mkcdir(BRUKER_f)

timings = []
timings.append(time.perf_counter())

#Here, reorganize Bruker processed data into BIDS format
overwrite = False
verbose=True
processed_folders = glob_remote(os.path.join(BRUKER_processed,'*'))
processed_folders.sort()
csv_summary = pd.read_excel(csv_summary_path)
for folder in processed_folders:

    folder_name = os.path.basename(folder)
    folder_id = folder_name.split('_apoe')[0]
    folder_id = folder_id.split('_18_APOE')[0]
    folder_id = folder_id.split('_APOE')[0]
    folder_id = folder_id.split('_18abb11')[0]
    line = csv_summary.loc[csv_summary['MRI_Name'] == str(folder_id)]
    try:
        id = str(list(line['Animal_ID'])[0])
    except:
        print(f'could not find associated animal ID for this folder {folder}, skipping')
        continue
    id = id.replace('_','')
    folder_ID = os.path.join(BRUKER_f, 'sub-'+id)
    folder_ses = os.path.join(folder_ID, 'ses-1')
    folder_anat = os.path.join(folder_ses, 'anat')
    folder_func = os.path.join(folder_ses, 'func')
    mkcdir([folder_ID,folder_ses,folder_anat,folder_func])
    session_rare = str(int(list(line['RARE'])[0]))
    session_T2S = str(int(list(line['T2S_EPI'])[0]))
    #rare_path = glob.glob(os.path.join(folder,session_rare+'_*'))[0]
    try:
        rare_path = glob_remote(os.path.join(folder,session_rare+'_*.nii.gz'))[0]
    except:
        print(f'could not find {os.path.join(folder, session_rare + "_*.nii.gz")}')
        continue
    try:
        func_path = glob_remote(os.path.join(folder,session_T2S+'_*.nii.gz'))[0]
    except:
        print(f'could not find {os.path.join(folder,session_T2S+"_*.nii.gz")}')
        continue
    rare_newpath = os.path.join(folder_anat, f'sub-{id}_ses-1_T1w.nii.gz')
    func_newpath = os.path.join(folder_func, f'sub-{id}_ses-1_bold.nii.gz')

    if not checkfile_exists_remote(func_newpath) or overwrite:
        shutil.copy(func_path, func_newpath)
        if verbose:
            print(f'Copied file {func_path} to {func_newpath}')
    else:
        print(f'File {func_newpath} already exists')

    if not checkfile_exists_remote(rare_newpath) or overwrite:
        shutil.copy(rare_path, rare_newpath)
        if verbose:
            print(f'Copied file {rare_path} to {rare_newpath}')
    else:
        print(f'File {rare_newpath} already exists')

