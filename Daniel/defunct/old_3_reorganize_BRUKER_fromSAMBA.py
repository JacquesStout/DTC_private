import os, glob, shutil
from DTC.file_manager.file_tools import mkcdir, getfromfile
import pandas as pd
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote, copy_loctoremote, checkfile_exists_remote


BRUKER_orig = ''
BRUKER_processed = '/Users/jas/jacques/Daniel_test/BRUKER/'
csv_summary_path = '/Users/jas/jacques/Daniel_test/FMRI_mastersheet.xlsx'

mkcdir(BRUKER_processed)

#Here, turn bruker raw data into processed nifti data

#######

remote=True
project='Daniel'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
if remote:
    _, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
    BRUKER_f = '/mnt/paros_WORK/daniel/project/BRUKER_organized_SAMBA_JS/'
else:
    BRUKER_f = '/Users/jas/jacques/Daniel_test/BRUKER_organized_SAMBA_JS/'
    sftp = None

mkcdir(BRUKER_f, sftp)

#Here, reorganize Bruker processed data into BIDS format
overwrite = False
verbose=True
processed_folders = glob.glob(os.path.join(BRUKER_processed,'*'))
processed_folders.sort()
csv_summary = pd.read_excel(csv_summary_path)
for folder in processed_folders:
    folder_name = os.path.basename(folder)
    folder_id = folder_name.split('_apoe')[0]
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
    mkcdir([folder_ID,folder_ses,folder_anat,folder_func],sftp)
    session_rare = str(list(line['RARE'])[0])
    session_T2S = str(list(line['T2S_EPI'])[0])
    #rare_path = glob.glob(os.path.join(folder,session_rare+'_*'))[0]
    try:
        rare_path = glob_remote(os.path.join(folder,session_rare+'_*'),None)[0]
    except:
        print('hi')
    try:
        t2S_path = glob_remote(os.path.join(folder,session_T2S+'_*'),None)[0]
    except:
        print('hi')
    rare_newpath = os.path.join(folder_anat, f'sub-{id}_ses-1_T1w.nii.gz')
    func_newpath = os.path.join(folder_func, f'sub-{id}_ses-1_bold.nii.gz')
    if 'RARE_MEMRI' in os.path.basename(rare_path):
        if not checkfile_exists_remote(rare_newpath, sftp) or overwrite:
            copy_loctoremote(rare_path, rare_newpath, sftp)
        elif verbose:
            print(f'already created {rare_newpath}')
    else:
        print(f'Problem with {rare_path}, check your excel reference!')
    if 'T2S_EPI' in os.path.basename(t2S_path):
        if not checkfile_exists_remote(func_newpath, sftp) or overwrite:
            copy_loctoremote(t2S_path, func_newpath, sftp)
        elif verbose:
            print(f'already created {func_newpath}')
    else:
        print(f'Problem with {t2S_path}, check your excel reference!')

