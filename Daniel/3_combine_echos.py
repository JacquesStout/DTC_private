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


#Here, turn bruker raw data into processed nifti data

#######

outpath_temp = tempfile.gettempdir()
outpath_temp = '/Users/jas/jacques/tempdir'

remote=True
project='Daniel'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
if remote:
    _, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
    BRUKER_processed = '/mnt/paros_DB/BRUKER/niftis/'
    BRUKER_f = '/mnt/paros_WORK/daniel/project/BRUKER_organized_JS_combined_v2/'
else:
    BRUKER_processed = '/Users/jas/jacques/Daniel_test/BRUKER/'
    BRUKER_f = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_combined/'
    sftp = None

mkcdir(BRUKER_f, sftp)

bonus_rare_folder = '/mnt/paros_WORK/jacques/RARE/'

timings = []
timings.append(time.perf_counter())


#Here, reorganize Bruker processed data into BIDS format
overwrite = False
verbose=True
processed_folders = glob_remote(os.path.join(BRUKER_processed), sftp)
processed_folders.sort()
csv_summary = pd.read_excel(csv_summary_path)
for folder in processed_folders:

    to_remove = []

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
    mkcdir([folder_ID,folder_ses,folder_anat,folder_func],sftp)
    session_rare = str(int(list(line['RARE'])[0]))
    session_T2S = str(int(list(line['T2S_EPI'])[0]))
    #rare_path = glob.glob(os.path.join(folder,session_rare+'_*'))[0]
    try:
        rare_path = glob_remote(os.path.join(folder,session_rare+'_*.nii.gz'),sftp)[0]
    except:
        print(f'could not find {os.path.join(folder, session_rare + "_*.nii.gz")}')
        continue
    try:
        t2S_path = glob_remote(os.path.join(folder,session_T2S+'_*.nii.gz'),sftp)[0]
    except:
        print(f'could not find {os.path.join(folder,session_T2S+"_*.nii.gz")}')
        continue
    rare_newpath = os.path.join(folder_anat, f'sub-{id}_ses-1_T1w.nii.gz')
    func_newpath = os.path.join(folder_func, f'sub-{id}_ses-1_bold.nii.gz')
    if not checkfile_exists_remote(func_newpath, sftp) or overwrite:
        print(f'Starting processing for {t2S_path} to {func_newpath}')
        data_multi_echo, affine, _, hdr, _ = load_nifti_remote(t2S_path, sftp)
        #img = nib.load('/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/sub-2204043_ses-1_run-1_bold.nii.gz')
        #data_truncated = data_multi_echo[:,:,:,:9]
        #truncated_nii = nib.Nifti1Image(data_truncated, affine, hdr)
        #nib.save(truncated_nii,'/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/trunc_{id}.nii.gz')
        timings.append(time.perf_counter())
        print(f'Loaded {t2S_path}, took {timings[-1] - timings[-2]} seconds')

        mask_threshold_path = os.path.join(outpath_temp, f'{id}_mask.nii.gz')
        if not checkfile_exists_remote(mask_threshold_path, sftp):
            if sftp is not None:
                t2S_path_local = os.path.join(outpath_temp, os.path.basename(t2S_path))
                sftp.get(t2S_path, t2S_path_local)
            else:
                t2S_path_local = t2S_path
            threshold_command = f'ThresholdImage 3 {t2S_path_local} {mask_threshold_path} 0 1000000'
            os.system(threshold_command)

        # specify the number of echos
        TEs = 3

        # iterate over the number of echos and create individual 4D NIfTIs for each echo
        echo_paths = []
        for i in np.arange(TEs):
            single_echo_path = os.path.join(outpath_temp, f'{id}_echo_' + str(i + 1) + '.nii.gz')
            if not os.path.exists(single_echo_path):
                data_single_echo = data_multi_echo[:,:,:,i::3]
                single_echo_nii = nib.Nifti1Image(data_single_echo, affine, hdr)
                nib.save(single_echo_nii, single_echo_path)
                timings.append(time.perf_counter())
                print(f'Saved at {single_echo_path}, took {timings[-1] - timings[-2]} seconds')
            echo_paths.append(single_echo_path)
            to_remove.append(single_echo_path)

        tedana_out_dir = os.path.join(outpath_temp, 'tedana_outputs')
        mkcdir(tedana_out_dir)
        tedana_command = f'tedana -d {os.path.join(outpath_temp,f"{id}_echo_1.nii.gz")} {os.path.join(outpath_temp,f"{id}_echo_2.nii.gz")} {os.path.join(outpath_temp,f"{id}_echo_3.nii.gz")} -e 5.0 19.315 33.63 --mask {mask_threshold_path} --out-dir {tedana_out_dir}'
        #maskpath = '/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/mask.nii.gz'
        #tedana_command = f'tedana -d {os.path.join(outpath_temp, f"{id}_echo_1.nii.gz")} {os.path.join(outpath_temp, f"{id}_echo_2.nii.gz")} {os.path.join(outpath_temp, f"{id}_echo_3.nii.gz")} -e 5.0 19.315 33.63 --mask {maskpath} --out-dir {tedana_out_dir}'
        os.system(tedana_command)
        timings.append(time.perf_counter())
        print(f'Combined echos, took {timings[-1] - timings[-2]} seconds')
        tedanafile_path = os.path.join(tedana_out_dir, 'desc-optcom_bold.nii.gz')

        #shutil.copy(tedanafile_path, func_newpath)
        copy_loctoremote(tedanafile_path, func_newpath, sftp)
        if checkfile_exists_remote(func_newpath, sftp):
            timings.append(time.perf_counter())
            print(f'Saved file at {func_newpath}, took {timings[-1] - timings[-2]} seconds')
        else:
            raise Exception
        tedana_files = glob.glob(os.path.join(tedana_out_dir, '*'))
        for tedana_file in tedana_files:
            to_remove.append(tedana_file)
    else:
        print(f'already created {func_newpath}')

    if not checkfile_exists_remote(rare_newpath, sftp) or overwrite:
        #shutil.copy(rare_path,rare_newpath)
        if sftp is not None:
            rare_path_temp = os.path.join(outpath_temp, os.path.basename(rare_path))
            sftp.get(rare_path,rare_path_temp)
            to_remove.append(rare_path_temp)
        else:
            rare_path_temp = rare_path
        copy_loctoremote(rare_path_temp, rare_newpath, sftp)

    for remove_file in to_remove:
        if os.path.isfile(remove_file):
            os.remove(remove_file)
        elif os.path.isdir(remove_file):
            if is_subpath(remove_file,tedana_out_dir):
                shutil.rmtree(remove_file)
