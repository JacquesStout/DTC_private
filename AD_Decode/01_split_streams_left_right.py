#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:33:41 2022

@author: ruidai
"""

import numpy as np
import os, socket, time
import nibabel as nib
from DTC.tract_manager.tract_handler import gettrkpath
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote, pickledump_remote
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from DTC.tract_manager.tract_save import save_trk_header
from DTC.file_manager.argument_tools import parse_arguments
import sys
from DTC.tract_manager.tract_to_roi_handler import filter_streamlines
from DTC.tract_manager.tract_handler import ratio_to_str


MDT_mask_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT'

test= False
project='AD_Decode'
type = 'mrtrix'
if project=='AD_Decode':
    #TRK_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT_real_testtemp'
    #TRK_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT'
    MDT_mask_folder
    if test:
        subjects = ['S01912', 'S02110', 'S02224', 'S02227']
    else:
        subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361',
                    'S02363',
                    'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451', 'S02469',
                    'S02473',
                    'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654', 'S02666', 'S02670',
                    'S02686',
                    'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771', 'S02781',
                    'S02802',
                    'S02804', 'S02813', 'S02812', 'S02817', 'S02840', 'S02842', 'S02871', 'S02877', 'S02898', 'S02926',
                    'S02938',
                    'S02939', 'S02954', 'S02967', 'S02987', 'S03010', 'S03017', 'S03028', 'S03033', 'S03034', 'S03045',
                    'S03048',
                    'S03069', 'S03225', 'S03265', 'S03293', 'S03308', 'S03321', 'S03343', 'S03350', 'S03378', 'S03391',
                    'S03394', 'S03847', 'S03866', 'S03867', 'S03889', 'S03890', 'S03896']
        #subjects = ['S02771']
    removed_list = ['S02230', 'S02654', 'S02490', 'S02523', 'S02745']
    for remove in removed_list:
        if remove in subjects:
            subjects.remove(remove)

    # Get the values from DTC_launcher_ADDecode. Should probalby create a single parameter file for each project one day
    stepsize = 2
    ratio = 100
    trkroi = ["wholebrain"]

    if type == 'mrtrix':
        prune = False
    else:
        prune = True
    str_identifier = get_str_identifier(stepsize, ratio, trkroi,type = 'mrtrix')
    SAMBA_MDT = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz'
    #subjects = ['S02715']

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv,subjects)
subjects = subjects[firstsubj:lastsubj]
print(subjects)

#method = ['trk_roiseeded', 'trk_roisplit']
ext = ".nii.gz"
computer_name = socket.gethostname()

remote=True
project='AD_Decode'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)


ratio_str = ratio_to_str(ratio,spec_all=False)

#path_TRK = os.path.join(inpath, 'TRK_MPCA_MDT'+ratio_str)
path_TRK = os.path.join(inpath, 'TRK_MDT'+ratio_str)
proj_path = os.path.join(inpath, 'TRK_bundle_splitter')

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratio_str)
outpath_trk = os.path.join(proj_path, 'trk_roi'+ratio_str)

overwrite = True
verbose = False

mkcdir([proj_path, pickle_folder, outpath_trk], sftp)

method = 'dwi_roi_to_trk'

if method=='dwi_roi_to_trk':

    for subject in subjects:

        picklepath_r = os.path.join(pickle_folder, f'{subject}_roi_rstream.p')
        trkpath_right = os.path.join(outpath_trk, f'{subject}_roi_rstream.trk')

        picklepath_l = os.path.join(pickle_folder, f'{subject}_roi_lstream.p')
        trkpath_left = os.path.join(outpath_trk, f'{subject}_roi_lstream.trk')

        _, exists = check_files([picklepath_r, trkpath_right, picklepath_l, trkpath_left], sftp)
        if not np.invert(exists).any():
            print(f'already did subject {subject}')
            continue
        right_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_right.nii.gz')
        left_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_left.nii.gz')

        roi_mask_right = nib.load(right_mask_path)
        roi_mask_left = nib.load(left_mask_path)
        print(f'Loaded masks {right_mask_path} and {left_mask_path}')

        subj_trk, trkexists = gettrkpath(path_TRK, subject, str_identifier, pruned=prune, verbose=False, sftp=sftp)

        t1 = time.perf_counter()

        if not trkexists:
            import warnings
            txt = f'Subject {subject} does not have {subj_trk} yet, skip'
            print(txt)
            continue

        streamlines_data = load_trk_remote(subj_trk, 'same', sftp)
        header = streamlines_data.space_attributes
        streamlines = streamlines_data

        ttrk = time.perf_counter()

        print(f'Loaded the streamlines in {subj_trk}, took {ttrk - t1:0.2f} seconds')
        #roi_rstream, roi_rreal = filter_streamlines(roi_mask_right, streamlines, world_coords = True, include='all')

        roi_rstream = filter_streamlines(roi_mask_right, streamlines, world_coords = True, include='only_mask')
        print('Right side done!')

        roi_lstream = filter_streamlines(roi_mask_left, streamlines, world_coords = True, include='only_mask')
        print('Left side done!')

        import pickle

        if (not checkfile_exists_remote(trkpath_right, sftp) or overwrite):
            save_trk_header(filepath=trkpath_right, streamlines=roi_rstream, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp)
            print(f'Writing subject {subject} for right side at {trkpath_right}')
        else:
            print(f'Already wrote {trkpath_right}')
        if not checkfile_exists_remote(picklepath_r, sftp):
            pickledump_remote(roi_rstream, picklepath_r, sftp)
        else:
            print(f'Already wrote {picklepath_r}')


        if (not checkfile_exists_remote(trkpath_left, sftp) or overwrite):
            save_trk_header(filepath=trkpath_left, streamlines=roi_lstream, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp)
            print(f'Writing subject {subject} for left side at {trkpath_left}')
        else:
            print(f'Already wrote {trkpath_left}')

        if not checkfile_exists_remote(picklepath_l, sftp):
            pickledump_remote(roi_lstream, picklepath_l, sftp)
        else:
            print(f'Already wrote {picklepath_l}')
