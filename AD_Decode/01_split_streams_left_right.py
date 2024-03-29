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
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, write_parameters_to_ini, read_parameters_from_ini
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from DTC.tract_manager.tract_save import save_trk_header
from DTC.file_manager.argument_tools import parse_arguments
import sys
from DTC.tract_manager.tract_handler import ratio_to_str, filter_streamlines
import configparser


if len(sys.argv)<2:
    project_headfile_folder = '/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/'
    project_run_identifier = 'V0_9_10template_100_72_interhe'
    project_summary_file = os.path.join(project_headfile_folder, project_run_identifier + '.ini')
else:
    project_summary_file = sys.argv[1]
    project_run_identifier = os.path.basename(project_summary_file).split('.')[0]

if not os.path.exists(project_summary_file):
    txt = f'Could not find configuration file at {project_summary_file}'
    raise Exception(txt)
else:
    params = read_parameters_from_ini(project_summary_file)


project = params['project']
streamline_type = params['streamline_type']
test = params['test']
ratio = params['ratio']
stepsize = params['stepsize']
template_subjects = params['template_subjects']
streamline_lr_inclusion  = params['streamline_lr_inclusion']
length_threshold = int(params['length_threshold'])
remote_input = bool(params['remote_input'])
remote_output = bool(params['remote_output'])
path_TRK = params['path_trk']

overwrite = False
overwrite = False
verbose = False

if 'santorini' in socket.gethostname().split('.')[0]:
    lab_folder = '/Volumes/Data/Badea/Lab'
if 'blade' in socket.gethostname().split('.')[0]:
    lab_folder = '/mnt/munin2/Badea/Lab'

if project == 'AD_Decode':
    SAMBA_MDT = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz')
    MDT_mask_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT')
    ref_MDT_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images')
    anat_path = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz')


if streamline_type == 'mrtrix':
    prune = False
    trkroi = [""]
else:
    prune = True
    trkroi = ["wholebrain"]

str_identifier = get_str_identifier(stepsize, ratio, trkroi, type='mrtrix')

#subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv,template_subjects)
#template_subjects = template_subjects[firstsubj:lastsubj]
#print(template_subjects)

#method = ['trk_roiseeded', 'trk_roisplit']
computer_name = socket.gethostname()

if remote_input or remote_output:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

_, _, _, sftp_in = get_mainpaths(remote_input,project = project, username=username,password=passwd)
outpath, _, _, sftp_out = get_mainpaths(remote_output,project = project, username=username,password=passwd)

ratio_str = ratio_to_str(ratio,spec_all=False)

outpath_all = os.path.join(outpath, 'TRK_bundle_splitter')
proj_path = os.path.join(outpath_all,project_run_identifier)

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratio_str)
outpath_trk = os.path.join(proj_path, 'trk_roi'+ratio_str)

mkcdir([outpath_all, proj_path, pickle_folder, outpath_trk], sftp_out)

method = 'dwi_roi_to_trk'

if len(sys.argv) >2:
    template_subjects = [sys.argv[2]]


if method=='dwi_roi_to_trk':

    for subject in template_subjects:

        picklepath_r = os.path.join(pickle_folder, f'{subject}_roi_rstream.p')
        trkpath_right = os.path.join(outpath_trk, f'{subject}_roi_rstream.trk')

        picklepath_l = os.path.join(pickle_folder, f'{subject}_roi_lstream.p')
        trkpath_left = os.path.join(outpath_trk, f'{subject}_roi_lstream.trk')

        _, exists = check_files([picklepath_r, trkpath_right, picklepath_l, trkpath_left], sftp_out)
        if not np.invert(exists).any():
            print(f'already did subject {subject}')
            continue
        right_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_right.nii.gz')
        left_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_left.nii.gz')

        roi_mask_right = nib.load(right_mask_path)
        roi_mask_left = nib.load(left_mask_path)
        print(f'Loaded masks {right_mask_path} and {left_mask_path}')

        subj_trk, trkexists = gettrkpath(path_TRK, subject, str_identifier, pruned=prune, verbose=False, sftp=sftp_in)

        t1 = time.perf_counter()

        if not trkexists:
            import warnings
            txt = f'Subject {subject} does not have {subj_trk} yet, skip'
            print(txt)
            continue

        streamlines_data = load_trk_remote(subj_trk, 'same', sftp_in)
        header = streamlines_data.space_attributes
        streamlines = streamlines_data

        ttrk = time.perf_counter()

        print(f'Loaded the streamlines in {subj_trk}, took {ttrk - t1:0.2f} seconds')
        #roi_rstream, roi_rreal = filter_streamlines(roi_mask_right, streamlines, world_coords = True, include='all')

        roi_rstream = filter_streamlines(streamlines, roi_mask = roi_mask_right , world_coords = True, include=streamline_lr_inclusion, threshold = length_threshold)
        print('Right side done!')

        roi_lstream = filter_streamlines(streamlines, roi_mask = roi_mask_left, world_coords = True, include=streamline_lr_inclusion, threshold = length_threshold)
        print('Left side done!')

        import pickle

        if (not checkfile_exists_remote(trkpath_right, sftp_out) or overwrite):
            save_trk_header(filepath=trkpath_right, streamlines=roi_rstream, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp_out)
            print(f'Writing subject {subject} for right side at {trkpath_right}')
        else:
            print(f'Already wrote {trkpath_right}')
        if not checkfile_exists_remote(picklepath_r, sftp_out):
            pickledump_remote(roi_rstream, picklepath_r, sftp_out)
        else:
            print(f'Already wrote {picklepath_r}')


        if (not checkfile_exists_remote(trkpath_left, sftp_out) or overwrite):
            save_trk_header(filepath=trkpath_left, streamlines=roi_lstream, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp_out)
            print(f'Writing subject {subject} for left side at {trkpath_left}')
        else:
            print(f'Already wrote {trkpath_left}')

        if not checkfile_exists_remote(picklepath_l, sftp_out):
            pickledump_remote(roi_lstream, picklepath_l, sftp_out)
        else:
            print(f'Already wrote {picklepath_l}')
