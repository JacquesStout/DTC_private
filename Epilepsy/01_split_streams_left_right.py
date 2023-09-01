#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:33:41 2022

@author: ruidai
"""

import numpy as np
from scipy.ndimage.morphology import binary_dilation
import os
from dipy.direction import peaks
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
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
from os.path import expanduser, join
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import create_backport_labels, create_MDT_labels


home = expanduser('~')

#subject = 'S01952'
subjects = ['S00393', 'S00613', 'S00680', 'S00699', 'S01952', 'S00795', 'S00490']
sftp = None

overwrite = False
verbose = False
prune = False
myiteration = 6

mainpath = '/Volumes/Data/Badea/Lab/'
project_folder = '/Volumes/Data/Badea/Lab/mouse/epilepsy_coreg/'

SAMBA_mainpath = '/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/'
#SAMBA_mainpath = os.path.join(mainpath, "mouse")
SAMBA_projectname = "VBM_19IntractEP01_IITmean_RPI_cleaned"

MDT_mask_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname+'-results')

atlas_labels = os.path.join(mainpath, "atlas", "IITmean_RPI", "IITmean_RPI_labels.nii.gz")
labels_folder = os.path.join(mainpath, "labels")
SAMBA_prep_folder = os.path.join(project_folder, "Reg")
SAMBA_headfile = '/Volumes/Data/Badea/Lab/samba_startup_cache/rja20_IntractEP.01_v2021_SAMBA_startup.headfile'

for subject in subjects:
    subject_num = subject[1:]
    dwi_folder = os.path.join(project_folder, 'Reg')
    print(dwi_folder)

    # flabel = join(dname, str(subject) + '_wm.nii')
    # print(flabel)

    # flabels = join(dname, str(subject) + '_IITmean_RPI_labels.nii')
    dwi_path = join(dwi_folder, f'{subject}_Reg_LPCA_nii4_RAS_mpca.nii')
    fbval = os.path.join(dwi_folder, str(subject) + '_bvals.txt')
    fbvec = os.path.join(dwi_folder, str(subject) + '_bvecs.txt')
    #labels_path = ''
    #print(flabels)

    path_TRK = os.path.join(project_folder, 'TRK')
    ratio=100

    if ratio > 1:
        path_TRK = path_TRK + f'_{ratio}'

    pickle_folder = os.path.join(project_folder, 'pickle_roi')
    outpath_trk = os.path.join(project_folder, 'trk_roi')
    if ratio > 1:
        pickle_folder = pickle_folder + f'_{ratio}'
        outpath_trk = outpath_trk + f'_{ratio}'


    mkcdir([pickle_folder, outpath_trk], sftp)

    method = 'dwi_roi_to_trk'

    if method == 'dwi_roi_to_trk':

        for subject in subjects:

            picklepath_r = os.path.join(pickle_folder, f'{subject}_roi_rstream.p')
            trkpath_right = os.path.join(outpath_trk, f'{subject}_roi_rstream.trk')

            picklepath_l = os.path.join(pickle_folder, f'{subject}_roi_lstream.p')
            trkpath_left = os.path.join(outpath_trk, f'{subject}_roi_lstream.trk')

            _, exists = check_files([picklepath_r, trkpath_right, picklepath_l, trkpath_left], sftp)
            if not np.invert(exists).any():
                print(f'already did subject {subject}')
                continue


            #create_MDT_labels(subject, SAMBA_mainpath, SAMBA_projectname, atlas_labels, reg_type = 'dwi', overwrite=overwrite)
            create_MDT_labels(subject, SAMBA_mainpath, SAMBA_projectname, atlas_labels, myiteration=myiteration,
                              reg_type='dwi', overwrite=overwrite, verbose=verbose)
            labelspath_remote = os.path.join(labels_folder, f'{subject}_labels.nii.gz')
            subject_notdone = create_backport_labels(subject, SAMBA_mainpath, SAMBA_projectname, SAMBA_prep_folder,
                                                     atlas_labels, headfile=SAMBA_headfile, overwrite=overwrite)


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
            # roi_rstream, roi_rreal = filter_streamlines(roi_mask_right, streamlines, world_coords = True, include='all')

            roi_rstream = filter_streamlines(roi_mask_right, streamlines, world_coords=True, include='only_mask')
            print('Right side done!')

            roi_lstream = filter_streamlines(roi_mask_left, streamlines, world_coords=True, include='only_mask')
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

