# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
import os, fury, sys

from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote, pickledump_remote, remote_pickle, copy_remotefiles, write_parameters_to_ini, read_parameters_from_ini
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from dipy.tracking.streamline import transform_streamlines
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
from dipy.tracking.streamline import set_number_of_points
from DTC.tract_manager.tract_save import save_trk_header
import time
import nibabel as nib
import copy
import socket
from DTC.tract_manager.tract_handler import ratio_to_str

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

#locals().update(params) #This line will add to the code the variables specified above from the config file, namely

project = params['project']
streamline_type = params['streamline_type']
test = params['test']
ratio = params['ratio']
stepsize = params['stepsize']
template_subjects = params['template_subjects']
setpoints = params['setpoints']
points_resample = int(params['points_resample'])
remote_output = bool(params['remote_output'])

overwrite=False
verbose = False

if remote_output:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

if streamline_type == 'mrtrix':
    prune = False
    trkroi = [""]
else:
    prune = True
    trkroi = ["wholebrain"]

str_identifier = get_str_identifier(stepsize, ratio, trkroi, type=streamline_type)

if 'santorini' in socket.gethostname().split('.')[0]:
    lab_folder = '/Volumes/Data/Badea/Lab'
if 'blade' in socket.gethostname().split('.')[0]:
    lab_folder = '/mnt/munin2/Badea/Lab'

if project == 'AD_Decode':
    SAMBA_MDT = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz')
    MDT_mask_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT')
    ref_MDT_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images')
    anat_path = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz')


outpath, _, _, sftp_out = get_mainpaths(remote_output,project = project, username=username,password=passwd)

ratiostr = ratio_to_str(ratio,spec_all=False)


outpath_all = os.path.join(outpath, 'TRK_bundle_splitter')
proj_path = os.path.join(outpath_all,project_run_identifier)

trk_proj_path = os.path.join(proj_path, 'trk_roi'+ratiostr)

srr = StreamlineLinearRegistration()

streams_dict = {}

streamlines_template = {}

combined_trk_folder = os.path.join(proj_path, 'combined_TRK')
mkcdir([combined_trk_folder],sftp_out)
trktemplate_paths = {}
streams_dict_picklepaths = {}

allsides = ['right', 'right_f', 'left', 'left_f', 'combined']

for side in allsides:
    trktemplate_paths[side] = os.path.join(combined_trk_folder,f'streamlines_template_{side}.trk')
    streams_dict_picklepaths[side] = os.path.join(combined_trk_folder, f'streams_dict_{side}.py')
    streamlines_template[side] = nib.streamlines.array_sequence.ArraySequence()

timings = []

sides = ['left', 'right', 'right_f', 'left_f', 'combined']

reversesides = ['right_f', 'left_f']
dict_revtracker = {'right_f': 'right', 'left_f': 'left', 'right': 'right', 'left': 'left'}

trkpaths = {}

for side in sides:
    streams_dict_side = {}
    timings.append(time.perf_counter())
    if not checkfile_exists_remote(trktemplate_paths[side], sftp_out) \
            or not checkfile_exists_remote(streams_dict_picklepaths[side], sftp_out) or overwrite:
        if side not in reversesides and side != 'combined':
            num_streamlines_all = 0
            i=1
            for subject in template_subjects:
                trkpaths[subject,'right'] = os.path.join(trk_proj_path, f'{subject}_roi_rstream.trk')
                trkpaths[subject,'left'] = os.path.join(trk_proj_path, f'{subject}_roi_lstream.trk')

                if not checkfile_exists_remote(trkpaths[subject, side], sftp_out):
                    print(f'skipped subject {subject}')
                    continue

                if 'header' not in locals():
                    streamlines_temp_data = load_trk_remote(trkpaths[subject, side], 'same', sftp_out)
                    header = streamlines_temp_data.space_attributes
                    streamlines_temp = streamlines_temp_data.streamlines
                    del streamlines_temp_data
                else:
                    streamlines_temp = load_trk_remote(trkpaths[subject, side], 'same', sftp_out).streamlines

                if setpoints:
                    streamlines_template[side].extend(set_number_of_points(streamlines_temp, points_resample))
                else:
                    streamlines_template[side].extend(streamlines_temp)

                num_streamlines_subj = len(streamlines_temp)

                del streamlines_temp

                streams_dict_side[side,subject] = np.arange(num_streamlines_all, num_streamlines_all + num_streamlines_subj)

                if verbose:
                    timings.append(time.perf_counter())
                    print(f'Loaded {side} side of subject {subject} from {trkpaths[subject, side]}, took {timings[-1] - timings[-2]} seconds')
                    print(f'{(1-((np.size(template_subjects)-i)/np.size(template_subjects)))*100}% done, has gone for {timings[-1] - timings[0]}, {(timings[-1] - timings[0])*((np.size(template_subjects)-i)/i)} seconds remaining')

                num_streamlines_all += num_streamlines_subj
                i += 1

            save_trk_header(filepath=trktemplate_paths[side], streamlines=streamlines_template[side], header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp_out)
            timings.append(time.perf_counter())
            print(f'Saved streamlines at {trktemplate_paths[side]}, took {timings[-1] - timings[-2]} seconds')
            pickledump_remote(streams_dict_side, streams_dict_picklepaths[side], sftp_out)
            timings.append(time.perf_counter())
            print(f'Saved dictionary at {streams_dict_picklepaths[side]}, took {timings[-1] - timings[0]} seconds')
        elif side in reversesides:  # if side is right_f
            if (not checkfile_exists_remote(trktemplate_paths[side], sftp_out) or overwrite):
                affine_flip = np.eye(4)
                affine_flip[0, 0] = -1
                affine_flip[0, 3] = 0
                if np.size(streamlines_template[dict_revtracker[side]]) == 0:
                    streamlines_template_data = load_trk_remote(trktemplate_paths[dict_revtracker[side]], 'same', sftp_out)
                    if 'header' not in locals():
                        header = streamlines_template_data.space_attributes
                    streamlines_template[dict_revtracker[side]] = streamlines_template_data.streamlines
                    timings.append(time.perf_counter())
                    print(
                        f'Loaded {dict_revtracker[side]} side from {trktemplate_paths[dict_revtracker[side]]}, took {timings[-1] - timings[-2]} seconds')
                    del streamlines_template_data
                streamlines_template[side] = transform_streamlines(streamlines_template[dict_revtracker[side]], affine_flip,
                                                                   in_place=False)
                if verbose:
                    timings.append(time.perf_counter())
                    print(f'Flipped {dict_revtracker[side]} side, took {timings[-1] - timings[-2]} seconds')
                save_trk_header(filepath=trktemplate_paths[side], streamlines=streamlines_template[side], header=header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp_out)
                timings.append(time.perf_counter())
                print(
                    f'Saved flipped {dict_revtracker[side]} side at {trktemplate_paths[side]}, took {timings[-1] - timings[-2]} seconds')
                copy_remotefiles(streams_dict_picklepaths[dict_revtracker[side]], streams_dict_picklepaths[side], sftp_out)
                timings.append(time.perf_counter())
                print(f'Saved dictionary at {streams_dict_picklepaths[side]} by copying {streams_dict_picklepaths[dict_revtracker[side]]}, took {timings[-1] - timings[-2]} seconds')
        elif side == 'combined':
            streams_dict_temp = {}
            for sidetemp in ['left','right']:
                streams_dict.update(remote_pickle(streams_dict_picklepaths[dict_revtracker[sidetemp]], sftp=sftp_out))
                timings.append(time.perf_counter())
                print(f'Loaded dictionary from {streams_dict_picklepaths[dict_revtracker[sidetemp]]}, took {timings[-1] - timings[-2]} seconds')
                if np.size(streamlines_template[sidetemp]) == 0:
                    streamlines_template_data = load_trk_remote(trktemplate_paths[sidetemp], 'same', sftp_out)
                    if 'header' not in locals():
                        header = streamlines_template_data.space_attributes
                    streamlines_template[sidetemp] = streamlines_template_data.streamlines
                    timings.append(time.perf_counter())
                    print(
                        f'Loaded {dict_revtracker[sidetemp]} side from {trktemplate_paths[sidetemp]}, took {timings[-1] - timings[-2]} seconds')
                    del streamlines_template_data

            for subject in template_subjects:
                #streams_dict_comb[subject] = streams_dict['left',subject] #streams_dict['left','right_f']
                streams_dict_side[side,subject] = np.array(list(streams_dict['left', subject]) + (list(streams_dict['right',subject] + streams_dict['left', template_subjects[-1]][-1])))

            streamlines_comb = copy.copy(streamlines_template['left'])
            streamlines_comb.extend(streamlines_template['right_f'])
            save_trk_header(filepath=trktemplate_paths[side], streamlines=streamlines_comb, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp_out)
            timings.append(time.perf_counter())
            print(f'Saved streamlines at {trktemplate_paths[side]}, took {timings[-1] - timings[-2]} seconds')
            pickledump_remote(streams_dict_side, streams_dict_picklepaths[side], sftp_out)
            timings.append(time.perf_counter())
            print(f'Saved dictionary at {streams_dict_picklepaths[side]}, took {timings[-1] - timings[-2]} seconds')
    else:
        print(f'already wrote {trktemplate_paths[side]} and {streams_dict_picklepaths[side]}')

