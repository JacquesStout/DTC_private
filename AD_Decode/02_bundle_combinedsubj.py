# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
import os, fury

from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote, pickledump_remote, remote_pickle, copy_remotefiles
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
"""
import pandas as pd
from dipy.segment.bundles import bundle_shape_similarity
import pickle
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
from  dipy.segment.clustering import ClusterCentroid, ClusterMapCentroid
import warnings
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from os.path import expanduser, join
"""

def set1(a,b):
    c = [value for value in a if value in b]
    return c


def get_indices(a,b):
    c = [idx for idx,value in enumerate(a) if value in b]
    return c


def set2(a,b):
    return list(set(a) & set(b))


def set3(a,b):
    c = set(a).intersection(b)
    return c


def show_both_bundles(bundles, colors=None, show=True, fname=None):

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
        lines_actor.RotateX(180)
        lines_actor.RotateZ(180)
        scene.add(lines_actor)
    if show:
        window.show(scene)
    if fname is not None:
        sleep(1)
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


MDT_mask_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT'
project='AD_Decode'
remote=True

if 'samos' in socket.gethostname():
    remote=False
else:
    remote=True

if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, _, _, sftp_in = get_mainpaths(remote,project = project, username=username,password=passwd)

group = 'test'
if group == 'test':
    group_str = '_test'
else:
    test_str = ''

project='AD_Decode'
type = 'mrtrix'
if project=='AD_Decode':
    #TRK_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT_real_testtemp'
    #TRK_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT'

    if group == 'test':
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

if group == 'test':
    sftp_out = None
    outpath = '/Users/jas/jacques/AD_Decode/AD_Decode_bundlesplit/Test'
else:
    sftp_out = sftp_in
    outpath = inpath

ratiostr = ratio_to_str(ratio,spec_all=False)

#path_TRK = os.path.join(inpath, 'TRK_MPCA_MDT'+ratio_str)
path_TRK = os.path.join(inpath, 'TRK_MDT'+ratio_str)
proj_path = os.path.join(inpath, 'TRK_bundle_splitter')

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratio_str)
inpath_trk = os.path.join(proj_path, 'trk_roi'+ratio_str)
outpath_trk = os.path.join(proj_path, 'trk_roi'+ratio_str)

srr = StreamlineLinearRegistration()

streams_dict = {}

streamlines_template = {}
num_streamlines_right_all = 0

num_points = 50

combined_trk_folder = os.path.join(outpath, 'combined_TRK')
mkcdir([outpath_trk,combined_trk_folder],sftp_out)
trktemplate_paths = {}
streams_dict_picklepaths = {}

allsides = ['right', 'right_f', 'left', 'left_f', 'combined']

for side in allsides:
    trktemplate_paths[side] = os.path.join(combined_trk_folder,f'streamlines_template_{side}{test_str}.trk')
    streams_dict_picklepaths[side] = os.path.join(combined_trk_folder, f'streams_dict_{side}{test_str}.py')
    streamlines_template[side] = nib.streamlines.array_sequence.ArraySequence()

verbose=True
setpoints=True
overwrite=False
saveflip = False

timings = []

sides = ['left', 'right', 'right_f', 'left_f', 'combined']

reversesides = ['right_f', 'left_f']
dict_revtracker = {'right_f': 'right', 'left_f': 'left', 'right': 'right', 'left': 'left'}

ratio_str = ratio_to_str(ratio)
print(ratio_str)

trkpaths = {}

for side in sides:
    streams_dict_side = {}
    timings.append(time.perf_counter())
    if not checkfile_exists_remote(trktemplate_paths[side], sftp_out) \
            or not checkfile_exists_remote(streams_dict_picklepaths[side], sftp_out) or overwrite:
        if side not in reversesides and side is not 'combined':
            num_streamlines_all = 0
            i=1
            for subject in template_subjects:
                trkpaths[subject,'right'] = os.path.join(inpath_trk, f'{subject}_roi_rstream.trk')
                trkpaths[subject,'left'] = os.path.join(inpath_trk, f'{subject}_roi_lstream.trk')

                if not checkfile_exists_remote(trkpaths[subject, side], sftp_in):
                    print(f'skipped subject {subject}')
                    continue

                if 'header' not in locals():
                    streamlines_temp_data = load_trk_remote(trkpaths[subject, side], 'same', sftp_in)
                    header = streamlines_temp_data.space_attributes
                    streamlines_temp = streamlines_temp_data.streamlines
                    del streamlines_temp_data
                else:
                    streamlines_temp = load_trk_remote(trkpaths[subject, side], 'same', sftp_in).streamlines

                if setpoints:
                    streamlines_template[side].extend(set_number_of_points(streamlines_temp, num_points))
                else:
                    streamlines_template[side].extend(streamlines_temp)

                num_streamlines_subj = np.size(streamlines_temp)

                del streamlines_temp

                #streams_dict[side, subject] = np.arange(num_streamlines_all,
                #                                        num_streamlines_all + num_streamlines_subj)
                streams_dict_side[side,subject] = np.arange(num_streamlines_all, num_streamlines_all + num_streamlines_subj)

                if verbose:
                    timings.append(time.perf_counter())
                    print(f'Loaded {side} side of subject {subject} from {trkpaths[subject, side]}, took {timings[-1] - timings[-2]} seconds')
                    print(f'{(1-((np.size(subjects)-i)/np.size(subjects)))*100}% done, has gone for {timings[-1] - timings[0]}, {(timings[-1] - timings[0])*((np.size(subjects)-i)/i)} seconds remaining')

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
                    streamlines_template_data = load_trk_remote(trktemplate_paths[dict_revtracker[side]], 'same', sftp_in)
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
                    streamlines_template_data = load_trk_remote(trktemplate_paths[sidetemp], 'same', sftp_in)
                    if 'header' not in locals():
                        header = streamlines_template_data.space_attributes
                    streamlines_template[sidetemp] = streamlines_template_data.streamlines
                    timings.append(time.perf_counter())
                    print(
                        f'Loaded {dict_revtracker[sidetemp]} side from {trktemplate_paths[sidetemp]}, took {timings[-1] - timings[-2]} seconds')
                    del streamlines_template_data

            for subject in subjects:
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
#pickledump_remote(streams_dict,streams_dict_picklepath, sftp)

"""
    else:
        streamlines_template_data = load_trk_remote(streamlines_template[side],'same', sftp)
        if 'header' not in locals():
            header = streamlines_template_data.space_attributes
        streamlines_template[side] = streamlines_template_data.streamlines
        timings.append(time.perf_counter())
        print(
            f'Loaded {side} side from {streamlines_template[side]}, took {timings[-1] - timings[-2]} seconds')
        del streamlines_template_data
"""


"""
distance = 50
num_bundles = 6

feature2 = ResampleFeature(nb_points=num_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
qb = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=num_bundles)
qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)

bundles = {}
selected_bundles = {}
num_streamlines = {}

selected_bundles['right'] = []
selected_bundles['left'] = []
selected_bundles['right_f'] = []

for side in sides:
    bundles = qb.cluster(streamlines_template[side])
    num_streamlines[side] = bundles.clusters_sizes()
    print(f'side {side} has {num_streamlines[side]}')
    top_bundles = sorted(range(len(num_streamlines[side])), key=lambda i: num_streamlines[side][i], reverse=True)[:]
    for bundle in top_bundles[:num_bundles]:
        selected_bundles[side].append(bundles.clusters[bundle])
    del bundles, top_bundles

    timings.append(time.perf_counter())
    print(f'Organized top {num_bundles} bundles of {side} side, took {timings[-1] - timings[-2]} seconds')


show_all = True
distance_stats = True

show_subj = template_subjects
more_images = True

anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz'


for subject in viewed_subjects:
    outpath_trk_subj_bundleset = os.path.join(outpath_trk, f'{subject}_bundles')
    for side in sides:
        selected_bundles_new = []
        #record_path = os.path.join(figures_outpath,
        #                           f'{subject}_{side}side_{num_bundles}_bundles_distance_{str(distance)}{ratiostr}_figure.png')
        for i, bundle in enumerate(selected_bundles[side]):
            filepath_bundle = os.path.join(outpath_trk_subj_bundleset, f'{side}_bundle_{i}.trk')
            if (not checkfile_exists_remote(filepath_bundle, sftp) or overwrite):
                test = set1(bundle.indices, streams_dict[side, subject])
                streamlines_set = streamlines_template[side][test]
                save_trk_header(filepath=filepath_bundle, streamlines=streamlines_set, header=header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp)
                timings.append(time.perf_counter())
                print(f'Saved bundle file {filepath_bundle} for subject {subject}, took {timings[-1] - timings[-2]} seconds')
            else:
                streamlines_set = load_trk_remote(filepath_bundle, 'same', sftp)
                timings.append(time.perf_counter())
                print(f'Loaded bundle file {filepath_bundle} for subject {subject}, took {timings[-1] - timings[-2]} seconds')
            new_bundle = qb_test.cluster(streamlines_set)[0]
            selected_bundles_new.append(new_bundle)

    if distance_stats:
        threshold = 30
        clust_thr = [3]

        bundle_shape_temp = np.zeros([num_bundles, num_bundles])

        near_flipb = {}

        for i in np.arange(num_bundles):
            for j in np.arange(num_bundles):
                bundle_shape_temp(selected_bundles['right'][i], selected_bundles['right_f'][j], None, clust_thr, threshold)
        for i in np.arange(num_bundles):
            near_flipb[i] = np.argmin(bundle_shape_temp[i,:])

        ba_scores_all = np.zeros([np.size(viewed_subjects), num_bundles])
        for i,subject in enumerate(viewed_subjects):
            for j in np.arange(num_bundles):
                ba_scores_all[i, j] = bundle_shape_similarity(selected_bundles['right'][i], selected_bundles['right'][i], None, clust_thr,
                                                              threshold)
        print("Shape similarity score = ", ba_score)
        print("Bundle shape similarity score = ", ba_scores_all)

if show_all:
    lut_cmap = None
    coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=num_bundles)
    colorbar = False
    plane = 'y'
    interactive = False
    scene = None

    for side in sides:
        record_path = os.path.join(figures_outpath,
                                   f'template_{side}side_{num_bundles}_bundles_distance_{str(distance)}{ratiostr}_figure.png')
        if not np.all(check_files(record_path, None)[1]) or more_images:
            scene = setup_view(selected_bundles[side], colors=coloring_vals, ref=anat_path, world_coords=True, objectvals=None,
                               colorbar=colorbar, record=record_path, scene=scene, plane=plane, interactive=interactive)

if np.size(show_subj)>0:
    for subject in show_subj:
        anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz'

        outpath_trk_subj_bundleset = os.path.join(outpath_trk, f'{subject}_bundles')
        mkcdir(outpath_trk_subj_bundleset, sftp)

        lut_cmap = None
        coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=num_bundles)
        colorbar = False
        plane = 'y'
        interactive = False
        scene = None

        for side in sides:
            selected_bundles_new = []
            record_path =  os.path.join(figures_outpath,
                                       f'{subject}_{side}side_{num_bundles}_bundles_distance_{str(distance)}{ratiostr}_figure.png')
            if not np.all(check_files(record_path, None)[1]) or more_images or interactive:
                for i, bundle in enumerate(selected_bundles[side]):
                    filepath_bundle_temp = os.path.join(outpath_trk_subj_bundleset, f'{side}_bundle_{i}.trk')
                    if (not checkfile_exists_remote(filepath_bundle_temp, sftp) or overwrite):
                        test = set1(bundle.indices, streams_dict[side,subject])
                        streamlines_set = streamlines_template[side][test]
                    else:
                        streamlines_set = load_trk_remote(filepath_bundle_temp,'same',sftp)
                    new_bundle = qb_test.cluster(streamlines_set)[0]
                    selected_bundles_new.append(new_bundle)

                scene = setup_view(selected_bundles_new, colors=coloring_vals, ref=anat_path, world_coords=True, objectvals=None,
                                   colorbar=colorbar, record=record_path, scene=scene, plane=plane, interactive=interactive)
        
"""