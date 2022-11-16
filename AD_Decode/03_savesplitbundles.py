# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
import os, fury
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, remote_pickle
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
from DTC.tract_manager.tract_save import save_trk_header
import time
import nibabel as nib
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
from dipy.tracking.streamline import transform_streamlines
import copy
"""
from os.path import expanduser, join
from dipy.segment.clustering import ClusterCentroid, ClusterMapCentroid
import warnings
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.bundles import bundle_shape_similarity
import pickle
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import transform_streamlines
import pandas as pd
"""

def set1(a, b):
    c = [value for value in a if value in b]
    return c


def get_indices(a, b):
    c = [idx for idx, value in enumerate(a) if value in b]
    return c


def set2(a, b):
    return list(set(a) & set(b))


def set3(a, b):
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


def add_value_dict(dict, value=0, ):
    for key in dict.keys():
        print("Key exist, ", end=" ")
        dict.update({key: dict[key]+value })
    else:
        print("Not Exist")


MDT_mask_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT'
project = 'AD_Decode'
remote = True

if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'], 'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, _, _, sftp = get_mainpaths(remote, project=project, username=username, password=passwd)


test=True
if test:
    test_str = '_test'
else:
    test_str = ''
if project == 'AD_Decode':
    TRK_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT_real_testtemp'


    if test:
        template_subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320',
                             'S02361',
                             'S02363',
                             'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451',
                             'S02469',
                             'S02473',
                             'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654', 'S02666',
                             'S02670',
                             'S02686',
                             'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771',
                             'S02781',
                             'S02802']
        template_subjects = ['S01912', 'S02110', 'S02224', 'S02227']
    else:
        template_subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320',
                             'S02361',
                             'S02363',
                             'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451',
                             'S02469',
                             'S02473',
                             'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654', 'S02666',
                             'S02670',
                             'S02686',
                             'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771',
                             'S02781',
                             'S02802',
                             'S02804', 'S02813', 'S02812', 'S02817', 'S02840', 'S02842', 'S02871', 'S02877', 'S02898',
                             'S02926',
                             'S02938',
                             'S02939', 'S02954', 'S02967', 'S02987', 'S03010', 'S03017', 'S03028', 'S03033', 'S03034',
                             'S03045',
                             'S03048',
                             'S03069', 'S03225', 'S03265', 'S03293', 'S03308', 'S03321', 'S03343', 'S03350', 'S03378',
                             'S03391',
                             'S03394', 'S03847']  # , 'S03866', 'S03867', 'S03889', 'S03890', 'S03896']
    viewed_subjects = template_subjects
    removed_list = ['S02230', 'S02654', 'S02490', 'S02523', 'S02745']

    for remove in removed_list:
        if remove in template_subjects:
            template_subjects.remove(remove)
    stepsize = 2
    ratio = 100
    trkroi = ["wholebrain"]
    prune = True
    str_identifier = get_str_identifier(stepsize, ratio, trkroi)
    SAMBA_MDT = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz'

    figures_outpath = '/Users/jas/jacques/Figures_ADDecode'

pickle_folder = os.path.join(inpath, 'pickle_roi')
outpath_trk = os.path.join(inpath, 'trk_roi')
if ratio > 1:
    pickle_folder = pickle_folder + f'_{ratio}'
    outpath_trk = outpath_trk + f'_{ratio}'
    ratiostr = f'_{ratio}'
else:
    ratiostr = ''

srr = StreamlineLinearRegistration()

streams_dict = {}
dict_revtracker = {'right_f': 'right', 'left_f': 'left', 'right': 'right', 'left': 'left', 'combined':'combined'}

streamlines_template = {}
num_streamlines_right_all = 0

num_points = 50
distance = 50
num_bundles = 6


combined_trk_folder = os.path.join(outpath_trk, 'combined_TRK')
mkcdir(combined_trk_folder, sftp)
trktemplate_paths = {}
allsides = ['right', 'right_f', 'left', 'left_f', 'combined']

bundles = {}
selected_bundles = {}
num_streamlines = {}
streams_dict_picklepaths = {}

for side in allsides:
    trktemplate_paths[side] = os.path.join(combined_trk_folder, f'streamlines_template_{side}{test_str}.trk')
    streams_dict_picklepaths[side] = os.path.join(combined_trk_folder, f'streams_dict_{side}{test_str}.py')
    streamlines_template[side] = nib.streamlines.array_sequence.ArraySequence()
    selected_bundles[side] = []

#streams_dict_picklepath = os.path.join(combined_trk_folder,'streams_dict.py')

verbose = False
setpoints = True
overwrite = True
saveflip = False

timings = []
timings.append(time.perf_counter())

bundle_lr_combined = True

if bundle_lr_combined:
    sides = ['combined','left', 'right_f']
else:
    sides = ['right_f', 'left', 'right', 'left_f']

reversesides = ['right_f', 'left_f']

trkpaths = {}

streams_dict = {}
for side in sides:
    if checkfile_exists_remote(streams_dict_picklepaths[dict_revtracker[side]], sftp=sftp):
        streams_dict.update(remote_pickle(streams_dict_picklepaths[dict_revtracker[side]], sftp=sftp))
    else:
        print('hi')

#streams_dict['left_f'] = streams_dict['left']

feature2 = ResampleFeature(nb_points=num_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
qb = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=num_bundles)
qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)
#qb = QuickBundles(metric=metric2, max_nb_clusters=num_bundles)
#qb_test = QuickBundles(metric=metric2, max_nb_clusters=1)

for side in sides:
    if (checkfile_exists_remote(trktemplate_paths[side], sftp)):
        streamlines_template_data = load_trk_remote(trktemplate_paths[side], 'same', sftp)
        if 'header' not in locals():
            header = streamlines_template_data.space_attributes
        streamlines_template[side] = streamlines_template_data.streamlines
        timings.append(time.perf_counter())
        print(f'Loaded {side} side from {trktemplate_paths[side]}, took {timings[-1] - timings[-2]} seconds')
        del streamlines_template_data
    else:
        print(f'run 02_bundle_combined for side {side}')

if bundle_lr_combined:
    side_type = 'bundle_lr'
else:
    side_type = 'bundle_ind'

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

save_img = True

anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz'

for subject in viewed_subjects:
    outpath_trk_subj_bundleset = os.path.join(outpath_trk, f'{subject}_{side_type}_{num_bundles}max')
    mkcdir(outpath_trk_subj_bundleset, sftp)
    if bundle_lr_combined:
        selected_bundles_side = {}
        selected_bundles_side['left'] = []
        selected_bundles_side['right'] = []

        # record_path = os.path.join(figures_outpath,
        #                           f'{subject}_{side}side_{num_bundles}_bundles_distance_{str(distance)}{ratiostr}_figure.png')
        for i, bundle in enumerate(selected_bundles['combined']):
            side = 'combined'
            filepath_bundle = os.path.join(outpath_trk_subj_bundleset, f'{side}_bundle_{i}.trk')
            indices = bundle.indices
            stream_combvals = streams_dict['combined', subject]

            if (not checkfile_exists_remote(filepath_bundle, sftp) or overwrite):
                test = set1(indices, stream_combvals)
                streamlines_set = streamlines_template[side][test]
                save_trk_header(filepath=filepath_bundle, streamlines=streamlines_set, header=header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp)
                timings.append(time.perf_counter())
                print(f'Saved bundle file {filepath_bundle} for subject {subject}, took '
                      f'{timings[-1] - timings[-2]} seconds')
            else:
                streamlines_set = load_trk_remote(filepath_bundle, 'same', sftp)
                timings.append(time.perf_counter())
                print(f'Loaded bundle file {filepath_bundle} for subject {subject}, took '
                      f'{timings[-1] - timings[-2]} seconds')

            side = 'left'
            filepath_bundle = os.path.join(outpath_trk_subj_bundleset, f'{side}_bundle_{i}.trk')
            if (not checkfile_exists_remote(filepath_bundle, sftp) or overwrite):
                test = set1(bundle.indices, stream_combvals[:np.max(streams_dict['left',subject])-np.min(streams_dict['left',subject])+1])
                streamlines_set = streamlines_template['combined'][test]
                save_trk_header(filepath=filepath_bundle, streamlines=streamlines_set, header=header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp)
                timings.append(time.perf_counter())
                print(f'Saved bundle file {filepath_bundle} for subject {subject}, took '
                      f'{timings[-1] - timings[-2]} seconds')
            else:
                streamlines_set = load_trk_remote(filepath_bundle, 'same', sftp)
                timings.append(time.perf_counter())
                print(f'Loaded bundle file {filepath_bundle} for subject {subject}, took '
                      f'{timings[-1] - timings[-2]} seconds')

            new_bundle = qb_test.cluster(streamlines_set)[0]
            selected_bundles_side['left'].append(new_bundle)

            side = 'right'
            filepath_bundle = os.path.join(outpath_trk_subj_bundleset, f'{side}_bundle_{i}.trk')
            if (not checkfile_exists_remote(filepath_bundle, sftp) or overwrite):
                test = set1(bundle.indices, stream_combvals[np.max(streams_dict['left',subject])-np.min(streams_dict['left',subject])+1:])
                streamlines_set = streamlines_template['combined'][test]
                affine_flip = np.eye(4)
                affine_flip[0, 0] = -1
                affine_flip[0, 3] = 0
                streamlines_set = transform_streamlines(streamlines_set, affine_flip,in_place=False)
                save_trk_header(filepath=filepath_bundle, streamlines=streamlines_set, header=header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp)
                timings.append(time.perf_counter())
                print(f'Saved bundle file {filepath_bundle} for subject {subject}, took '
                      f'{timings[-1] - timings[-2]} seconds')
            else:
                streamlines_set = load_trk_remote(filepath_bundle, 'same', sftp)
                timings.append(time.perf_counter())
                print(f'Loaded bundle file {filepath_bundle} for subject {subject}, took '
                      f'{timings[-1] - timings[-2]} seconds')
            if np.size(streamlines_set)>0:
                new_bundle = qb_test.cluster(streamlines_set)[0]
                selected_bundles_side['right'].append(new_bundle)
            else:
                print(f'bad subject {subject}')

        if save_img:
            lut_cmap = None
            coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=num_bundles)
            colorbar = False
            plane = 'y'
            interactive = False
            scene = None
            for side in ['left','right']:
                record_path = os.path.join(figures_outpath,
                                           f'{subject}_{side}_side_{num_bundles}_bundles_distance_'
                                           f'{str(distance)}{ratiostr}_figure.png')
                scene = setup_view(selected_bundles_side[side], colors=coloring_vals, ref=anat_path, world_coords=True,
                                   objectvals=None,
                                   colorbar=colorbar, record=record_path, scene=scene, plane=plane,
                                   interactive=interactive)
    else:
        for side in sides:
            selected_bundles_new = []
            # record_path = os.path.join(figures_outpath,
            #                           f'{subject}_{side}side_{num_bundles}_bundles_distance_{str(distance)}{ratiostr}_figure.png')
            for i, bundle in enumerate(selected_bundles[side]):
                filepath_bundle = os.path.join(outpath_trk_subj_bundleset, f'{side}_bundle_{i}.trk')
                if (not checkfile_exists_remote(filepath_bundle, sftp) or overwrite):
                    test = set1(bundle.indices, streams_dict[dict_revtracker[side], subject])
                    streamlines_set = streamlines_template[side][test]
                    save_trk_header(filepath=filepath_bundle, streamlines=streamlines_set, header=header,
                                    affine=np.eye(4), verbose=verbose, sftp=sftp)
                    timings.append(time.perf_counter())
                    print(f'Saved bundle file {filepath_bundle} for subject {subject}, took '
                          f'{timings[-1] - timings[-2]} seconds')
                else:
                    streamlines_set = load_trk_remote(filepath_bundle, 'same', sftp)
                    timings.append(time.perf_counter())
                    print(f'Loaded bundle file {filepath_bundle} for subject {subject}, took '
                          f'{timings[-1] - timings[-2]} seconds')
                new_bundle = qb_test.cluster(streamlines_set)[0]
                selected_bundles_new.append(new_bundle)

        if save_img:
            lut_cmap = None
            coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=num_bundles)
            colorbar = False
            plane = 'y'
            interactive = False
            scene = None
            record_path = os.path.join(figures_outpath,
                                       f'{subject}_{side}_side_{num_bundles}_bundles_distance_'
                                       f'{str(distance)}{ratiostr}_figure.png')
            scene = setup_view(selected_bundles_new, colors=coloring_vals, ref=anat_path, world_coords=True,
                               objectvals=None,
                               colorbar=colorbar, record=record_path, scene=scene, plane=plane,
                               interactive=True)

"""
if show_all:
    lut_cmap = None
    coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=num_bundles)
    colorbar = False
    plane = 'y'
    interactive = False
    scene = None

    for side in sides:
        record_path = os.path.join(figures_outpath,
                                   f'template_{side}side_{num_bundles}'
                                   f'_bundles_distance_{str(distance)}{ratiostr}_figure.png')
                                  
        if viewing == 'left':
            selected_bundles = selected_bundles_L
            record_path = os.path.join(figures_outpath,
                                       f'{subject}_leftside_{num_bundles}_bundles_distance_{str(
                                           distance)}{ratiostr}_figure.png')

        if viewing == 'all':
            selected_bundles = selected_bundles_R + selected_bundles_L
            coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=num_bundles * 2)
            record_path = os.path.join(figures_outpath,
                                       f'{subject}_bothside_{num_bundles}_bundles_distance_{str(
                                           distance)}{ratiostr}_figure.png')
            record_path = os.path.join(figures_outpath,
                                       f'{subject}_bothside_{num_bundles}_bundles_distance_{str(
                                           distance)}{ratiostr}_test_maxedout_6_figure.png')
            more_images = True
  
        scene = setup_view(selected_bundles[side], colors=coloring_vals, ref=anat_path, world_coords=True,
                           objectvals=None,
                           colorbar=colorbar, record=record_path, scene=scene, plane=plane, interactive=interactive)
"""