# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
from os.path import expanduser, join
import os, fury
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
import pandas as pd
from dipy.viz import window, actor
from dipy.segment.bundles import bundle_shape_similarity
import pickle
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, remote_pickle
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from dipy.tracking.streamline import transform_streamlines
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
from dipy.tracking.streamline import set_number_of_points
from DTC.tract_manager.tract_save import save_trk_header
import time
import nibabel as nib
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
from dipy.segment.clustering import ClusterCentroid, ClusterMapCentroid
import warnings
from dipy.segment.bundles import bundle_shape_similarity
from DTC.tract_manager.tract_handler import ratio_to_str

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


MDT_mask_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT'
project = 'AD_Decode'
remote = True

if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'], 'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, _, _, sftp = get_mainpaths(remote, project=project, username=username, password=passwd)

if project == 'AD_Decode':
    TRK_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT_real_testtemp'

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
    template_subjects = ['S01912', 'S02110', 'S02224', 'S02227']
    template_subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361',
                'S02363',
                'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451', 'S02469',
                'S02473',
                'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654', 'S02666', 'S02670',
                'S02686',
                'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771', 'S02781',
                'S02802']
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

streamlines_template = {}
num_streamlines_right_all = 0

num_points = 50

combined_trk_folder = os.path.join(outpath_trk, 'combined_TRK')
mkcdir(combined_trk_folder, sftp)
trktemplate_paths = {}
trktemplate_paths['right'] = os.path.join(combined_trk_folder, 'streamlines_template_right.trk')
trktemplate_paths['right_f'] = os.path.join(combined_trk_folder, 'streamlines_template_right_f.trk')
trktemplate_paths['left'] = os.path.join(combined_trk_folder, 'streamlines_template_left.trk')

verbose = False
setpoints = True
overwrite = False
saveflip = False

timings = []

streamlines_template['right'] = nib.streamlines.array_sequence.ArraySequence()
streamlines_template['left'] = nib.streamlines.array_sequence.ArraySequence()
streamlines_template['right_f'] = nib.streamlines.array_sequence.ArraySequence()

sides = ['left', 'right_f']

reverseside = {}
reverseside['right_f'] = 'right'

trkpaths = {}
timings = []
timings.append(time.perf_counter())
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

show_all = True
distance_stats = True

show_subj = template_subjects
more_images = True

anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz'

for subject in viewed_subjects:
    outpath_trk_subj_bundleset = os.path.join(outpath_trk, f'{subject}_bundles')
    selected_bundles_subject = {}
    for side in ['right_f', 'left']:
        selected_bundles_subject[side] = []
        # record_path = os.path.join(figures_outpath,
        #                           f'{subject}_{side}side_{num_bundles}_bundles_distance_{str(distance)}{ratiostr}_figure.png')
        for i in np.arange(0,num_bundles):
            filepath_bundle = os.path.join(outpath_trk_subj_bundleset, f'{side}_bundle_{i}.trk')
            if (not checkfile_exists_remote(filepath_bundle, sftp)):
                print(f'run savesplitbundle on side {side} for subject {subject}, missing {filepath_bundle}')
            else:
                streamlines_set = load_trk_remote(filepath_bundle, 'same', sftp)
                timings.append(time.perf_counter())
                print(f'Loaded bundle file {filepath_bundle} for subject {subject}, took '
                      f'{timings[-1] - timings[-2]} seconds')
            new_bundle = qb_test.cluster(streamlines_set.streamlines)[0]
            selected_bundles_subject[side].append(new_bundle)

    if distance_stats:
        threshold = 30
        clust_thr = [3]

        bundle_shape_temp = np.zeros([num_bundles, num_bundles])
        bundle_shape_temp_mdf = np.zeros([num_bundles, num_bundles])
        near_flipb = {}
        similarity_dict_path = os.path.join(combined_trk_folder,'similarity_dict.py')

        from dipy.segment.metric import mdf

        if not checkfile_exists_remote(similarity_dict_path, sftp):
            similarity_dict = {}
            for i in np.arange(num_bundles):
                for j in np.arange(num_bundles):
                    #bundle_shape_temp[i,j] = bundle_shape_similarity(selected_bundles_subject['left'][i], selected_bundles_subject['right_f'][j], None, clust_thr,
                    #                  threshold)
                    bundle_shape_temp_mdf[i, j] = mdf(selected_bundles_subject['left'][i].centroid, selected_bundles_subject['right_f'][j].centroid)
                #similarity_dict['left',i] = np.argmax(bundle_shape_temp[i,:])
                similarity_dict['left',i] = np.argmin(bundle_shape_temp_mdf[i,:])
                pickledump_remote(similarity_dict, similarity_dict_path, sftp)
        else:
            similarity_dict = remote_pickle(similarity_dict_path, sftp=sftp)

        for i in np.arange(num_bundles):
            near_flipb[i] = np.argmin(bundle_shape_temp[i, :])

        ba_scores_all = np.zeros([np.size(viewed_subjects), num_bundles])
        for i, subject in enumerate(viewed_subjects):
            for j in np.arange(num_bundles):
                ba_scores_all[i, j] = bundle_shape_similarity(selected_bundles_subject['left'][i],
                                                              selected_bundles_subject['right_f'][near_flipb[i]], None, clust_thr,
                                                              threshold)

                print("Bundle shape similarity score = ", ba_scores_all)