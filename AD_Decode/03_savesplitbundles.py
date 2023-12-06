# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
import os, fury, sys
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, remote_pickle, write_parameters_to_ini, read_parameters_from_ini
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
from DTC.tract_manager.tract_save import save_trk_header
import time, socket
import nibabel as nib
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
from dipy.tracking.streamline import transform_streamlines
from DTC.tract_manager.tract_handler import ratio_to_str
from dipy.segment.metric import mdf

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
import copy
"""


if len(sys.argv)<2:
    project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/BuSA_headfiles'
    #project_run_identifier = '202311_10template_test01'
    project_run_identifier = '202311_10template_1000_newcentroids'
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
#project, streamline_type, text_folder, test, MDT_mask_folder, ratio, stepsize

project = params['project']
streamline_type = params['streamline_type']
test = params['test']
ratio = params['ratio']
stepsize = params['stepsize']
template_subjects = params['template_subjects']
setpoints = params['setpoints']

overwrite=True
verbose = False
saveflip = False

if 'santorini' in socket.gethostname().split('.')[0]:
    lab_folder = '/Volumes/Data/Badea/Lab'
if 'blade' in socket.gethostname().split('.')[0]:
    lab_folder = '/mnt/munin2/Badea/Lab'

if project == 'AD_Decode':
    SAMBA_MDT = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz')
    MDT_mask_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT')
    ref_MDT_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images')
    anat_path = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz')

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
sftp_out = sftp_in



if streamline_type == 'mrtrix':
    prune = False
    trkroi = [""]
else:
    prune = True
    trkroi = ["wholebrain"]

str_identifier = get_str_identifier(stepsize, ratio, trkroi, type=streamline_type)

ratiostr = ratio_to_str(ratio,spec_all=False)


path_TRK = os.path.join(inpath, 'TRK_MDT'+ratiostr)
outpath_all = os.path.join(inpath, 'TRK_bundle_splitter')
proj_path = os.path.join(outpath_all,project_run_identifier)
figures_proj_path = os.path.join(proj_path, 'Figures')
mkcdir([figures_proj_path],sftp_out)

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratiostr)
trk_proj_path = os.path.join(proj_path, 'trk_roi'+ratiostr)


srr = StreamlineLinearRegistration()

streams_dict = {}
dict_revtracker = {'right_f': 'right', 'left_f': 'left', 'right': 'right', 'left': 'left', 'combined':'combined'}

streamlines_template = {}
num_streamlines_right_all = 0

num_points = 50
distance = 50
num_bundles = 6



combined_trk_folder = os.path.join(proj_path, 'combined_TRK')
mkcdir(combined_trk_folder, sftp_out)
trktemplate_paths = {}
allsides = ['right', 'right_f', 'left', 'left_f', 'combined']

bundles = {}
selected_bundles = {}
num_streamlines = {}
streams_dict_picklepaths = {}

for side in allsides:
    trktemplate_paths[side] = os.path.join(combined_trk_folder, f'streamlines_template_{side}.trk')
    streams_dict_picklepaths[side] = os.path.join(combined_trk_folder, f'streams_dict_{side}.py')
    streamlines_template[side] = nib.streamlines.array_sequence.ArraySequence()
    selected_bundles[side] = []

#streams_dict_picklepath = os.path.join(combined_trk_folder,'streams_dict.py')

timings = []
timings.append(time.perf_counter())

bundle_lr_combined = True

if bundle_lr_combined:
    sides = ['combined','left','right']
else:
    sides = ['right_f', 'left', 'right', 'left_f']

reversesides = ['right_f', 'left_f']

trkpaths = {}

streams_dict = {}
for side in sides:
    if checkfile_exists_remote(streams_dict_picklepaths[dict_revtracker[side]], sftp=sftp_out):
        streams_dict.update(remote_pickle(streams_dict_picklepaths[dict_revtracker[side]], sftp=sftp_out))
    else:
        print(f'Could not find {streams_dict_picklepaths[dict_revtracker[side]]}')

#streams_dict['left_f'] = streams_dict['left']

feature2 = ResampleFeature(nb_points=num_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
qb = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=num_bundles)
qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)
#qb = QuickBundles(metric=metric2, max_nb_clusters=num_bundles)
#qb_test = QuickBundles(metric=metric2, max_nb_clusters=1)

for side in sides:
    if (checkfile_exists_remote(trktemplate_paths[side], sftp_out)):
        streamlines_template_data = load_trk_remote(trktemplate_paths[side], 'same', sftp_out)
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

centroids_perside = {}
if bundle_lr_combined:
    bundles = qb.cluster(streamlines_template['combined'])
    num_streamlines[side] = bundles.clusters_sizes()
    print(f'side {side} has {num_streamlines[side]}')

    top_bundles = sorted(range(len(num_streamlines[side])), key=lambda i: num_streamlines[side][i], reverse=True)[:]
    for bundle in top_bundles[:num_bundles]:
        selected_bundles['combined'].append(bundles.clusters[bundle])

    centroids_perside['combined'] = []
    for i in np.arange(num_bundles):
        centroids_perside['combined'].append(selected_bundles['combined'][i].centroid)

    centroids_perside['left'] = centroids_perside['combined']

    centroids_perside['right'] = []

    affine_flip = np.eye(4)
    affine_flip[0, 0] = -1
    affine_flip[0, 3] = 0
    for bundle_toflip in centroids_perside['left']:
        centroids_perside['right'].append(np.array(transform_streamlines(bundle_toflip, affine_flip, in_place=False)))

    """
    bundle_centroids = bundle_centroids_flipped
    for comb_bundle in selected_bundles['combined']:
        dist_min = 100000
        bundle_id = -1
        combcentroid = comb_bundle.centroid
        for i, centroid in enumerate(bundle_centroids):
            dist = (mdf(combcentroid, centroid))
            if dist < dist_min:
                bundle_id = i
                dist_min = dist
    """

    timings.append(time.perf_counter())
    print(f'Organized top {num_bundles} bundles for left and right side, took {timings[-1] - timings[-2]} seconds')

    for side in sides:
        pickled_centroids = os.path.join(pickle_folder, f'bundles_centroids_{side}.py')
        if not checkfile_exists_remote(pickled_centroids, sftp_out) or overwrite:
            # pickledump_remote(bundles.centroids,pickled_centroids,sftp_out)
            pickledump_remote(centroids_perside[side], pickled_centroids, sftp_out)
        timings.append(time.perf_counter())
        print(f'Saved centroids at {pickled_centroids}, took {timings[-1] - timings[-2]} seconds')

    del bundles
else:
    raise Exception('No longer functional for other versions, need to make sure that the sides compared to each other are equivalent '
                    'for example left to right_f, right to left_f, etc')


