# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
import os, fury
import nibabel as nib
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.tract_manager.DTC_manager import get_str_identifier
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    checkfile_exists_all, save_df_remote, load_df_remote, write_parameters_to_ini, read_parameters_from_ini
from DTC.file_manager.file_tools import mkcdir, getfromfile
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
import socket
from DTC.tract_manager.tract_handler import ratio_to_str
import pandas as pd
from dipy.viz import window, actor

from DTC.tract_manager.tract_handler import gettrkpath
from DTC.nifti_handlers.nifti_handler import get_diff_ref
from DTC.diff_handlers.connectome_handlers.connectome_handler import retweak_points
import xlsxwriter
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
from collections import defaultdict, OrderedDict
from itertools import combinations, groupby
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.utils import length as tract_length
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
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


project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/Bundle_project_heafile'
project_run_identifier = '202311_10template_test02_configtest'

project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')

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
added_subjects = params['added_subjects']
removed_list = params['removed_list']
setpoints = params['setpoints']
figures_outpath = params['figures_outpath']
distance = params['distance']
num_points = params['num_points']
num_bundles = params['num_bundles']

overwrite=False
verbose = False

save_img = True

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

trkroi = ["wholebrain"]
str_identifier = get_str_identifier(stepsize, ratio, trkroi, type='mrtrix')

ratiostr = ratio_to_str(ratio,spec_all=False)


if project == 'AD_Decode':
    SAMBA_MDT = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz'
    MDT_mask_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT'
    ref_MDT_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images/'
    anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz'


path_TRK = os.path.join(inpath, 'TRK_MDT'+ratiostr)
outpath_all = os.path.join(inpath, 'TRK_bundle_splitter')
proj_path = os.path.join(outpath_all,project_run_identifier)
figures_proj_path = os.path.join(figures_outpath, project_run_identifier)
small_streamlines_testzone = os.path.join(figures_proj_path,'single_streamlines')

mkcdir([figures_outpath,figures_proj_path, small_streamlines_testzone])

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratiostr)
trk_proj_path = os.path.join(proj_path, 'trk_roi'+ratiostr)
stat_folder = os.path.join(proj_path, 'stats')

mkcdir([stat_folder],sftp_out)

srr = StreamlineLinearRegistration()

streams_dict = {}
dict_revtracker = {'right_f': 'right', 'left_f': 'left', 'right': 'right', 'left': 'left', 'combined':'combined'}

streamlines_template = {}
num_streamlines_right_all = 0

feature2 = ResampleFeature(nb_points=num_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)

combined_trk_folder = os.path.join(proj_path, 'combined_TRK')

sides = ['left','right']
centroids_sidedic = {}
centroids_all = []
centroid_all_side_tracker = {}
streamline_bundle = {}
centroids = {}

full_subjects_list = template_subjects + added_subjects

for remove in removed_list:
    if remove in full_subjects_list:
        full_subjects_list.remove(remove)


qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)

right_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_right.nii.gz')
left_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_left.nii.gz')

roi_mask_right = nib.load(right_mask_path)
roi_mask_left = nib.load(left_mask_path)

references = ['fa']

for side in sides:
    for i in np.arange(num_bundles):
        streamline_bundle[side,i] = []

for side,bundle_id in streamline_bundle.keys():
    stat_path_all = os.path.join(stat_folder, f'allsubj_side_{side}_bundle_{bundle_id}.xlsx')

    for subject in full_subjects_list:
        stat_path_subject = os.path.join(stat_folder, f'{subject}_{side}_bundle_{bundle_id}.xlsx')
        df_subj = load_df_remote(stat_path_subject,sftp_out)
        df_subj['Subject'] = subject
        if 'df_all' not in locals():
            df_all = df_subj
        else:
            df_all = pd.concat([df_all, df_subj], ignore_index=True)

    save_df_remote(df_all,stat_path_all,sftp_out)
    print(f'Saved the concatenated file at {stat_path_all}')