# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
import sys
import os, fury
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, remote_pickle, checkfile_exists_all, write_parameters_to_ini, \
    read_parameters_from_ini
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
from DTC.tract_manager.tract_save import save_trk_header
import time, socket
import nibabel as nib
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
from dipy.tracking.streamline import transform_streamlines
from DTC.tract_manager.tract_handler import ratio_to_str
from DTC.tract_manager.tract_handler import gettrkpath
from dipy.segment.metric import mdf
from dipy.tracking.streamline import set_number_of_points
from DTC.tract_manager.tract_to_roi_handler import filter_streamlines


if len(sys.argv)<2:
    project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/BuSA_headfiles'
    project_run_identifier = '202311_10template_test01'
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
added_subjects = params['added_subjects']
setpoints = params['setpoints']
figures_outpath = params['figures_outpath']
distance = params['distance']
removed_list = params['removed_list']
num_points = 50
distance = 50
num_bundles = 6
fury.colormap.distinguishable_colormap(nb_colors=num_bundles)
overwrite=False
verbose = False

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

ratiostr = ratio_to_str(ratio,spec_all=False)

path_TRK = os.path.join(inpath, 'TRK_MDT'+ratiostr)
outpath_all = os.path.join(inpath, 'TRK_bundle_splitter')
proj_path = os.path.join(outpath_all,project_run_identifier)
figures_proj_path = os.path.join(figures_outpath, project_run_identifier)
mkcdir([figures_outpath,figures_proj_path])

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratiostr)
trk_proj_path = os.path.join(proj_path, 'trk_roi'+ratiostr)


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

for side in sides:
    pickled_centroids = os.path.join(pickle_folder, f'bundles_centroids_{side}.py')
    centroids_side = remote_pickle(pickled_centroids,sftp_out,erase_temp=False)
    bundles_num = np.shape(centroids_side)[0]
    centroids[side] = centroids_side
    for i in np.arange(bundles_num):
        #centroids_sidedic[side,i] = centroids_side[i]
        #centroids_all.append(centroids_side[i])
        #centroid_all_side_tracker[np.shape(centroids_all)[0]-1] = (side,i)
        streamline_bundle[side,i] = []

verbose = True
overwrite=False

save_img = False

qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)

right_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_right.nii.gz')
left_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_left.nii.gz')

roi_mask_right = nib.load(right_mask_path)
roi_mask_left = nib.load(left_mask_path)

scene = None
interactive = False

if len(sys.argv) >2:
    full_subjects_list = [sys.argv[2]]
else:
    full_subjects_list = template_subjects + added_subjects

for remove in removed_list:
    if remove in full_subjects_list:
        full_subjects_list.remove(remove)

for subject in full_subjects_list:

    files_subj = []
    for side, bundle_id in streamline_bundle.keys():
        files_subj.append(os.path.join(trk_proj_path, f'{subject}_{side}_bundle_{bundle_id}.trk'))
    check_all = checkfile_exists_all(files_subj,sftp_out)
    #filepath_bundle = os.path.join(trk_proj_path, f'{subject}_{side}_bundle_{bundle_id}.trk')
    if check_all and not overwrite:
        print(f'Already ran succesfully for subject {subject}')
        if verbose:
            print(f'Example of file found {files_subj[0]}')
        continue
    else:
        print(f'Starting run for subject {subject}')
    subj_trk, trkexists = gettrkpath(path_TRK, subject, str_identifier, pruned=prune, verbose=False, sftp=sftp_in)
    streamlines_data = load_trk_remote(subj_trk, 'same', sftp_in)
    if verbose:
        print(f'Finished loading of {subj_trk}')
    header = streamlines_data.space_attributes
    streamlines = streamlines_data.streamlines
    streamlines_side = {}
    streamlines_numpoints = set_number_of_points(streamlines, nb_points=num_points)
    del(streamlines)
    streamlines_side['right'] = filter_streamlines(roi_mask_right, streamlines_numpoints, world_coords=True, include='only_mask')
    streamlines_side['left'] = filter_streamlines(roi_mask_left, streamlines_numpoints, world_coords=True, include='only_mask')
    del(streamlines_numpoints)

    for side in sides:
        for streamline in streamlines_side[side]:
            dist_min = 100000
            bundle_id = -1
            for i,centroid in enumerate(centroids[side]):
                dist = (mdf(streamline, centroid))
                if dist<dist_min:
                    bundle_id = i
                    dist_min = dist
            streamline_bundle[side,bundle_id].append(streamline)
    if verbose:
        print(f'Finished streamline prep')
    for side,bundle_id in streamline_bundle.keys():
        sg = lambda: (s for i, s in enumerate(streamline_bundle[side,bundle_id]))
        filepath_bundle = os.path.join(trk_proj_path, f'{subject}_{side}_bundle_{bundle_id}.trk')
        save_trk_header(filepath=filepath_bundle, streamlines=sg, header=header,
                    affine=np.eye(4), verbose=verbose, sftp=sftp_out)
    if verbose:
        print(f'Finished saving trk files')
    if save_img:
        lut_cmap = None
        coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=num_bundles)
        colorbar = False
        plane = 'y'
        for side in ['left', 'right']:
            streamlines_side = []
            for bundle_id in np.arange(num_bundles):
                new_bundle = qb_test.cluster(streamline_bundle[side,bundle_id])[0]
                streamlines_side.append(new_bundle)
            record_path = os.path.join(figures_proj_path,
                                       f'{subject}_{side}_side_{num_bundles}_bundles_distance_'
                                       f'{str(distance)}{ratiostr}_figure.png')
            scene = setup_view(streamlines_side, colors=coloring_vals, ref=anat_path, world_coords=True,
                               objectvals=None,
                               colorbar=colorbar, record=record_path, scene=scene, plane=plane,
                               interactive=interactive)
    interactive = False
    print(f'Finished for subject {subject}')
