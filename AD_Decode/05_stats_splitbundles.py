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
from DTC.tract_manager.DTC_manager import get_str_identifier
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    checkfile_exists_all, save_df_remote, write_parameters_to_ini, read_parameters_from_ini
from DTC.file_manager.file_tools import mkcdir, getfromfile
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
import socket
import nibabel as nib
from dipy.tracking.streamline import transform_streamlines
from DTC.tract_manager.tract_handler import ratio_to_str
from DTC.tract_manager.tract_handler import gettrkpath
from DTC.nifti_handlers.nifti_handler import get_diff_ref
from DTC.diff_handlers.connectome_handlers.connectome_handler import retweak_points
import xlsxwriter
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
from collections import defaultdict, OrderedDict
from itertools import combinations, groupby
import pandas as pd
import sys
from dipy.viz import window, actor
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
import nibabel as nib
from dipy.tracking.utils import length as tract_length

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


project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/Bundle_project_headfile'

if len(sys.argv)<2:
    project_run_identifier = '202311_10template_test01'
else:
    project_run_identifier = sys.argv[1]

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
references = params['references']

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


if streamline_type == 'mrtrix':
    prune = False
    trkroi = [""]
else:
    prune = True
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

num_points = 50
distance = 50
num_bundles = 6

feature2 = ResampleFeature(nb_points=num_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)

combined_trk_folder = os.path.join(proj_path, 'combined_TRK')

centroids_sidedic = {}
centroids_all = []
centroid_all_side_tracker = {}
streamline_bundle = {}
centroids = {}

#full_subjects_list = ['S02373']

verbose = False
overwrite=False

clustering = False

save_img = True

qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)

right_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_right.nii.gz')
left_mask_path = os.path.join(MDT_mask_folder, 'IITmean_RPI_MDT_mask_left.nii.gz')

roi_mask_right = nib.load(right_mask_path)
roi_mask_left = nib.load(left_mask_path)

#full_subjects_list = ['S02390']

if len(sys.argv) >2:
    full_subjects_list = [sys.argv[2]]
    sides = [sys.argv[3]]
    bundle_ids = [sys.argv[4]]
else:
    full_subjects_list = template_subjects + added_subjects
    sides = ['left', 'right']
    bundle_ids = np.arange(num_bundles)

for remove in removed_list:
    if remove in full_subjects_list:
        full_subjects_list.remove(remove)


for subject in full_subjects_list:

    files_subj = []
    for side, bundle_id in streamline_bundle.keys():
        files_subj.append(os.path.join(trk_proj_path, f'{subject}_{side}_bundle_{bundle_id}.trk'))
    check_all = checkfile_exists_all(files_subj,sftp_out)
    #filepath_bundle = os.path.join(trk_proj_path, f'{subject}_{side}_bundle_{bundle_id}.trk')
    if not check_all:
        print(f'Missing trk files for subject {subject} in {trk_proj_path}, please rerun bundle creator')
        continue
    subj_trk, trkexists = gettrkpath(path_TRK, subject, str_identifier, pruned=prune, verbose=False, sftp=sftp_in)

    column_names = ['Streamline_ID']
    for ref in references:
        column_names+=([f'point_{ID}_fa' for ID in np.arange(num_points)])

    print(f'Files will be saved at {stat_folder}')
    for side in sides:
        for bundle_id in bundle_ids:
            stat_path_subject = os.path.join(stat_folder, f'{subject}_{side}_bundle_{bundle_id}.xlsx')

            if not overwrite and checkfile_exists_remote(stat_path_subject, sftp_out):
                print(f'Already created file for subject {subject}, side {side} and {bundle_id}')
                continue
    
            dataf_subj = pd.DataFrame(columns=column_names)
    
            filepath_bundle = os.path.join(trk_proj_path, f'{subject}_{side}_bundle_{bundle_id}.trk')
            bundle_data = load_trk_remote(filepath_bundle, 'same', sftp_in)
    
            bundle_streamlines = bundle_data.streamlines
            num_streamlines = np.shape(bundle_streamlines)[0]
            header = bundle_data.space_attributes
    
            dataf_subj['Streamline_ID'] = np.arange(num_streamlines)
            #dataf_subj.set_index('Streamline_ID', inplace=True)
    
            #workbook = xlsxwriter.Workbook(stat_path_subject)
            #worksheet = workbook.add_worksheet()
    
            for ref in references:
                if ref=='ln':
                    dataf_subj[dataf_subj['Streamline_ID'] == sl][column_names_ref] = ref_values
                    dataf_subj['Length'] = list(tract_length(bundle_streamlines[:]))
                if ref != 'ln':
                    ref_img_path = get_diff_ref(ref_MDT_folder, subject, ref, sftp=None)
                    ref_data, ref_affine, _, _, _ = load_nifti_remote(ref_img_path, sftp=None)
    
                    bundle_streamlines_transformed = transform_streamlines(bundle_streamlines,
                                                                           np.linalg.inv(ref_affine))
    
                    edges = np.ndarray(shape=(3, 0), dtype=int)
                    lin_T, offset = _mapping_to_voxel(bundle_data.space_attributes[0])
                    stream_ref = []
                    stream_point_ref = []
                    from time import time
    
                    time1 = time()
                    testmode = False
    
                    for sl, _ in enumerate(bundle_streamlines_transformed):
                        # Convert streamline to voxel coordinates
                        # entire = _to_voxel_coordinates(target_streamlines_set[sl], lin_T, offset)
    
                        voxel_coords = np.round(bundle_streamlines_transformed[sl]).astype(int)
                        voxel_coords_tweaked = retweak_points(voxel_coords, np.shape(ref_data))
                        ref_values = ref_data[
                            voxel_coords_tweaked[:, 0], voxel_coords_tweaked[:, 1], voxel_coords_tweaked[:, 2]]
    
                        stream_point_ref.append(ref_values)
                        stream_ref.append(np.mean(ref_values))
    
                        if np.mean(ref_values) == 0:
                            if verbose:
                                print('too low a value for new method')
                            testmode = True
    
                        if testmode:
                            from DTC.tract_manager.tract_save import save_trk_header
    
    
                            streamline_file_path = os.path.join(small_streamlines_testzone,
                                                                f'{subject}_streamline_{sl}.trk')
                            # sg = lambda: (s for i, s in enumerate(trkobject[0]))
                            from dipy.tracking import streamline
    
                            streamlines = streamline.Streamlines([bundle_streamlines[sl]])
                            save_trk_header(filepath=streamline_file_path, streamlines=streamlines, header=header,
                                            affine=np.eye(4), verbose=verbose, sftp=None)
                            testmode = False
    
                            lut_cmap = actor.colormap_lookup_table(
                                scale_range=(0.05, 0.3))
    
                            #scene = setup_view(nib.streamlines.ArraySequence(bundle_streamlines_transformed), colors=lut_cmap,
                            #                   ref=ref_img_path, world_coords=False,
                            #                   objectvals=[None], colorbar=True, record=None, scene=None, interactive=True, value_range = (0,0.8))
    
                        #new_row = {'Streamline_ID': sl}
                        column_names_ref = [f'point_{i}_{ref}' for i,_ in enumerate(ref_values)]
                        row_index = dataf_subj.index[dataf_subj['Streamline_ID'] == sl].tolist()[0]
                        column_indices = [dataf_subj.columns.get_loc(col) for col in column_names_ref]
                        dataf_subj.iloc[row_index, column_indices] = ref_values
                        #dataf_subj.loc[dataf_subj[row_index],column_indices] = ref_values
                        #dataf_subj[dataf_subj['Streamline_ID'] == sl][column_names_ref] = ref_values
                        #new_row.update({f'point_{i}_{ref}': value for i, value in enumerate(ref_values)})
                        #new_row.update({'Length':list(tract_length(bundle_streamlines[sl:sl+1]))[0]})
                        #dataf_subj.loc[np.shape(dataf_subj)[0][]] = new_row
    
            save_df_remote(dataf_subj,stat_path_subject,sftp_out)
            print(f'Wrote file for subject {subject}, side {side} and {bundle_id}')