# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:05 2022

@author: ruidai
"""

import numpy as np
import os
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
from dipy.tracking.streamline import transform_streamlines, cluster_confidence
from DTC.tract_manager.tract_handler import ratio_to_str
from DTC.tract_manager.tract_handler import gettrkpath
from DTC.nifti_handlers.nifti_handler import get_diff_ref
from DTC.diff_handlers.connectome_handlers.connectome_handler import retweak_points
import xlsxwriter
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
from collections import defaultdict, OrderedDict
from itertools import combinations, groupby
import pandas as pd
import sys, warnings
from dipy.viz import window, actor
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
import nibabel as nib
from dipy.tracking.utils import length as tract_length
from dipy.segment.bundles import bundle_shape_similarity
import argparse
from DTC.wrapper_tools import parse_list_arg
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from dipy.tracking.utils import length


def convert_tck_to_trk(input_file, output_file, ref):
    header = {}

    nii = nib.load(ref)
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    tck = nib.streamlines.load(input_file)
    nib.streamlines.save(tck.tractogram, output_file, header=header)


project = ''
root = f'/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/TractSeg_analysis'
metadata_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx'

if metadata_path.split('.')[1]=='csv':
    master_df = pd.read_csv(metadata_path)
elif metadata_path.split('.')[1]=='xlsx':
    master_df = pd.read_excel(metadata_path)
else:
    txt = f'Unidentifiable data file path {metadata_path}'
    raise Exception(txt)

master_df = master_df.dropna(subset=['MRI_Exam'])
full_subjects_list = list(master_df['MRI_Exam'].astype(int).astype(str))
for i in range(len(full_subjects_list)):
    if len(full_subjects_list[i]) < 4:
        full_subjects_list[i] = '0' + full_subjects_list[i]

full_subjects_list = ['S0' + s for s in full_subjects_list]


stat_folder = os.path.join(root, 'stats')
tract_seg_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/TractSeg_outputs/'
ref_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/TractSeg_outputs/'


#references = ['fa']
references = []
ref='fa'
verbose = False
points_resample = 50

region_list_pairs = ['SLF_II','STR','FX']

sides = ['left','right']

calc_BUAN = True
overwrite = False

bundle_compare_summary = os.path.join(stat_folder,
                                      f'BUAN_wmatter_comparison.xlsx')

for subject in full_subjects_list:

    bundle_data_dic = {}
    missing_data = False

    for region in region_list_pairs:

        if missing_data:
            break

        ref_img_path = os.path.join(ref_folder, subject,f'{subject}_MNI_fa.nii.gz')

        for side in sides:

            trk_reg_subj_path = os.path.join(tract_seg_folder,subject,'TOM_trackings',f'{region}_{side}.trk')
            if not os.path.exists(trk_reg_subj_path):
                tck_reg_subj_path = trk_reg_subj_path.replace('.trk','.tck')
                if not os.path.exists(tck_reg_subj_path):
                    warning_txt = f'tck for subject {subject} at side {side} not found at {tck_reg_subj_path}, ignoring subject {subject}'
                    missing_data = True
                    warnings.warn(warning_txt)
                    break
                convert_tck_to_trk(tck_reg_subj_path,trk_reg_subj_path,ref_img_path)

            column_names = ['Streamline_ID']
            for ref in references:
                #if ref not in unique_refs:
                column_names += ([f'point_{ID}_{ref}' for ID in np.arange(points_resample)])
                #if ref in unique_refs:
                #    column_names += ([ref])

            print(f'Files will be saved at {stat_folder}')

            bundle_data = load_trk_remote(trk_reg_subj_path, 'same', None)
            bundle_data_dic[region,side] = bundle_data
            bundle_streamlines = bundle_data.streamlines
            num_streamlines = len(bundle_streamlines)
            length_streamlines = list(length(bundle_streamlines))
            header = bundle_data.space_attributes

            dataf_subj = pd.DataFrame(columns=column_names)

            dataf_subj['Streamline_ID'] = np.arange(num_streamlines)

            for ref in references:

                if ref == 'Length':

                    column_indices = dataf_subj.columns.get_loc('Length')
                    dataf_subj.iloc[:, column_indices] = list(tract_length(bundle_streamlines[:]))

                elif ref == 'CCI':
                    column_indices = dataf_subj.columns.get_loc('CCI')
                    # print(tract_length(bundle_streamlines))
                    # print([[i] for i in tract_length(bundle_streamlines)])
                    try:
                        cci = cluster_confidence(bundle_streamlines, override=True)
                        dataf_subj.iloc[:, column_indices] = cci
                    except TypeError:
                        warningstxt = f'Could not work for subject {subject} at bundle_id {region}'
                        warnings.warn(warningstxt)
                else:
                    ref_img_path = os.path.join(ref_folder, subject, f'{subject}_MNI_{ref}.nii.gz')
                    ref_data, ref_affine, _, _, _ = load_nifti_remote(ref_img_path, sftp=None)

                    bundle_streamlines_transformed = transform_streamlines(bundle_streamlines,
                                                                           np.linalg.inv(ref_affine))

                    edges = np.ndarray(shape=(3, 0), dtype=int)
                    lin_T, offset = _mapping_to_voxel(bundle_data.space_attributes[0])
                    # stream_ref = []
                    # stream_point_ref = []
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

                        # new_row = {'Streamline_ID': sl}
                        column_names_ref = [f'point_{i}_{ref}' for i, _ in enumerate(ref_values)]
                        row_index = dataf_subj.index[dataf_subj['Streamline_ID'] == sl].tolist()[0]
                        column_indices = [dataf_subj.columns.get_loc(col) for col in column_names_ref]
                        dataf_subj.iloc[row_index, column_indices] = ref_values
                        # dataf_subj.loc[dataf_subj[row_index],column_indices] = ref_values
                        # dataf_subj[dataf_subj['Streamline_ID'] == sl][column_names_ref] = ref_values
                        # new_row.update({f'point_{i}_{ref}': value for i, value in enumerate(ref_values)})
                        # new_row.update({'Length':list(tract_length(bundle_streamlines[sl:sl+1]))[0]})
                        # dataf_subj.loc[np.shape(dataf_subj)[0][]] = new_row

                save_df_remote(dataf_subj, stat_path_subject, sftp_out)
                print(f'Wrote file for subject {subject} and {full_bundle_id}')

    if not missing_data and calc_BUAN and (not os.path.exists(bundle_compare_summary) or overwrite):
        BUANs = []
        #column_names_ref = [f'BUAN_{bundle_id_orig_txt + f"_{new_bundle_id}"}' for new_bundle_id in new_bundle_ids]

        BUAN_ids = {}

        affine_flip = np.eye(4)
        affine_flip[0, 0] = -1
        affine_flip[0, 3] = 0

        for region in region_list_pairs:
            rng = np.random.RandomState()
            clust_thr = [5, 3, 1.5]
            threshold = 6

            streamlines_left = bundle_data_dic[region, 'left'].streamlines
            streamlines_right = bundle_data_dic[region, 'right'].streamlines
            streamlines_right_flipped = transform_streamlines(streamlines_right, affine_flip,
                                                              in_place=False)
            BUAN_id = bundle_shape_similarity(streamlines_left, streamlines_right_flipped, rng, clust_thr, threshold)

            BUAN_ids[region] = (BUAN_id)

        subj_data = {'Subject': subject, **{f'BUAN_{region}': BUAN_ids[region] for region in region_list_pairs}}
        if not 'BUAN_df' in locals():
            BUAN_df = pd.DataFrame.from_records([subj_data])
        else:
            BUAN_df = pd.concat([BUAN_df, pd.DataFrame.from_records([subj_data])], ignore_index=True)


if np.size(full_subjects_list) > 0 and calc_BUAN and (not os.path.exists(bundle_compare_summary) or overwrite):
    save_df_remote(BUAN_df, bundle_compare_summary, None)
