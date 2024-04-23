import re, os
import nibabel as nib
from dipy.tracking import utils
from DTC.file_manager.computer_nav import load_trk_remote
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
from DTC.file_manager.file_tools import mkcdir
import numpy as np
import pandas as pd
from dipy.segment.bundles import bundle_shape_similarity
from dipy.tracking.streamline import transform_streamlines, cluster_confidence
from dipy.tracking.utils import length as tract_length
import warnings, socket
import glob
from dipy.io.streamline import load_tractogram


# volume_excel_path = '/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/density_maps/volume_tract_summary.xlsx'
# volume_excel_path = '/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/stats/Tractometry_'

computer_name = socket.gethostname()

if 'santorini' in computer_name:
    root_folder = '/Volumes/Data/Badea/Lab/'
if 'blade' in computer_name:
    root_folder = '/mnt/munin2/Badea/Lab/'

proj_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/TractSeg_outputs/'

pattern = 'S\d{5}'

folders = os.listdir(proj_folder)

matching_folders = [filename for filename in folders if re.match(pattern, filename)]

stats_summary_folder = os.path.join(proj_folder, 'excel_summary')
mkcdir(stats_summary_folder)

numsl_df = pd.DataFrame(columns=['Subj'])
lensl_df = pd.DataFrame(columns=['Subj'])

volume_excel_path = os.path.join(stats_summary_folder, f'volume_tract_summary.xlsx')
num_streamlines_path = os.path.join(stats_summary_folder, f'numstreamlines_summary.xlsx')
len_streamlines_path = os.path.join(stats_summary_folder, f'lenstreamlines_summary.xlsx')
BUAN_summary_path = os.path.join(stats_summary_folder, f'BUAN_summary.xlsx')

overwrite = False

"""
for folder_name in matching_folders:

    folder_path = os.path.join(proj_folder,folder_name)
    #trk_folder = os.path.join(proj_folder, 'trk_roi_ratio_100/')

    densitymap_folder = os.path.join(folder_path, 'density_maps')

    mkcdir([densitymap_folder])

    subj_name = folder_name

    fa_file = os.path.join(folder_path,f'{folder_name}_MNI_fa.nii.gz')

    fa_nii = nib.load(fa_file)
    fa_shape = fa_nii.shape
    # trk_pattern = '_\d+_\d+_\d+'
    # trk_pattern = '_\d+_\d+_\d+'

    tck_folder = os.path.join(folder_path,'TOM_trackings')
    tck_files = glob.glob(os.path.join(tck_folder,'*.tck'))

    #pattern = 'S\d{5}_bundle_[a-zA-Z0-9]{3,5}' + tck_pattern + '.tck'

    overwrite_denmaps = False

    # np.max([int(matching_file.split('.trk')[0][-1]) for matching_file in matching_files])+1

    for tck_file in tck_files:

        streamline_name = os.path.basename(tck_file.split('.tck')[0])
        density_map_path = os.path.join(densitymap_folder, f'{streamline_name}_dm.nii.gz')
        density_map_mask_path = os.path.join(densitymap_folder, f'{streamline_name}_dm_binary.nii.gz')

        if not os.path.exists(density_map_path) or not os.path.exists(density_map_mask_path) or \
                not os.path.exists(num_streamlines_path) or not os.path.exists(len_streamlines_path) or overwrite:
            streamlines_data = load_tractogram(tck_file, fa_file)

            header = streamlines_data.space_attributes
            streamlines = streamlines_data.streamlines
            affine = streamlines_data.affine

            if not os.path.exists(density_map_path) or not os.path.exists(density_map_mask_path) or overwrite:
                dm = utils.density_map(streamlines, affine, fa_shape)

                threshold = 0.5
                binary_dm = (dm > threshold).astype(int)

                if not os.path.exists(density_map_path) or overwrite_denmaps:
                    save_nifti(density_map_path, dm.astype("int16"), affine)
                if not os.path.exists(density_map_mask_path) or overwrite_denmaps:
                    save_nifti(density_map_mask_path, binary_dm.astype("int16"), affine)

            if not os.path.exists(num_streamlines_path) or not os.path.exists(len_streamlines_path) or overwrite:

                ROI_name = streamline_name

                if not os.path.exists(num_streamlines_path) or overwrite:

                    num_streamlines = len(streamlines)

                    if ROI_name not in numsl_df.columns:
                        # Add an empty column if it doesn't exist
                        numsl_df[ROI_name] = 0
                        # numsl_df[bundle_name].astype(int)

                    if subj_name not in numsl_df['Subj'].values:
                        numsl_df = pd.concat([numsl_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

                    row_index = numsl_df.index[numsl_df['Subj'] == subj_name]
                    col_index = numsl_df.columns.get_loc(ROI_name)
                    numsl_df.iloc[row_index, col_index] = int(np.round(num_streamlines))

                if not os.path.exists(len_streamlines_path) or overwrite:
                    avg_streamlines = np.mean(list(tract_length(streamlines)))

                    if ROI_name not in lensl_df.columns:
                        # Add an empty column if it doesn't exist
                        lensl_df[ROI_name] = 0
                        # lensl_df[ROI_name].astype(int)

                    if subj_name not in lensl_df['Subj'].values:
                        lensl_df = pd.concat([lensl_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

                    row_index = lensl_df.index[lensl_df['Subj'] == subj_name]
                    col_index = lensl_df.columns.get_loc(ROI_name)
                    if len(streamlines) == 0:
                        warnings.warn(f'Empty file {tck_file}')
                        lensl_df.iloc[row_index, col_index] = 0
                    else:
                        lensl_df.iloc[row_index, col_index] = int(np.round(avg_streamlines))

if not os.path.exists(num_streamlines_path) or overwrite:

    for col in list(numsl_df.columns):
        if col != 'Subj':
            numsl_df[col].astype(int)

    numsl_df.to_excel(num_streamlines_path, index=False)

if not os.path.exists(len_streamlines_path) or overwrite:

    for col in list(lensl_df.columns):
        if col != 'Subj':
            lensl_df[col].astype(int)

    lensl_df.to_excel(len_streamlines_path, index=False)
"""

if not os.path.exists(volume_excel_path) or overwrite:
    vol_df = pd.DataFrame(columns=['Subj'])
    for folder_name in matching_folders:
        folder_path = os.path.join(proj_folder, folder_name)

        densitymap_folder = os.path.join(folder_path, 'density_maps')

        tck_folder = os.path.join(folder_path, 'TOM_trackings')
        tck_files = glob.glob(os.path.join(tck_folder, '*.tck'))

        subj_name = folder_name

        for file_name in tck_files:
            ROI_name = os.path.basename(file_name.split('.tck')[0])
            density_map_mask_path = os.path.join(densitymap_folder, f'{ROI_name}_dm_binary.nii.gz')

            density_data, density_affine = load_nifti(density_map_mask_path)

            vox_size = nib.load(density_map_mask_path).header.get_zooms()
            vox_size = [np.round(vox, 2) for vox in vox_size]
            subj_bundle_volume = np.sum(density_data) * vox_size[0] * vox_size[1] * vox_size[2]

            if ROI_name not in vol_df.columns:
                # Add an empty column if it doesn't exist
                vol_df[ROI_name] = 0
                # vol_df[ROI_name].astype(int)

            if subj_name not in vol_df['Subj'].values:
                vol_df = pd.concat([vol_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

            row_index = vol_df.index[vol_df['Subj'] == subj_name]
            col_index = vol_df.columns.get_loc(ROI_name)
            vol_df.iloc[row_index, col_index] = int(np.round(subj_bundle_volume))

    for col in list(vol_df.columns):
        if col != 'Subj':
            vol_df[col].astype(int)

    vol_df.to_excel(volume_excel_path, index=False)


if not os.path.exists(BUAN_summary_path) or overwrite:

    BUAN_df = pd.DataFrame(columns=['Subj'])

    for folder_name in matching_folders:

        subj_name = folder_name
        folder_path = os.path.join(proj_folder, folder_name)

        tck_folder = os.path.join(folder_path, 'TOM_trackings')
        tck_files = glob.glob(os.path.join(tck_folder, '*.tck'))

        tck_files_left = [tck_file for tck_file in tck_files if 'left' in tck_file]

        affine_flip = np.eye(4)
        affine_flip[0, 0] = -1
        affine_flip[0, 3] = 0

        rng = np.random.RandomState()
        clust_thr = [5, 3, 1.5]
        threshold = 6
        fa_file = os.path.join(folder_path, f'{folder_name}_MNI_fa.nii.gz')

        for file_name_left in tck_files_left:

            streamline_name_left = file_name_left.split('.tck')[0]
            streamline_name_right = streamline_name_left.replace('left', 'right')

            file_name_right = file_name_left.replace('left', 'right')
            tck_file_left = os.path.join(tck_folder, file_name_left)
            tck_file_right = os.path.join(tck_folder, file_name_right)

            subj_name = re.search('S\d{5}', streamline_name_left)[0]
            ROI_name = os.path.basename(file_name_left.split('.tck')[0])

            streamlines_data_left = load_tractogram(tck_file_left, fa_file)

            header = streamlines_data_left.space_attributes
            streamlines_left = streamlines_data_left.streamlines
            streamlines_right = load_tractogram(tck_file_right, fa_file).streamlines
            affine = streamlines_data_left.affine

            streamlines_right_flipped = transform_streamlines(streamlines_right, affine_flip,
                                                              in_place=False)

            BUAN_id = bundle_shape_similarity(streamlines_left, streamlines_right_flipped, rng, clust_thr, threshold)

            if ROI_name not in BUAN_df.columns:
                # Add an empty column if it doesn't exist
                BUAN_df[ROI_name] = None
                # BUAN_df[ROI_name].astype(int)

            if subj_name not in BUAN_df['Subj'].values:
                BUAN_df = pd.concat([BUAN_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

            row_index = BUAN_df.index[BUAN_df['Subj'] == subj_name]
            col_index = BUAN_df.columns.get_loc(ROI_name)
            BUAN_df.iloc[row_index, col_index] = (np.round(BUAN_id, 3))

    BUAN_df.to_excel(BUAN_summary_path, index=False)
