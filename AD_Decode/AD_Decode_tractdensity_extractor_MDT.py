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

#volume_excel_path = '/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/density_maps/volume_tract_summary.xlsx'
#volume_excel_path = '/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/stats/Tractometry_'

computer_name = socket.gethostname()

if 'santorini' in computer_name:
    root_folder = '/Volumes/Data/Badea/Lab/'
if 'blade' in computer_name:
    root_folder = '/mnt/munin2/Badea/Lab/'

trk_folder = os.path.join(root_folder,'AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/trk_roi_ratio_100/')
densitymap_folder = os.path.join(root_folder,'AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/density_maps/')

stats_summary_folder = os.path.join(root_folder,'AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/stats/excel_summary')

mkcdir(densitymap_folder)

label_file = os.path.join(root_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/labels_MDT/MDT_IITmean_RPI_labels.nii.gz')

label_nii = nib.load(label_file)
label_shape = label_nii.shape
#trk_pattern = '_\d+'
trk_pattern = '_\d+_\d+_\d+'
#trk_pattern = '_\d+_\d+_\d+'

files = os.listdir(trk_folder)

pattern = 'S\d{5}_bundle_[a-zA-Z0-9]{3,5}'+trk_pattern+'.trk'

overwrite=True
overwrite_denmaps = False

matching_files = [filename for filename in files if re.match(pattern, filename)]

bundle_names = [matching_file[1].split('.trk')[0][-1] for matching_file in matching_files]

trk_pattern_name = trk_pattern.replace('\d+','all')
#np.max([int(matching_file.split('.trk')[0][-1]) for matching_file in matching_files])+1

volume_excel_path = os.path.join(stats_summary_folder,f'volume_tract_summary{trk_pattern_name}.xlsx')
num_streamlines_path = os.path.join(stats_summary_folder,f'numstreamlines_summary{trk_pattern_name}.xlsx')
len_streamlines_path = os.path.join(stats_summary_folder,f'lenstreamlines_summary{trk_pattern_name}.xlsx')
BUAN_summary_path = os.path.join(stats_summary_folder,f'BUAN_summary{trk_pattern_name}.xlsx')

numsl_df = pd.DataFrame(columns=['Subj'])
lensl_df = pd.DataFrame(columns=['Subj'])

#matching_files = matching_files[5227:]

print(matching_files)
print(pattern)

for file_name in matching_files:

    streamline_name = file_name.split('.trk')[0]
    trk_file_path = os.path.join(trk_folder,file_name)
    density_map_path = os.path.join(densitymap_folder,f'{streamline_name}_dm.nii.gz')
    density_map_mask_path = os.path.join(densitymap_folder,f'{streamline_name}_dm_binary.nii.gz')

    if not os.path.exists(density_map_path) or not os.path.exists(density_map_mask_path) or \
            not os.path.exists(num_streamlines_path) or not os.path.exists(len_streamlines_path) or overwrite:
        streamlines_data = load_trk_remote(trk_file_path, 'same', None)

        header = streamlines_data.space_attributes
        streamlines = streamlines_data.streamlines
        affine = streamlines_data.affine
        
        if not os.path.exists(density_map_path) or not os.path.exists(density_map_mask_path) or overwrite:
            dm = utils.density_map(streamlines, affine, label_shape)
    
            threshold = 0.5
            binary_dm = (dm > threshold).astype(int)
    
            if not os.path.exists(density_map_path) or overwrite_denmaps:
                save_nifti(density_map_path, dm.astype("int16"), affine)
            if not os.path.exists(density_map_mask_path) or overwrite_denmaps:
                save_nifti(density_map_mask_path, binary_dm.astype("int16"), affine)
                
        if not os.path.exists(num_streamlines_path) or not os.path.exists(len_streamlines_path) or overwrite:

            streamline_name = file_name.split('.trk')[0]
            subj_name = re.search('S\d{5}', streamline_name)[0]
            bundle_name = streamline_name.split(subj_name + '_')[1]

            if not os.path.exists(num_streamlines_path) or overwrite:

                num_streamlines = len(streamlines)

                if bundle_name not in numsl_df.columns:
                    # Add an empty column if it doesn't exist
                    numsl_df[bundle_name] = 0
                    # numsl_df[bundle_name].astype(int)


                if subj_name not in numsl_df['Subj'].values:
                    numsl_df = pd.concat([numsl_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

                row_index = numsl_df.index[numsl_df['Subj'] == subj_name]
                col_index = numsl_df.columns.get_loc(bundle_name)
                numsl_df.iloc[row_index, col_index] = int(np.round(num_streamlines))

            if not os.path.exists(len_streamlines_path) or overwrite:
                avg_streamlines = np.mean(list(tract_length(streamlines)))

                if bundle_name not in lensl_df.columns:
                    # Add an empty column if it doesn't exist
                    lensl_df[bundle_name] = 0
                    # lensl_df[bundle_name].astype(int)

                if subj_name not in lensl_df['Subj'].values:
                    lensl_df = pd.concat([lensl_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

                row_index = lensl_df.index[lensl_df['Subj'] == subj_name]
                col_index = lensl_df.columns.get_loc(bundle_name)
                if len(streamlines)==0:
                    warnings.warn(f'Empty file {trk_file_path}')
                    lensl_df.iloc[row_index, col_index] = 0
                else:
                    lensl_df.iloc[row_index, col_index] = int(np.round(avg_streamlines))


if not os.path.exists(num_streamlines_path) or overwrite:

    for col in list(numsl_df.columns):
        if col!='Subj':
            numsl_df[col].astype(int)

    numsl_df.to_excel(num_streamlines_path,index = False)

if not os.path.exists(len_streamlines_path) or overwrite:

    for col in list(lensl_df.columns):
        if col!='Subj':
            lensl_df[col].astype(int)

    lensl_df.to_excel(len_streamlines_path,index = False)

if not os.path.exists(volume_excel_path) or overwrite:
    vol_df = pd.DataFrame(columns=['Subj'])

    for file_name in matching_files:
        streamline_name = file_name.split('.trk')[0]
        subj_name = re.search('S\d{5}', streamline_name)[0]
        bundle_name = streamline_name.split(subj_name+'_')[1]
        density_map_mask_path = os.path.join(densitymap_folder, f'{streamline_name}_dm_binary.nii.gz')


        density_data, density_affine = load_nifti(density_map_mask_path)

        vox_size = nib.load(density_map_mask_path).header.get_zooms()
        vox_size = [np.round(vox,2) for vox in vox_size]
        subj_bundle_volume = np.sum(density_data) * vox_size[0] * vox_size[1] * vox_size[2]

        if bundle_name not in vol_df.columns:
            # Add an empty column if it doesn't exist
            vol_df[bundle_name] = 0
            #vol_df[bundle_name].astype(int)

        if subj_name not in vol_df['Subj'].values:
            vol_df = pd.concat([vol_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

        row_index = vol_df.index[vol_df['Subj'] == subj_name]
        col_index = vol_df.columns.get_loc(bundle_name)
        vol_df.iloc[row_index,col_index] = int(np.round(subj_bundle_volume))

    for col in list(vol_df.columns):
        if col!='Subj':
            vol_df[col].astype(int)

    vol_df.to_excel(volume_excel_path,index=False)


matching_files_left = [matching_file for matching_file in matching_files if 'left' in matching_file]


if not os.path.exists(BUAN_summary_path) or overwrite:
    BUAN_df = pd.DataFrame(columns=['Subj'])

    affine_flip = np.eye(4)
    affine_flip[0, 0] = -1
    affine_flip[0, 3] = 0

    rng = np.random.RandomState()
    clust_thr = [5, 3, 1.5]
    threshold = 6

    for file_name_left in matching_files_left:

        streamline_name_left = file_name_left.split('.trk')[0]
        streamline_name_right = streamline_name_left.replace('left','right')

        file_name_right = file_name_left.replace('left','right')
        trk_file_path_left = os.path.join(trk_folder, file_name_left)
        trk_file_path_right = os.path.join(trk_folder, file_name_right)

        subj_name = re.search('S\d{5}', streamline_name_left)[0]
        bundle_name = streamline_name_left.split(subj_name+'_')[1].replace('_left','')

        streamlines_data_left = load_trk_remote(trk_file_path_left, 'same', None)

        header = streamlines_data_left.space_attributes
        streamlines_left = streamlines_data_left.streamlines
        streamlines_right = load_trk_remote(trk_file_path_right, 'same', None).streamlines
        affine = streamlines_data_left.affine

        streamlines_right_flipped = transform_streamlines(streamlines_right, affine_flip,
                                                          in_place=False)

        BUAN_id = bundle_shape_similarity(streamlines_left, streamlines_right_flipped, rng, clust_thr, threshold)


        if bundle_name not in BUAN_df.columns:
            # Add an empty column if it doesn't exist
            BUAN_df[bundle_name] = None
            #BUAN_df[bundle_name].astype(int)

        if subj_name not in BUAN_df['Subj'].values:
            BUAN_df = pd.concat([BUAN_df, pd.DataFrame({'Subj': [subj_name]})], ignore_index=True)

        row_index = BUAN_df.index[BUAN_df['Subj'] == subj_name]
        col_index = BUAN_df.columns.get_loc(bundle_name)
        BUAN_df.iloc[row_index,col_index] = (np.round(BUAN_id,3))

    BUAN_df.to_excel(BUAN_summary_path,index=False)
