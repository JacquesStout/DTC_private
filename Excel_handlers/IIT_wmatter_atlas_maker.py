import os
import pandas as pd
from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile, glob_remote
import nibabel as nib
import numpy as np
from DTC.nifti_handlers.atlas_handlers.mask_handler import create_mask_threshold
import copy


bundles_folder_path = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas'
bundles_excel_path = os.path.join(bundles_folder_path,'bundle_summary.xlsx')
atlas_excel_path = os.path.join(bundles_folder_path,'atlas_summary.xlsx')
bundles_orignii_path = os.path.join(bundles_folder_path,'IIT_bundles')

bundles_mask_path = os.path.join(bundles_folder_path,'IIT_bundles_masks')

atlas_f_path = os.path.join(bundles_folder_path,'atlas_f_test_2.nii.gz')

bundles_df = pd.read_excel(bundles_excel_path)

boi_names = bundles_df.Name #boi means Bundle of interest, might want ot subselect specific ones down the line. If want all of them: boi_names = bundles_df.Name

mkcdir(bundles_mask_path)

overwrite=False

region_dict = {}

atlas_df = copy.deepcopy(bundles_df)
atlas_df = atlas_df.drop('Tract density threshold', axis=1)
atlas_df['Label number']=''


for i,boi in enumerate(boi_names):
    boi_nii_path = os.path.join(bundles_orignii_path,f'{boi}.nii.gz')
    if not os.path.exists(boi_nii_path):
        errortxt = f'Could not find {boi_nii_path}'
        raise FileNotFoundError(errortxt)

    threshold = bundles_df[bundles_df['Name']==boi]['Tract density threshold'].values[0]

    boi_nii = nib.load(boi_nii_path)
    boi_data = boi_nii.get_fdata()
    boi_data[boi_data==0] = -100
    boi_data[boi_data>0] = boi_data[boi_data>0]-threshold
    if i==0:
        ROI_all = np.zeros(list(boi_nii.shape)+[np.size(boi_names)+1])
        ROI_all[:,:,:,i]=-90
        atlas_shape = boi_nii.shape

    ROI_all[:,:,:,i+1] = boi_data
    #atlas_df[atlas_df['Name'] == boi]['Label number'] =
    atlas_df.loc[atlas_df['Name'] == boi, 'Label number'] = i+1
    print(f'Ran through {boi}')
    #boi_nii = nib.load(boi_nii_path)
    #boi_data = boi_nii.get_fdata()
    #mask_data = np.zeros(boi_data)
    #if not os.path.exists(bart_mask_path) or overwrite:
        #create_mask_threshold(boi_nii_path,threshold = threshold,outpath = bart_mask_path)


label_array = np.argmax(ROI_all, axis=3)
label_array = label_array.astype(np.int8)
atlas_nii = nib.Nifti1Image(label_array, boi_nii.affine)
nib.save(atlas_nii, atlas_f_path)

atlas_df.to_excel(atlas_excel_path)

"""
for boi in boi_names:
    boi_nii_path = os.path.join(bundles_orignii_path,f'{boi}.nii.gz')
    if not os.path.exists(boi_nii_path):
        errortxt = f'Could not find {boi_nii_path}'
        raise FileNotFoundError(errortxt)
    bart_mask_path = os.path.join(bundles_mask_path,f'{boi}_mask.nii.gz')
    try:
        threshold = bundles_df[bundles_df['Name']==boi]['Tract density threshold'].values[0]
    except:
        print('hi')
    #boi_nii = nib.load(boi_nii_path)
    #boi_data = boi_nii.get_fdata()
    #mask_data = np.zeros(boi_data)
    if not os.path.exists(bart_mask_path) or overwrite:
        create_mask_threshold(boi_nii_path,threshold = threshold,outpath = bart_mask_path)

label_data = np.zeros(nib.load(boi_nii_path).shape)

for i,boi in enumerate(boi_names):
    atlas_df = copy.deepcopy(bundles_df)
    atlas_df.drop_columns('Tract density threshold')
    atlas_df.add_columns('Label number')
    atlas_df[atlas_df['Name']==boi]['Label number'] = i

    for
"""