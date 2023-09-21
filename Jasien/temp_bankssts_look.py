import nibabel as nib
import numpy as np
import os
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter


label_folder_path = '/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/'

subjects = ['J04086', 'J04129', 'J04300', 'J01257', 'J01277', 'J04472', 'J01402']
subjects = ['S03866']
outpath = '/Users/jas/jacques/Jasien/bankssts_tests'

ROI_legends = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx'
index1_to_2, _, index2_to_struct, _ = atlas_converter(ROI_legends)

#ROI_val = 53
ROI_val = 2001
#struct = index2_to_struct[index1_to_2[ROI_val]]
struct = index2_to_struct[index1_to_2[ROI_val]]

SAMBA_path_results = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/connectomics/'

for subj in subjects:
    subj_temp = subj.replace('J','T')
    label_path = os.path.join(SAMBA_path_results, subj_temp, subj_temp + '_IITmean_RPI_labels.nii.gz')
    nifti_img = nib.load(label_path)
    nifti_data = nifti_img.get_fdata()
    mask = (nifti_data == ROI_val)
    mask_path = os.path.join(outpath, f'{subj}_{struct}.nii.gz')
    mask_nifti = nib.Nifti1Image(mask.astype(np.uint8), nifti_img.affine)
    nib.save(mask_nifti, mask_path)

subj_temp = subj.replace('J','T')
label_path = os.path.join(SAMBA_path_results, subj_temp, subj_temp + '_IITmean_RPI_labels.nii.gz')
nifti_img = nib.load(label_path)
nifti_data = nifti_img.get_fdata()
mask = (nifti_data == ROI_val)
mask_path = os.path.join(outpath, f'{subj}_{struct}.nii.gz')
mask_nifti = nib.Nifti1Image(mask.astype(np.uint8), nifti_img.affine)
nib.save(mask_nifti, mask_path)
