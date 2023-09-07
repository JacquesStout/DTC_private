import os, glob
import nibabel as nib
import numpy as np

MDT_folder = '/Users/jas/jacques/AD_Decode/QSM_MDT/smoothed_1_5/'
MDT_output = '/Users/jas/jacques/AD_Decode/QSM_MDT/smoothed_1_5_masked/'
MDT_files = glob.glob(os.path.join(MDT_folder,'*.nii.gz'))

mask_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_mask.nii.gz'

overwrite=False

for image_path in MDT_files:

    maskedimage_path = os.path.join(MDT_output, os.path.basename(image_path))

    if not os.path.exists(maskedimage_path) or overwrite:
        if maskedimage_path == image_path:
            raise Exception('Wrong folder output is same as input folder')

        # Load the image and mask using nibabel
        image_nifti = nib.load(image_path)
        mask_nifti = nib.load(mask_path)

        # Extract the data arrays from the NIfTI files
        image_data = image_nifti.get_fdata()
        mask_data = mask_nifti.get_fdata()

        # Apply the mask to the image
        masked_image_data = np.multiply(image_data, mask_data)

        # Create a new NIfTI image object with the masked data
        masked_image_nifti = nib.Nifti1Image(masked_image_data, affine=image_nifti.affine)

        # Save the masked image to a new NIfTI file (optional)
        nib.save(masked_image_nifti, maskedimage_path)
        print(f'Saved masked image {maskedimage_path}')
    else:
        print(f'Masked image {maskedimage_path} already exists')