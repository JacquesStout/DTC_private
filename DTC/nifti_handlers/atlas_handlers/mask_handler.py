
"""
Created by Jacques Stout
Functions for applying and creating masks from MRI acquisitions
"""


import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
import numpy as np
import warnings, shutil


def mask_fixer(maskpath, outpath=None):
    mask_nii = nib.load(maskpath)
    mask_data = mask_nii.get_fdata()
    if mask_data.dtype == 'bool':
        if outpath is None:
            return
        else:
            shutil.copy(maskpath, outpath)
    values = np.unique(mask_data)
    values.sort()
    if np.size(values)==1:
        if values[0]==0:
            warnings.warn('Only zeroes found in mask, will erase all images when applied')
        else:
            warnings.warn('Only a single value found in mask, converting to 1')
            mask_data_new = mask_data/values[0]
    elif np.size(values)==2:
        if 0 not in values:
            warnings.warn('Could not find 0, will assume smallest value is 0')
        if np.sum(values)>0:
            mask_data_new = np.zeros(np.shape(mask_data))
            #mask_data_new[mask_data == values[0]] = 0
            mask_data_new[mask_data == values[1]] = 1
        else:
            warnings.warn('Found negative value in mask, 0 still treated as 0 and other value treated as positive')
            mask_data_new = np.zeros(np.shape(mask_data))
            #mask_data_new[mask_data == values[1]] = 0
            mask_data_new[mask_data == values[0]] = 1
    else:
        warnings.warn('Found multiple values in mask, will treat any positive values as mask, and all else as 0')
        mask_shape = np.shape(mask_data)
        mask_data_new = np.zeros(mask_shape)
        for i in np.arange(mask_shape[0]):
            for j in np.arange(mask_shape[1]):
                for k in np.arange(mask_shape[2]):
                    if mask_data[i, j, k] >= 1:
                        mask_data_new = 1

    img_nii_new = nib.Nifti1Image(mask_data_new, mask_nii.affine, mask_nii.header)
    if outpath is None:
        outpath = maskpath
    nib.save(img_nii_new, outpath)

def applymask_array(data, mask):

    data_new = data
    dims = np.size(np.shape(data))
    data_shape = np.shape(data)
    mask_shape = np.shape(mask)

    if (dims == 3 or dims == 4) and mask_shape[0:3] == data_shape[0:3]:
        for i in np.arange(data_shape[0]):
            for j in np.arange(data_shape[1]):
                for k in np.arange(data_shape[2]):
                    if mask[i, j, k] >= 1:
                        if dims == 3:
                            data_new[i, j, k] = data[i, j, k]
                        else:
                            for l in range(data_shape[3]):
                                data_new[i, j, k, l] = data[i, j, k, l]
                    else:
                        if dims == 3:
                            data_new[i, j, k] = 0
                        else:
                            for l in range(data_shape[3]):
                                data_new[i, j, k, l] = 0
    return data_new

def applymask_samespace(file, mask, outpath=None):
    #note: there should definitely be a bet option which would probably end up faster, but for some insane reason there is no
    #obvious documentation on it, so this will do for now -_-
    img_nii= nib.load(file)
    mask_nii = nib.load(mask)
    img_data = img_nii.get_fdata()
    img_data_new = img_data
    mask_data = mask_nii.get_fdata()
    img_shape = img_nii.shape
    dims = np.size(img_shape)
    if (dims == 3 or dims == 4) and mask_nii.shape[0:3] == img_nii.shape[0:3]:
        img_data_new = applymask_array(img_data, mask_data)
    elif dims not in [3,4]:
        raise TypeError("Could not interpret the dimensions of the entering image")
    elif mask_nii.shape[0:3] != img_nii.shape[0:3]:
        raise TypeError("The shape of the mask and the image are not equal, therefore readjustments are needed")
    img_nii_new = nib.Nifti1Image(img_data_new, img_nii.affine, img_nii.header)
    if outpath is None:
        outpath = file
    nib.save(img_nii_new, outpath)


def median_mask_make(inpath, outpath, outpathmask=None, median_radius=4, numpass=4,binary_dilation=None, vol_idx = None):

    if outpathmask is None:
        outpathmask=outpath.replace(".nii","_mask.nii")
    data, affine = load_nifti(inpath)
    data = np.squeeze(data)
    data_masked, mask = median_otsu(data, median_radius=median_radius, numpass=numpass, dilate=binary_dilation, vol_idx=vol_idx)
    save_nifti(outpath, data_masked.astype(np.float32), affine)
    save_nifti(outpathmask, mask.astype(np.float32), affine)
