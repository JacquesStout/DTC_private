
"""
Created by Jacques Stout
Functions for applying and creating masks from MRI acquisitions
"""


import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
import numpy as np
import warnings, shutil
import os
import copy

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
                    if mask_data[i, j, k] > 0:
                        mask_data_new[i,j,k] = 1

    img_nii_new = nib.Nifti1Image(mask_data_new, mask_nii.affine, mask_nii.header)
    if outpath is None:
        outpath = maskpath
    nib.save(img_nii_new, outpath)


def unite_labels(list_atlases, outpath):
    cur_max=0
    for atlas_path in list_atlases:
        atlas_nii = nib.load(atlas_path)
        mask_data = atlas_nii.get_fdata()
        mask_shape = np.shape(mask_data)
        if not 'atlas_mask' in locals():
            atlas_mask = np.zeros(mask_shape)

        for i in np.arange(mask_shape[0]):
            for j in np.arange(mask_shape[1]):
                for k in np.arange(mask_shape[2]):
                    if mask_data[i, j, k] > 0 and atlas_mask[i,j,k]==0:
                        atlas_mask[i,j,k] = cur_max + mask_data[i,j,k]

        vals = cur_max + np.unique(mask_data)
        cur_max = np.max(vals)

    img_nii_new = nib.Nifti1Image(atlas_mask, atlas_nii.affine, atlas_nii.header)
    nib.save(img_nii_new, outpath)
    return outpath


def create_basemask(imgpath, outpath=None):
    if outpath is None:
        outpath = imgpath.replace('.nii','_mask.nii')
    if outpath==imgpath:
        raise Exception('Cant replace nifti by mask of nifti')
    if 'mask' in imgpath:
        warnings.warn('Creating a mask from a mask, seems dicey')
        return outpath
    if os.path.exists(outpath):
        print(f'Already created mask {outpath}')
        return outpath
    img_nii = nib.load(imgpath)
    img_data = img_nii.get_fdata()
    if np.size(np.shape(img_data))==4:
        img_data = img_data[:,:,:,0]
    mask_shape = np.shape(img_data)
    mask_data = np.zeros(mask_shape)
    for i in np.arange(mask_shape[0]):
        for j in np.arange(mask_shape[1]):
            for k in np.arange(mask_shape[2]):
                if img_data[i, j, k] > 0:
                    mask_data[i,j,k] = 1

    img_nii_new = nib.Nifti1Image(mask_data, img_nii.affine, img_nii.header)
    nib.save(img_nii_new, outpath)
    return outpath


def create_mask_threshold(imgpath, threshold = 1, outpath = None):
    if outpath is None:
        outpath = imgpath.replace('.nii','_mask.nii')
    if outpath==imgpath:
        raise Exception('Cant replace nifti by mask of nifti')
    if 'mask' in imgpath:
        warnings.warn('Creating a mask from a mask, seems dicey')
        return outpath
    if os.path.exists(outpath):
        print(f'Already created mask {outpath}')
        return outpath

    img_nii = nib.load(imgpath)
    img_data = img_nii.get_fdata()
    mask_shape = np.shape(img_data)
    mask_data = np.zeros(mask_shape)
    for i in np.arange(mask_shape[0]):
        for j in np.arange(mask_shape[1]):
            for k in np.arange(mask_shape[2]):
                if img_data[i, j, k] >= threshold:
                    mask_data[i,j,k] = 1

    img_nii_new = nib.Nifti1Image(mask_data, img_nii.affine, img_nii.header)
    nib.save(img_nii_new, outpath)
    return outpath


def applymask_array(data, mask):

    data_new = copy.deepcopy(data)
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
    if mask is str:
        mask_nii = nib.load(mask)
    else:
        mask_nii = mask
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


def median_mask_make(inpath, outpath=None, outpathmask=None, median_radius=4, numpass=4,binary_dilation=None, vol_idx = None, affine=None):

    if type(inpath)==str:
        data, affine = load_nifti(inpath)
        if outpath is None:
            outpath = inpath.replace(".nii", "_masked.nii")
        elif outpath is None and outpathmask is None:
            outpath = inpath.replace(".nii", "_masked.nii")
            outpathmask = inpath.replace(".nii", "_mask.nii")
        elif outpathmask is None:
            outpathmask = outpath.replace(".nii", "_mask.nii")
    else:
        data = inpath
        if affine is None:
            raise Exception('Needs affine')
        if outpath is None:
            raise Exception('Needs outpath')
    if os.path.exists(outpath) and os.path.exists(outpathmask):
        print('Already wrote mask')
        return outpath, outpathmask
    data = np.squeeze(data)
    data_masked, mask = median_otsu(data, median_radius=median_radius, numpass=numpass, dilate=binary_dilation, vol_idx=vol_idx)
    save_nifti(outpath, data_masked.astype(np.float32), affine)
    save_nifti(outpathmask, mask.astype(np.float32), affine)
    return outpath, outpathmask