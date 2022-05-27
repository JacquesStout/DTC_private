import warnings
import nibabel as nib
import numpy as np

def header_superpose(target_path, origin_path, outpath=None, verbose=False):
    target_nii=nib.load(target_path)
    origin_nii=nib.load(origin_path)
    if np.shape(target_nii._data)[0:3] != np.shape(origin_nii._data)[0:3]:
        raise TypeError('not implemented')
    else:
        target_affine=target_nii._affine
        target_header = target_nii._header
        if np.any(target_affine != origin_nii._affine) or np.any(target_header != origin_nii._header):
            new_nii = nib.Nifti1Image(origin_nii._data, target_affine, target_header)
            if outpath is None:
                outpath = origin_path
                txt= (f'Overwriting original file {origin_path}')
                warnings.warn(txt)
            if verbose:
                print(f'Saving nifti file to {outpath}')
            nib.save(new_nii, outpath)
            if verbose:
                print(f'Saved')
        else:
            print('Same header for target_path and origin_path, skipping')

test_file = '/Users/jas/jacques/AD_Decode_warp_test/H22102_labels.nii.gz'
outpath = '/Users/jas/jacques/AD_Decode_warp_test/H22102_labels_new.nii.gz'
header_superpose(subjspace_coreg_RAS,coreg_RAS_tomove,outpath = testoutpath, verbose=True)