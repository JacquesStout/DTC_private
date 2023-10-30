import os, glob
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from DTC.nifti_handlers.atlas_handlers.mask_handler import applymask_array
import pandas as pd
from DTC.diff_handlers.bvec_handler import read_bvecs
from DTC.file_manager.file_tools import buildlink, mkcdir


def median_mask_make(inpath, outpath=None, outpathmask=None, median_radius=4, numpass=4, binary_dilation_val=None,
                     vol_idx=None, affine=None, verbose=False, overwrite=False):
    if type(inpath) == str:
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
    if os.path.exists(outpath) and os.path.exists(outpathmask) and not overwrite:
        print('Already wrote mask')
        return outpath, outpathmask
    data = np.squeeze(data)
    data_masked, mask = median_otsu(data, median_radius=median_radius, numpass=numpass, dilate=binary_dilation_val,
                                    vol_idx=vol_idx)
    save_nifti(outpath, data_masked.astype(np.float32), affine)
    save_nifti(outpathmask, mask.astype(np.float32), affine)
    if verbose:
        print(f'Saved masked file to {outpath}, saved mask to {outpathmask}')
    return outpath, outpathmask


input_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_niftis/20231017_140603_230925_16_apoe_18abb11_1_1/'
output_path = '/Users/jas/jacques/Bruker_DTI_EPI/'
group_name = 'DTI_230905_16'
subjects = ['23','26','19','20','21','24','25','27','28','29']


input_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_niftis/20231018_133130_221003_8_apoe_18_abb_11_1_1/'
output_path = '/Users/jas/jacques/Bruker_DTI_EPI/'
group_name = 'DTI_221003_8'
subjects = ['11','13','14']


input_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_niftis/20231024_090511_221003_10_apoe_18abb11_1_1/'
output_path = '/Users/jas/jacques/Bruker_DTI_EPI/'
group_name = 'DTI_221003_10'
subjects = ['6','18','19']
"""
"""

inpath_excel = f'/Users/jas/jacques/Bruker_tracker/{group_name}.xlsx'

dataframe = pd.read_excel(inpath_excel)

group_outpath = os.path.join(output_path,group_name)

outpath_excel = os.path.join(output_path,f'{group_name}_snr.xlsx')

mkcdir(group_outpath)

snr_b0 = True
snr_diff = True

test=True

median_radius = 4
numpass = 6
binary_dilation_val = 1
verbose = True

for subject in subjects:
    nii_path = glob.glob(os.path.join(input_path,f'{subject}_*nii.gz'))

    if np.size(nii_path)>1:
        raise Exception('More than one subject with that name found')
    else:
        nii_path = nii_path[0]

    bval_path = nii_path.replace('.nii.gz','.bval')
    bvec_path = nii_path.replace('.nii.gz','.bvec')
    bvals = read_bvecs(bval_path)

    b0s_bool = list(bvals < np.max(bvals) / 10).index(True)
    try:
        b0_slice = list(bvals < np.max(bvals) / 10).index(True)
    except ValueError:
        bo_slice = None
    try:
        b1_slice = list(bvals < np.max(bvals) / 10).index(False)
    except ValueError:
        b1_slice = None

    image_nifti = nib.load(nii_path)
    image_data = image_nifti.get_fdata()
    image_affine = image_nifti.affine
    image_shape = np.shape(image_data)

    masked_nii_path = os.path.join(group_outpath,f'{subject}_b0_masked.nii.gz')
    mask_nii_path = os.path.join(group_outpath,f'{subject}_b0_mask.nii.gz')
    b0_slice_data = image_data[:,:,:,b0_slice]

    if not os.path.exists(mask_nii_path):
        median_mask_make(b0_slice_data, outpath=masked_nii_path, median_radius=median_radius,
                         binary_dilation_val=binary_dilation_val, affine=image_affine,
                         numpass=numpass, outpathmask=mask_nii_path, verbose=verbose, overwrite=True)

    mask = nib.load(mask_nii_path).get_fdata()
    reverse_mask = 1-mask

    if snr_b0 and b0_slice is not None:
        b0_slice_data = image_data[:,:,:,b0_slice]
        data_masked = applymask_array(image_data[:,:,:,b0_slice], mask)
        data_noise = applymask_array(b0_slice_data, reverse_mask)
        # Calculate the mean signal
        center_slice = data_masked[3*int(image_shape[0]/8):5*int(image_shape[0]/8), 3*int(image_shape[1]/8):5*int(image_shape[1]/8), int(image_shape[2]/2)]
        mean_signal = np.mean(center_slice[center_slice>0])

        # Calculate the standard deviation of the background noise
        background_slice = data_noise[0:int(image_shape[0]/4), 0:int(image_shape[1]/4), int(image_shape[2]/2)]  # Define your own background slice
        noise_std_dev = np.std(background_slice[background_slice>0])

        snr_b0 = mean_signal / noise_std_dev

        dataframe.loc[dataframe['Subjects'] == int(subject), 'b0SNR'] = snr_b0
        #dataframe.loc[dataframe['Subjects'] == int(subject), 'b0mean'] = mean_signal
        #dataframe.loc[dataframe['Subjects'] == int(subject), 'bostd'] = noise_std_dev

        if test:
            test_noise_path = os.path.join(group_outpath,f'{subject}_b0noise.nii.gz')
            data_noise[int(image_shape[0]/8):, int(image_shape[1]/8):,int(image_shape[2]/8):]=0
            nib_background = nib.Nifti1Image(data_noise, image_affine)
            nib.save(nib_background, test_noise_path)

    if snr_diff and b1_slice is not None:
        diff_data = image_data[:,:,:,b1_slice]

        data_masked = applymask_array(diff_data, mask)
        data_noise = applymask_array(diff_data, reverse_mask)
        # Calculate the mean signal
        center_slice = data_masked[3*int(image_shape[0]/8):5*int(image_shape[0]/8), 3*int(image_shape[1]/8):5*int(image_shape[1]/8), int(image_shape[2]/2)]
        mean_signal = np.mean(center_slice[center_slice>0])

        # Calculate the standard deviation of the background noise
        background_slice = data_noise[0:int(image_shape[0]/4), 0:int(image_shape[1]/4), int(image_shape[2]/2)]  # Define your own background slice
        noise_std_dev = np.std(background_slice[background_slice>0])

        snr_diff = mean_signal / noise_std_dev

        dataframe.loc[dataframe['Subjects'] == int(subject), 'diffSNR'] = snr_diff
        #dataframe.loc[dataframe['Subjects'] == int(subject), 'diffmean'] = mean_signal
        #dataframe.loc[dataframe['Subjects'] == int(subject), 'diffstd'] = noise_std_dev

    if verbose:
        print(f'Finished for subject {subject}')

dataframe.to_excel(outpath_excel)