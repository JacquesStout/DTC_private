import os, shutil, glob
from DTC.file_manager.file_tools import buildlink, mkcdir
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import sys
from DTC.nifti_handlers.transform_handler import img_transform_exec


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


verbose = False
overwrite = False

in_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone'
out_path = os.path.join(in_path, 'trks')

orient_orig = 'RIA'
orient = 'RAS'


if orient == orient_orig:
    orient_str = ''
else:
    orient_str = f'_{orient}'

subj = '18'
4
bval_path = glob.glob(os.path.join(in_path,f'{subj}*0.bval'))[0]
bvec_path = glob.glob(os.path.join(in_path,f'{subj}*0.bvec'))[0]

subj_nii_orig = glob.glob(os.path.join(in_path,f'{subj}*0.nii.gz'))[0]
subj_nii_path = subj_nii_orig.replace('.nii.gz',orient_str+'.nii.gz')

mkcdir(out_path)

out_mif_temp = subj_nii_path.replace('.nii.gz', '.mif.gz')

mask_nii_path = subj_nii_path.replace('.nii.gz','_mask.nii.gz')
mask_nii_orig_path = mask_nii_path.replace(orient_str,'')
mask_mif_path = mask_nii_path.replace('.nii.gz','.mif.nii.gz')

masked_nii_path = subj_nii_path.replace('.nii.gz','_masked.nii.gz')

bval_checked_path = bval_path.replace('.bval','_checked.bval')
bvec_checked_path = bvec_path.replace('.bvec','_checked.bvec')

median_radius = 4
numpass = 7
binary_dilation_val = 1

cleanup = False

checked_bvecs = True

index_gz = '.gz'

if orient != orient_orig and not os.path.exists(subj_nii_path):
    print(f'Reorienting {subj_nii_path}')
    subj_nii_orig = subj_nii_path.replace(orient_str, '')
    img_transform_exec(subj_nii_orig, orient_orig, orient, output_path=subj_nii_path, recenter_test=True)


if not os.path.exists(out_mif_temp) or overwrite:
    os.system(
        'mrconvert ' + subj_nii_path + ' ' + out_mif_temp + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000


if not (os.path.exists(mask_nii_path) and not os.path.exists(mask_nii_orig_path)) or overwrite:

    b0_dwi_mif_temp = os.path.join(in_path, f'orig_b0_mean_temp.mif')
    if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        command = 'dwiextract ' + out_mif_temp + ' - -bzero | mrmath - mean ' + b0_dwi_mif_temp + ' -axis 3 -force'
        # if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        print(command)
        os.system(command)

    b0_dwi_nii_temp = os.path.join(in_path, f'orig_b0_mean_temp.nii.gz')
    # if not os.path.exists(b0_dwi_nii_temp):
    if not os.path.exists(b0_dwi_nii_temp) or overwrite:
        os.system(f'mrconvert {b0_dwi_mif_temp} {b0_dwi_nii_temp} -force')

    median_mask_make(b0_dwi_nii_temp, masked_nii_path, median_radius=median_radius,
                     binary_dilation_val=binary_dilation_val,
                     numpass=numpass, outpathmask=mask_nii_path, verbose=verbose, overwrite=True)

    if cleanup:
        os.remove(b0_dwi_mif_temp)
        os.remove(b0_dwi_nii_temp)


if orient != orient_orig and not os.path.exists(mask_nii_path):
    print(f'Reorienting {mask_nii_orig_path}')
    if not os.path.exists(mask_nii_path) or overwrite:
        img_transform_exec(mask_nii_orig_path, orient_orig, orient, output_path=mask_nii_path, recenter_test=True)

if not os.path.exists(mask_mif_path):
    os.system(f'mrconvert {mask_nii_path} {mask_mif_path}')

if not os.path.exists(bval_checked_path) or not os.path.exists(bvec_checked_path) and checked_bvecs:
    os.system(
        f'dwigradcheck ' + out_mif_temp + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -mask ' + mask_mif_path + ' -number 100000 -export_grad_fsl ' + bvec_checked_path + ' ' + bval_checked_path + ' -force')

fastrun = False

if checked_bvecs:
    coreg_bvecs = bvec_checked_path
    coreg_bvals = bval_checked_path
    bvec_string = ''
else:
    coreg_bvecs = bvec_path
    coreg_bvals = bval_path
    bvec_string = '_orig'

subj_out_folder = os.path.join(in_path, f'{subj}_temp{orient_str}{bvec_string}')
perm_subj_output = os.path.join(in_path, f'{subj}_perm{orient_str}{bvec_string}')

mkcdir([subj_out_folder, perm_subj_output])

print(f'Starting process for {subj_nii_path} using {coreg_bvecs}')



wmfod_norm_mif = os.path.join(subj_out_folder, subj + '_wmfod_norm.mif' + index_gz)
gmfod_norm_mif = os.path.join(subj_out_folder, subj + '_gmfod_norm.mif' + index_gz)
csffod_norm_mif = os.path.join(subj_out_folder, subj + '_csffod_norm.mif' + index_gz)

denoise = True

if denoise:
    denoise_str = '_denoised'
else:
    denoise_str = ''

if fastrun:
    smallerTracks = os.path.join(perm_subj_output, subj + f'_smallerTracks10000{bvec_string}{denoise_str}.tck')
else:
    smallerTracks = os.path.join(perm_subj_output, subj + f'_smallerTracks2mill{bvec_string}{denoise_str}.tck')

if not os.path.exists(smallerTracks):


    if denoise:
        output_denoise = subj_nii_path.replace('.nii.gz','_denoised.nii.gz')
        if not os.path.exists(output_denoise) or overwrite:
            os.system('dwidenoise ' + subj_nii_path + ' ' + output_denoise + ' -force')
        output_denoise = subj_nii_path

        # Estimating the Basis Functions:
    if not os.path.exists(wmfod_norm_mif) or overwrite:
        wm_txt = os.path.join(subj_out_folder, subj + '_wm.txt')
        gm_txt = os.path.join(subj_out_folder, subj + '_gm.txt')
        csf_txt = os.path.join(subj_out_folder, subj + '_csf.txt')
        voxels_mif = os.path.join(subj_out_folder, subj + '_voxels.mif' + index_gz)

        subj_mif_path = subj_nii_path.replace('.nii', f'{bvec_string}.mif')

        overwrite = True

        if not os.path.exists(subj_mif_path) or overwrite:
            os.system(
                f'mrconvert {subj_nii_path} {subj_mif_path} -fslgrad {coreg_bvecs} {coreg_bvals} -bvalue_scaling 0 -force')

        ##Right now we are using the RESAMPLED mif of the 4D miff, to be discussed
        if not os.path.exists(voxels_mif) or not os.path.exists(wm_txt) or not os.path.exists(
                gm_txt) or not os.path.exists(csf_txt) or overwrite:
            command = f'dwi2response dhollander {subj_mif_path} {wm_txt} {gm_txt} {csf_txt} -voxels {voxels_mif} -mask {mask_mif_path} -scratch {subj_out_folder} -fslgrad {coreg_bvecs} {coreg_bvals} -force'
            print(command)
            os.system(command)

        # Applying the basis functions to the diffusion data:
        wmfod_mif = os.path.join(subj_out_folder, subj + '_wmfod.mif' + index_gz)
        gmfod_mif = os.path.join(subj_out_folder, subj + '_gmfod.mif' + index_gz)
        csffod_mif = os.path.join(subj_out_folder, subj + '_csffod.mif' + index_gz)

        # os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force' )
        if not os.path.exists(wmfod_mif) or overwrite:
            # command = 'dwi2fod msmt_csd ' + subj_mif_path + ' -mask ' + mask_mif_path + ' ' + wm_txt + ' ' + wmfod_mif + ' ' + gm_txt + ' ' + gmfod_mif + ' ' + csf_txt + ' ' + csffod_mif + ' -force'
            # Only doing white matter in mouse brain
            command = f'dwi2fod msmt_csd {subj_mif_path} -mask {mask_mif_path} {wm_txt} {wmfod_mif} -force'
            print(command)
            os.system(command)

        """
        # combine to single image to view them
        # Concatenating the FODs:
        vf_mif = os.path.join(subj_out_folder, subj + '_vf.mif')
        if not os.path.exists(vf_mif) or overwrite:
            command = 'mrconvert -coord 3 0 ' + wmfod_mif + ' -| mrcat ' + csffod_mif + ' ' + gmfod_mif + ' - ' + vf_mif + ' -force'
            print(command)
            os.system(command)
        # os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat ' +gmfod_mif+ ' - ' + vf_mif+' -force' ) # without csf
        """

        if not os.path.exists(wmfod_norm_mif) or not os.path.exists(gmfod_norm_mif) or not os.path.exists(
                csffod_norm_mif) or overwrite:
            # command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' ' + gmfod_mif + ' ' + gmfod_norm_mif + ' ' + csffod_mif + ' ' + csffod_norm_mif + ' -mask ' + mask_mif_path + '  -force'
            command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' -mask ' + mask_mif_path + '  -force'
            print(command)
            os.system(command)

    gmwmSeed_coreg_mif = mask_mif_path

    if fastrun:
        command = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.1 -select 10000 ' + wmfod_norm_mif + ' ' + smallerTracks + ' -force'
        print(command)
        os.system(command)
    else:
        tracks_10M_tck = os.path.join(subj_out_folder, subj + '_tracks_10M.tck')
        if not os.path.exists(tracks_10M_tck):
            command = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.1 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force'
            print(command)
            os.system(command)

        if not os.path.exists(smallerTracks):
            command = 'tckedit ' + tracks_10M_tck + ' -number 2000000 ' + smallerTracks + ' -force'
            print(command)
            os.system(command)

if verbose:
    print(f'Created {smallerTracks}')

if cleanup:
    shutil.rmtree(subj_out_folder)