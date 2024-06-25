import os, shutil
from DTC.file_manager.file_tools import buildlink, mkcdir
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import sys
from DTC.nifti_handlers.transform_handler import img_transform_exec
from DTC.diff_handlers.bvec_handler import extractbvals_from_method, reorient_bvecs_files,fix_bvals_bvecs, cut_bvecs_bvals, read_bval_file
from DTC.nifti_handlers.nifti_handler import cut_4D_img

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


verbose=False
overwrite=False
masking = True
cleanup = False
checked_bvecs = True
#
subj = '20220905_14'
orig_recon_nii = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/11_CS_DWI_bart_recon.nii.gz'
method_path = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke/20220905_14/11/method'
in_path = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/'
out_path = os.path.join(in_path,'trkscutdirs_improved')
temp_path = os.path.join(in_path,'temp')

mkcdir([out_path,temp_path])

orient_start = 'ARI'
orient_end = 'RAS'

basename = os.path.basename(orig_recon_nii)

if orient_start!=orient_end:
    recon_nii = os.path.join(temp_path,basename.replace('.nii.gz',f'_{orient_end}.nii.gz'))
    if not os.path.exists(recon_nii) or overwrite:
        img_transform_exec(orig_recon_nii, orient_start, orient_end, output_path=recon_nii, recenter_test=True)
else:
    recon_nii = os.path.join(temp_path,basename)
    if not os.path.exists(recon_nii) or overwrite:
        shutil.copy(orig_recon_nii,recon_nii)

mask_nii_path = recon_nii.replace('.nii.gz','_mask.nii.gz')
masked_nii_path = recon_nii.replace('.nii.gz','_masked.nii.gz')

recon_mif = recon_nii.replace('.nii.gz','.mif')

bval_orig_path = os.path.join(in_path, f'{subj}_bval.txt')
bvec_orig_path = os.path.join(in_path, f'{subj}_bvec.txt')

bval_path = os.path.join(temp_path,f'{subj}_bval_fix.txt')
bvec_path = os.path.join(temp_path,f'{subj}_bvec_fix.txt')

bval_checked_path = os.path.join(temp_path,f'{subj}_checked.bval')
bvec_checked_path = os.path.join(temp_path,f'{subj}_checked.bvec')

if not os.path.exists(bvec_orig_path) or not os.path.exists(bvec_orig_path) or overwrite:
    extractbvals_from_method(method_path, outpath=os.path.join(in_path, f'{subj}'), tonorm=True, verbose=False)

if orient_start!=orient_end:
    bvecs_reoriented_path = os.path.join(temp_path,f'{subj}_{orient_end}.bvec')
    if not os.path.exists(bvecs_reoriented_path) or overwrite:
        reorient_bvecs_files(bvec_orig_path,bvecs_reoriented_path,orient_start,orient_end)
    bvec_orig_path = bvecs_reoriented_path

if not os.path.exists(bval_path) or not os.path.exists(bvec_path) or overwrite:
    fix_bvals_bvecs(bval_orig_path, bvec_orig_path, outpath=temp_path, b0_threshold=300,writeformat='mrtrix')

if not os.path.exists(recon_mif) or overwrite:
    os.system(
        'mrconvert ' + recon_nii + ' ' + recon_mif + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000


if not os.path.exists(mask_nii_path) or overwrite:

    median_radius = 4
    numpass = 7
    binary_dilation_val = 1

    b0_dwi_mif_temp = os.path.join(temp_path, f'orig_b0_mean_temp.mif')
    if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        command = 'dwiextract ' + recon_mif + ' - -bzero | mrmath - mean ' + b0_dwi_mif_temp + ' -axis 3 -force'
        #if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        print(command)
        os.system(command)


    b0_dwi_nii_temp = os.path.join(temp_path, f'orig_b0_mean_temp.nii.gz')
    #if not os.path.exists(b0_dwi_nii_temp):
    if not os.path.exists(b0_dwi_nii_temp) or overwrite:
        os.system(f'mrconvert {b0_dwi_mif_temp} {b0_dwi_nii_temp} -force')

    median_mask_make(b0_dwi_nii_temp, masked_nii_path, median_radius=median_radius,
                     binary_dilation_val=binary_dilation_val,
                     numpass=numpass, outpathmask=mask_nii_path, verbose=verbose, overwrite=True)

    if cleanup:
        os.remove(b0_dwi_mif_temp)
        os.remove(b0_dwi_nii_temp)


if not os.path.exists(bval_checked_path) or not os.path.exists(bvec_checked_path) and checked_bvecs:
    os.system(
        f'dwigradcheck ' + recon_mif + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -mask ' + mask_nii_path + ' -number 100000 -export_grad_fsl ' + bvec_checked_path + ' ' + bval_checked_path + ' -force')


#cutdirs = 21
cutdirs = [0, 5, 6, 8, 9, 10, 11, 12, 14, 23, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

fastrun = False

keep_10mil = True

denoise = True


if checked_bvecs:
    coreg_bvecs = bvec_checked_path
    coreg_bvals = bval_checked_path
    bvec_string = ''
else:
    coreg_bvecs = bvec_path
    coreg_bvals = bval_path
    bvec_string = '_orig'


basename = os.path.basename(recon_nii)

if cutdirs is not None:
    if type(cutdirs) != list:
        bvec_checked_cut_path = os.path.join(out_path, f'{subj}_checked_{cutdirs}.bvec')
        bval_checked_cut_path = os.path.join(out_path, f'{subj}_checked_{cutdirs}.bval')
        recon_nii_cut = os.path.join(out_path, basename.replace('.nii.gz', f'_{cutdirs}.nii.gz'))
    else:
        bvec_checked_cut_path = os.path.join(out_path, f'{subj}_checked_{np.size(cutdirs)}.bvec')
        bval_checked_cut_path = os.path.join(out_path, f'{subj}_checked_{np.size(cutdirs)}.bval')
        recon_nii_cut = os.path.join(out_path, basename.replace('.nii.gz', f'_{np.size(cutdirs)}.nii.gz'))

    if not os.path.exists(bvec_checked_cut_path) or not os.path.exists(bval_checked_cut_path) or overwrite:
        cut_bvecs_bvals(coreg_bvals,coreg_bvecs,bvec_checked_cut_path,bval_checked_cut_path,cutdirs,writeformat='mrtrix', b0_threshold=300)
    if not os.path.exists(recon_nii_cut) or overwrite:
        cut_4D_img(recon_nii,recon_nii_cut,cutdirs,bval_path = bval_checked_cut_path,b0_threshold=300)
    cut_bvecs_bvals(
        '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/20220905_14_bval.txt',
        '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/20220905_14_bvec.txt',
        '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/20220905_14_cut_bvec.txt',
        '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/20220905_14_cut_bval.txt',
        cutdirs, writeformat='mrtrix', b0_threshold=300)
    recon_nii = recon_nii_cut
    coreg_bvals = bval_checked_cut_path
    coreg_bvecs = bvec_checked_cut_path

if denoise:
    denoise_str = '_denoised'
else:
    denoise_str = ''

wmfod_norm_mif = os.path.join(out_path, f'{subj}_wmfod_norm.mif')
gmfod_norm_mif = os.path.join(out_path, f'{subj}_gmfod_norm.mif')
csffod_norm_mif = os.path.join(out_path, f'{subj}_csffod_norm.mif')

if denoise:
    output_denoise_nii = os.path.join(out_path, basename.replace('.nii.gz','_denoised.nii.gz'))
    output_denoise_mif = os.path.join(out_path, basename.replace('.nii.gz','_denoised.mif'))
    if (not os.path.exists(output_denoise_mif) and not os.path.exists(output_denoise_nii)) or overwrite:
        os.system('dwidenoise ' + recon_nii + ' ' + output_denoise_nii + ' -force')
    recon_nii = output_denoise_nii
    recon_mif = output_denoise_mif
    if not os.path.exists(recon_mif) or overwrite:
        os.system(
            'mrconvert ' + recon_nii + ' ' + recon_mif + ' -fslgrad ' + coreg_bvecs + ' ' + coreg_bvals + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000

make_fa = True
if make_fa:
    bvals = read_bval_file(coreg_bvals)

    dt_mif = os.path.join(out_path,f'{subj}_dt.mif')
    fa_mif = os.path.join(out_path,f'{subj}_fa.mif')
    dk_mif = os.path.join(out_path,f'{subj}_dk.mif')
    mk_mif = os.path.join(out_path,f'{subj}_mk.mif')
    md_mif = os.path.join(out_path,f'{subj}_md.mif')
    ad_mif = os.path.join(out_path,f'{subj}_ad.mif')
    rd_mif = os.path.join(out_path,f'{subj}_rd.mif')
    from DTC.file_manager.computer_nav import checkfile_exists_all

    checked_all = checkfile_exists_all([dt_mif,fa_mif, rd_mif, ad_mif, md_mif])

    """
    if np.unique(bvals).shape[0] > 2 :
        os.system('dwi2tensor ' + recon_nii + ' ' + dt_mif + ' -dkt ' +  dk_mif +' -fslgrad ' +  coreg_bvecs + ' ' + coreg_bvals + ' -force'  )
        os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -adc '  + md_mif+' -ad '  + ad_mif + ' -rd '  + rd_mif   + ' -force' )
    else:
    """
    if not checked_all or overwrite:
        os.system('dwi2tensor ' + recon_nii + ' ' + dt_mif  +' -fslgrad ' +  coreg_bvecs + ' ' + coreg_bvals + ' -force'  )
        os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -force' )
        os.system('tensor2metric  -rd ' + rd_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
        os.system('tensor2metric  -ad ' + ad_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
        os.system('tensor2metric  -adc ' + md_mif  + ' '+ dt_mif + ' -force' )


if fastrun:
    smallerTracks = os.path.join(out_path, f'{subj}_smallerTracks10000{bvec_string}{denoise_str}.tck')
else:
    smallerTracks = os.path.join(out_path, f'{subj}_smallerTracks2mill{bvec_string}{denoise_str}.tck')

tracks_10M_tck = os.path.join(out_path, f'{subj}_smallerTracks10mill{bvec_string}{denoise_str}.tck')

if not os.path.exists(smallerTracks) or (keep_10mil and not os.path.exists(tracks_10M_tck)):

    # Estimating the Basis Functions:
    if not os.path.exists(wmfod_norm_mif) or overwrite:
        wm_txt = os.path.join(out_path, subj + '_wm.txt')
        gm_txt = os.path.join(out_path, subj + '_gm.txt')
        csf_txt = os.path.join(out_path, subj + '_csf.txt')
        voxels_mif = os.path.join(out_path, subj + '_voxels.mif')

        ##Right now we are using the RESAMPLED mif of the 4D miff, to be discussed
        if not os.path.exists(voxels_mif) or not os.path.exists(wm_txt) or not os.path.exists(
                gm_txt) or not os.path.exists(csf_txt) or overwrite:
            command = f'dwi2response dhollander {recon_mif} {wm_txt} {gm_txt} {csf_txt} -voxels {voxels_mif} -mask {mask_nii_path} -scratch {out_path} -fslgrad {coreg_bvecs} {coreg_bvals} -force'
            print(command)
            os.system(command)

        # Applying the basis functions to the diffusion data:
        wmfod_mif = os.path.join(out_path, subj + '_wmfod.mif')
        gmfod_mif = os.path.join(out_path, subj + '_gmfod.mif')
        csffod_mif = os.path.join(out_path, subj + '_csffod.mif')


        if not os.path.exists(wmfod_mif) or overwrite:
            #command = 'dwi2fod msmt_csd ' + subj_mif_path + ' -mask ' + mask_mif_path + ' ' + wm_txt + ' ' + wmfod_mif + ' ' + gm_txt + ' ' + gmfod_mif + ' ' + csf_txt + ' ' + csffod_mif + ' -force'
            #Only doing white matter in mouse brain
            command = f'dwi2fod msmt_csd {recon_mif} -mask {mask_nii_path} {wm_txt} {wmfod_mif} -force'
            print(command)
            os.system(command)

        if not os.path.exists(wmfod_norm_mif) or not os.path.exists(gmfod_norm_mif) or not os.path.exists(
                csffod_norm_mif) or overwrite:
            #command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' ' + gmfod_mif + ' ' + gmfod_norm_mif + ' ' + csffod_mif + ' ' + csffod_norm_mif + ' -mask ' + mask_mif_path + '  -force'
            command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' -mask ' + mask_nii_path + '  -force'
            print(command)
            os.system(command)

    gmwmSeed_coreg_mif = mask_nii_path

    if fastrun:
        command = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.1 -select 10000 ' + wmfod_norm_mif + ' ' + smallerTracks + ' -force'
        print(command)
        os.system(command)
    else:

        if not os.path.exists(tracks_10M_tck):
            command = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.1 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force'
            print(command)
            os.system(command)

        if not os.path.exists(smallerTracks):
            command = 'tckedit ' + tracks_10M_tck + ' -number 2000000 ' + smallerTracks + ' -force'
            print(command)
            os.system(command)

        smallerTracks_50000 = os.path.join(out_path, f'{subj}_smallerTracks50000{bvec_string}{denoise_str}.tck')
        if not os.path.exists(smallerTracks_50000):
            command = 'tckedit ' + smallerTracks + ' -number 50000 ' + smallerTracks_50000 + ' -force'
            print(command)
            os.system(command)

if verbose:
    print(f'Created {smallerTracks}')

if cleanup:
    shutil.rmtree(out_path)
