import os, shutil
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


verbose=False
overwrite=False

in_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/'
out_path = os.path.join(in_path,'trks')

orient = 'RAS'
if orient =='PLS':
    orient_str = ''
else:
    orient_str = f'_{orient}'

cs_vals = [4,6,8,10,12,1]
cs_vals = []
cs_vals.append(sys.argv[1])
subjects = [f'Bruker_diffusion_test_15_0.045_af_{cs_val}x__TV_and_L1_wavelet_0.0025_0.0025_bart_recon{orient_str}.nii.gz' for cs_val in cs_vals]

bval_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/15_1_fully_sampled_DWI_16p25_120_21directions_Brukerdirs.bval'
bvec_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/15_1_fully_sampled_DWI_16p25_120_21directions_Brukerdirs.bvec'

mkcdir(out_path)

orig_nii_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded.nii.gz'

out_mif_temp = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded{orient_str}.mif.gz'

mask_nii_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded_mask.nii.gz'
mask_mif_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded_mask{orient_str}.mif.gz'

masked_nii_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded_masked{orient_str}.nii.gz'

bval_checked_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/15_dirs_checked{orient_str}.bval'
bvec_checked_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/15_dirs_checked{orient_str}.bvec'

median_radius = 4
numpass = 7
binary_dilation_val = 1

dilation = '30'
redilated_mask = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded_mask_dilated_{dilation}.mif.gz'
redilated_mask_nii = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded_mask_dilated_{dilation}.nii'
cleanup = False

checked_bvecs = True

index_gz = '.gz'

new_orig_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded{orient_str}.nii.gz'
if orient != 'PLS':
    if not os.path.exists(new_orig_path):
        print(f'Reorienting {orig_nii_path}')
        img_transform_exec(orig_nii_path, 'PLS', orient, output_path=new_orig_path, recenter_test=True)
    orig_nii_path = new_orig_path

if not os.path.exists(out_mif_temp) or overwrite:
    os.system(
        'mrconvert ' + orig_nii_path + ' ' + out_mif_temp + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000


if not os.path.exists(mask_nii_path) or overwrite:

    b0_dwi_mif_temp = os.path.join(in_path, f'orig_b0_mean_temp.mif')
    if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        command = 'dwiextract ' + out_mif_temp + ' - -bzero | mrmath - mean ' + b0_dwi_mif_temp + ' -axis 3 -force'
        #if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        print(command)
        os.system(command)


    b0_dwi_nii_temp = os.path.join(in_path, f'orig_b0_mean_temp.nii.gz')
    #if not os.path.exists(b0_dwi_nii_temp):
    if not os.path.exists(b0_dwi_nii_temp) or overwrite:
        os.system(f'mrconvert {b0_dwi_mif_temp} {b0_dwi_nii_temp} -force')

    median_mask_make(b0_dwi_nii_temp, masked_nii_path, median_radius=median_radius,
                     binary_dilation_val=binary_dilation_val,
                     numpass=numpass, outpathmask=mask_nii_path, verbose=verbose, overwrite=True)

    if cleanup:
        os.remove(b0_dwi_mif_temp)
        os.remove(b0_dwi_nii_temp)

    new_mask_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded_mask{orient_str}.nii.gz'
    if orient != 'PLS':
        if not os.path.exists(new_mask_path) or overwrite:
            img_transform_exec(mask_nii_path, 'PLS', orient, output_path=new_mask_path, recenter_test=True)
        mask_nii_path = new_mask_path
        if not os.path.exists(mask_mif_path):
            os.system(f'mrconvert {mask_nii_path} {mask_mif_path}')

if not os.path.exists(mask_mif_path):
    os.system(f'mrconvert {mask_nii_path} {mask_mif_path}')

if not os.path.exists(bval_checked_path) or not os.path.exists(bvec_checked_path) and checked_bvecs:

    os.system(
        f'dwigradcheck ' + out_mif_temp + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -mask ' + mask_mif_path + ' -number 100000 -export_grad_fsl ' + bvec_checked_path + ' ' + bval_checked_path + ' -force')


if not os.path.exists(redilated_mask):
    if not os.path.exists(mask_nii_path):
        os.system(f'mrconvert {mask_mif_path} {mask_nii_path}')

    img = nib.load(mask_nii_path)
    mask_data = img.get_fdata()

    # Perform binary dilation
    dilated_mask = binary_dilation(mask_data, iterations=int(dilation)).astype(np.uint8)

    # Create a new NIfTI image with the dilated mask
    dilated_img = nib.Nifti1Image(dilated_mask, img.affine)

    nib.save(dilated_img, redilated_mask_nii)
    os.system(f'mrconvert {redilated_mask_nii} {redilated_mask}')

if cleanup and os.path.exists(redilated_mask_nii):
    shutil.remove(redilated_mask_nii)


fastrun = False

keep_10mil = True

denoise = True

for subj_nii in subjects:

    if checked_bvecs:
        coreg_bvecs = bvec_checked_path
        coreg_bvals = bval_checked_path
        bvec_string = ''
    else:
        coreg_bvecs = bvec_path
        coreg_bvals = bval_path
        bvec_string = '_orig'

    subj_min = subj_nii.split('af_')[1].split('x')[0]
    subj_nii_path = os.path.join(in_path, subj_nii)
    subj_out_folder = os.path.join(in_path,f'{subj_min}_temp{orient_str}{bvec_string}')
    perm_subj_output = os.path.join(in_path,f'{subj_min}_perm{orient_str}{bvec_string}')

    if orient != 'LPS' and not os.path.exists(subj_nii_path):
        print(f'Reorienting {subj_nii_path}')
        subj_nii_orig = subj_nii_path.replace(orient_str, '')
        img_transform_exec(subj_nii_orig, 'PLS', orient, output_path=subj_nii_path, recenter_test=True)

    mkcdir([subj_out_folder,perm_subj_output])

    print(f'Starting process for {subj_nii} using {coreg_bvecs}')

    if denoise:
        denoise_str = '_denoised'
    else:
        denoise_str = ''

    """

    dt_mif = os.path.join(perm_subj_output, subj + '_dt.mif' + index_gz)
    fa_mif = os.path.join(perm_subj_output, subj + '_fa.mif' + index_gz)
    dk_mif = os.path.join(perm_subj_output, subj + '_dk.mif' + index_gz)
    mk_mif = os.path.join(perm_subj_output, subj + '_mk.mif' + index_gz)
    md_mif = os.path.join(perm_subj_output, subj + '_md.mif' + index_gz)
    ad_mif = os.path.join(perm_subj_output, subj + '_ad.mif' + index_gz)
    rd_mif = os.path.join(perm_subj_output, subj + '_rd.mif' + index_gz)

    fa_nii = os.path.join(perm_subj_output, subj + '_fa.nii' + index_gz)

    wmfod_norm_mif = os.path.join(subj_out_folder, subj + '_wmfod_norm.mif' + index_gz)
    gmfod_norm_mif = os.path.join(subj_out_folder, subj + '_gmfod_norm.mif')
    csffod_norm_mif = os.path.join(subj_out_folder, subj + '_csffod_norm.mif')

    if not os.path.exists(fa_nii) or overwrite:

        # making fa and Kurt:

        dt_mif = os.path.join(perm_subj_output, subj + '_dt.mif' + index_gz)
        fa_mif = os.path.join(perm_subj_output, subj + '_fa.mif' + index_gz)
        dk_mif = os.path.join(perm_subj_output, subj + '_dk.mif' + index_gz)
        md_mif = os.path.join(perm_subj_output, subj + '_md.mif' + index_gz)
        ad_mif = os.path.join(perm_subj_output, subj + '_ad.mif' + index_gz)
        rd_mif = os.path.join(perm_subj_output, subj + '_rd.mif' + index_gz)

        # mk_mif = os.path.join(perm_subj_output,subj+'_mk.mif'+index_gz)

        # output_denoise = '/Users/ali/Desktop/Feb23/mrtrix_pipeline/temp/N59141/N59141_subjspace_dwi_copy.mif.gz'#


        # exists_all = checkallfiles([dt_mif, fa_mif, dk_mif, mk_mif, md_mif, ad_mif, rd_mif])
        exists_all = checkallfiles([dt_mif, fa_mif, dk_mif, md_mif, ad_mif, rd_mif])

        if not exists_all:

            if np.unique(bval_orig).shape[0] > 2:
                os.system(
                    'dwi2tensor ' + resampled_mif_path + ' ' + dt_mif + ' -dkt ' + dk_mif + ' -fslgrad ' + coreg_bvecs + ' ' + coreg_bvals + ' -force')
                os.system(
                    'tensor2metric  -fa ' + fa_mif + ' ' + dt_mif + ' -adc ' + md_mif + ' -ad ' + ad_mif + ' -rd ' + rd_mif + ' -force')

                # os.system('mrview '+ fa_mif) #inspect residual
            else:
                os.system(
                    'dwi2tensor ' + resampled_mif_path + ' ' + dt_mif + ' -fslgrad ' + coreg_bvecs + ' ' + coreg_bvals + ' -force')
                os.system('tensor2metric  -fa ' + fa_mif + ' ' + dt_mif + ' -force')
                os.system('tensor2metric  -rd ' + rd_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
                os.system('tensor2metric  -ad ' + ad_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
                os.system(
                    'tensor2metric  -adc ' + md_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
                # os.system('mrview '+ fa_mif) #inspect residual

            command = 'mrconvert ' + fa_mif + ' ' + fa_nii + ' -force'
            print(command)
            os.system(command)

    if verbose:
        print(f'Created the file {fa_nii}')

    # T1_orig_res = os.path.join(subj_out_folder,subj+'_T1_res.nii.gz')
    # T1_orig = T1_orig_res

    if cleanup:
        shutil.rmtree(scratch_path)
        
    """

    wmfod_norm_mif = os.path.join(subj_out_folder, subj_min + '_wmfod_norm.mif' + index_gz)
    gmfod_norm_mif = os.path.join(subj_out_folder, subj_min + '_gmfod_norm.mif' + index_gz)
    csffod_norm_mif = os.path.join(subj_out_folder, subj_min + '_csffod_norm.mif' + index_gz)

    if denoise:
        output_denoise = subj_nii_path.replace('.nii.gz','_denoised.nii.gz')
        if not os.path.exists(output_denoise) or overwrite:
            os.system('dwidenoise ' + subj_nii_path + ' ' + output_denoise + ' -force')
        output_denoise = subj_nii_path

    if fastrun:
        smallerTracks = os.path.join(perm_subj_output, subj_min + f'_smallerTracks10000{bvec_string}{denoise_str}.tck')
    else:
        smallerTracks = os.path.join(perm_subj_output, subj_min + f'_smallerTracks2mill{bvec_string}{denoise_str}.tck')

    if keep_10mil:
        tracks_10M_tck = os.path.join(perm_subj_output, subj_min + denoise_str + '_tracks_10M.tck')
    else:
        tracks_10M_tck = os.path.join(subj_out_folder, subj_min + denoise_str + '_tracks_10M.tck')


    if not os.path.exists(smallerTracks) or (keep_10mil and not os.path.exists(tracks_10M_tck)):

        # Estimating the Basis Functions:
        if not os.path.exists(wmfod_norm_mif) or overwrite:
            wm_txt = os.path.join(subj_out_folder, subj_min + '_wm.txt')
            gm_txt = os.path.join(subj_out_folder, subj_min + '_gm.txt')
            csf_txt = os.path.join(subj_out_folder, subj_min + '_csf.txt')
            voxels_mif = os.path.join(subj_out_folder, subj_min + '_voxels.mif' + index_gz)

            subj_mif_path = subj_nii_path.replace('.nii',f'{bvec_string}.mif')

            overwrite=True

            if not os.path.exists(subj_mif_path) or overwrite:
                os.system(f'mrconvert {subj_nii_path} {subj_mif_path} -fslgrad {coreg_bvecs} {coreg_bvals} -bvalue_scaling 0 -force')

            ##Right now we are using the RESAMPLED mif of the 4D miff, to be discussed
            if not os.path.exists(voxels_mif) or not os.path.exists(wm_txt) or not os.path.exists(
                    gm_txt) or not os.path.exists(csf_txt) or overwrite:
                command = f'dwi2response dhollander {subj_mif_path} {wm_txt} {gm_txt} {csf_txt} -voxels {voxels_mif} -mask {mask_mif_path} -scratch {subj_out_folder} -fslgrad {coreg_bvecs} {coreg_bvals} -force'
                print(command)
                os.system(command)

            # Applying the basis functions to the diffusion data:
            wmfod_mif = os.path.join(subj_out_folder, subj_min + '_wmfod.mif' + index_gz)
            gmfod_mif = os.path.join(subj_out_folder, subj_min + '_gmfod.mif' + index_gz)
            csffod_mif = os.path.join(subj_out_folder, subj_min + '_csffod.mif' + index_gz)


            # os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force' )
            if not os.path.exists(wmfod_mif) or overwrite:
                #command = 'dwi2fod msmt_csd ' + subj_mif_path + ' -mask ' + mask_mif_path + ' ' + wm_txt + ' ' + wmfod_mif + ' ' + gm_txt + ' ' + gmfod_mif + ' ' + csf_txt + ' ' + csffod_mif + ' -force'
                #Only doing white matter in mouse brain
                command = f'dwi2fod msmt_csd {subj_mif_path} -mask {mask_mif_path} {wm_txt} {wmfod_mif} -force'
                print(command)
                os.system(command)

            """
            # combine to single image to view them
            # Concatenating the FODs:
            vf_mif = os.path.join(subj_out_folder, subj_min + '_vf.mif')
            if not os.path.exists(vf_mif) or overwrite:
                command = 'mrconvert -coord 3 0 ' + wmfod_mif + ' -| mrcat ' + csffod_mif + ' ' + gmfod_mif + ' - ' + vf_mif + ' -force'
                print(command)
                os.system(command)
            # os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat ' +gmfod_mif+ ' - ' + vf_mif+' -force' ) # without csf
            """

            if not os.path.exists(wmfod_norm_mif) or not os.path.exists(gmfod_norm_mif) or not os.path.exists(
                    csffod_norm_mif) or overwrite:
                #command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' ' + gmfod_mif + ' ' + gmfod_norm_mif + ' ' + csffod_mif + ' ' + csffod_norm_mif + ' -mask ' + mask_mif_path + '  -force'
                command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' -mask ' + mask_mif_path + '  -force'
                print(command)
                os.system(command)

        gmwmSeed_coreg_mif = mask_mif_path

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

    if verbose:
        print(f'Created {smallerTracks}')

    if cleanup:
        shutil.rmtree(subj_out_folder)