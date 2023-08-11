
import os, subprocess
from DTC.file_manager.file_tools import mkcdir
import socket

from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
import numpy as np


def mkcdir(folderpaths, sftp=None):
    #creates new folder only if it doesnt already exists

    if sftp is None:
        if np.size(folderpaths) == 1:
            if not os.path.exists(folderpaths):
                os.mkdir(folderpaths)
        else:
            for folderpath in folderpaths:
                if not os.path.exists(folderpath):
                    os.mkdir(folderpath)
    else:
        if np.size(folderpaths) == 1:
            try:
                sftp.chdir(folderpaths)
            except:
                sftp.mkdir(folderpaths)
        else:
            for folderpath in folderpaths:
                try:
                    sftp.chdir(folderpath)
                except:
                    sftp.mkdir(folderpath)


def median_mask_make(inpath, outpath=None, outpathmask=None, median_radius=4, numpass=4, binary_dilation=None,
                     vol_idx=None, affine=None, verbose = False):
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
    if os.path.exists(outpath) and os.path.exists(outpathmask):
        print('Already wrote mask')
        return outpath, outpathmask
    data = np.squeeze(data)
    data_masked, mask = median_otsu(data, median_radius=median_radius, numpass=numpass, dilate=binary_dilation,
                                    vol_idx=vol_idx)
    save_nifti(outpath, data_masked.astype(np.float32), affine)
    save_nifti(outpathmask, mask.astype(np.float32), affine)
    if verbose:
        print(f'Saved masked file to {outpath}, saved mask to {outpathmask}')
    return outpath, outpathmask

data_path = '/Volumes/dusom_mousebrains/All_Staff/Projects/ADRC/Data/raw_ADRC_data/ADRC-20230511/'
se_epi_manual_pe_scheme_txt = '/Volumes/Data/Badea/Lab/mouse/ADRC_mrtrix_dwifsl/se_epi_manual_pe_scheme.txt'

data_path_output = '/Volumes/dusom_mousebrains/All_Staff/Projects/ADRC/Data/raw_ADRC_data/ADRC_JS_tests/'

root = '/Volumes/Data/Badea/Lab/mouse/ADRC_mrtrix_dwifsl/'
dwi_manual_pe_scheme_txt = os.path.join(root, 'dwi_manual_pe_scheme.txt')
bvec_folder = os.path.join(root, 'perm_files/')
path_perm = os.path.join(root, 'perm_files/')

mkcdir(data_path_output)

subjects = ['ADRC0001']

index_gz = '.gz'
overwrite = False


for subj in subjects:

    subj_folder = os.path.join(data_path, subj,'visit1')
    subj_out_folder = os.path.join(data_path_output, subj)
    scratch_path = os.path.join(data_path_output, 'scratch')
    mkcdir([subj_out_folder,scratch_path])

    bvec_path_orig = os.path.join(bvec_folder, subj + '_bvec.txt')
    bval_path_orig = os.path.join(bvec_folder, subj + '_bvals.txt')
    bvec_path_PA = os.path.join(bvec_folder, subj+'_bvec_rvrs.txt')
    bval_path_PA = os.path.join(bvec_folder, subj+'_bvals_rvrs.txt')

    DTI_forward_nii_gz = os.path.join(subj_folder,'HCP_DTI.nii.gz')

    ######### denoise:

    out_mif = os.path.join(subj_out_folder, subj + '_subjspace_dwi.mif' + index_gz)
    if not os.path.exists(out_mif) or overwrite:
        os.system(
            'mrconvert ' + DTI_forward_nii_gz + ' ' + out_mif + ' -fslgrad ' + bvec_path_orig + ' ' + bval_path_orig + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000

    output_denoise = os.path.join(subj_out_folder, subj+'_den.mif')
    if not os.path.exists(output_denoise) or overwrite:
        os.system('dwidenoise ' + out_mif + ' ' + output_denoise + ' -force')  # denoising
    # compute residual to check if any resdiual is loaded on anatomy
    output_residual = os.path.join(subj_out_folder, subj + 'residual.mif')
    if not os.path.exists(output_residual) or overwrite:
        os.system(
            'mrcalc ' + out_mif + ' ' + output_denoise + ' -subtract ' + output_residual + ' -force')  # compute residual
    # os.system('mrview '+ output_residual) #inspect residual

    ##################
    ##### b0 warp unphasing

    nii_gz_path_PA = os.path.join(subj_folder, 'HCP_DTI_reverse_phase.nii.gz')

    PA_mif = os.path.join(subj_out_folder,subj+'_PA.mif')
    if not os.path.exists(PA_mif) or overwrite:
        os.system('mrconvert '+nii_gz_path_PA+ ' ' +PA_mif + ' -force') #PA to mif

    mean_b0_PA_mif = os.path.join(subj_out_folder,subj+'mean_b0_PA.mif')
        #take mean of PA and AP to unwarp
    if not os.path.exists(mean_b0_PA_mif) or overwrite:
        os.system('mrconvert '+PA_mif+ ' -fslgrad '+bvec_path_PA+ ' '+ bval_path_PA + ' - | mrmath - mean '+ mean_b0_PA_mif+' -axis 3 -force')

    #Extracting b0 images from the AP dataset, and concatenating the b0 images across both AP and PA images:
    mean_b0_AP_mif = os.path.join(subj_out_folder,subj+'mean_b0_AP.mif')
    if not os.path.exists(mean_b0_AP_mif) or overwrite:
        os.system('dwiextract '+output_denoise+ ' - -bzero | mrmath - mean '+ mean_b0_AP_mif+' -axis 3 -force')
    b0_pair_mif = os.path.join(subj_out_folder,subj+'b0_pair.mif')
    if not os.path.exists(b0_pair_mif) or overwrite:
        os.system('mrcat '+mean_b0_AP_mif+ ' ' +mean_b0_PA_mif+' -axis 3 '+ b0_pair_mif+' -force')

    ########### converting the b0 unwarped

    se_epi_mif = os.path.join(data_path_output,'se_epi.mif')
    if not os.path.exists(se_epi_mif) or overwrite:
        command = 'mrconvert ' + b0_pair_mif + " " + se_epi_mif + ' -force'
        os.system(command)

    ###### padding the unwarped b0s

    se_epi_pad2_mif = os.path.join(data_path_output, 'se_epi_pad2.mif')
    end_vol = subprocess.check_output('mrinfo -size ' + out_mif + " | awk '{print $3}' ", shell=True)
    end_vol = end_vol.decode()
    end_vol = str(int(end_vol[0:2]) - 1)
    if not os.path.exists(se_epi_pad2_mif) or overwrite:
        command = 'mrconvert ' + se_epi_mif + ' -coord 2 ' + end_vol + ' - | mrcat ' + se_epi_mif + ' - ' + se_epi_pad2_mif + ' -axis 2 -force'
        print(command)
        os.system(command)

    ##### converting unwarped b0s to 'topup'
    topup_in_nii = os.path.join(data_path_output, 'topup_in.nii')
    topup_datain_txt =  os.path.join(data_path_output, 'topup_datain.txt')
    if not os.path.exists(topup_in_nii) or not os.path.exists(topup_datain_txt) or overwrite:
        command = 'mrconvert '+ se_epi_pad2_mif +' ' + topup_in_nii + ' -import_pe_table ' + se_epi_manual_pe_scheme_txt + ' -strides -1,+2,+3,+4 -export_pe_table '+ topup_datain_txt + ' -force'
        print(command)
        os.system(command)

    #### getting the field map out of the topup info
    field_map_nii_gz = os.path.join(data_path_output, 'field_map.nii.gz')
    dwi_topup = os.path.join(data_path_output, 'dwi_topup')
    dwi_topup_nii = dwi_topup + '_fieldcoef.nii.gz'
    command = 'topup --imain=' + topup_in_nii + ' --datain=' + topup_datain_txt + ' --out=field --fout=' + field_map_nii_gz + ' --config=$FSLDIR/src/fsl-topup/flirtsch/b02b0.cnf --out=' + dwi_topup + ' --verbose'
    if not os.path.exists(field_map_nii_gz) or not os.path.exists(dwi_topup_nii) or overwrite:
        print(command)
        os.system(command)


    #### converting the original diff to a mif
    dwi_mif = os.path.join(subj_out_folder, 'dwi.mif')
    dwi_json = os.path.join(subj_out_folder, 'dwi.json')
    if not os.path.exists(dwi_mif) or overwrite:
        command = 'mrconvert ' + output_denoise + " " + dwi_mif + ' -fslgrad ' + bvec_path_orig + ' ' + bval_path_orig + ' -json_export ' + dwi_json + ' -force'
        print(command)
        os.system(command)


    #### padding the diff mif
    dwi_pad2_mif = scratch_path + 'dwi_pad2.mif'
    if not os.path.exists(dwi_pad2_mif):
        command = 'mrconvert ' + dwi_mif + ' -coord 2 ' + end_vol + '  -clear dw_scheme - | mrcat ' + dwi_mif + ' - ' + dwi_pad2_mif + ' -axis 2 -force'
        print(command)
        os.system(command)

    #### Command: Getting the popup config
    applytopup_config_txt = os.path.join(scratch_path, 'applytopup_config.txt')
    applytopup_indices_txt = os.path.join(scratch_path, 'applytopup_indices.txt')
    dwi_manual_pe_scheme_txt = os.path.join(root, 'dwi_manual_pe_scheme.txt')
    if not os.path.exists(applytopup_config_txt):
        command = 'mrconvert ' + dwi_pad2_mif + ' -import_pe_table ' + dwi_manual_pe_scheme_txt + ' - | mrinfo - -export_pe_eddy ' + applytopup_config_txt + ' ' + applytopup_indices_txt + ' -force'
        print(command)
        os.system(command)


    #### Converting the padded diff back to a nii
    dwi_pad2_pe_0_nii = os.path.join(subj_out_folder, 'dwi_pad2_pe_0.nii')
    dwi_pad2_pe_0_json = os.path.join(subj_out_folder, 'dwi_pad2_pe_0.json')
    if not os.path.exists(dwi_pad2_pe_0_nii) or overwrite:
        command = 'mrconvert ' + dwi_pad2_mif + ' ' + dwi_pad2_pe_0_nii + ' -strides -1,+2,+3,+4 -json_export ' + dwi_pad2_pe_0_json + ' -force'
        print(command)
        os.system(command)

    #### Command:  applytopup --imain=dwi_pad2_pe_0.nii --datain=applytopup_config.txt --inindex=1 --topup=field --out=dwi_pad2_pe_0_applytopup.nii --method=jac
    dwi_pad2_pe_0_applytopup_nii = os.path.join(subj_out_folder, 'dwi_pad2_pe_0_applytopup.nii')
    command = 'applytopup --imain=' + dwi_pad2_pe_0_nii + ' --datain=' + applytopup_config_txt + ' --inindex=1 --topup=' + dwi_topup + ' --out=' + dwi_pad2_pe_0_applytopup_nii + ' --method=jac'
    if not os.path.exists(dwi_pad2_pe_0_applytopup_nii+index_gz) or overwrite:
        print(command)
        os.system(command)

    #### Converting dwi topupped nifti to mif
    dwi_pad2_pe_0_applytopup_nii = dwi_pad2_pe_0_applytopup_nii + '.gz'
    dwi_pad2_pe_0_applytopup_mif = os.path.join(subj_out_folder, 'dwi_pad2_pe_0_applytopup.mif')
    if not os.path.exists(dwi_pad2_pe_0_applytopup_mif) or overwrite:
        command = 'mrconvert ' + dwi_pad2_pe_0_applytopup_nii + ' ' + dwi_pad2_pe_0_applytopup_mif + ' -json_import ' + dwi_pad2_pe_0_json + ' -force'
        print(command)
        os.system(command)

    #### Command: Create the mask for the eddy
    eddy_mask_nii = os.path.join(subj_out_folder, 'eddy_mask.nii')
    if not os.path.exists(eddy_mask_nii) or overwrite:
        command = 'dwi2mask ' + dwi_pad2_pe_0_applytopup_mif + ' - | maskfilter - dilate - | mrconvert - ' + eddy_mask_nii + ' -datatype float32 -strides -1,+2,+3 -force'
        print(command)
        os.system(command)

    #### Command: Preparing the padded dwi for eddy command
    bvecs_eddy = path_perm + subj + "_bvecs_eddy.txt"
    bvals_eddy = path_perm + subj + "_bvals_eddy.txt"
    eddy_config_txt = os.path.join(subj_out_folder, 'eddy_config.txt')
    eddy_indices_txt = os.path.join(subj_out_folder, 'eddy_indices.txt')
    eddy_in_nii = os.path.join(subj_out_folder, 'eddy_in.nii')
    if not os.path.exists(eddy_in_nii) or overwrite:
        command = 'mrconvert ' + dwi_pad2_mif + ' -import_pe_table ' + dwi_manual_pe_scheme_txt + ' ' + eddy_in_nii + ' -strides -1,+2,+3,+4 -export_grad_fsl ' + bvecs_eddy + ' ' + bvals_eddy + ' -export_pe_eddy ' + eddy_config_txt + " " + eddy_indices_txt + ' -force'
        print(command)
        os.system(command)

    #### Command: Running the eddy command
    dwi_post_eddy = scratch_path + "dwi_post_eddy.nii"
    dwi_post_eddy_gz = dwi_post_eddy + '.gz'
    if not os.path.exists(dwi_post_eddy_gz) or overwrite:
        if socket.gethostname().split('.')[0]=='santorini':
            command = 'eddy --imain=' + eddy_in_nii + ' --mask=' + eddy_mask_nii + ' --acqp=' + eddy_config_txt + ' --index=' + eddy_indices_txt + ' --bvecs=' + bvecs_eddy + ' --bvals=' + bvals_eddy + ' --topup=' + dwi_topup + ' --slm=linear --data_is_shelled --out=' + dwi_post_eddy + ' --verbose'
        else:
            command = 'eddy_openmp --imain=' + eddy_in_nii + ' --mask=' + eddy_mask_nii + ' --acqp=' + eddy_config_txt + ' --index=' + eddy_indices_txt + ' --bvecs=' + bvecs_eddy + ' --bvals=' + bvals_eddy + ' --topup=' + dwi_topup + ' --slm=linear --data_is_shelled --out=' + dwi_post_eddy + ' --verbose'
        print(command)
        os.system(command)

    #### Command: Converting the eddy corrected diffusion file nifti into the mif version, also unpadded the file at same time (using command  -coord 2 0:' + end_vol)
    den_preproc_mif = os.path.join(subj_out_folder,subj + '_den_preproc.mif')
    dwi_post_eddy_eddy_rotated_bvecs = scratch_path + 'dwi_post_eddy.bvecs'
    if not os.path.exists(den_preproc_mif) or not os.path.exists(dwi_post_eddy_eddy_rotated_bvecs) or overwrite:
        command = 'mrconvert ' + dwi_post_eddy_gz + ' ' + den_preproc_mif + ' -coord 2 0:' + end_vol + ' -strides -1,-2,3,4 -fslgrad ' + dwi_post_eddy_eddy_rotated_bvecs + ' ' + bvals_eddy + ' -force'
        print(command)
        os.system(command)

    #### Command: Running the dwi bias correction on the eddy corrected image
    den_unbiased_mif = os.path.join(subj_out_folder, subj + '_den_preproc_unbiased.mif')
    bias_mif = os.path.join(subj_out_folder, subj + '_bias.mif')
    if not os.path.exists(den_unbiased_mif) or overwrite:
        command = 'dwibiascorrect ants ' + den_preproc_mif + ' ' + den_unbiased_mif + ' -scratch ' + subj_out_folder + ' -bias ' + bias_mif + ' -force'
        print(command)
        os.system(command)

    #### Command: Extracting the b0s from the bias corrected diffusion file and making the average diffusion weighted image based on thsoe
    dwi_mif = os.path.join(subj_out_folder, subj + '_dwi.mif')
    command = 'dwiextract ' + den_unbiased_mif + ' - -no_bzero | mrmath - mean ' + dwi_mif + ' -axis 3 -force'
    if not os.path.exists(dwi_mif) or overwrite:
        print(command)
        os.system('echo ' + command)
        os.system(command)

    #### Command: Converting the diffusion mean of b0s into a nifti
    dwi_nii_gz = os.path.join(subj_out_folder, subj + '_dwi.nii.gz')
    if not os.path.exists(dwi_nii_gz) or overwrite:
        command = 'mrconvert ' + dwi_mif + ' ' + dwi_nii_gz + ' -force'
        print(command)
        os.system(command)

    #### Command: Converting the bias corrected diffusion mif file into the the nifti version (and extracting bvals/bvecs)
    coreg_bvecs = os.path.join(subj_out_folder, subj + '_coreg_bvecs.txt')
    coreg_bvals = os.path.join(subj_out_folder, subj + '_coreg_bvals.txt')
    coreg_nii_gz = os.path.join(subj_out_folder, subj + '_coreg.nii.gz')
    command = 'mrconvert ' + den_unbiased_mif + ' ' + coreg_nii_gz + ' -export_grad_fsl ' + coreg_bvecs + ' ' + coreg_bvals + ' -force'
    if not os.path.exists(coreg_nii_gz) or overwrite:
        print(command)
        os.system(command)

    """
    #### Command: Making the mask (mrtrix version)
    mask_mif = os.path.join(path_perm, subj + '_mask.mif')
    if not os.path.exists(mask_mif) or overwrite:
        command = 'dwi2mask ' + den_unbiased_mif + ' ' + mask_mif + ' -force'
        print(command)
        os.system(command)

    #### Command: Converting the mask into a
    mask_mrtrix_nii = os.path.join(path_perm, subj + '_mask.nii.gz')
    if not os.path.exists(mask_mrtrix_nii) or overwrite:
        command = 'mrconvert ' + mask_mif + ' ' + mask_mrtrix_nii + ' -force'
        print(command)
        os.system(command)
    """

    masked_nii_path = os.path.join(subj_out_folder, subj + '_dwi_masked.nii.gz')
    mask_nii_path = os.path.join(subj_out_folder, subj + '_mask.nii.gz')

    median_radius = 4
    numpass = 7
    binary_dilation = 3
    full_name = False
    verbose = False

    if not os.path.exists(masked_nii_path) or not os.path.exists(mask_nii_path) or overwrite:
        median_mask_make(dwi_nii_gz, masked_nii_path, median_radius=median_radius, binary_dilation=binary_dilation,
                         numpass=numpass, outpathmask=mask_nii_path, verbose=verbose)

    resampled_path = os.path.join(subj_out_folder, subj + '_coreg_resampled.nii.gz')

    command = f'ResampleImage 4 {coreg_nii_gz} {resampled_path} 1.5x1.5x1.5x1 0 0 2'
    if not os.path.exists(resampled_path) or overwrite:
        print(command)
        os.system(command)