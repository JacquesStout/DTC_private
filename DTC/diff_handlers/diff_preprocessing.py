
"""
Created by Jacques Stout
Part of the DTC pipeline, mostly handles dwi files before calculating trk.
Tries to create masks, determines the parameters of a denoising request, handles fa files, etc
"""

"""
import matplotlib.pyplot as plt
from dipy.core.histeq import histeq
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
import os
from DTC.diff_handlers.denoise_processes import mppca
from dipy.denoise.gibbs import gibbs_removal
from time import time
from DTC.visualization_tools.figures_handler import denoise_fig
import glob
from DTC.nifti_handlers.atlas_handlers.mask_handler import applymask_array
import nibabel as nib
from DTC.file_manager.computer_nav import checkfile_exists_remote, load_nifti_remote, save_nifti_remote
import shutil
import warnings
#from os.path import join as pjoin
#from dipy.data import get_fnames
#from nifti_handlers.nifti_handler import getdwidata
#from gunnies.basic_LPCA_denoise import basic_LPCA_denoise_func
"""

#from nifti_handlers.transform_handler import img_transform_exec, space_transpose, affine_superpose, header_superpose
from DTC.nifti_handlers.transform_handler import *
import subprocess
from DTC.file_manager.file_tools import largerfile, mkcdir, getext, buildlink
import glob
from DTC.nifti_handlers.atlas_handlers.mask_handler import applymask_samespace, median_mask_make, mask_fixer
import time
from DTC.gunnies.basic_LPCA_denoise import basic_LPCA_denoise_func
from numpy import sum
import numpy as np

def string_inclusion(string_option,allowed_strings,option_name):
    "checks if option string is part of the allowed list of strings for that option"
    try:
        string_option=string_option.lower()
    except AttributeError:
        if string_option is None:
            #raise Warning(option_name + " stated as None value, option will not be implemented")
            print(option_name + " stated as None value, option will not be implemented")
            return
        else:
            raise AttributeError('Unrecognized value for ' + option_name)

    if not any(string_option == x for x in allowed_strings):
        raise ValueError(string_option + " is an unrecognized string, please check your input for " + option_name)
    if string_option == "none":
        print(option_name + " stated as None value, option will not be implemented")


def dwi_to_mask(data, subject, affine, outpath, masking='median', makefig=False, vol_idx=None, median_radius = 5,
                numpass=6, dilate = 2, forcestart = False, header = None, verbose = False, sftp=None):

    data = np.squeeze(data)
    binarymask_path = os.path.join(outpath, subject + '_dwi_binary_mask.nii.gz')
    maskeddwi_path = os.path.join(outpath, subject + '_dwi_mask.nii.gz')
    if not checkfile_exists_remote(binarymask_path,sftp) and not checkfile_exists_remote(maskeddwi_path, sftp) \
            and not forcestart:
        if masking == 'median':
            b0_mask, mask = median_otsu(data, vol_idx=vol_idx, median_radius=median_radius, numpass=numpass,
                                        dilate=dilate)
        if masking == 'extract':
            if np.size(np.shape(data)) == 3:
                mask=data>0
            if np.size(np.shape(data)) == 4:
                mask=data[:,:,:,0]>0
            mask = mask.astype(np.float32)
            b0_mask = applymask_array(data,mask)

        if verbose:
            txt = f"Creating binarymask at {binarymask_path} and masked data at {maskeddwi_path}"
            print(txt)
        if header is None:
            save_nifti(binarymask_path, mask, affine)
            save_nifti(maskeddwi_path, b0_mask.astype(np.float32), affine)
        else:
            binarymask_nii = nib.Nifti1Image(mask, affine, header)
            save_nifti_remote(binarymask_nii, binarymask_path, sftp=sftp)
            #maskeddwi_nii = nib.Nifti1Image(b0_mask, affine, header)
            #save_nifti_remote(maskeddwi_nii, maskeddwi_path, sftp=sftp)
    else:
        mask = load_nifti_remote(binarymask_path,sftp=sftp)
        mask = mask[0]
        b0_mask = load_nifti_remote(maskeddwi_path,sftp=sftp)
        b0_mask = b0_mask[0]

    if makefig:
        sli = data.shape[2] // 2
        if len(b0_mask.shape) ==4:
            b0_mask_2 = b0_mask[:,:,:,0]
        else:
            b0_mask_2 = b0_mask
        if len(data.shape) ==4:
            data = data[:,:,:,0]
        plt.figure('Brain segmentation')
        plt.subplot(1, 2, 1).set_axis_off()
        plt.imshow(histeq(data[:, :, sli].astype('float')).T,
                   cmap='gray', origin='lower')

        plt.subplot(1, 2, 2).set_axis_off()
        plt.imshow(histeq(b0_mask_2[:, :, sli].astype('float')).T,
                   cmap='gray', origin='lower')
        plt.savefig(outpath + 'median_otsu.png')

    return(mask.astype(np.float32), b0_mask.astype(np.float32))

def dwi_to_mask_old(data, subject, affine, outpath, masking='median', makefig=False, vol_idx=None, median_radius = 5,
                numpass=6, dilate = 2, forcestart = False, header = None, verbose = False, sftp=None):

    data = np.squeeze(data)
    binarymask_path = os.path.join(outpath, subject + '_dwi_binary_mask.nii.gz')
    maskeddwi_path = os.path.join(outpath, subject + '_dwi_mask.nii.gz')
    if not os.path.exists(binarymask_path) and not os.path.exists(maskeddwi_path) and not forcestart:
        if masking == 'median':
            b0_mask, mask = median_otsu(data, vol_idx=vol_idx, median_radius=median_radius, numpass=numpass,
                                        dilate=dilate)
        if masking == 'extract':
            if np.size(np.shape(data)) == 3:
                mask=data>0
            if np.size(np.shape(data)) == 4:
                mask=data[:,:,:,0]>0
            mask = mask.astype(np.float32)
            b0_mask = applymask_array(data,mask)

        if verbose:
            txt = f"Creating binarymask at {binarymask_path} and masked data at {maskeddwi_path}"
            print(txt)
        if header is None:
            save_nifti(binarymask_path, mask, affine)
            save_nifti(maskeddwi_path, b0_mask.astype(np.float32), affine)
        else:
            binarymask_nii = nib.Nifti1Image(mask, affine, header)
            nib.save(binarymask_nii, binarymask_path)
            maskeddwi_nii = nib.Nifti1Image(b0_mask, affine, header)
            nib.save(maskeddwi_nii, maskeddwi_path)
    else:
        mask = load_nifti(binarymask_path)
        mask = mask[0]
        b0_mask = load_nifti(maskeddwi_path)
        b0_mask = b0_mask[0]

    if makefig:
        sli = data.shape[2] // 2
        if len(b0_mask.shape) ==4:
            b0_mask_2 = b0_mask[:,:,:,0]
        else:
            b0_mask_2 = b0_mask
        if len(data.shape) ==4:
            data = data[:,:,:,0]
        plt.figure('Brain segmentation')
        plt.subplot(1, 2, 1).set_axis_off()
        plt.imshow(histeq(data[:, :, sli].astype('float')).T,
                   cmap='gray', origin='lower')

        plt.subplot(1, 2, 2).set_axis_off()
        plt.imshow(histeq(b0_mask_2[:, :, sli].astype('float')).T,
                   cmap='gray', origin='lower')
        plt.savefig(outpath + 'median_otsu.png')

    return(mask.astype(np.float32), b0_mask.astype(np.float32))


def check_for_fa(outpath, subject, getdata=False):
    #Checks for fa files ('bmfa') in specified outpath folder. Returns with the path
    #whether it exists or not, and the fa nifti if specified to do so
    if os.path.isdir(outpath):
        outpathglobfa = os.path.join(outpath, subject + '_*fa.nii.gz')
    elif os.path.isfile(outpath):
        outpathglobfa = os.path.join(os.path.dirname(outpath), subject + '_*fa.nii.gz')
    outpathfa = glob.glob(outpathglobfa)
    if np.size(outpathfa) == 1:
        outpathfa=outpathfa[0]
        if getdata is True:
            fa = load_nifti(outpathfa)
            return outpathfa, True, fa
        else:
            return outpathfa, True, None
    elif np.size(outpathfa) == 0:
        outpathfa = outpathglobfa.replace("*fa","bmfa")
        return outpathfa, False, None

def make_tensorfit(data,mask,gtab,affine,subject,outpath, overwrite=False, forcestart = False, verbose=None):
    #Given dwi data, a mask, and other relevant information, creates the fa and saves it to outpath, unless
    #if it already exists, in which case it simply returns the fa

    from dipy.reconst.dti import TensorModel
    from dipy.io.image import load_nifti, save_nifti
    outpathbmfa, exists, _ = check_for_fa(outpath, subject, getdata = False)
    if exists and not forcestart:
        fa = load_nifti(outpathbmfa)
        fa_array = fa[0]
        if verbose:
            txt = "FA already computed at " + outpathbmfa
            print(txt)
        return outpathbmfa, fa_array
    else:
        if verbose:
            print('Calculating the tensor model from bval/bvec values of ', subject)
        tensor_model = TensorModel(gtab)

        t1 = time.time()
        if len(mask.shape) == 4:
            mask = mask[...,0]
        tensor_fit = tensor_model.fit(data, mask)

        duration1 = time.time() - t1
        if verbose:
            print(subject + ' DTI duration %.3f' % (duration1,))

        save_nifti(outpathbmfa, tensor_fit.fa, affine)
        if verbose:
            print('Saving subject'+ subject+ ' at ' + outpathbmfa)

        return outpathbmfa, tensor_fit.fa


def denoise_pick(data, affine, hdr, outpath, mask, type_denoise='macenko', processes=1, savedenoise=True, verbose=False,
                 forcestart=False, datareturn=False, display=None):
    allowed_strings = ['mpca', 'yes', 'all', 'gibbs', 'none', 'macenko']
    string_inclusion(type_denoise, allowed_strings, "type_denoise")

    if type_denoise == 'macenko' or type_denoise == 'mpca' or type_denoise == 'yes' or type_denoise == 'all':
        type_denoise = '_mpca_'
    if type_denoise == 'gibbs':
        type_denoise = "_gibbs"
    if type_denoise is None or type_denoise == 'none':
        type_denoise = "_"

    outpath_denoise = outpath + type_denoise + 'dwi.nii.gz'
    if os.path.exists(outpath_denoise) and not forcestart:
        if verbose:
            txt = "Denoising already done at " + outpath_denoise
            print(txt)
        if datareturn:
            data = load_nifti(outpath_denoise)
    else:
        if type_denoise == '_mpca_':
            # data, snr = marcenko_denoise(data, False, verbose=verbose)
            t = time()
            denoised_arr, sigma = mppca(data, patch_radius=2, return_sigma=True, processes=processes, verbose=verbose)
            save_nifti(outpath_denoise, denoised_arr, affine, hdr=hdr)
            if verbose:
                txt = ("Saved image at " + outpath_denoise)
                print(txt)
            mask = np.array(mask, dtype=bool)
            mean_sigma = np.mean(sigma[mask])
            b0 = denoised_arr[..., 0]

            mean_signal = np.mean(b0[mask])

            snr = mean_signal / mean_sigma
            if verbose:
                print("Time taken for local MP-PCA ", -t +
                      time())
                print("The SNR of the b0 image appears to be at " + str(snr))
            if display:
                denoise_fig(data, denoised_arr, type='macenko')
            data = denoised_arr

        if type_denoise == 'gibbs':
            outpath_denoise = outpath + '_gibbs.nii.gz'
            if os.path.exists(outpath_denoise) and not forcestart:
                if verbose:
                    txt = "Denoising already done at " + outpath_denoise
                    print(txt)
                if datareturn:
                    data = load_nifti(outpath_denoise)
            t = time()
            data_corrected = gibbs_removal(data, slice_axis=2)
            save_nifti(outpath_denoise, denoised_arr, affine, hdr=hdr)
            if verbose:
                print("Time taken for the gibbs removal ", - t + time())
            if display:
                denoise_fig(data, data_corrected, type='gibbs')

            data = data_corrected

        if type_denoise == "_":
            print('No denoising was done')
            save_nifti(outpath_denoise, data, affine, hdr=hdr)

    return data, outpath_denoise

def launch_preprocessing(subj, raw_nii, outpath, cleanup=False, nominal_bval=4000, SAMBA_inputs_folder=None,
                         shortcuts_all_folder = None, gunniespath="~/gunnies/", processes=1, masking="bet", ref=None,
                         transpose=None, overwrite=False, denoise='None', recenter=0, verbose=False):

    overwrite=False
    proc_name ="diffusion_prep_" # Not gonna call it diffusion_calc so we don't assume it does the same thing as the civm pipeline
    work_dir=os.path.join(outpath,proc_name+subj)
    """
    for filePath in glob.glob(os.path.join(work_dir,'*')):
        modTimesinceEpoc = os.path.getmtime(filePath)
        modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
        if modificationTime[5:7]=='09' and int(modificationTime[8:10])>9:
            os.remove(filePath)
    """
    if verbose:
        print(f"Processing diffusion data with runno/subj: {subj}")
        print(f"Work directory is {work_dir}")
    mkcdir(work_dir)

    sbatch_folder =os.path.join(work_dir,"sbatch")
    mkcdir(sbatch_folder)
    #nii_path = os.path.join(work_dir,'nii4D_'+subj + '.nii.gz')
    #if not os.path.exists(nii_path):
    #    shutil.copy(raw_nii, nii_path)
    nii_name=os.path.split(raw_nii)[1]
    niifolder = os.path.dirname(raw_nii)
    ext = ".nii.gz"
    nii_ext=getext(nii_name)
    bxheader = nii_name.replace(nii_ext,".bxh")
    bxheader = os.path.join(niifolder, bxheader)
    bvecs = os.path.join(work_dir, subj+"_bvecs.txt")
    bvals =bvecs.replace("bvecs","bvals")

    from DTC.diff_handlers.bvec_handler import writebfiles, extractbvals
    bvals, bvecs = extractbvals(work_dir, subj, outpath=work_dir, writeformat='dsi')

    if verbose:
        print(f"Original nifti is at {nii_name}\nbvecs are at {bvecs}\nbvals are at {bvals}\n")
    if not os.path.exists(bvecs):
        if verbose:
            print("Extracting diff directions")
        #print("Bvals and bvecs not found, using extractdiffdirs, however it it NOT RELIABLE, beware!")
        bvec_cmd = (f"extractdiffdirs --colvectors --writebvals --fieldsep='\t' --space=RAI {bxheader} {bvecs} {bvals}")
        os.system(bvec_cmd)

    # Make dwi for mask generation purposes.
    if np.size(masking.split('_')) > 1:
        median_radius = int(masking.split('_')[1])
    else:
        median_radius = 4
    binary_dilation = 8
    #tmp_mask = os.path.join(work_dir,f"{subj}_tmp_{str(median_radius)}_bd_{binary_dilation}_mask{ext}")
    tmp_mask = os.path.join(work_dir, f"{subj}_tmp_mask{ext}")
    tmp = tmp_mask.replace("_mask", "")

    raw_dwi = os.path.join(work_dir,f"{subj}_raw_dwi.nii.gz")
    b0_dwi = os.path.join(work_dir,f"{subj}_b0_dwi.nii.gz")  #test average of the b0 images to make a better mask
    orient_string = os.path.join(work_dir,"relative_orientation.txt")

    if shortcuts_all_folder is not None:
        #nii_path_link = os.path.join(shortcuts_all_folder, f"{subj}_rawnii{ext}")
        #if not os.path.exists(nii_path_link) or overwrite:
        #    buildlink(nii_path, nii_path_link)
        bvecs_new = os.path.join(shortcuts_all_folder, subj + "_bvecs.txt")
        bvals_new = os.path.join(shortcuts_all_folder, subj + "_bvals.txt")
        if not os.path.exists(bvecs_new) or not os.path.exists(bvals_new) or overwrite:
            shutil.copyfile(bvecs, bvecs_new)
            shutil.copyfile(bvals, bvals_new)

    final_mask = os.path.join(work_dir, f'{subj}_mask{ext}')

    #if (not os.path.exists(final_mask) and not os.path.exists(tmp_mask)) or overwrite:

    if not os.path.exists(tmp_mask) or not os.path.exists(tmp) or overwrite:
        if not os.path.exists(raw_dwi) or overwrite:
            select_cmd = f"select_dwi_vols {raw_nii} {bvals} {raw_dwi} {nominal_bval} -m"
            os.system(select_cmd)
        if not os.path.exists(b0_dwi) or overwrite:
            b0_val = 0
            while not os.path.exists(b0_dwi) or b0_val > (nominal_bval / 2):
                cmd = f'select_dwi_vols {raw_nii} {bvals} {b0_dwi} {b0_val}  -m;'
                os.system(cmd)
                b0_val += 50
        if not os.path.exists(tmp_mask) or overwrite:
            if 'median' in masking:
                median_mask_make(raw_dwi, tmp, median_radius = median_radius, binary_dilation = binary_dilation, numpass=7, outpathmask=tmp_mask)
                #median_mask_make(b0_dwi, tmp, outpathmask=tmp_mask, median_radius = median_radius, numpass=median_radius)
                #Rmedian_mask_make(b0_dwi, tmp, outpathmask='/Users/jas/jacques/Chavez_test_temp/b0_test.nii.gz', median_radius = median_radius, numpass=median_radius)
                #median_mask_make(raw_dwi, tmp, outpathmask='/Users/jas/jacques/Chavez_test_temp/007_mask_rad7.nii.gz',
                #                 median_radius=4, numpass=7)
            elif masking=="bet":
                bet_cmd = f"bet {raw_dwi} {tmp} -m -n -R"
                os.system(bet_cmd)
            elif masking=='None':
                shutil.copy(raw_dwi, tmp)
                data, affine, _, hdr, _ = load_nifti_remote(raw_dwi)
                data_mask = np.ones(np.shape(data))
                newnii = nib.Nifti1Image(data_mask, affine, hdr)
                nib.save(newnii, tmp_mask)
            elif masking=='premade': #temp solution for when matlab or other method made the mask
                mask_fixer(tmp_mask)
                if not os.path.exists(tmp) or overwrite:
                    applymask_samespace(raw_nii, tmp_mask, outpath=tmp)
            else:
                raise Exception("Unrecognized masking type")

    # I think this part is done later more properly:     if create_subj_space_files: for contrast in ['dwi', 'b0', 'mask']:
    #if SAMBA_inputs_folder is not None:
    #    mask_subj_link = os.path.join(SAMBA_inputs_folder,f'{subj}_subjspace_mask{ext}')
    #    if not os.path.exists(mask_subj_link) or overwrite:
    #        shutil.copy(tmp_mask, mask_subj_link)

    #if cleanup and (os.path.exists(tmp_mask) and os.path.exists(raw_dwi)):
    #    os.remove(raw_dwi)
    #overwrite=False
    # Run Local PCA Denoising algorithm on 4D nifti:



    

    masked_nii = os.path.join(work_dir, nii_name)
    if not "nii.gz" in masked_nii:
        masked_nii = masked_nii.replace(".nii", ".nii.gz")
    masked_nii = masked_nii.replace(ext, "_masked" + ext)

    if denoise.lower()=='lpca':
        D_subj=f'LPCA_{subj}';
    elif denoise.lower()=='mpca':
        D_subj=f'MPCA_{subj}';
    elif denoise=="None" or denoise is None:
        D_subj = f'{subj}'

    if denoise=="None" or denoise is None:
        denoised_nii = masked_nii
        if not os.path.exists(masked_nii) or overwrite:
            fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
            os.system(fsl_cmd)
    else:
        denoised_nii = os.path.join(work_dir,f"{D_subj}_nii4D.nii.gz")
        if not os.path.exists(denoised_nii) or overwrite:
            if not os.path.exists(masked_nii) or overwrite:
                fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
                os.system(fsl_cmd)
            basic_LPCA_denoise_func(subj,masked_nii,bvecs,denoised_nii, processes=processes,
                                    denoise=denoise, verbose=False) #to improve and make multiprocessing

    #if cleanup and os.path.exists(denoised_nii) and os.path.exists(masked_nii) and denoised_nii!=masked_nii:
    #    os.remove(masked_nii)

    # Run coregistration/eddy current correction:

    coreg_nii_old = f'{outpath}/co_reg_{D_subj}_m00-results/Reg_{D_subj}_nii4D{ext}';
    coreg_nii = os.path.join(work_dir,f'Reg_{D_subj}_nii4D{ext}')
    if not cleanup:
        coreg_nii=coreg_nii_old
    if not os.path.exists(coreg_nii) or overwrite:
        if not os.path.exists(coreg_nii_old) or overwrite:
            temp_cmd = os.path.join(gunniespath,'co_reg_4d_stack_tmpnew.bash')+f' {denoised_nii} {D_subj} 0 {outpath} 0';
            os.system(temp_cmd)
        if cleanup:
            shutil.move(coreg_nii_old,coreg_nii)

    if shortcuts_all_folder is not None:
        coreg_link = os.path.join(shortcuts_all_folder,f'{subj}_subjspace_coreg{ext}')
        if not os.path.exists(coreg_link) or overwrite:
            buildlink(coreg_nii, coreg_link)

    toeddy=False
    if toeddy:
        #fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
        #os.system(fsl_cmd)
        eddy_cmd = f"eddy --imain={coreg_nii} --mask={tmp_mask} --acqp=acq_params.txt --index={os.path.join(work_dir,'index.txt')} --bvecs={bvecs} --bvals={bvals} --topup=topup_results --repol --out = {os.path.join(work_dir,f'Reg_{D_subj}_nii4D_eddy{ext}')}"
        os.system(eddy_cmd)

    coreg_inputs=os.path.join(outpath,f'co_reg_{D_subj}_m00-inputs')
    coreg_work=coreg_inputs.replace('-inputs','-work')
    coreg_results=coreg_inputs.replace('-inputs','-results')
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_inputs):
        shutil.rmtree(coreg_inputs)
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_work):
        shutil.rmtree(coreg_work)
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_results):
        shutil.rmtree(coreg_results)

    # Generate tmp DWI:

    tmp_dwi_out=os.path.join(work_dir, f'{subj}_tmp_dwi{ext}')
    dwi_out=os.path.join(work_dir,f'{subj}_dwi{ext}')

    if not os.path.exists(tmp_dwi_out) or overwrite:
        cmd=f'select_dwi_vols {coreg_nii} {bvals} {tmp_dwi_out} {nominal_bval}  -m'
        os.system(cmd)

    # Generate tmp B0:
    tmp_b0_out=os.path.join(work_dir,f'{subj}_tmp_b0{ext}')
    b0_out = os.path.join(work_dir, f'{subj}_b0{ext}')
    #if (not os.path.exists(b0_out) and not os.path.exists(tmp_b0_out)) or overwrite:
    if not os.path.exists(tmp_b0_out) or overwrite:
        b0_val = 0
        while not os.path.exists(tmp_b0_out) or b0_val>(nominal_bval/2):
            cmd=f'select_dwi_vols {coreg_nii} {bvals} {tmp_b0_out} {b0_val}  -m;'
            os.system(cmd)
            b0_val+=50

    #overwrite=False
    #elif cleanup and os.path.exists(tmp_b0_out):
    #    os.remove(tmp_b0_out)

    # Generate DTI contrasts and perform some tracking QA:
    if cleanup:
        c_string=' --cleanup '
    else:
        c_string=''

    #Important note: this is what first creates the fa, md, etc
    if len(glob.glob(os.path.join(work_dir,f'*.fib.gz.md{ext}'))) == 0 or overwrite:
        if overwrite:
            oldfiles = glob.glob(os.path.join(work_dir, f'*.fib.gz*'))
            for oldfile in oldfiles:
                os.remove(oldfile)
        cmd = 'bash ' + os.path.join(gunniespath,'dti_qa_with_dsi_studio_weirdcall.bash')+f' {coreg_nii} {bvecs} {tmp_mask} {work_dir} {c_string}';
        os.system(cmd)

    #Save the subject space dti results

    #Generate tmp MD:
    for contrast in ['md']:
        real_file=largerfile(os.path.join(work_dir,f'*.fib.gz.{contrast}{ext}'))  #Catch the 'real file' for each contrast
        tmp_file = f'{work_dir}/{subj}_tmp_{contrast}{ext}';
        if not os.path.exists(tmp_file):
            shutil.copy(real_file,tmp_file)

    tmp_md = f'{work_dir}/{subj}_tmp_md{ext}';

    if ref=="md" or ref is None:
        reference=tmp_md
    elif ref=="coreg":
        reference=coreg_nii
    elif os.path.exists(ref):
        reference=ref

    reference_file = os.path.join(work_dir, f'{subj}_reference{ext}')
    if not os.path.exists(reference_file):
        shutil.copy(reference, reference_file)

    if shortcuts_all_folder is not None:
        bonus_ref_link = os.path.join(shortcuts_all_folder, f'{subj}_reference{ext}')
        if not os.path.exists(bonus_ref_link) or overwrite:
            buildlink(reference_file,bonus_ref_link)

    #give new header to the non-dti files using md as reference


    for contrast in ['dwi', 'b0', 'mask']:
        tmp_file=os.path.join(work_dir,f'{subj}_tmp_{contrast}{ext}')
        tmp2_file=os.path.join(work_dir,f'{subj}_tmp2_{contrast}{ext}')
        final_file=os.path.join(work_dir,f'{subj}_{contrast}{ext}')
        if ((not os.path.exists(tmp2_file) and not os.path.exists(final_file)) or overwrite):
            if not os.path.exists(tmp_file):
                txt = f"Tmp file was not created for subject {subj}, need to rerun previous processes"
                raise Exception(txt)
            else:
                header_superpose(reference, tmp_file, outpath=tmp2_file)

    create_subj_space_files = True
    if create_subj_space_files:
        for contrast in ['dwi', 'b0', 'mask']:
            tmp_file = os.path.join(work_dir, f'{subj}_tmp_{contrast}{ext}')
            subj_file = os.path.join(work_dir, f'{subj}_subjspace_{contrast}{ext}')
            if not os.path.exists(subj_file) or overwrite:
                if not os.path.exists(tmp_file):
                    raise Exception("Tmp file was not created, need to rerun previous processes")
                else:
                    header_superpose(raw_dwi, tmp_file, outpath=subj_file)
            if shortcuts_all_folder is not None:
                subj_link = os.path.join(shortcuts_all_folder, f'{subj}_subjspace_{contrast}{ext}')
                if not os.path.exists(subj_link) or overwrite:
                    buildlink(subj_file, subj_link)

    #write the relative orientation file here
    if not os.path.isfile(orient_string) or overwrite:
        if os.path.isfile(orient_string):
            os.remove(orient_string)
        file = os.path.join(work_dir,subj+'_tmp_mask'+ext);
        cmd = 'bash ' + os.path.join(gunniespath,'find_relative_orientation_by_CoM.bash') + f' {reference_file} {file}'
        orient_relative = subprocess.getoutput(cmd)

        with open(orient_string, 'w') as f:
            f.write(orient_relative)
    else:
        orient_relative = open(orient_string, mode='r').read()

    if SAMBA_inputs_folder is not None:
        subj_orient_string = os.path.join(SAMBA_inputs_folder, f'{subj}_relative_orientation.txt')
        shutil.copy(orient_string, subj_orient_string)

    if shortcuts_all_folder is not None:
        subj_orient_string = os.path.join(shortcuts_all_folder, f'{subj}_relative_orientation.txt')
        shutil.copy(orient_string, subj_orient_string)

    #check extracted values from relative orientation vals
    orientation_out = orient_relative.split(',')[0]
    orientation_out = orientation_out.split(':')[1]
    orientation_in = orient_relative.split(',')[1]
    orientation_in = orientation_in.split(':')[1]
    if verbose:
        print(f'flexible orientation: {orientation_in}');
        print(f'reference orientation: {orientation_out}');

    #apply the orientation modification to specified contrasts
    for contrast in ['dwi', 'b0', 'mask']:
        img_in=os.path.join(work_dir,f'{subj}_tmp2_{contrast}{ext}')
        img_out=os.path.join(work_dir,f'{subj}_{contrast}{ext}')
        if not os.path.isfile(img_out) or overwrite:
            if orientation_out != orientation_in:
                print('TRYING TO REORIENT...b0 and dwi and mask')
                if os.path.exists(img_in) and (not os.path.exists(img_out) or overwrite):
                    img_transform_exec(img_in, orientation_in, orientation_out, img_out)
                    if os.path.exists(img_out):
                        os.remove(img_in)
                elif os.path.exists(img_out) and cleanup:
                    os.remove(img_in)
            else:
                shutil.move(img_in,img_out)

        if SAMBA_inputs_folder is not None:
            inputs_space_link = os.path.join(SAMBA_inputs_folder, f'{subj}_{contrast}{ext}')
            if not os.path.exists(inputs_space_link) or overwrite:
                buildlink(img_out, inputs_space_link)

        if shortcuts_all_folder is not None:
            inputs_space_link = os.path.join(shortcuts_all_folder, f'{subj}_{contrast}{ext}')
            if not os.path.exists(inputs_space_link) or overwrite:
                buildlink(img_out, inputs_space_link)


    mask = os.path.join(work_dir,f'{subj}_mask{ext}')
    b0 = os.path.join(work_dir,f'{subj}_b0{ext}')

    #if cleanup and os.path.exists(dwi_out) and os.path.exists(tmp_dwi_out):
    #    os.remove(tmp_dwi_out)

    for contrast in ['fa0', 'rd', 'ad', 'md']:
        real_file=largerfile(os.path.join(work_dir,f'*.fib.gz.{contrast}{ext}'))  # It will be fun times if we ever have more than one match to this pattern...
        #inputspace = real_file
        inputspace = os.path.join(work_dir, f'{subj}_inputspace_{contrast}{ext}')

        contrast=contrast.replace('0','')
        #linked_file=os.path.join(shortcut_dir,f'{subj}_{contrast}{ext}')
        linked_file_w=os.path.join(work_dir,f'{subj}_{contrast}{ext}')
        if not os.path.exists(inputspace) or overwrite:
            if orientation_out != orientation_in:
                affine_dwi = nib.load(dwi_out).affine
                if sum(affine_dwi[0,:3])<0 or sum(affine_dwi[1,:3])<0 or sum(affine_dwi[2,:3])<0:
                    real_reorient = real_file.replace(f'nii4D', f'nii4D_tmp')
                    real_reorient_2 = real_file.replace(f'nii4D', f'nii4D_tmp_2')
                    img_transform_affreset(real_file, affine_dwi, output_path=real_reorient)
                    img_transform_exec(real_reorient, orientation_in, orientation_out, real_reorient_2)
                    affine_superpose(dwi_out, real_reorient_2, outpath=inputspace, transpose=transpose)
                    os.remove(real_reorient)
                    os.remove(real_reorient_2)
                else:
                    affine_superpose(dwi_out, real_file, outpath=inputspace, transpose=transpose)
            else:
                affine_superpose(dwi_out, real_file, outpath = inputspace, transpose=transpose)
        if not os.path.isfile(linked_file_w) or overwrite:
            buildlink(inputspace, linked_file_w)
        if SAMBA_inputs_folder is not None:
            #warnings.warn('should reach this!')
            blinked_file = os.path.join(SAMBA_inputs_folder, f'{subj}_{contrast}{ext}')
            if not os.path.exists(blinked_file) or overwrite:
                buildlink(inputspace, blinked_file)
                print(f'build link from {inputspace} to {blinked_file}')

        if shortcuts_all_folder is not None:
            #warnings.warn('should reach this!')
            blinked_file = os.path.join(shortcuts_all_folder, f'{subj}_{contrast}{ext}')
            if not os.path.exists(blinked_file) or overwrite:
                buildlink(inputspace, blinked_file)
                print(f'build link from {inputspace} to {blinked_file}')




    if create_subj_space_files:
        for contrast in ['fa0', 'rd', 'ad', 'md']:

            real_file = largerfile(os.path.join(work_dir,
                                                f'*.fib.gz.{contrast}{ext}'))  # It will be fun times if we ever have more than one match to this pattern...
            contrast = contrast.replace('0', '')
            subj_file_tmp = os.path.join(work_dir, f'{subj}_subjspace_tmp_{contrast}{ext}')
            subj_file = os.path.join(work_dir, f'{subj}_subjspace_{contrast}{ext}')
            if not os.path.exists(subj_file) or overwrite:
                if not os.path.exists(subj_file_tmp):
                    if orientation_out != orientation_in:
                        print('TRYING TO REORIENT.ReineR..b0 and dwi and mask')
                        if os.path.exists(real_file) and (not os.path.exists(subj_file) or overwrite):
                            img_transform_exec(real_file, orientation_out, orientation_in, subj_file_tmp)
                    else:
                        shutil.copy(real_file,subj_file_tmp)
                header_superpose(raw_dwi, subj_file_tmp, outpath=subj_file)

            if shortcuts_all_folder is not None:
                subj_link = os.path.join(shortcuts_all_folder, f'{subj}_subjspace_{contrast}{ext}')
                buildlink(subj_file, subj_link)

    #if cleanup:
    #    tmp_files = glob.glob(os.path.join(work_dir, '*tmp*'))
    #    for file in tmp_files:
    #        os.remove(file)

    nii_path = os.path.join(work_dir,'nii4D_'+subj + ext)
    if os.path.exists(nii_path):
        os.remove(nii_path)

    if cleanup:
        reg_src = os.path.join(work_dir,'Reg_' + subj + f'_nii4D{ext}.src.gz')
        if os.path.exists(reg_src):
            os.remove(reg_src)
        reg_src_fib = os.path.join(work_dir,'Reg_' + subj + f'_nii4D{ext}.src.gz.dti.fib.gz')
        if os.path.exists(reg_src):
            os.remove(reg_src)


def launch_preprocessing_onlydwi(subj, raw_nii, outpath, cleanup=False, nominal_bval=4000, SAMBA_inputs_folder=None,
                         shortcuts_all_folder=None, gunniespath="~/gunnies/", processes=1, masking="bet", ref=None,
                         transpose=None, overwrite=False, denoise='None', recenter=0, verbose=False):
    overwrite = False
    proc_name = "diffusion_prep_"  # Not gonna call it diffusion_calc so we don't assume it does the same thing as the civm pipeline
    work_dir = os.path.join(outpath, proc_name + subj)
    """
    for filePath in glob.glob(os.path.join(work_dir,'*')):
        modTimesinceEpoc = os.path.getmtime(filePath)
        modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
        if modificationTime[5:7]=='09' and int(modificationTime[8:10])>9:
            os.remove(filePath)
    """
    if verbose:
        print(f"Processing diffusion data with runno/subj: {subj}")
        print(f"Work directory is {work_dir}")
    mkcdir(work_dir)

    sbatch_folder = os.path.join(work_dir, "sbatch")
    mkcdir(sbatch_folder)
    # nii_path = os.path.join(work_dir,'nii4D_'+subj + '.nii.gz')
    # if not os.path.exists(nii_path):
    #    shutil.copy(raw_nii, nii_path)
    nii_name = os.path.split(raw_nii)[1]
    niifolder = os.path.dirname(raw_nii)
    ext = ".nii.gz"
    nii_ext = getext(nii_name)
    bxheader = nii_name.replace(nii_ext, ".bxh")
    bxheader = os.path.join(niifolder, bxheader)
    bvecs = os.path.join(work_dir, subj + "_bvecs.txt")
    bvals = bvecs.replace("bvecs", "bvals")

    from DTC.diff_handlers.bvec_handler import writebfiles, extractbvals
    bvals, bvecs = extractbvals(work_dir, subj)

    if verbose:
        print(f"Original nifti is at {nii_name}\nbvecs are at {bvecs}\nbvals are at {bvals}\n")
    if not os.path.exists(bvecs):
        if verbose:
            print("Extracting diff directions")
        # print("Bvals and bvecs not found, using extractdiffdirs, however it it NOT RELIABLE, beware!")
        bvec_cmd = (f"extractdiffdirs --colvectors --writebvals --fieldsep='\t' --space=RAI {bxheader} {bvecs} {bvals}")
        os.system(bvec_cmd)

    # Make dwi for mask generation purposes.
    if np.size(masking.split('_')) > 1:
        median_radius = int(masking.split('_')[1])
    else:
        median_radius = 4
    binary_dilation = 8
    # tmp_mask = os.path.join(work_dir,f"{subj}_tmp_{str(median_radius)}_bd_{binary_dilation}_mask{ext}")
    tmp_mask = os.path.join(work_dir, f"{subj}_tmp_mask{ext}")
    tmp = tmp_mask.replace("_mask", "")

    raw_dwi = os.path.join(work_dir, f"{subj}_raw_dwi.nii.gz")
    b0_dwi = os.path.join(work_dir, f"{subj}_b0_dwi.nii.gz")  # test average of the b0 images to make a better mask
    orient_string = os.path.join(work_dir, "relative_orientation.txt")

    if shortcuts_all_folder is not None:
        # nii_path_link = os.path.join(shortcuts_all_folder, f"{subj}_rawnii{ext}")
        # if not os.path.exists(nii_path_link) or overwrite:
        #    buildlink(nii_path, nii_path_link)
        bvecs_new = os.path.join(shortcuts_all_folder, subj + "_bvecs.txt")
        bvals_new = os.path.join(shortcuts_all_folder, subj + "_bvals.txt")
        if not os.path.exists(bvecs_new) or not os.path.exists(bvals_new) or overwrite:
            shutil.copyfile(bvecs, bvecs_new)
            shutil.copyfile(bvals, bvals_new)

    final_mask = os.path.join(work_dir, f'{subj}_mask{ext}')

    # if (not os.path.exists(final_mask) and not os.path.exists(tmp_mask)) or overwrite:
    if not os.path.exists(tmp_mask) or not os.path.exists(tmp) or overwrite:
        overwrite = False
        if not os.path.exists(raw_dwi) or overwrite:
            select_cmd = f"select_dwi_vols {raw_nii} {bvals} {raw_dwi} {nominal_bval} -m"
            os.system(select_cmd)
        if not os.path.exists(b0_dwi) or overwrite:
            b0_val = 0
            while not os.path.exists(b0_dwi) or b0_val > (nominal_bval / 2):
                cmd = f'select_dwi_vols {raw_nii} {bvals} {b0_dwi} {b0_val}  -m;'
                os.system(cmd)
                b0_val += 50
        if not os.path.exists(tmp_mask) or overwrite:
            if 'median' in masking:
                median_mask_make(raw_dwi, tmp, median_radius=median_radius, binary_dilation=binary_dilation, numpass=7,
                                 outpathmask=tmp_mask)
                # median_mask_make(b0_dwi, tmp, outpathmask=tmp_mask, median_radius = median_radius, numpass=median_radius)
                # Rmedian_mask_make(b0_dwi, tmp, outpathmask='/Users/jas/jacques/Chavez_test_temp/b0_test.nii.gz', median_radius = median_radius, numpass=median_radius)
                # median_mask_make(raw_dwi, tmp, outpathmask='/Users/jas/jacques/Chavez_test_temp/007_mask_rad7.nii.gz',
                #                 median_radius=4, numpass=7)
            elif masking == "bet":
                bet_cmd = f"bet {raw_dwi} {tmp} -m -n -R"
                os.system(bet_cmd)
            elif masking == 'None':
                shutil.copy(raw_dwi, tmp)
                data, affine, _, hdr, _ = load_nifti_remote(raw_dwi)
                data_mask = np.ones(np.shape(data))
                newnii = nib.Nifti1Image(data_mask, affine, hdr)
                nib.save(newnii, tmp_mask)
            elif masking == 'premade':  # temp solution for when matlab or other method made the mask
                mask_fixer(tmp_mask)
                if not os.path.exists(tmp) or overwrite:
                    applymask_samespace(raw_nii, tmp_mask, outpath=tmp)
            else:
                raise Exception("Unrecognized masking type")

    # I think this part is done later more properly:     if create_subj_space_files: for contrast in ['dwi', 'b0', 'mask']:
    # if SAMBA_inputs_folder is not None:
    #    mask_subj_link = os.path.join(SAMBA_inputs_folder,f'{subj}_subjspace_mask{ext}')
    #    if not os.path.exists(mask_subj_link) or overwrite:
    #        shutil.copy(tmp_mask, mask_subj_link)

    # if cleanup and (os.path.exists(tmp_mask) and os.path.exists(raw_dwi)):
    #    os.remove(raw_dwi)
    # overwrite=False
    # Run Local PCA Denoising algorithm on 4D nifti:

    masked_nii = os.path.join(work_dir, nii_name)
    if not "nii.gz" in masked_nii:
        masked_nii = masked_nii.replace(".nii", ".nii.gz")
    masked_nii = masked_nii.replace(ext, "_masked" + ext)

    if denoise.lower() == 'lpca':
        D_subj = f'LPCA_{subj}';
    elif denoise.lower() == 'mpca':
        D_subj = f'MPCA_{subj}';
    elif denoise == "None" or denoise is None:
        D_subj = f'{subj}'

    if denoise == "None" or denoise is None:
        denoised_nii = masked_nii
        if not os.path.exists(masked_nii) or overwrite:
            fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
            os.system(fsl_cmd)
    else:
        denoised_nii = os.path.join(work_dir, f"{D_subj}_nii4D.nii.gz")
        if not os.path.exists(denoised_nii) or overwrite:
            if not os.path.exists(masked_nii) or overwrite:
                fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
                os.system(fsl_cmd)
            basic_LPCA_denoise_func(subj, masked_nii, bvecs, denoised_nii, processes=processes,
                                    denoise=denoise, verbose=False)  # to improve and make multiprocessing

    # if cleanup and os.path.exists(denoised_nii) and os.path.exists(masked_nii) and denoised_nii!=masked_nii:
    #    os.remove(masked_nii)

    # Run coregistration/eddy current correction:

    coreg_nii_old = f'{outpath}/co_reg_{D_subj}_m00-results/Reg_{D_subj}_nii4D{ext}';
    coreg_nii = os.path.join(work_dir, f'Reg_{D_subj}_nii4D{ext}')
    if not cleanup:
        coreg_nii = coreg_nii_old
    if not os.path.exists(coreg_nii) or overwrite:
        if not os.path.exists(coreg_nii_old) or overwrite:
            temp_cmd = os.path.join(gunniespath,
                                    'co_reg_4d_stack_tmpnew.bash') + f' {denoised_nii} {D_subj} 0 {outpath} 0';
            os.system(temp_cmd)
        if cleanup:
            shutil.move(coreg_nii_old, coreg_nii)

    if shortcuts_all_folder is not None:
        coreg_link = os.path.join(shortcuts_all_folder, f'{subj}_subjspace_coreg{ext}')
        if not os.path.exists(coreg_link) or overwrite:
            buildlink(coreg_nii, coreg_link)

    toeddy = False
    if toeddy:
        # fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
        # os.system(fsl_cmd)
        eddy_cmd = f"eddy --imain={coreg_nii} --mask={tmp_mask} --acqp=acq_params.txt " \
            f"--index={os.path.join(work_dir,'index.txt')} --bvecs={bvecs} --bvals={bvals} --topup=topup_results --repol " \
            f"--out = {os.path.join(work_dir, f'Reg_{D_subj}_nii4D_eddy{ext}')}"
        os.system(eddy_cmd)

    coreg_inputs = os.path.join(outpath, f'co_reg_{D_subj}_m00-inputs')
    coreg_work = coreg_inputs.replace('-inputs', '-work')
    coreg_results = coreg_inputs.replace('-inputs', '-results')
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_inputs):
        shutil.rmtree(coreg_inputs)
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_work):
        shutil.rmtree(coreg_work)
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_results):
        shutil.rmtree(coreg_results)

    # Generate tmp DWI:

    tmp_dwi_out = os.path.join(work_dir, f'{subj}_tmp_dwi{ext}')
    dwi_out = os.path.join(work_dir, f'{subj}_dwi{ext}')

    if not os.path.exists(tmp_dwi_out) or overwrite:
        cmd = f'select_dwi_vols {coreg_nii} {bvals} {tmp_dwi_out} {nominal_bval}  -m'
        os.system(cmd)

    # Generate tmp B0:
    tmp_b0_out = os.path.join(work_dir, f'{subj}_tmp_b0{ext}')
    b0_out = os.path.join(work_dir, f'{subj}_b0{ext}')
    # if (not os.path.exists(b0_out) and not os.path.exists(tmp_b0_out)) or overwrite:
    if not os.path.exists(tmp_b0_out) or overwrite:
        b0_val = 0
        while not os.path.exists(tmp_b0_out) or b0_val > (nominal_bval / 2):
            cmd = f'select_dwi_vols {coreg_nii} {bvals} {tmp_b0_out} {b0_val}  -m;'
            os.system(cmd)
            b0_val += 50

    # overwrite=False
    # elif cleanup and os.path.exists(tmp_b0_out):
    #    os.remove(tmp_b0_out)

    # Generate DTI contrasts and perform some tracking QA:
    if cleanup:
        c_string = ' --cleanup '
    else:
        c_string = ''

    # Important note: this is what first creates the fa, md, etc
    if len(glob.glob(os.path.join(work_dir, f'*.fib.gz.md{ext}'))) == 0 or overwrite:
        if overwrite:
            oldfiles = glob.glob(os.path.join(work_dir, f'*.fib.gz*'))
            for oldfile in oldfiles:
                os.remove(oldfile)
        cmd = 'bash ' + os.path.join(gunniespath,
                                     'dti_qa_with_dsi_studio_weirdcall.bash') + f' {coreg_nii} {bvecs} {tmp_mask} {work_dir} {c_string}';
        os.system(cmd)

    # Save the subject space dti results

    # Generate tmp MD:
    for contrast in ['md']:
        real_file = largerfile(
            os.path.join(work_dir, f'*.fib.gz.{contrast}{ext}'))  # Catch the 'real file' for each contrast
        tmp_file = f'{work_dir}/{subj}_tmp_{contrast}{ext}';
        if not os.path.exists(tmp_file):
            shutil.copy(real_file, tmp_file)

    tmp_md = f'{work_dir}/{subj}_tmp_md{ext}';

    if ref == "md" or ref is None:
        reference = tmp_md
    elif ref == "coreg":
        reference = coreg_nii
    elif os.path.exists(ref):
        reference = ref

    reference_file = os.path.join(work_dir, f'{subj}_reference{ext}')
    if not os.path.exists(reference_file):
        shutil.copy(reference, reference_file)

    if shortcuts_all_folder is not None:
        bonus_ref_link = os.path.join(shortcuts_all_folder, f'{subj}_reference{ext}')
        if not os.path.exists(bonus_ref_link) or overwrite:
            buildlink(reference_file, bonus_ref_link)

    # give new header to the non-dti files using md as reference

    for contrast in ['dwi', 'b0', 'mask']:
        tmp_file = os.path.join(work_dir, f'{subj}_tmp_{contrast}{ext}')
        tmp2_file = os.path.join(work_dir, f'{subj}_tmp2_{contrast}{ext}')
        final_file = os.path.join(work_dir, f'{subj}_{contrast}{ext}')
        if ((not os.path.exists(tmp2_file) and not os.path.exists(final_file)) or overwrite):
            if not os.path.exists(tmp_file):
                txt = f"Tmp file was not created for subject {subj}, need to rerun previous processes"
                raise Exception(txt)
            else:
                header_superpose(reference, tmp_file, outpath=tmp2_file)

    create_subj_space_files = True
    if create_subj_space_files:
        for contrast in ['dwi', 'b0', 'mask']:
            tmp_file = os.path.join(work_dir, f'{subj}_tmp_{contrast}{ext}')
            subj_file = os.path.join(work_dir, f'{subj}_subjspace_{contrast}{ext}')
            if not os.path.exists(subj_file) or overwrite:
                if not os.path.exists(tmp_file):
                    raise Exception("Tmp file was not created, need to rerun previous processes")
                else:
                    header_superpose(raw_dwi, tmp_file, outpath=subj_file)
            if shortcuts_all_folder is not None:
                subj_link = os.path.join(shortcuts_all_folder, f'{subj}_subjspace_{contrast}{ext}')
                if not os.path.exists(subj_link) or overwrite:
                    buildlink(subj_file, subj_link)


def launch_preprocessing_simple(raw_nii, outpath, bvalpath, bvecpath, cleanup=False, nominal_bval=4000,
                                SAMBA_inputs_folder=None, transpose=None, gunniespath="~/gunnies/", processes=1,
                                masking="bet", denoise='None',verbose=False):
    overwrite = False
    work_dir = outpath
    """
    for filePath in glob.glob(os.path.join(work_dir,'*')):
        modTimesinceEpoc = os.path.getmtime(filePath)
        modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
        if modificationTime[5:7]=='09' and int(modificationTime[8:10])>9:
            os.remove(filePath)
    """
    if verbose:
        print(f"Work directory is {work_dir}")
    mkcdir(work_dir)

    sbatch_folder = os.path.join(work_dir, "sbatch")
    mkcdir(sbatch_folder)
    # nii_path = os.path.join(work_dir,'nii4D_'+subj + '.nii.gz')
    # if not os.path.exists(nii_path):
    #    shutil.copy(raw_nii, nii_path)
    nii_name = os.path.split(raw_nii)[1]
    niifolder = os.path.dirname(raw_nii)
    ext = ".nii.gz"
    nii_ext = getext(nii_name)
    bxheader = nii_name.replace(nii_ext, ".bxh")
    bxheader = os.path.join(niifolder, bxheader)
    bvecs = bvecpath
    bvals = bvalpath

    if verbose:
        print(f"Original nifti is at {nii_name}\nbvecs are at {bvecs}\nbvals are at {bvals}\n")
    if not os.path.exists(bvecs):
        if verbose:
            print("Extracting diff directions")
        # print("Bvals and bvecs not found, using extractdiffdirs, however it it NOT RELIABLE, beware!")
        bvec_cmd = (f"extractdiffdirs --colvectors --writebvals --fieldsep='\t' --space=RAI {bxheader} {bvecs} {bvals}")
        os.system(bvec_cmd)

    # Make dwi for mask generation purposes.
    if np.size(masking.split('_')) > 1:
        median_radius = int(masking.split('_')[1])
    else:
        median_radius = 4
    binary_dilation = 8
    # tmp_mask = os.path.join(work_dir,f"{subj}_tmp_{str(median_radius)}_bd_{binary_dilation}_mask{ext}")
    tmp_mask = os.path.join(work_dir, f"4Dnii_tmp_mask{ext}")
    tmp = tmp_mask.replace("_mask", "")

    raw_dwi = os.path.join(work_dir, f"4Dnii_raw_dwi.nii.gz")
    b0_dwi = os.path.join(work_dir, f"4Dnii_b0_dwi.nii.gz")  # test average of the b0 images to make a better mask

    final_mask = os.path.join(work_dir, f'4Dnii_mask{ext}')

    # if (not os.path.exists(final_mask) and not os.path.exists(tmp_mask)) or overwrite:
    if not os.path.exists(tmp_mask) or not os.path.exists(tmp) or overwrite:
        overwrite = False
        if not os.path.exists(raw_dwi) or overwrite:
            select_cmd = f"select_dwi_vols {raw_nii} {bvals} {raw_dwi} {nominal_bval} -m"
            os.system(select_cmd)
        if not os.path.exists(b0_dwi) or overwrite:
            b0_val = 0
            while not os.path.exists(b0_dwi) or b0_val > (nominal_bval / 2):
                cmd = f'select_dwi_vols {raw_nii} {bvals} {b0_dwi} {b0_val}  -m;'
                os.system(cmd)
                b0_val += 50
        if not os.path.exists(tmp_mask) or overwrite:
            if 'median' in masking:
                median_mask_make(raw_dwi, tmp, median_radius=median_radius, binary_dilation=binary_dilation, numpass=7,
                                 outpathmask=tmp_mask)
                # median_mask_make(b0_dwi, tmp, outpathmask=tmp_mask, median_radius = median_radius, numpass=median_radius)
                # Rmedian_mask_make(b0_dwi, tmp, outpathmask='/Users/jas/jacques/Chavez_test_temp/b0_test.nii.gz', median_radius = median_radius, numpass=median_radius)
                # median_mask_make(raw_dwi, tmp, outpathmask='/Users/jas/jacques/Chavez_test_temp/007_mask_rad7.nii.gz',
                #                 median_radius=4, numpass=7)
            elif masking == "bet":
                bet_cmd = f"bet {raw_dwi} {tmp} -m -n -R"
                os.system(bet_cmd)
            elif masking == 'None':
                shutil.copy(raw_dwi, tmp)
                data, affine, _, hdr, _ = load_nifti_remote(raw_dwi)
                data_mask = np.ones(np.shape(data))
                newnii = nib.Nifti1Image(data_mask, affine, hdr)
                nib.save(newnii, tmp_mask)
            elif masking == 'premade':  # temp solution for when matlab or other method made the mask
                mask_fixer(tmp_mask)
                if not os.path.exists(tmp) or overwrite:
                    applymask_samespace(raw_nii, tmp_mask, outpath=tmp)
            else:
                raise Exception("Unrecognized masking type")

    # I think this part is done later more properly:     if create_subj_space_files: for contrast in ['dwi', 'b0', 'mask']:
    # if SAMBA_inputs_folder is not None:
    #    mask_subj_link = os.path.join(SAMBA_inputs_folder,f'{subj}_subjspace_mask{ext}')
    #    if not os.path.exists(mask_subj_link) or overwrite:
    #        shutil.copy(tmp_mask, mask_subj_link)

    # if cleanup and (os.path.exists(tmp_mask) and os.path.exists(raw_dwi)):
    #    os.remove(raw_dwi)
    # overwrite=False
    # Run Local PCA Denoising algorithm on 4D nifti:

    masked_nii = os.path.join(work_dir, nii_name)
    if not "nii.gz" in masked_nii:
        masked_nii = masked_nii.replace(".nii", ".nii.gz")
    masked_nii = masked_nii.replace(ext, "_masked" + ext)

    if denoise.lower() == 'lpca':
        D_subj = f'LPCA_4Dnii';
    elif denoise.lower() == 'mpca':
        D_subj = f'MPCA_4Dnii';
    elif denoise == "None" or denoise is None:
        D_subj = f'4Dnii'

    if denoise == "None" or denoise is None:
        denoised_nii = masked_nii
        if not os.path.exists(masked_nii) or overwrite:
            fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
            os.system(fsl_cmd)
    else:
        denoised_nii = os.path.join(work_dir, f"{D_subj}_nii4D.nii.gz")
        if not os.path.exists(denoised_nii) or overwrite:
            if not os.path.exists(masked_nii) or overwrite:
                fsl_cmd = f"fslmaths {raw_nii} -mas {tmp_mask} {masked_nii} -odt 'input'";
                os.system(fsl_cmd)
            basic_LPCA_denoise_func('4Dnii', masked_nii, bvecs, denoised_nii, processes=processes,
                                    denoise=denoise, verbose=False)  # to improve and make multiprocessing

    # if cleanup and os.path.exists(denoised_nii) and os.path.exists(masked_nii) and denoised_nii!=masked_nii:
    #    os.remove(masked_nii)

    # Run coregistration/eddy current correction:

    coreg_nii_old = f'{outpath}/co_reg_{D_subj}_m00-results/Reg_{D_subj}_nii4D{ext}';
    coreg_nii = os.path.join(work_dir, f'Reg_{D_subj}_nii4D{ext}')
    if not cleanup:
        coreg_nii = coreg_nii_old
    if not os.path.exists(coreg_nii) or overwrite:
        if not os.path.exists(coreg_nii_old) or overwrite:
            temp_cmd = os.path.join(gunniespath,
                                    'co_reg_4d_stack_tmpnew.bash') + f' {denoised_nii} {D_subj} 0 {outpath} 0';
            os.system(temp_cmd)
        if cleanup:
            shutil.move(coreg_nii_old, coreg_nii)

    coreg_inputs = os.path.join(outpath, f'co_reg_{D_subj}_m00-inputs')
    coreg_work = coreg_inputs.replace('-inputs', '-work')
    coreg_results = coreg_inputs.replace('-inputs', '-results')
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_inputs):
        shutil.rmtree(coreg_inputs)
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_work):
        shutil.rmtree(coreg_work)
    if cleanup and os.path.exists(coreg_nii) and os.path.isdir(coreg_results):
        shutil.rmtree(coreg_results)

    # Generate tmp DWI:

    tmp_dwi_out = os.path.join(work_dir, f'4Dnii_tmp_dwi{ext}')
    dwi_out = os.path.join(work_dir, f'4Dnii_dwi{ext}')

    if not os.path.exists(tmp_dwi_out) or overwrite:
        cmd = f'select_dwi_vols {coreg_nii} {bvals} {tmp_dwi_out} {nominal_bval}  -m'
        os.system(cmd)

    # Generate tmp B0:
    tmp_b0_out = os.path.join(work_dir, f'4Dnii_tmp_b0{ext}')
    b0_out = os.path.join(work_dir, f'4Dnii_b0{ext}')
    # if (not os.path.exists(b0_out) and not os.path.exists(tmp_b0_out)) or overwrite:
    if not os.path.exists(tmp_b0_out) or overwrite:
        b0_val = 0
        while not os.path.exists(tmp_b0_out) or b0_val > (nominal_bval / 2):
            cmd = f'select_dwi_vols {coreg_nii} {bvals} {tmp_b0_out} {b0_val}  -m;'
            os.system(cmd)
            b0_val += 50

    # overwrite=False
    # elif cleanup and os.path.exists(tmp_b0_out):
    #    os.remove(tmp_b0_out)

    # Generate DTI contrasts and perform some tracking QA:
    if cleanup:
        c_string = ' --cleanup '
    else:
        c_string = ''

    # Important note: this is what first creates the fa, md, etc
    if len(glob.glob(os.path.join(work_dir, f'*.fib.gz.md{ext}'))) == 0 or overwrite:
        if overwrite:
            oldfiles = glob.glob(os.path.join(work_dir, f'*.fib.gz*'))
            for oldfile in oldfiles:
                os.remove(oldfile)
        cmd = 'bash ' + os.path.join(gunniespath,
                                     'dti_qa_with_dsi_studio_weirdcall.bash') + f' {coreg_nii} {bvecs} {tmp_mask} {work_dir} {c_string}';
        os.system(cmd)

    # Save the subject space dti results

    # Generate tmp MD:
    for contrast in ['md']:
        real_file = largerfile(
            os.path.join(work_dir, f'*.fib.gz.{contrast}{ext}'))  # Catch the 'real file' for each contrast
        tmp_file = f'{work_dir}/4Dnii_tmp_{contrast}{ext}';
        if not os.path.exists(tmp_file):
            shutil.copy(real_file, tmp_file)

    tmp_md = f'{work_dir}/4Dnii_tmp_md{ext}';

    ref='md'
    if ref == "md" or ref is None:
        reference = tmp_md
    elif ref == "coreg":
        reference = coreg_nii
    elif os.path.exists(ref):
        reference = ref

    reference_file = os.path.join(work_dir, f'4Dnii_reference{ext}')
    if not os.path.exists(reference_file):
        shutil.copy(reference, reference_file)


    # give new header to the non-dti files using md as reference

    for contrast in ['dwi', 'b0', 'mask']:
        tmp_file = os.path.join(work_dir, f'4Dnii_tmp_{contrast}{ext}')
        tmp2_file = os.path.join(work_dir, f'4Dnii_tmp2_{contrast}{ext}')
        final_file = os.path.join(work_dir, f'4Dnii_{contrast}{ext}')
        if ((not os.path.exists(tmp2_file) and not os.path.exists(final_file)) or overwrite):
            if not os.path.exists(tmp_file):
                txt = f"Tmp file was not created for subject 4Dnii, need to rerun previous processes"
                raise Exception(txt)
            else:
                header_superpose(reference, tmp_file, outpath=tmp2_file)

    create_subj_space_files = True
    if create_subj_space_files:
        for contrast in ['dwi', 'b0', 'mask']:
            tmp_file = os.path.join(work_dir, f'4Dnii_tmp_{contrast}{ext}')
            subj_file = os.path.join(work_dir, f'4Dnii_subjspace_{contrast}{ext}')
            if not os.path.exists(subj_file) or overwrite:
                if not os.path.exists(tmp_file):
                    raise Exception("Tmp file was not created, need to rerun previous processes")
                else:
                    header_superpose(raw_dwi, tmp_file, outpath=subj_file)

    """
    # write the relative orientation file here
    if not os.path.isfile(orient_string) or overwrite:
        if os.path.isfile(orient_string):
            os.remove(orient_string)
        file = os.path.join(work_dir, subj + '_tmp_mask' + ext);
        cmd = 'bash ' + os.path.join(gunniespath, 'find_relative_orientation_by_CoM.bash') + f' {reference_file} {file}'
        orient_relative = subprocess.getoutput(cmd)

        with open(orient_string, 'w') as f:
            f.write(orient_relative)
    else:
        orient_relative = open(orient_string, mode='r').read()

    if SAMBA_inputs_folder is not None:
        subj_orient_string = os.path.join(SAMBA_inputs_folder, f'4Dnii_relative_orientation.txt')
        shutil.copy(orient_string, subj_orient_string)

    if shortcuts_all_folder is not None:
        subj_orient_string = os.path.join(shortcuts_all_folder, f'4Dnii_relative_orientation.txt')
        shutil.copy(orient_string, subj_orient_string)

    # check extracted values from relative orientation vals
    orientation_out = orient_relative.split(',')[0]
    orientation_out = orientation_out.split(':')[1]
    orientation_in = orient_relative.split(',')[1]
    orientation_in = orientation_in.split(':')[1]
    if verbose:
        print(f'flexible orientation: {orientation_in}');
        print(f'reference orientation: {orientation_out}');
    """
    orientation_in='RAS'
    orientation_out='RAS'
    # apply the orientation modification to specified contrasts
    for contrast in ['dwi', 'b0', 'mask']:
        img_in = os.path.join(work_dir, f'4Dnii_tmp2_{contrast}{ext}')
        img_out = os.path.join(work_dir, f'4Dnii_{contrast}{ext}')
        if not os.path.isfile(img_out) or overwrite:
            if orientation_out != orientation_in:
                print('TRYING TO REORIENT...b0 and dwi and mask')
                if os.path.exists(img_in) and (not os.path.exists(img_out) or overwrite):
                    img_transform_exec(img_in, orientation_in, orientation_out, img_out)
                    if os.path.exists(img_out):
                        os.remove(img_in)
                elif os.path.exists(img_out) and cleanup:
                    os.remove(img_in)
            else:
                shutil.move(img_in, img_out)

        if SAMBA_inputs_folder is not None:
            inputs_space_link = os.path.join(SAMBA_inputs_folder, f'4Dnii_{contrast}{ext}')
            if not os.path.exists(inputs_space_link) or overwrite:
                buildlink(img_out, inputs_space_link)


    mask = os.path.join(work_dir, f'4Dnii_mask{ext}')
    b0 = os.path.join(work_dir, f'4Dnii_b0{ext}')

    # if cleanup and os.path.exists(dwi_out) and os.path.exists(tmp_dwi_out):
    #    os.remove(tmp_dwi_out)

    for contrast in ['fa0', 'rd', 'ad', 'md']:
        real_file = largerfile(os.path.join(work_dir,
                                            f'*.fib.gz.{contrast}{ext}'))  # It will be fun times if we ever have more than one match to this pattern...
        # inputspace = real_file
        inputspace = os.path.join(work_dir, f'4Dnii_inputspace_{contrast}{ext}')

        contrast = contrast.replace('0', '')
        # linked_file=os.path.join(shortcut_dir,f'4Dnii_{contrast}{ext}')
        linked_file_w = os.path.join(work_dir, f'4Dnii_{contrast}{ext}')
        if not os.path.exists(inputspace) or overwrite:
            if orientation_out != orientation_in:
                affine_dwi = nib.load(dwi_out).affine
                if sum(affine_dwi[0, :3]) < 0 or sum(affine_dwi[1, :3]) < 0 or sum(affine_dwi[2, :3]) < 0:
                    real_reorient = real_file.replace(f'nii4D', f'nii4D_tmp')
                    real_reorient_2 = real_file.replace(f'nii4D', f'nii4D_tmp_2')
                    img_transform_affreset(real_file, affine_dwi, output_path=real_reorient)
                    img_transform_exec(real_reorient, orientation_in, orientation_out, real_reorient_2)
                    affine_superpose(dwi_out, real_reorient_2, outpath=inputspace, transpose=transpose)
                    os.remove(real_reorient)
                    os.remove(real_reorient_2)
                else:
                    affine_superpose(dwi_out, real_file, outpath=inputspace, transpose=transpose)
            else:
                affine_superpose(dwi_out, real_file, outpath=inputspace, transpose=transpose)
        if not os.path.isfile(linked_file_w) or overwrite:
            buildlink(inputspace, linked_file_w)
        if SAMBA_inputs_folder is not None:
            # warnings.warn('should reach this!')
            blinked_file = os.path.join(SAMBA_inputs_folder, f'4Dnii_{contrast}{ext}')
            if not os.path.exists(blinked_file) or overwrite:
                buildlink(inputspace, blinked_file)
                print(f'build link from {inputspace} to {blinked_file}')


    if create_subj_space_files:
        for contrast in ['fa0', 'rd', 'ad', 'md']:

            real_file = largerfile(os.path.join(work_dir,
                                                f'*.fib.gz.{contrast}{ext}'))  # It will be fun times if we ever have more than one match to this pattern...
            contrast = contrast.replace('0', '')
            subj_file_tmp = os.path.join(work_dir, f'4Dnii_subjspace_tmp_{contrast}{ext}')
            subj_file = os.path.join(work_dir, f'4Dnii_subjspace_{contrast}{ext}')
            if not os.path.exists(subj_file) or overwrite:
                if not os.path.exists(subj_file_tmp):
                    if orientation_out != orientation_in:
                        print('TRYING TO REORIENT.ReineR..b0 and dwi and mask')
                        if os.path.exists(real_file) and (not os.path.exists(subj_file) or overwrite):
                            img_transform_exec(real_file, orientation_out, orientation_in, subj_file_tmp)
                    else:
                        shutil.copy(real_file, subj_file_tmp)
                header_superpose(raw_dwi, subj_file_tmp, outpath=subj_file)
