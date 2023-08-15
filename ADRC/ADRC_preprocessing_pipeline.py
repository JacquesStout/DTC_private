
import os, subprocess, sys, shutil, glob, re, fnmatch
import socket

from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
import numpy as np
from scipy.cluster.vq import kmeans, vq


def regexify(string):
    newstring = ('^'+string+'$').replace('*','.*')
    return newstring

def glob_remote(path, sftp=None):
    match_files = []
    if sftp is not None:
        if '*' in path:
            pathdir, pathname = os.path.split(path)
            pathname = regexify(pathname)
            allfiles = sftp.listdir(pathdir)
            for file in allfiles:
                if re.search(pathname, file) is not None:
                    match_files.append(os.path.join(pathdir,file))
        elif '.' not in path:
            allfiles = sftp.listdir(path)
            for filepath in allfiles:
                match_files.append(os.path.join(path, filepath))
            return match_files
        else:
            dirpath = os.path.dirname(path)
            try:
                sftp.stat(dirpath)
            except:
                return match_files
            allfiles = sftp.listdir(dirpath)
            #if '*' in path:
            #    for filepath in allfiles:
            #            match_files.append(os.path.join(dirpath,filepath))
            #else:
            for filepath in allfiles:
                if fnmatch.fnmatch(os.path.basename(filepath), os.path.basename(path)):
                    match_files.append(os.path.join(dirpath, filepath))
    else:
        if '.' not in path:
            match_files = glob.glob(path)
        else:
            dirpath = os.path.dirname(path)
            if not os.path.exists(dirpath):
                return(match_files)
            else:
                allfiles = glob.glob(os.path.join(dirpath,'*'))
                for filepath in allfiles:
                    if fnmatch.fnmatch(os.path.basename(filepath), os.path.basename(path)):
                        match_files.append(os.path.join(dirpath, filepath))
    return(match_files)


def checkallfiles(paths, sftp=None):
    existing = True
    for path in paths:
        match_files = glob_remote(path, sftp)
        if not match_files:
            existing= False
    return existing


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

#inputs
root = '/mnt/munin2/Badea/Lab/mouse/ADRC_jacques_pipeline/'
data_path = '/mnt/munin2/Badea/Lab/ADRC-20230511/'
dwi_manual_pe_scheme_txt = os.path.join(root, 'dwi_manual_pe_scheme.txt')
perm_output = os.path.join(root, 'perm_files/')
se_epi_manual_pe_scheme_txt = os.path.join(root, 'se_epi_manual_pe_scheme.txt')

#outputs
data_path_output = '/mnt/munin2/Badea/Lab/mouse/ADRC_jacques_pipeline/'

mkcdir(data_path_output)

#subjects to run
subjects = list(sys.argv[1])
#subjects = ['ADRC0001']

index_gz = '.gz'
overwrite = False

cleanup = False

for subj in subjects:

    subj_folder = os.path.join(data_path, subj,'visit1')
    subj_out_folder = os.path.join(data_path_output, subj)
    scratch_path = os.path.join(subj_out_folder, 'scratch')
    perm_subj_output = os.path.join(perm_output, subj)
    mkcdir([subj_out_folder,scratch_path, perm_subj_output])

    bvec_path_AP = os.path.join(perm_subj_output, subj + '_bvecs.txt')
    bval_path_AP = os.path.join(perm_subj_output, subj + '_bvals.txt')
    bvec_path_PA = os.path.join(perm_subj_output, subj+'_bvecs_rvrs.txt')
    bval_path_PA = os.path.join(perm_subj_output, subj+'_bvals_rvrs.txt')

    DTI_forward_nii_gz = os.path.join(subj_folder, 'HCP_DTI.nii.gz')
    nii_gz_path_PA = os.path.join(subj_folder, 'HCP_DTI_reverse_phase.nii.gz')

    if not os.path.exists(DTI_forward_nii_gz):
        print(f'Missing {DTI_forward_nii_gz}')
    if not os.path.exists(nii_gz_path_PA):
        print(f'Missing {nii_gz_path_PA}')

    if not os.path.exists(bvec_path_AP) or not os.path.exists(bval_path_AP) or overwrite:
        bvec = []  # bvec extraction
        with open(os.path.join(subj_folder, "HCP_DTI_reverse_phase.bxh")) as file:
            for line in file:
                if "<value>" in line:
                    temp_loop = line
                    temp_loop_split = temp_loop.split()
                    temp_loop_2 = [i.replace("<value>", "") for i in temp_loop_split]
                    temp_loop_2 = [i.replace("</value>", "") for i in temp_loop_2]
                    temp_loop_2 = [float(i) for i in temp_loop_2]
                    bvec.append(temp_loop_2)
                    # print( temp_loop_2)

        norms = np.linalg.norm(bvec, axis=1)
        norms[norms == 0] = 1
        bvec = np.array(bvec)
        bvec = bvec / norms.reshape(len(norms), 1)
        bvec = bvec.transpose()

        np.savetxt(bvec_path_AP, bvec, fmt='%.2f')

        with open(os.path.join(subj_folder, "HCP_DTI_reverse_phase.bxh")) as file:
            line = file.readline()
            while line:
                if "bvalues" in line:
                    temp_loop = line
                    temp_loop_split = temp_loop.split()
                    temp_loop_2 = [i.replace("<bvalues>", "") for i in temp_loop_split]
                    temp_loop_2 = [i.replace('</bvalues>', "") for i in temp_loop_2]

        temp_loop_2 = [float(i) for i in temp_loop_2]
        bvals = temp_loop_2
        bvals = bvals * (norms ** 2)

        codebook, _ = kmeans(bvals, 3)
        ccodebook2 = codebook.round()
        cluster_indices, _ = vq(bvals, codebook)
        rnd_bvals = ccodebook2[cluster_indices]
        bvals = rnd_bvals
        bvals = bvals.reshape(1, len(bvals))
        new_bval = bvals
        np.savetxt(bval_path_AP, bvals, fmt='%.2f')


    if not os.path.exists(bvec_path_PA) or not os.path.exists(bval_path_PA) or overwrite:
        bvec_rvrs = [] # bvec extraction
        with open(os.path.join(subj_folder, "HCP_DTI_reverse_phase.bxh")) as file:
            for line in file:
                if "<value>" in line:
                    temp_loop = line
                    temp_loop_split = temp_loop.split()
                    temp_loop_2  = [ i.replace("<value>" , "") for i in temp_loop_split ]
                    temp_loop_2  = [ i.replace("</value>" , "") for i in temp_loop_2 ]
                    temp_loop_2 = [float(i) for i in temp_loop_2]
                    bvec_rvrs.append(temp_loop_2)
                    #print( temp_loop_2)


        norms  = np.linalg.norm(bvec_rvrs, axis= 1)
        norms [norms ==0 ] = 1 
        bvec_rvrs  = np.array(bvec_rvrs )
        bvec_rvrs = bvec_rvrs /   norms.reshape(len(norms),1)   
        bvec_rvrs  = bvec_rvrs .transpose()   

        bvec_rvrs[1, 1:] = -bvec_rvrs[1, 1:]

        np.savetxt(bvec_path_PA ,bvec_rvrs,fmt='%.2f')


        bvals_rvrs = [] # bval extraction
        with open(os.path.join(subj_folder, "HCP_DTI_reverse_phase.bxh")) as file:
            line = file.readline()
            while line:
                if "bvalues" in line:
                    temp_loop = line
                    temp_loop_split = temp_loop.split()
                    temp_loop_2  = [ i.replace("<bvalues>" , "") for i in temp_loop_split ]
                    temp_loop_2  = [ i.replace('</bvalues>' , "") for i in temp_loop_2 ]

        temp_loop_2 = [float(i) for i in temp_loop_2]            
        bvals_rvrs =  temp_loop_2
        bvals_rvrs  =  bvals_rvrs *(norms**2)

        codebook, _ = kmeans(bvals_rvrs, 3) 
        ccodebook2 = codebook.round()
        cluster_indices, _ = vq(bvals_rvrs, codebook)
        rnd_bvals= ccodebook2[cluster_indices]
        bvals = rnd_bvals
        bvals = bvals.reshape(1,len(bvals_rvrs))

        np.savetxt(bval_path_PA ,bvals,fmt='%.2f')


    DTI_forward_nii_gz = os.path.join(subj_folder,'HCP_DTI.nii.gz')

    resampled_path = os.path.join(perm_subj_output, subj + '_coreg_resampled.nii.gz')
    dwi_nii_gz = os.path.join(perm_subj_output, subj + '_dwi.nii.gz')
    mask_nii_path = os.path.join(perm_subj_output, subj + '_mask.nii.gz')
    mask_mif_path = os.path.join(perm_subj_output, subj + '_mask.mif')
    resampled_mif_path = os.path.join(perm_subj_output, subj + '_coreg_resampled.mif')

    coreg_bvecs = os.path.join(perm_subj_output, subj + '_coreg_bvecs.txt')
    coreg_bvals = os.path.join(perm_subj_output, subj + '_coreg_bvals.txt')

    if not os.path.exists(resampled_path) or not os.path.exists(dwi_nii_gz) or not os.path.exists(mask_mif_path) \
            or not os.path.exists(resampled_mif_path) or not os.path.exists(coreg_bvecs) or overwrite:

        ######### denoise:

        out_mif = os.path.join(subj_out_folder, subj + '_subjspace_diff.mif' + index_gz)
        if not os.path.exists(out_mif) or overwrite:
            os.system(
                'mrconvert ' + DTI_forward_nii_gz + ' ' + out_mif + ' -fslgrad ' + bvec_path_AP + ' ' + bval_path_AP + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000

        output_denoise = os.path.join(subj_out_folder, subj+'_den.mif')
        if not os.path.exists(output_denoise) or overwrite:
            os.system('dwidenoise ' + out_mif + ' ' + output_denoise + ' -force')  # denoising
        # compute residual to check if any residual is loaded on anatomy
        output_residual = os.path.join(subj_out_folder, subj + 'residual.mif')
        if not os.path.exists(output_residual) or overwrite:
            os.system(
                'mrcalc ' + out_mif + ' ' + output_denoise + ' -subtract ' + output_residual + ' -force')  # compute residual
        # os.system('mrview '+ output_residual) #inspect residual

        ##################
        ##### b0 warp unphasing

        PA_mif = os.path.join(subj_out_folder,subj+'_PA.mif')
        if not os.path.exists(PA_mif) or overwrite:
            os.system('mrconvert '+nii_gz_path_PA+ ' ' +PA_mif + ' -force') #PA to mif

        mean_b0_PA_mif = os.path.join(subj_out_folder,subj+'mean_b0_PA.mif')
            #take mean of PA and AP to unwarp
        if not os.path.exists(mean_b0_PA_mif) or overwrite:
            os.system('mrconvert '+PA_mif+ ' -fslgrad '+bvec_path_PA+ ' '+ bval_path_PA + ' - | mrmath - mean '+ mean_b0_PA_mif+' -axis 3 -force')

        ##### Command: Extracting b0 images from the AP dataset, and concatenating the b0 images across both AP and PA images:
        mean_b0_AP_mif = os.path.join(subj_out_folder,subj+'mean_b0_AP.mif')
        if not os.path.exists(mean_b0_AP_mif) or overwrite:
            os.system('dwiextract '+output_denoise+ ' - -bzero | mrmath - mean '+ mean_b0_AP_mif+' -axis 3 -force')
        b0_pair_mif = os.path.join(subj_out_folder,subj+'b0_pair.mif')
        if not os.path.exists(b0_pair_mif) or overwrite:
            os.system('mrcat '+mean_b0_AP_mif+ ' ' +mean_b0_PA_mif+' -axis 3 '+ b0_pair_mif+' -force')


        #### Command: converting the original diff to a mif

        diff_mif = os.path.join(subj_out_folder, subj+'_denoised_diff.mif')
        diff_json = os.path.join(subj_out_folder, subj+'_denoised_diff.json')
        if not os.path.exists(diff_mif) or overwrite:
            command = 'mrconvert ' + output_denoise + " " + diff_mif + ' -fslgrad ' + bvec_path_AP + ' ' + bval_path_AP + ' -json_export ' + diff_json + ' -force'
            print(command)
            os.system(command)


        ##### Command: converting the b0 unwarped

        se_epi_mif = os.path.join(scratch_path,'se_epi.mif')
        if not os.path.exists(se_epi_mif) or overwrite:
            command = 'mrconvert ' + b0_pair_mif + " " + se_epi_mif + ' -force'
            os.system(command)

        ##### Command: padding the unwarped b0s

        se_epi_pad2_mif = os.path.join(scratch_path, 'se_epi_pad2.mif')
        end_vol = subprocess.check_output('mrinfo -size ' + out_mif + " | awk '{print $3}' ", shell=True)
        end_vol = end_vol.decode()
        end_vol = str(int(end_vol[0:2]) - 1)
        if not os.path.exists(se_epi_pad2_mif) or overwrite:
            command = 'mrconvert ' + se_epi_mif + ' -coord 2 ' + end_vol + ' - | mrcat ' + se_epi_mif + ' - ' + se_epi_pad2_mif + ' -axis 2 -force'
            print(command)
            os.system(command)


        #### Command: padding the diff mif

        diff_pad2_mif = os.path.join(scratch_path, 'diff_pad2.mif')
        if not os.path.exists(diff_pad2_mif):
            command = 'mrconvert ' + diff_mif + ' -coord 2 ' + end_vol + '  -clear dw_scheme - | mrcat ' + diff_mif + ' - ' + diff_pad2_mif + ' -axis 2 -force'
            print(command)
            os.system(command)

        ##### Command: converting unwarped b0s to 'topup'

        topup_in_nii = os.path.join(scratch_path, 'topup_in.nii')
        topup_datain_txt =  os.path.join(data_path_output, 'topup_datain.txt')
        if not os.path.exists(topup_in_nii) or not os.path.exists(topup_datain_txt) or overwrite:
            command = 'mrconvert '+ se_epi_pad2_mif +' ' + topup_in_nii + ' -import_pe_table ' + se_epi_manual_pe_scheme_txt + ' -strides -1,+2,+3,+4 -export_pe_table '+ topup_datain_txt + ' -force'
            print(command)
            os.system(command)

        #### Command: getting the field map out of the topup info
        field_map_nii_gz = os.path.join(scratch_path, 'field_map.nii.gz')
        diff_topup = os.path.join(scratch_path, 'diff_topup')
        diff_topup_nii = diff_topup + '_fieldcoef.nii.gz'
        command = 'topup --imain=' + topup_in_nii + ' --datain=' + topup_datain_txt + ' --out=field --fout=' + field_map_nii_gz + ' --config=$FSLDIR/src/fsl-topup/flirtsch/b02b0.cnf --out=' + diff_topup + ' --verbose'
        if not os.path.exists(field_map_nii_gz) or not os.path.exists(diff_topup_nii) or overwrite:
            print(command)
            os.system(command)


        #### Command: Getting the popup config
        applytopup_config_txt = os.path.join(scratch_path, 'applytopup_config.txt')
        applytopup_indices_txt = os.path.join(scratch_path, 'applytopup_indices.txt')
        dwi_manual_pe_scheme_txt = os.path.join(root, 'dwi_manual_pe_scheme.txt')
        if not os.path.exists(applytopup_config_txt):
            command = 'mrconvert ' + diff_pad2_mif + ' -import_pe_table ' + dwi_manual_pe_scheme_txt + ' - | mrinfo - -export_pe_eddy ' + applytopup_config_txt + ' ' + applytopup_indices_txt + ' -force'
            print(command)
            os.system(command)


        #### Converting the padded diff back to a nii
        diff_pad2_pe_0_nii = os.path.join(subj_out_folder, 'diff_pad2_pe_0.nii')
        diff_pad2_pe_0_json = os.path.join(subj_out_folder, 'diff_pad2_pe_0.json')
        if not os.path.exists(diff_pad2_pe_0_nii) or overwrite:
            command = 'mrconvert ' + diff_pad2_mif + ' ' + diff_pad2_pe_0_nii + ' -strides -1,+2,+3,+4 -json_export ' + diff_pad2_pe_0_json + ' -force'
            print(command)
            os.system(command)

        #### Command: Applying the topup command on the padded diff path
        diff_pad2_pe_0_applytopup_nii = os.path.join(subj_out_folder, 'diff_pad2_pe_0_applytopup.nii')
        command = 'applytopup --imain=' + diff_pad2_pe_0_nii + ' --datain=' + applytopup_config_txt + ' --inindex=1 --topup=' + diff_topup + ' --out=' + diff_pad2_pe_0_applytopup_nii + ' --method=jac'
        if not os.path.exists(diff_pad2_pe_0_applytopup_nii+index_gz) or overwrite:
            print(command)
            os.system(command)

        #### Command: Converting dwi topupped nifti to mif
        diff_pad2_pe_0_applytopup_nii = diff_pad2_pe_0_applytopup_nii + '.gz'
        diff_pad2_pe_0_applytopup_mif = os.path.join(subj_out_folder, 'diff_pad2_pe_0_applytopup.mif')
        if not os.path.exists(diff_pad2_pe_0_applytopup_mif) or overwrite:
            command = 'mrconvert ' + diff_pad2_pe_0_applytopup_nii + ' ' + diff_pad2_pe_0_applytopup_mif + ' -json_import ' + diff_pad2_pe_0_json + ' -force'
            print(command)
            os.system(command)

        #### Command: Create the mask for the eddy
        eddy_mask_nii = os.path.join(subj_out_folder, 'eddy_mask.nii')
        if not os.path.exists(eddy_mask_nii) or overwrite:
            command = 'dwi2mask ' + diff_pad2_pe_0_applytopup_mif + ' - | maskfilter - dilate - | mrconvert - ' + eddy_mask_nii + ' -datatype float32 -strides -1,+2,+3 -force'
            print(command)
            os.system(command)

        #### Command: Preparing the padded diff for eddy command
        bvecs_eddy = os.path.join(perm_subj_output, subj + "_bvecs_eddy.txt")
        bvals_eddy = os.path.join(perm_subj_output, subj + "_bvals_eddy.txt")
        eddy_config_txt = os.path.join(subj_out_folder, 'eddy_config.txt')
        eddy_indices_txt = os.path.join(subj_out_folder, 'eddy_indices.txt')
        eddy_in_nii = os.path.join(subj_out_folder, 'eddy_in.nii')
        if not os.path.exists(eddy_in_nii) or overwrite:
            command = 'mrconvert ' + diff_pad2_mif + ' -import_pe_table ' + dwi_manual_pe_scheme_txt + ' ' + eddy_in_nii + ' -strides -1,+2,+3,+4 -export_grad_fsl ' + bvecs_eddy + ' ' + bvals_eddy + ' -export_pe_eddy ' + eddy_config_txt + " " + eddy_indices_txt + ' -force'
            print(command)
            os.system(command)

        #### Command: Running the eddy command
        diff_post_eddy = os.path.join(scratch_path, "diff_post_eddy.nii")
        diff_post_eddy_gz = diff_post_eddy + '.gz'
        if not os.path.exists(diff_post_eddy_gz) or overwrite:
            if socket.gethostname().split('.')[0]=='santorini':
                command = 'eddy --imain=' + eddy_in_nii + ' --mask=' + eddy_mask_nii + ' --acqp=' + eddy_config_txt + \
                          ' --index=' + eddy_indices_txt + ' --bvecs=' + bvecs_eddy + ' --bvals=' + bvals_eddy + \
                          ' --topup=' + diff_topup + ' --slm=linear --data_is_shelled --out=' + diff_post_eddy + \
                          ' --verbose'
            else:
                command = 'eddy_openmp --imain=' + eddy_in_nii + ' --mask=' + eddy_mask_nii + ' --acqp=' + \
                          eddy_config_txt + ' --index=' + eddy_indices_txt + ' --bvecs=' + bvecs_eddy + ' --bvals=' + \
                          bvals_eddy + ' --topup=' + diff_topup + ' --slm=linear --data_is_shelled --out=' + \
                          diff_post_eddy + ' --verbose'
            print(command)
            os.system(command)

        #### Command: Converting the eddy corrected diffusion file nifti into the mif version, also unpadded the file at same time (using command  -coord 2 0:' + end_vol)
        den_preproc_mif = os.path.join(subj_out_folder,subj + '_den_preproc.mif')
        diff_post_eddy_eddy_rotated_bvecs = os.path.join(scratch_path, 'diff_post_eddy.bvecs')
        if not os.path.exists(den_preproc_mif) or not os.path.exists(diff_post_eddy_eddy_rotated_bvecs) or overwrite:
            command = 'mrconvert ' + diff_post_eddy_gz + ' ' + den_preproc_mif + ' -coord 2 0:' + end_vol + \
                      ' -strides -1,-2,3,4 -fslgrad ' + diff_post_eddy_eddy_rotated_bvecs + ' ' + bvals_eddy + ' -force'
            print(command)
            os.system(command)

        #### Command: Running the dwi bias correction on the eddy corrected image
        den_unbiased_mif = os.path.join(subj_out_folder, subj + '_den_preproc_unbiased.mif')
        bias_mif = os.path.join(subj_out_folder, subj + '_bias.mif')
        if not os.path.exists(den_unbiased_mif) or overwrite:
            command = 'dwibiascorrect ants ' + den_preproc_mif + ' ' + den_unbiased_mif + ' -scratch ' + \
                      subj_out_folder + ' -bias ' + bias_mif + ' -force'
            print(command)
            os.system(command)

        #### Command: Converting the bias corrected diffusion mif file into the the nifti version (and extracting bvals/bvecs)

        coreg_nii_gz = os.path.join(perm_subj_output, subj + '_coreg.nii.gz')
        command = 'mrconvert ' + den_unbiased_mif + ' ' + coreg_nii_gz + ' -export_grad_fsl ' + coreg_bvecs + ' ' + \
                  coreg_bvals + ' -force'
        if not os.path.exists(coreg_nii_gz) or overwrite:
            print(command)
            os.system(command)

        """
        #### Command: Making the mask (mrtrix version)
        mask_mif = os.path.join(subj_outpath, subj + '_mask.mif')
        if not os.path.exists(mask_mif) or overwrite:
            command = 'dwi2mask ' + den_unbiased_mif + ' ' + mask_mif + ' -force'
            print(command)
            os.system(command)
    
        #### Command: Converting the mask into a
        mask_mrtrix_nii = os.path.join(subj_outpath, subj + '_mask.nii.gz')
        if not os.path.exists(mask_mrtrix_nii) or overwrite:
            command = 'mrconvert ' + mask_mif + ' ' + mask_mrtrix_nii + ' -force'
            print(command)
            os.system(command)
        """

        median_radius = 4
        numpass = 7
        binary_dilation = 3
        full_name = False
        verbose = False

        #### Command: Resampling the diffusion into the 1x1x1x1 dimensions
        command = f'ResampleImage 4 {coreg_nii_gz} {resampled_path} 1x1x1x1 0 0 2'
        if not os.path.exists(resampled_path) or overwrite:
            print(command)
            os.system(command)

        #### Command: Converting the nifti of resampled diffusion images to mif

        command = 'mrconvert ' + resampled_path + ' ' + resampled_mif_path + ' -fslgrad ' + coreg_bvecs + ' ' + coreg_bvals + ' -bvalue_scaling false -force'
        if not os.path.exists(resampled_mif_path) or overwrite:
            print(command)
            os.system(command)

        #### Command: Extracting the b0s from the bias resampled diffusion file and making the average diffusion weighted image based on thsoe
        dwi_mif = os.path.join(perm_subj_output, subj + '_dwi.mif')
        command = 'dwiextract ' + resampled_mif_path + ' - -no_bzero | mrmath - mean ' + dwi_mif + ' -axis 3 -force'
        if not os.path.exists(dwi_mif) or overwrite:
            print(command)
            os.system(command)

        #### Command: Converting the diffusion mean of b0s into a nifti
        dwi_nii_gz = os.path.join(perm_subj_output, subj + '_dwi.nii.gz')
        if not os.path.exists(dwi_nii_gz) or overwrite:
            command = 'mrconvert ' + dwi_mif + ' ' + dwi_nii_gz + ' -force'
            print(command)
            os.system(command)


        masked_dwi_path = os.path.join(subj_out_folder, subj + '_dwi_masked.nii.gz')

        #### Command: Create the mask for the diffusion images from the resampled diffusion images
        if not os.path.exists(masked_dwi_path) or not os.path.exists(mask_nii_path) or overwrite:
            median_mask_make(dwi_nii_gz, masked_dwi_path, median_radius=median_radius, binary_dilation=binary_dilation,
                             numpass=numpass, outpathmask=mask_nii_path, verbose=verbose)

        if not os.path.exists(mask_mif_path) or overwrite:
            command = 'mrconvert ' + mask_nii_path + ' ' + mask_mif_path + ' -force'
            os.system(command)

    dt_mif = os.path.join(perm_subj_output, subj + '_dt.mif' + index_gz)
    fa_mif = os.path.join(perm_subj_output, subj + '_fa.mif' + index_gz)
    dk_mif = os.path.join(perm_subj_output, subj + '_dk.mif' + index_gz)
    mk_mif = os.path.join(perm_subj_output, subj + '_mk.mif' + index_gz)
    md_mif = os.path.join(perm_subj_output, subj + '_md.mif' + index_gz)
    ad_mif = os.path.join(perm_subj_output, subj + '_ad.mif' + index_gz)
    rd_mif = os.path.join(perm_subj_output, subj + '_rd.mif' + index_gz)

    fa_nii = os.path.join(perm_subj_output, subj + '_fa.nii' + index_gz)

    if not os.path.exists(fa_nii) or overwrite:

        #Estimating the Basis Functions:
        wm_txt = os.path.join(subj_out_folder,subj+'_wm.txt' )
        gm_txt = os.path.join(subj_out_folder+subj+'_gm.txt')
        csf_txt = os.path.join(subj_out_folder+subj+'_csf.txt')
        voxels_mif = os.path.join(subj_out_folder, subj+'_voxels.mif'+index_gz)

        ##Right now we are using the RESAMPLED mif of the 4D miff, to be discussed
        if not os.path.exists(voxels_mif) or not os.path.exists(wm_txt) or not os.path.exists(gm_txt) or not os.path.exists(csf_txt) or overwrite:
            command = 'dwi2response dhollander '+resampled_mif_path+ ' ' +wm_txt+ ' ' + gm_txt + ' ' + csf_txt + ' -voxels ' + voxels_mif+' -mask '+ mask_mif_path + ' -scratch ' +subj_out_folder + ' -fslgrad ' +coreg_bvecs + ' '+ coreg_bvals   +'  -force'
            print(command)
            os.system(command)

        #not on spider but on terminal
        #Viewing the Basis Functions:
        #os.system('mrview '+den_unbiased_mif+ ' -overlay.load '+ voxels_mif)
        #os.system('shview '+wm_txt)
        #os.system('shview '+gm_txt)
        #os.system('shview '+csf_txt)

        #Applying the basis functions to the diffusion data:
        wmfod_mif = subj_out_folder+subj+'_wmfod.mif'+index_gz
        gmfod_mif = subj_out_folder+subj+'_gmfod.mif'+index_gz
        csffod_mif = subj_out_folder+subj+'_csffod.mif'+index_gz

        #os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force' )
        if not os.path.exists(wmfod_mif) or not os.path.exists(gmfod_mif) or not os.path.exists(csffod_mif) or overwrite:
            command = 'dwi2fod msmt_csd ' +resampled_mif_path+ ' -mask '+mask_mif_path+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force'
            print(command)
            os.system(command)

        #combine to single image to view them
        #Concatenating the FODs:
        vf_mif =   subj_out_folder+subj+'_vf.mif'
        if not os.path.exists(vf_mif) or overwrite:
            command = 'mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat '+csffod_mif+ ' ' +gmfod_mif+ ' - ' + vf_mif+' -force'
            print(command)
            os.system(command)
        #os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat ' +gmfod_mif+ ' - ' + vf_mif+' -force' ) # without csf

        #Viewing the FODs:
        #os.system('mrview ' +fa_mif+ ' -odf.load_sh '+wmfod_mif )

        #Normalizing the FODs:
        wmfod_norm_mif =  subj_out_folder+subj+'_wmfod_norm.mif'+index_gz
        gmfod_norm_mif = subj_out_folder+subj+'_gmfod_norm.mif'
        csffod_norm_mif = subj_out_folder+subj+'_csffod_norm.mif'
        if not os.path.exists(wmfod_norm_mif) or not os.path.exists(gmfod_norm_mif) or not os.path.exists(csffod_norm_mif) or overwrite:
            command = 'mtnormalise ' +wmfod_mif+ ' '+wmfod_norm_mif+ ' ' +gmfod_mif+ ' '+gmfod_norm_mif + ' ' +csffod_mif+ ' '+csffod_norm_mif +' -mask ' + mask_mif_path + '  -force'
            print(command)
            os.system(command)


        #making fa and Kurt:
            
        dt_mif = os.path.join(perm_subj_output,subj+'_dt.mif'+index_gz)
        fa_mif = os.path.join(perm_subj_output,subj+'_fa.mif'+index_gz)
        dk_mif = os.path.join(perm_subj_output,subj+'_dk.mif'+index_gz)
        md_mif = os.path.join(perm_subj_output,subj+'_md.mif'+index_gz)
        ad_mif = os.path.join(perm_subj_output,subj+'_ad.mif'+index_gz)
        rd_mif = os.path.join(perm_subj_output,subj+'_rd.mif'+index_gz)

        #mk_mif = os.path.join(perm_subj_output,subj+'_mk.mif'+index_gz)


        #output_denoise = '/Users/ali/Desktop/Feb23/mrtrix_pipeline/temp/N59141/N59141_subjspace_dwi_copy.mif.gz'#

        bval_orig = np.loadtxt(bval_path_AP)

        #exists_all = checkallfiles([dt_mif, fa_mif, dk_mif, mk_mif, md_mif, ad_mif, rd_mif])
        exists_all = checkallfiles([dt_mif, fa_mif, dk_mif, md_mif, ad_mif, rd_mif])

        if not exists_all:

            if np.unique(bval_orig).shape[0] > 2 :
                os.system('dwi2tensor ' + resampled_mif_path + ' ' + dt_mif + ' -dkt ' +  dk_mif +' -fslgrad ' +  coreg_bvecs + ' ' + coreg_bvals + ' -force'  )
                os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -adc '  + md_mif+' -ad '  + ad_mif + ' -rd '  + rd_mif   + ' -force' )

                #os.system('mrview '+ fa_mif) #inspect residual
            else:
                os.system('dwi2tensor ' + resampled_mif_path + ' ' + dt_mif  +' -fslgrad ' +  coreg_bvecs + ' ' + coreg_bvals + ' -force'  )
                os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -force' )
                os.system('tensor2metric  -rd ' + rd_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
                os.system('tensor2metric  -ad ' + ad_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
                os.system('tensor2metric  -adc ' + md_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
                #os.system('mrview '+ fa_mif) #inspect residual

            command = 'mrconvert ' +fa_mif+ ' '+fa_nii + ' -force'
            print(command)
            os.system(command)

    #T1_orig_res = os.path.join(subj_out_folder,subj+'_T1_res.nii.gz')
    #T1_orig = T1_orig_res

    if cleanup:
        shutil.rmtree(scratch_path)

    smallerTracks = os.path.join(perm_subj_output, subj + '_smallerTracks2mill.tck')

    if not os.path.exists(smallerTracks):

        # boundry of white and grey matter used for streamlines
        # Converting the anatomical image to MRtrix format:

        # Segmenting the anatomical image with FSL's FAST to 5 different classes
        # fivett_nocoreg_mif  = subj_path+subj+'5tt_nocoreg.mif'
        # os.system('5ttgen fsl '  +T1_mif+ ' '+fivett_nocoreg_mif + ' -force')
        # cannot be done here go on on terminal after echoing and python it

        # os.system('mrview ' +fivett_nocoreg_mif  )

        # mean_b0_mif = subj_path+subj+'_mean_b0.mif'
        # Extracting the b0 images: for Coregistering the anatomical and diffusion datasets:
        # os.system('dwiextract '+ den_unbiased_mif+' - -bzero | mrmath - mean '+ mean_b0_mif +' -axis 3 -force')

        # Converting the b0 and 5tt images bc we wanna use fsl this part and fsl does not accept mif:
        # mean_b0_nii_gz    = subj_path+subj+'_mean_b0.nii.gz'
        # fivett_nocoreg_nii_gz = subj_path+subj+'_5tt_nocoreg.nii.gz'
        # os.system('mrconvert ' +mean_b0_mif + ' '+ mean_b0_nii_gz + ' -force')
        # os.system('mrconvert ' + fivett_nocoreg_mif + ' ' + fivett_nocoreg_nii_gz + ' -force')

        # now Extracting the grey matter segmentation with fsl:
        # fivett_vol0_nii_gz    =  subj_path+subj+'_5tt_vol0.nii.gz'
        # os.system('fslroi '+ fivett_nocoreg_nii_gz+ ' '+ fivett_vol0_nii_gz + ' 0 1')
        # if not working here, works on terminal after echoing and copy\pasting


        # Coregistering the anatomical and diffusion datasets:
        # diff2struct_fsl_mat =subj_path+subj+'_diff2struct_fsl.mat'
        # os.system('flirt -in '+ mean_b0_nii_gz + ' -ref ' + fivett_vol0_nii_gz + ' -interp nearestneighbour -dof 6 -omat ' + diff2struct_fsl_mat  )
        # if not working here, works on terminal after echoing and copy\pasting

        # Converting the transformation matrix to MRtrix format:
        # diff2struct_mrtrix_txt = subj_path+subj+'_diff2struct_mrtrix.txt'
        # os.system('transformconvert ' + diff2struct_fsl_mat + ' '+ mean_b0_nii_gz+ ' '+ fivett_nocoreg_nii_gz + ' flirt_import '+ diff2struct_mrtrix_txt + ' -force' )

        # Applying the transformation matrix to the non-coregistered segmentation data:
        # using the iverse transfomration coregsiter anatomiacl to dwi
        # fivett_coreg_mif   = subj_path+subj+'_fivett_coreg.mif'
        # os.system('mrtransform ' + fivett_nocoreg_mif + ' -linear ' + diff2struct_mrtrix_txt + ' -inverse '+ fivett_coreg_mif + ' -force')

        # Viewing the coregistration in mrview:
        # os.system( 'mrview '+ den_unbiased_mif +' -overlay.load ' + fivett_nocoreg_mif + ' -overlay.colourmap 2 -overlay.load ' + fivett_coreg_mif + ' -overlay.colourmap 1 ')
        # os.system( 'mrview '+ T1_mif +' -overlay.load ' + fivett_nocoreg_mif + ' -overlay.colourmap 2 -overlay.load ' + fivett_coreg_mif + ' -overlay.colourmap 1 ')
        # fivett_coreg_mif = fivett_nocoreg_mif

        # Creating the grey matter / white matter boundary: seed boundery bc they're used to create seeds for streamlines
        # gmwmSeed_coreg_mif = os.path.join(subj_out_folder,subj+'_gmwmSeed_coreg.mif')
        # os.system( '5tt2gmwmi ' +  fivett_coreg_mif+ ' '+ gmwmSeed_coreg_mif + ' -force')


        gmwmSeed_coreg_mif  = mask_mif_path

        tracks_10M_tck = os.path.join(subj_out_folder, subj + '_tracks_10M.tck')

        # os.system('tckgen -act ' + fivett_coreg_mif + '  -backtrack -seed_gmwmi '+ gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.06 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')
        # seconds1 = time.time()
        command = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.1 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force'
        print(command)
        os.system(command)

        # os.system('tckgen -backtrack -seed_image '+ gmwmSeed_coreg_mif + ' -maxlength 1000 -cutoff 0.3 -select 50k ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')
        # seconds2 = time.time()
        # (seconds2 - seconds1)/360 # a million track in hippo takes 12.6 mins


        # Extracting a subset of tracks:
        # os.system('echo tckedit '+ tracks_10M_tck + ' -number 2000000 -minlength 0.1 ' + smallerTracks + ' -force')

        command = 'tckedit ' + tracks_10M_tck + ' -number 2000000 ' + smallerTracks + ' -force'
        print(command)
        os.system(command)

        # os.system('tckedit '+ tracks_10M_tck + ' -number 2000000 -minlength 2 ' + smallerTracks + ' -force')
        # os.system('mrview ' + den_unbiased_mif + ' -tractography.load '+ smallerTracks)
        # os.system('mrview ' + den_unbiased_mif + ' -tractography.load '+ smallerTracks)

    if cleanup:
        shutil.rmtree(subj_out_folder)
