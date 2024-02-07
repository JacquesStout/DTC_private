#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:21:13 2023

@author: ali
"""

import os
import nibabel as nib
from nibabel import load, save, Nifti1Image, squeeze_image
#import multiprocessing
import numpy as np
import pandas as pd
import shutil
import sys
import socket
from DTC.file_manager.computer_nav import checkfile_exists_all
from DTC.diff_handlers.bvec_handler import fix_bvals_bvecs
import subprocess, re

def get_num_streamlines(tracks_path):
    cmd = f'tckinfo {tracks_path} -count'
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        # Split the output into lines and extract the last line
        output_lines = result.stdout.split('\n')
        last_line = output_lines[-2] if output_lines else ''
        numbers = re.findall(r'\d+', last_line)
        num_streamlines = max(map(int, numbers))
        return num_streamlines
    else:
        # Handle the case where the command failed
        print(f"Error: {result.stderr}")
        return None


#n_threds = str(multiprocessing.cpu_count())

subj = sys.argv[1] #reads subj number with s... from input of python file 
#subj = "S01912"

subj_ref = subj

overwrite=False
cleanup = False
make_connectomes = True
make_wmconnectomes = True

test = 'test'
test = 'T1'

if np.size(sys.argv)>2:
    act = bool(sys.argv[2])
else:
    act = True

if act:
    act_string = '_act'
else:
    act_string = ''

if np.size(sys.argv)>3:
    overwrite = bool(sys.argv[3])
else:
    overwrite = False

index_gz = ".gz"

if 'santorini' in socket.gethostname().split('.')[0]:
    root = '/Volumes/Data/Badea/Lab/mouse/mrtrix_ad_decode/'
    orig_subj_path = '/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/'
    temp_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/T1_transforms/'
    scratch_folder = '/Volumes/Data/Badea/Lab/jacques/temp/'

if 'blade' in socket.gethostname().split('.')[0]:
    root= '/mnt/munin2/Badea/Lab/mouse/mrtrix_ad_decode/'
    orig_subj_path = '/mnt/munin2/Badea/ADdecode.01/Analysis/DWI/'
    temp_folder = '/mnt/munin2/Badea/ADdecode.01/Analysis/T1_transforms/'
    scratch_folder = '/mnt/munin2/Badea/Lab/jacques/temp/'

#root= '/Users/ali/Desktop/Mar23/mrtrixc_ad_decode/'


#orig_subj_path = '/Users/ali/Desktop/Mar23/mrtrixc_ad_decode/DWI/'

bvec_path_orig = orig_subj_path+subj_ref+'_bvecs_fix.txt' 
if not os.path.isfile(bvec_path_orig) :
    bvec_path_orig_prefix = orig_subj_path+subj_ref+'_bvecs.txt'
    bval_path_orig_prefix = orig_subj_path+subj_ref+'_bvals.txt'

    if not os.path.exists(bvec_path_orig_prefix):
        print('where is original bvec?')
    else:
        fix_bvals_bvecs(bval_path_orig_prefix, bvec_path_orig_prefix, outpath=orig_subj_path)

nii_gz_path_orig =  orig_subj_path  + subj +'_subjspace_coreg.nii.gz'
if not os.path.isfile(nii_gz_path_orig) : print('where is original 4d?')
bval_path_orig = orig_subj_path + subj_ref +'_bvals_fix.txt'
if not os.path.isfile(bval_path_orig) : print('where is original bval?')

b0_orig =  orig_subj_path+subj+'_subjspace_b0.nii.gz'
if not os.path.isfile(b0_orig) : print('where is original b0?')


T1_orig_test =  orig_subj_path+subj+'_T1_test.nii.gz'
#if not os.path.isfile(T1_orig_test) : print('where is original T1?')

T1_orig =  orig_subj_path+subj+'_T1.nii.gz'
#if not os.path.isfile(T1_orig) : print('where is original T1?')


subjspace_dwi =  orig_subj_path+subj+'_subjspace_dwi.nii.gz'
if not os.path.isfile(subjspace_dwi) : print('where is subjspace dwi?')

subjspace_mask = orig_subj_path+subj+'_subjspace_mask.nii.gz'
if not os.path.isfile(subjspace_mask) : print('where is original mask?')

path_perm = root + 'perm_files/'

#path_trk = path_perm
path_trk = '/Volumes/Data/Badea/Lab/human/AD_Decode_trk_transfer/TRK'

if not os.path.isdir(path_perm) : os.mkdir(path_perm)
bval_path = path_perm  + subj + '_bvals_RAS.txt'
old_bval = np.loadtxt(bval_path_orig)
new_bval = np.round(old_bval)
#new_bval.shape
np.savetxt(bval_path, new_bval, newline=" ", fmt='%f') # saving the RAS bvec

#bval_path = '/Users/ali/Downloads/N59066_bval.txt'
if not os.path.isdir(root +  'temp/' ) : os.mkdir(root +  'temp/' )


if test == 'test':
    T1 =T1_orig_test
    subj_path = root + 'temp/' + subj + f'_test{act_string}/'
elif test == 'dwi':
    T1 =subjspace_dwi
    subj_path = root + 'temp/' + subj + f'_dwi{act_string}/'
else:
    T1 = T1_orig
    subj_path = root + 'temp/' + subj + f'{act_string}/'
#os.system('/Applications/Convert3DGUI.app/Contents/bin/c3d ' + T1_orig +" -orient RAS -o "+T1) # RAS rotation T1

conn_folder = root + f'connectome{act_string}/'
distances_csv = conn_folder +subj+'_distances.csv'

mean_FA_connectome =  conn_folder+subj+'_mean_FA_connectome.csv'

parcels_csv = conn_folder+subj+'_conn_sift_node.csv'
assignments_parcels_csv = path_perm +subj+f'_assignments_con_sift_node{act_string}.csv'

parcels_csv_2 = conn_folder+subj+'_conn_plain.csv'
assignments_parcels_csv2 = path_perm +subj+f'_assignments_con_plain{act_string}.csv'

parcels_csv_wm_2 = conn_folder+subj+'_whitematter_conn_plain.csv'
assignments_parcels_wm_csv2 = path_perm +subj+f'_assignments_wm_con_plain{act_string}.csv'

parcels_csv_3 = conn_folder+subj+'_conn_sift.csv'
assignments_parcels_csv3 = path_perm +subj+f'_assignments_con_sift{act_string}.csv'

list_outputs_all = [distances_csv,mean_FA_connectome,parcels_csv,assignments_parcels_csv,parcels_csv_2,
                    assignments_parcels_csv2,parcels_csv_3,assignments_parcels_csv3]
if make_wmconnectomes:
    list_outputs_all.append(parcels_csv_wm_2)
    list_outputs_all.append(assignments_parcels_wm_csv2)

alloutputs_found = checkfile_exists_all(list_outputs_all)

coreg_T1 = True
skip_T1 = False

label_whitematter_path = os.path.join(orig_subj_path, f'{subj}_labels_whitematter.nii.gz')

if make_wmconnectomes and not os.path.exists(label_whitematter_path):
    txt = f'For subject {subj}, could not find {label_whitematter_path}'
    raise Exception(txt)

if not os.path.exists(T1):
    fivett_nocoreg_nii_gz = orig_subj_path + subj + '_5tt_nocoreg.nii.gz'
    fivett_nocoreg_mif = subj_path + subj + '_5tt_nocoreg.mif'
    if os.path.exists(fivett_nocoreg_nii_gz):
        coreg_T1 = False
        skip_T1 = True
        if not os.path.isdir(subj_path): os.mkdir(subj_path)
        if not os.path.exists(fivett_nocoreg_mif) or overwrite:
            cmd = f'mrconvert {fivett_nocoreg_nii_gz} {fivett_nocoreg_mif} -force'
            os.system(cmd)
    else:
        txt = f'Could not find either T1 or 5tt for subject {subj}'
        raise Exception(txt)
else:
    fivett_nocoreg_nii_gz = subj_path + subj + '_5tt_nocoreg.nii.gz'
    fivett_nocoreg_mif = subj_path + subj + '5tt_nocoreg.mif'

overwrite=True
if alloutputs_found and not overwrite:
    print(f'All outputs found, subject {subj} is already done!')

else:
    overwrite=False
    if not os.path.isdir(subj_path) : os.mkdir(subj_path)

    nii_gz_path = nii_gz_path_orig

    bvec_path = path_perm+subj+'_bvecs_RAS.txt'
    old_bvec = np.loadtxt(bvec_path_orig)
    new_bvec = old_bvec
    #new_bvec = old_bvec [:, [2,1,0] ] # swap x and y
    new_bvec[1:,0] = -new_bvec[1:,0] # flip y sign
    #new_bvec[1:,1] = -new_bvec[1:,1] # flip x sign
    #new_bvec[1:,2] = -new_bvec[1:,2] # flip z sign
    new_bvec=new_bvec.transpose()
    np.savetxt(bvec_path, new_bvec, fmt='%f') # saving the RAS bvec
    #bvec_path = bvec_path_orig
    #bvec_path  = '/Users/ali/Downloads/N59066_bvecs.txt'

    #changng to mif format

    if coreg_T1:
        T1_transform_cut_path = os.path.join(temp_folder, f'{subj}_T1_to_dwi_')
        T1_transform_path = os.path.join(temp_folder, f'{subj}_T1_to_dwi_0GenericAffine.mat')
        T1_reggedtodwi = os.path.join(orig_subj_path,f'{subj}_T1_registered.nii.gz')
        subjspace_T1 = T1

        if not os.path.exists(T1_reggedtodwi) or overwrite:
            if not os.path.exists(T1_transform_path) or overwrite:
                command = f'antsRegistration -v 1 -d 3 -m Mattes[{subjspace_dwi}, {subjspace_T1}] -t affine[0.1] -c [3000x3000x0x0, 1e-8, 20] -s 4x2x1x0.5vox -f 6x4x2x1 -u 1 -z 1 -l 1 -x {subjspace_mask} -o {    T1_transform_cut_path}'
                os.system(command)
            command = f'antsApplyTransforms -d 3 -e 0 -i {subjspace_T1} -r {subjspace_dwi} -u float -o {T1_reggedtodwi} -t {T1_transform_path}'
            os.system(command)
        T1 = T1_reggedtodwi
        if not os.path.exists(T1_reggedtodwi):
            raise Exception('Could not create registered T1, investigate')

    #T1_mif = subj_path+subj+'_T1.mif'+index_gz


    out_mif = subj_path + subj+'_subjspace_dwi.mif'+index_gz
    if not os.path.exists(out_mif) or overwrite:
        os.system('mrconvert '+nii_gz_path+ ' ' +out_mif+' -fslgrad '+bvec_path+ ' '+ bval_path+' -bvalue_scaling 0 -force') #turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000

    """
    #preprocessing
    #denoise:
        #####skip denoise for mouse so far
    output_denoise =   subj_path+subj+'_den.mif'
    #os.system('dwidenoise '+out_mif + ' ' + output_denoise+' -force') #denoising
    #compute residual to check if any resdiual is loaded on anatomy
    output_residual = subj_path+subj+'residual.mif'
    os.system('mrcalc '+out_mif + ' ' + output_denoise+ ' -subtract '+ output_residual+ ' -force') #compute residual
    os.system('mrview '+ output_denoise) #inspect residual
    """
    output_denoise = out_mif #####skip denoise

    dt_mif = path_perm+subj+'_dt.mif'+index_gz
    fa_mif = path_perm+subj+'_fa.mif'+index_gz
    dk_mif = path_perm+subj+'_dk.mif'+index_gz
    mk_mif = path_perm+subj+'_mk.mif'+index_gz
    md_mif = path_perm+subj+'_md.mif'+index_gz
    ad_mif = path_perm+subj+'_ad.mif'+index_gz
    rd_mif = path_perm+subj+'_rd.mif'+index_gz

    checked_all = checkfile_exists_all([dt_mif,fa_mif, rd_mif, ad_mif, md_mif])

    if np.unique(new_bval).shape[0] > 2 :
        os.system('dwi2tensor ' + output_denoise + ' ' + dt_mif + ' -dkt ' +  dk_mif +' -fslgrad ' +  bvec_path + ' ' + bval_path + ' -force'  )
        os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -adc '  + md_mif+' -ad '  + ad_mif + ' -rd '  + rd_mif   + ' -force' )
    else:
        if not checked_all or overwrite:
            os.system('dwi2tensor ' + output_denoise + ' ' + dt_mif  +' -fslgrad ' +  bvec_path + ' ' + bval_path + ' -force'  )
            os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -force' )
            os.system('tensor2metric  -rd ' + rd_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
            os.system('tensor2metric  -ad ' + ad_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
            os.system('tensor2metric  -adc ' + md_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(

    den_preproc_mif = output_denoise # already skipping preprocessing (always)

    """
    #createing mask after bias correction:
    #den_unbiased_mif = subj_path+subj+'_den_preproc_unbiased.mif'
    #bias_mif = subj_path+subj+'_bias.mif'
    #os.system('dwibiascorrect ants '+den_preproc_mif+' '+den_unbiased_mif+ ' -bias '+ bias_mif + ' -force')
    #cannot be done here go on on terminal after echoing and python it
    """
    den_unbiased_mif = den_preproc_mif  # bypassing

    mask_mif  =  path_perm+subj+'_mask.mif'
    if not os.path.exists(mask_mif) or overwrite:
        os.system('dwi2mask '+den_unbiased_mif+  ' '+ mask_mif + ' -force')
    #os.system('mrview '+fa_mif + ' -overlay.load '+ mask_mif )
    mask_mrtrix_nii = subj_path +subj+'_mask_mrtrix.nii.gz'

    if not os.path.exists(mask_mrtrix_nii) or overwrite:
        os.system('mrconvert ' +mask_mif+ ' '+mask_mrtrix_nii + ' -force' )


    #making mask

    mask_nii_gz = path_perm +subj+'_mask.nii.gz'
    csf_nii_gz = subj_path +subj+'_csf.nii.gz'

    if not os.path.exists(csf_nii_gz):
        os.system('ImageMath 3 ' + csf_nii_gz + ' ThresholdAtMean '+ b0_orig + ' 10')
        os.system('ImageMath 3 ' + csf_nii_gz + ' MD '+ csf_nii_gz + ' 1')
        os.system('ImageMath 3 ' + csf_nii_gz + ' ME '+ csf_nii_gz + ' 1')

    if not os.path.exists(mask_mif) or overwrite:

        os.system('ImageMath 3 ' + mask_nii_gz + ' - '+ mask_mrtrix_nii +  ' '+ csf_nii_gz)
        os.system('ThresholdImage 3 ' + mask_nii_gz + ' '+ mask_nii_gz +  ' 0.0001 1 1 0')
        os.system('ImageMath 3 ' + mask_nii_gz + ' ME '+ mask_nii_gz + ' 1')
        os.system('ImageMath 3 ' + mask_nii_gz + ' MD '+ mask_nii_gz + ' 1')

        os.system('mrconvert ' +mask_nii_gz+ ' '+mask_mif + ' -force' )

    ########### making a mask out of labels

    label_path_orig = orig_subj_path +subj+'_labels.nii.gz'
    #label_path = path_perm +subj+'_labels.nii.gz'
    #os.system("/Applications/Convert3DGUI.app/Contents/bin/c3d "+label_path_orig+" -orient RAS -o "+label_path)

    label_path = label_path_orig  #if not doing the rotation seen above
    """
    #mask_output = subj_path +subj+'_mask_of_label.nii.gz'
    #mask_labels_data = label_nii.get_fdata()
    #mask_labels = np.unique(mask_labels_data)
    #mask_labels=np.delete(mask_labels, 0)
    #mask_of_label =label_nii.get_fdata()*0
    """

    path_atlas_legend = root+ 'IIT/IITmean_RPI_index.xlsx'
    legend  = pd.read_excel(path_atlas_legend)

    """
    #new_bval_path = path_perm+subj+'_new_bvals.txt' 
    #new_bvec_path = path_perm+subj+'_new_bvecs.txt' 
    #os.system('dwigradcheck ' + out_mif +  ' -fslgrad '+bvec_path+ ' '+ bval_path +' -mask '+ mask_mif + ' -number 100000 -export_grad_fsl '+ new_bvec_path + ' '  + new_bval_path  +  ' -force' )
    #bvec_temp=np.loadtxt(new_bvec_path)
    """

    #Estimating the Basis Functions:
    wm_txt =   subj_path+subj+'_wm.txt'
    gm_txt =  subj_path+subj+'_gm.txt'
    csf_txt = subj_path+subj+'_csf.txt'
    voxels_mif =  subj_path+subj+'_voxels.mif'+index_gz
    if not os.path.exists(wm_txt) or not os.path.exists(gm_txt) or not os.path.exists(csf_txt) or overwrite:
        os.system('dwi2response dhollander '+den_unbiased_mif+ ' ' +wm_txt+ ' ' + gm_txt + ' ' + csf_txt + ' -voxels ' + voxels_mif+' -mask '+ mask_mif + ' -scratch ' +subj_path + ' -fslgrad ' +bvec_path + ' '+ bval_path   +'  -force' )

    #Viewing the Basis Functions:
    #os.system('mrview '+den_unbiased_mif+ ' -overlay.load '+ voxels_mif)
    #os.system('shview '+wm_txt)
    #os.system('shview '+gm_txt)
    #os.system('shview '+csf_txt)

    #Applying the basis functions to the diffusion data:
    wmfod_mif =  subj_path+subj+'_wmfod.mif'+index_gz
    gmfod_mif = subj_path+subj+'_gmfod.mif'+index_gz
    csffod_mif = subj_path+subj+'_csffod.mif'+index_gz

    #os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force' )
    if not os.path.exists(wmfod_mif):
        os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' -force' )

    #combine to single image to view them
    #Concatenating the FODs:
    ##vf_mif =   subj_path+subj+'_vf.mif'
    #os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat '+csffod_mif+ ' ' +gmfod_mif+ ' - ' + vf_mif+' -force' )
    ##os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat ' +gmfod_mif+ ' - ' + vf_mif+' -force' ) # without csf

    #Viewing the FODs:
    #os.system('mrview ' +fa_mif+ ' -odf.load_sh '+wmfod_mif )

    #Normalizing the FODs:
    wmfod_norm_mif =  subj_path+subj+'_wmfod_norm.mif'+index_gz
    #gmfod_norm_mif = subj_path+subj+'_gmfod_norm.mif'
    #csffod_norm_mif = subj_path+subj+'_csffod_norm.mif'
    if not os.path.exists(wmfod_norm_mif) or overwrite:
        os.system('mtnormalise ' +wmfod_mif+ ' '+wmfod_norm_mif+' -mask ' + mask_mif + '  -force')
    #Viewing the normalise FODs:
    #os.system('mrview ' +fa_mif+ ' -odf.load_sh '+wmfod_norm_mif )

    #Segmenting the anatomical image with FSL's FAST to 5 different classes


    tracks_10M_tck  = subj_path +subj+'_tracks_10M.tck'

    if test == 'test':
        #smallerTracks = path_trk + subj + f'_smallerTracks2mill_test{act_string}.tck'
        smallerTracks = os.path.join(path_trk, f'{subj}_smallerTracks2mill_test{act_string}.tck')
    elif test == 'dwi':
        #smallerTracks = path_trk + subj + f'_smallerTracks2mill_dwi{act_string}.tck'
        smallerTracks = os.path.join(path_trk, f'{subj}_smallerTracks2mill_dwi{act_string}.tck')
    else:
        smallerTracks = os.path.join(path_trk, f'{subj}_smallerTracks2mill.tck')

    if os.path.exists(smallerTracks) and not overwrite:
        num_streamlines = get_num_streamlines(smallerTracks)
        if num_streamlines < int(2000000):
            print(f'Bad file at {smallerTracks}, trying again')
        else:
            trk_already_made = True

    if not trk_already_made:
        if act:
            subjspace_mask_mif = os.path.join(subj_path,f'{subj}_subjspace_mask.mif')
            if not os.path.exists(subjspace_mask_mif) or overwrite:
                os.system(f'mrconvert {subjspace_mask} {subjspace_mask_mif} -force')

            if (not os.path.exists(fivett_nocoreg_mif) or overwrite) and not skip_T1:
                os.system('5ttgen fsl '  +T1+ ' '+fivett_nocoreg_mif + f' -mask {subjspace_mask_mif} -scratch {scratch_folder} -force')

            #Extracting the b0 images: for Coregistering the anatomical and diffusion datasets:
            mean_b0_mif = subj_path+subj+'_mean_b0.mif'
            if not os.path.exists(mean_b0_mif):
                os.system('dwiextract '+ den_unbiased_mif+' - -bzero | mrmath - mean '+ mean_b0_mif +' -axis 3 -force')

            #Converting the b0 and 5tt images bc we wanna use fsl this part and fsl does not accept mif:
            mean_b0_nii_gz = subj_path+subj+'_mean_b0.nii.gz'
            if not os.path.exists(mean_b0_nii_gz) or overwrite:
                os.system('mrconvert ' +mean_b0_mif + ' '+ mean_b0_nii_gz + ' -force')
            if (not os.path.exists(fivett_nocoreg_nii_gz) or overwrite) and not skip_T1:
                os.system('mrconvert ' + fivett_nocoreg_mif + ' ' + fivett_nocoreg_nii_gz + ' -force')

            fivett_vol0_nii_gz = subj_path+subj+'_5tt_vol0.nii.gz'
            if not os.path.exists(fivett_vol0_nii_gz) or overwrite:
                os.system('fslroi '+ fivett_nocoreg_nii_gz+ ' '+ fivett_vol0_nii_gz + ' 0 1')

            #Coregistering the anatomical and diffusion datasets: #skip
            diff2struct_fsl_mat =subj_path+subj+'_diff2struct_fsl.mat'
            if not os.path.exists(diff2struct_fsl_mat) or overwrite:
                os.system('flirt -in '+ mean_b0_nii_gz + ' -ref ' + fivett_vol0_nii_gz + ' -interp nearestneighbour -dof 6 -omat ' + diff2struct_fsl_mat)


            #Converting the transformation matrix to MRtrix format:
            diff2struct_mrtrix_txt = subj_path+subj+'_diff2struct_mrtrix.txt'
            if not os.path.exists(diff2struct_mrtrix_txt) or overwrite:
                os.system('transformconvert ' + diff2struct_fsl_mat + ' '+ mean_b0_nii_gz+ ' '+ fivett_nocoreg_nii_gz + ' flirt_import '+ diff2struct_mrtrix_txt + ' -force' )

            #Applying the transformation matrix to the non-coregistered segmentation data:
            #using the inverse transfomration coregsiter anatomiacl to dwi
            fivett_coreg_mif = subj_path+subj+'_fivett_coreg.mif'
            if not os.path.exists(fivett_coreg_mif) or overwrite:
                os.system('mrtransform ' + fivett_nocoreg_mif + ' -linear ' + diff2struct_mrtrix_txt + ' -inverse '+ fivett_coreg_mif + ' -force')

            #os.system( 'mrview '+ den_unbiased_mif +' -overlay.load ' + fivett_nocoreg_mif + ' -overlay.colourmap 2 -overlay.load ' + fivett_coreg_mif + ' -overlay.colourmap 1 ')

            #Creating the grey matter / white matter boundary: seed boundery bc they're used to create seeds for streamlines
            gmwmSeed_coreg_mif = subj_path+subj+'_gmwmSeed_coreg.mif'
            if not os.path.exists(gmwmSeed_coreg_mif) or overwrite:
                os.system( '5tt2gmwmi ' +  fivett_coreg_mif+ ' '+ gmwmSeed_coreg_mif + ' -force')

            if not os.path.exists(tracks_10M_tck) or overwrite:
                os.system('tckgen -act ' + fivett_coreg_mif + '  -backtrack -seed_gmwmi '+ gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.06 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')

            cmd = 'tckgen -act ' + fivett_coreg_mif + '  -backtrack -seed_gmwmi ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.06 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force'
            if not os.path.exists(tracks_10M_tck) or overwrite:
                os.system(cmd)
            else:
                num_streamlines = get_num_streamlines(tracks_10M_tck)
                if num_streamlines < int(10000000):
                    print(f'Bad file at {tracks_10M_tck}, trying again')
                    os.system(cmd)

        else:
            gmwmSeed_coreg_mif = mask_mif
            cmd = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + '  -maxlength 410 -cutoff 0.05 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force'
            if not os.path.exists(tracks_10M_tck) or overwrite:
                os.system(cmd)
            else:
                num_streamlines = get_num_streamlines(tracks_10M_tck)
                if num_streamlines < int(10000000):
                    print(f'Bad file at {tracks_10M_tck}, trying again')
                    os.system(cmd)

        #Extracting a subset of tracks:

        #os.system('echo tckedit '+ tracks_10M_tck + ' -number 2000000 -minlength 0.1 ' + smallerTracks + ' -force')

        if not os.path.exists(smallerTracks) or overwrite:
            os.system('tckedit '+ tracks_10M_tck + ' -number 2000000 -minlength 2 ' + smallerTracks + ' -force')
        else:
            num_streamlines = get_num_streamlines(smallerTracks)
            if num_streamlines<int(2000000):
                print(f'Bad file at {smallerTracks}, trying again')
                os.system('tckedit ' + tracks_10M_tck + ' -number 2000000 -minlength 2 ' + smallerTracks + ' -force')


    #os.system('mrview ' + den_unbiased_mif + ' -tractography.load '+ smallerTracks)
    #os.system('mrview ' + den_unbiased_mif + ' -tractography.load '+ smallerTracks)

    if make_connectomes:
        try:
            label_nii = nib.load(label_path)
        except:
            print(f'Could not find {label_nii}, skipping the connectomes creation')
            make_connectomes = False

    if cleanup and not make_connectomes:
        os.remove(tracks_10M_tck)

    if make_connectomes or make_wmconnectomes:
        mean_FA_per_streamline = subj_path + subj + '_per_strmline_mean_FA.csv'

        #Sifting the tracks with tcksift2: bc some wm tracks are over or underfitted
        sift_mu_txt = subj_path+subj+'_sift_mu.txt'
        sift_coeffs_txt = subj_path+subj+'_sift_coeffs.txt'
        sift_1M_txt = subj_path+subj+'_sift_1M.txt'

        if not os.path.exists(sift_mu_txt) or overwrite:
            os.system('tcksift2  -out_mu '+ sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif+ ' ' + sift_1M_txt  + ' -force')

        if not os.path.exists(sift_coeffs_txt) or overwrite:
            os.system('tcksift2  -out_mu '+ sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif+ ' ' + sift_1M_txt  + ' -force')

        if not os.path.exists(mean_FA_per_streamline):
            os.system(
                'tcksample ' + smallerTracks + ' ' + fa_mif + ' ' + mean_FA_per_streamline + ' -stat_tck mean ' + ' -force')

    if make_connectomes:
        #convert subj labels to mif
        parcels_mif = subj_path + subj + '_parcels.mif' + index_gz
        if not os.path.exists(parcels_mif) or overwrite:
            labels_data = label_nii.get_fdata()
            labels = np.unique(labels_data)
            labels=np.delete(labels, 0)
            label_nii_order = labels_data*0.0

            #sum(legend['index2'] == labels)
            for i in labels:
                leg_index = np.where(legend['index2'] == i )
                leg_index = leg_index [0][0]
                ordered_num = legend['index'][leg_index]
                label3d_index = np.where( labels_data == i )
                label_nii_order[label3d_index] = ordered_num


            file_result= nib.Nifti1Image(label_nii_order, label_nii.affine, label_nii.header)
            new_label = path_perm +subj+'_new_label.nii.gz'
            nib.save(file_result, new_label)

            #new_label = label_path
            os.system('mrconvert '+new_label+ ' ' +parcels_mif + ' -force' )



        #Creating the connectome without coregistration:
        ### connectome folders :

        mean_FA_per_streamline =  subj_path+subj+'_per_strmline_mean_FA.csv'

        if not os.path.isdir(conn_folder) : os.mkdir(conn_folder)

        if not os.path.exists(distances_csv) or overwrite:
            os.system('tck2connectome ' + smallerTracks + ' ' + parcels_mif+ ' ' + distances_csv + ' -zero_diagonal -symmetric -scale_length -stat_edge  mean' + ' -force')

        if not os.path.exists(mean_FA_per_streamline) or not os.path.exists(mean_FA_connectome) or overwrite:
            os.system('tcksample '+ smallerTracks+ ' '+ fa_mif + ' ' + mean_FA_per_streamline + ' -stat_tck mean ' + ' -force')
            os.system('tck2connectome '+ smallerTracks + ' ' + parcels_mif + ' '+ mean_FA_connectome + ' -zero_diagonal -symmetric -scale_file ' + mean_FA_per_streamline + ' -stat_edge mean '+ ' -force')



        if not os.path.exists(parcels_csv) or not os.path.exists(assignments_parcels_csv) or overwrite:
            os.system('tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in '+ sift_1M_txt+ ' '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv + ' -out_assignment ' + assignments_parcels_csv + ' -force')



        if not os.path.exists(parcels_csv_2) or not os.path.exists(assignments_parcels_csv2) or overwrite:
            os.system('tck2connectome -symmetric -zero_diagonal '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv_2 + ' -out_assignment ' + assignments_parcels_csv2 + ' -force')


        if not os.path.exists(parcels_csv_3) or not os.path.exists(assignments_parcels_csv3) or overwrite:
            os.system('tck2connectome -symmetric -zero_diagonal -tck_weights_in '+ sift_1M_txt+ ' '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv_3 + ' -out_assignment ' + assignments_parcels_csv3 + ' -force')

    if make_wmconnectomes:
        # convert subj labels to mif
        parcels_wm_mif = subj_path + subj + '_wm_parcels.mif' + index_gz
        if not os.path.exists(parcels_wm_mif) or overwrite:

            os.system('mrconvert ' + label_whitematter_path + ' ' + parcels_wm_mif + ' -force')

        # Creating the connectome without coregistration:
        ### connectome folders :


        if not os.path.isdir(conn_folder): os.mkdir(conn_folder)

        #assignments_parcels_wm_csv2 = path_perm + subj + f'_assignments_wm_con_plain{act_string}.csv'

        if not os.path.exists(parcels_csv_wm_2) or not os.path.exists(assignments_parcels_wm_csv2) or overwrite:
            cmd = f'tck2connectome -symmetric -assignment_forward_search 50 -zero_diagonal {smallerTracks} {parcels_wm_mif} {parcels_csv_wm_2} -out_assignment {assignments_parcels_wm_csv2} -force'
            os.system(cmd)


    if cleanup:
        shutil.rmtree(subj_path)


        #scale_invnodevol scale connectome by the inverse of size of each node
        #tck_weights_in weight each connectivity by sift
        #out assignment helo converting connectome to tracks

        #Viewing the connectome in Matlab:

        #connectome = importdata('sub-CON02_parcels.csv');
        #imagesc(connectome, [0 1])

        #Viewing the lookup labels:

