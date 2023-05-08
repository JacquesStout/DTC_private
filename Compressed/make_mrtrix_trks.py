#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:21:13 2023
@author: ali
"""

import os
import nibabel as nib
from nibabel import load, save, Nifti1Image, squeeze_image
# import multiprocessing
import numpy as np
import pandas as pd
import shutil

work_dir = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/trk_dir'

orig_folder = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/'

nominal_bval = 2000

diff_name = 'Bruker_diffusion_test_15_0.045_af_4x__TV_and_L1_wavelet_0.01_0.01_bart_recon.nii.gz'
diff_path = os.path.join(orig_folder, diff_name)

dwi_name = diff_name.replace('.nii','_dwi.nii')
dwi_path = os.path.join(work_dir, dwi_name)

bval_name = '15_1_fully_sampled_DWI_16p25_120_21directions_Brukerdirs.bval'
bvec_name = '15_1_fully_sampled_DWI_16p25_120_21directions_Brukerdirs.bvec'

bval_path_orig = os.path.join(orig_folder, bval_name)
bvec_path_orig = os.path.join(orig_folder, bvec_name)

if not os.path.isfile(bval_path_orig): print('where is original bval?')
if not os.path.isfile(bvec_path_orig): print('where is original bvec?')
if not os.path.isfile(diff_path): print('where is original 4d?')
if not os.path.isfile(dwi_path): print('where is original DWI?')

bval_checked = os.path.join(work_dir, bval_name)
bvec_checked = os.path.join(work_dir, bvec_name)

# changing to mif format
dwi_mif = dwi_path.replace('.nii.gz','.mif')
diff_mif = os.path.join(work_dir, diff_name.replace('.nii.gz','.mif'))

os.system('mrconvert ' + dwi_path + ' ' + dwi_mif + ' -force')

if not os.path.exists(dwi_path):
    cmd = f'select_dwi_vols {diff_path} {bval_checked} {dwi_path} {nominal_bval}  -m'
    os.system(cmd)

os.system('mrconvert ' + diff_path + ' ' + diff_mif + ' -fslgrad ' + bvec_checked + ' ' + bval_checked + ' -bvalue_scaling 0 -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000

if not os.path.exists(bval_checked) and not os.path.exists(bvec_checked):
    os.system('dwigradcheck ' + dwi_mif + ' -fslgrad '+bvec_path_orig + ' ' + bval_path_orig +
              ' -number 50000 -export_grad_fsl ' + bvec_checked + ' ' + bval_checked + ' -force')

output_denoise = dwi_mif  #####skip denoise

# making fa and Kurt:

dt_mif = os.path.join(work_dir, diff_path.replace('_dwi.mif','_dt.mif'))
fa_mif = os.path.join(work_dir, diff_path.replace('_dwi.mif','_fa.mif'))
dk_mif = os.path.join(work_dir, diff_path.replace('_dwi.mif','_dk.mif'))
mk_mif = os.path.join(work_dir, diff_path.replace('_dwi.mif','_mk.mif'))
md_mif = os.path.join(work_dir, diff_path.replace('_dwi.mif','_md.mif'))
ad_mif = os.path.join(work_dir, diff_path.replace('_dwi.mif','_ad.mif'))
rd_mif = os.path.join(work_dir, diff_path.replace('_dwi.mif','_rd.mif'))

# output_denoise = '***/mrtrix_pipeline/temp/N59141/N59141_subjspace_dwi_copy.mif.gz'#

os.system(
    'dwi2tensor ' + output_denoise + ' ' + dt_mif + ' -dkt ' + dk_mif + ' -fslgrad ' + bvec_checked + ' ' + bval_checked + ' -force')
os.system(
    'tensor2metric  -fa ' + fa_mif + ' ' + dt_mif + ' -adc ' + md_mif + ' -ad ' + ad_mif + ' -rd ' + rd_mif + ' -force')

# os.system('mrview '+ fa_mif) #inspect residual
"""
else:
    os.system('dwi2tensor ' + output_denoise + ' ' + dt_mif + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -force')
    os.system('tensor2metric  -fa ' + fa_mif + ' ' + dt_mif + ' -force')
    os.system('tensor2metric  -rd ' + rd_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
    os.system('tensor2metric  -ad ' + ad_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
    os.system('tensor2metric  -adc ' + md_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
"""

den_preproc_mif = output_denoise  # already skipping preprocessing (always)

den_unbiased_mif = den_preproc_mif  # bypassing


"""
########### making a mask out of labels

label_path_orig = orig_subj_path + subj + '_labels_RAS.nii.gz'
label_path = label_path_orig
mask_output = subj_path + subj + '_mask_of_label.nii.gz'
label_nii = nib.load(label_path)
mask_labels_data = label_nii.get_fdata()
mask_labels = np.unique(mask_labels_data)
mask_labels = np.delete(mask_labels, 0)
mask_of_label = label_nii.get_fdata() * 0

path_atlas_legend = root + 'chass/CHASSSYMM3AtlasLegends.xlsx'
legend = pd.read_excel(path_atlas_legend)
index_csf = legend['Subdivisions_7'] == '8_CSF'
# index_wm = legend [ 'Subdivisions_7' ] == '7_whitematter'

vol_index_csf = legend[index_csf]
vol_index_csf = vol_index_csf['index2']

mask_labels_no_csf = set(mask_labels) - set(vol_index_csf)
for vol in mask_labels_no_csf: mask_of_label[mask_labels_data == int(vol)] = int(1)
mask_of_label = mask_of_label.astype(int)

file_result = nib.Nifti1Image(mask_of_label, label_nii.affine, label_nii.header)
nib.save(file_result, mask_output)
mask_mif = subj_path + subj + 'mask_of_label.mif' + index_gz
os.system('mrconvert ' + mask_output + ' ' + mask_mif + ' -datatype uint16 -force')
# os.system('mrview '+fa_mif + ' -overlay.load '+ mask_mif )


# new_bval_path = path_perm+subj+'_new_bvals.txt'
# new_bvec_path = path_perm+subj+'_new_bvecs.txt'
# os.system('dwigradcheck ' + dwi_mif +  ' -fslgrad '+bvec_path+ ' '+ bval_path +' -mask '+ mask_mif + ' -number 50000 -export_grad_fsl '+ new_bvec_path + ' '  + new_bval_path  +  ' -force' )
# bvec_temp=np.loadtxt(new_bvec_path)


# Estimating the Basis Functions:
wm_txt = subj_path + subj + '_wm.txt'
gm_txt = subj_path + subj + '_gm.txt'
csf_txt = subj_path + subj + '_csf.txt'
voxels_mif = subj_path + subj + '_voxels.mif' + index_gz
os.system(
    'dwi2response dhollander ' + den_unbiased_mif + ' ' + wm_txt + ' ' + gm_txt + ' ' + csf_txt + ' -voxels ' + voxels_mif + ' -mask ' + mask_mif + ' -scratch ' + subj_path + ' -fslgrad ' + bvec_path + ' ' + bval_path + '  -force')

# Viewing the Basis Functions:
# os.system('mrview '+den_unbiased_mif+ ' -overlay.load '+ voxels_mif)
# os.system('shview '+wm_txt)
# os.system('shview '+gm_txt)
# os.system('shview '+csf_txt)

# Applying the basis functions to the diffusion data:
wmfod_mif = subj_path + subj + '_wmfod.mif' + index_gz
gmfod_mif = subj_path + subj + '_gmfod.mif' + index_gz
csffod_mif = subj_path + subj + '_csffod.mif' + index_gz

# os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force' )
os.system('dwi2fod msmt_csd ' + den_unbiased_mif + ' -mask ' + mask_mif + ' ' + wm_txt + ' ' + wmfod_mif + ' -force')

# combine to single image to view them
# Concatenating the FODs:
##vf_mif =   subj_path+subj+'_vf.mif'
# os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat '+csffod_mif+ ' ' +gmfod_mif+ ' - ' + vf_mif+' -force' )
##os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat ' +gmfod_mif+ ' - ' + vf_mif+' -force' ) # without csf

# Viewing the FODs:
# os.system('mrview ' +fa_mif+ ' -odf.load_sh '+wmfod_mif )

# Normalizing the FODs:
wmfod_norm_mif = subj_path + subj + '_wmfod_norm.mif' + index_gz
# gmfod_norm_mif = subj_path+subj+'_gmfod_norm.mif'
# csffod_norm_mif = subj_path+subj+'_csffod_norm.mif'
os.system('mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' -mask ' + mask_mif + '  -force')
# Viewing the normalise FODs:
# os.system('mrview ' +fa_mif+ ' -odf.load_sh '+wmfod_norm_mif )


gmwmSeed_coreg_mif = mask_mif

####read to create streamlines
# Creating streamlines with tckgen: be carefull about number of threads on server
tracks_10M_tck = subj_path + subj + '_tracks_10M.tck'
# os.system('tckgen -act ' + fivett_coreg_mif + '  -backtrack -seed_gmwmi '+ gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.06 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')
# seconds1 = time.time()
os.system(
    'echo tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 1000 -cutoff 0.1 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')

os.system(
    'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 1000 -cutoff 0.1 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')

# os.system('tckgen -backtrack -seed_image '+ gmwmSeed_coreg_mif + ' -maxlength 1000 -cutoff 0.3 -select 50k ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')
# seconds2 = time.time()
# (seconds2 - seconds1)/360 # a million track in hippo takes 12.6 mins


# Extracting a subset of tracks:
smallerTracks = path_perm + subj + '_smallerTracks2mill.tck'
os.system('echo tckedit ' + tracks_10M_tck + ' -number 2000000 -minlength 0.1 ' + smallerTracks + ' -force')
os.system('tckedit ' + tracks_10M_tck + ' -number 2000000 -minlength 0.1 ' + smallerTracks + ' -force')
# os.system('mrview ' + den_unbiased_mif + ' -tractography.load '+ smallerTracks)


# Sifting the tracks with tcksift2: bc some wm tracks are over or underfitted
sift_mu_txt = subj_path + subj + '_sift_mu.txt'
sift_coeffs_txt = subj_path + subj + '_sift_coeffs.txt'
sift_1M_txt = subj_path + subj + '_sift_1M.txt'

os.system(
    'echo tcksift2  -out_mu ' + sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif + ' ' + sift_1M_txt + ' -force')
os.system(
    'tcksift2  -out_mu ' + sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif + ' ' + sift_1M_txt + ' -force')

#####connectome
##Running recon-all:

# os.system("SUBJECTS_DIR=`pwd`")
# sub_recon = subj_path+subj+'_recon3'
# os.system('recon-all -i '+ T1 +' -s '+ sub_recon +' -all -force')
# cant run here so do on command line


# Converting the labels:
# parcels_mif = subj_path+subj+'_parcels.mif'
# os.system('labelconvert '+ ' /Users/ali/sub-CON02_recon3/mri/aparc+aseg.mgz' + ' /Applications/freesurfer/7.3.2/FreeSurferColorLUT.txt ' +  '/Users/ali/opt/anaconda3/pkgs/mrtrix3-3.0.3-ha664bf1_0/share/mrtrix3/labelconvert/fs_default.txt '+ parcels_mif)


# Coregistering the parcellation:
# diff2struct_mrtrix_txt = subj_path+subj+'_diff2struct_mrtrix.txt'
# parcels_coreg_mif = subj_path+subj+'_parcels_coreg.mif'
# os.system('mrtransform '+parcels_mif + ' -interp nearest -linear ' + diff2struct_mrtrix_txt + ' -inverse -datatype uint32 ' + parcels_coreg_mif )


# convert subj labels to mif

labels_data = label_nii.get_fdata()
labels = np.unique(labels_data)
labels = np.delete(labels, 0)
for i in labels:
    if i > 166:  labels_data[labels_data == i] = i + 166 - 1000
file_result = nib.Nifti1Image(labels_data, label_nii.affine, label_nii.header)
new_label = subj_path + subj + '_new_label.nii.gz'
nib.save(file_result, new_label)
parcels_mif = subj_path + subj + '_parcels.mif' + index_gz
os.system('mrconvert ' + new_label + ' ' + parcels_mif + ' -force')

# os.system('mrview '+ fa_mif + ' -tractography.load '+ smallerTracks)


# Creating the connectome without coregistration:
### connectome folders :

conn_folder = root + 'connectome/'
if not os.path.isdir(conn_folder): os.mkdir(conn_folder)

distances_csv = conn_folder + subj + '_distances.csv'
os.system(
    'tck2connectome ' + smallerTracks + ' ' + parcels_mif + ' ' + distances_csv + ' -zero_diagonal -symmetric -scale_length -stat_edge  mean' + ' -force')
mean_FA_per_streamline = subj_path + subj + '_per_strmline_mean_FA.csv'
mean_FA_connectome = conn_folder + subj + '_mean_FA_connectome.csv'
os.system('tcksample ' + smallerTracks + ' ' + fa_mif + ' ' + mean_FA_per_streamline + ' -stat_tck mean ' + ' -force')
os.system(
    'tck2connectome ' + smallerTracks + ' ' + parcels_mif + ' ' + mean_FA_connectome + ' -zero_diagonal -symmetric -scale_file ' + mean_FA_per_streamline + ' -stat_edge mean ' + ' -force')

parcels_csv = conn_folder + subj + '_conn_sift_node.csv'
assignments_parcels_csv = path_perm + subj + '_assignments_con_sift_node.csv'
os.system(
    'tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in ' + sift_1M_txt + ' ' + smallerTracks + ' ' + parcels_mif + ' ' + parcels_csv + ' -out_assignment ' + assignments_parcels_csv + ' -force')

parcels_csv_2 = conn_folder + subj + '_conn_plain.csv'
assignments_parcels_csv2 = path_perm + subj + '_assignments_con_plain.csv'
os.system(
    'tck2connectome -symmetric -zero_diagonal ' + smallerTracks + ' ' + parcels_mif + ' ' + parcels_csv_2 + ' -out_assignment ' + assignments_parcels_csv2 + ' -force')

parcels_csv_3 = conn_folder + subj + '_conn_sift.csv'
assignments_parcels_csv3 = path_perm + subj + '_assignments_con_sift.csv'
os.system(
    'tck2connectome -symmetric -zero_diagonal -tck_weights_in ' + sift_1M_txt + ' ' + smallerTracks + ' ' + parcels_mif + ' ' + parcels_csv_3 + ' -out_assignment ' + assignments_parcels_csv3 + ' -force')

shutil.rmtree(subj_path)

"""