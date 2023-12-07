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

#n_threds = str(multiprocessing.cpu_count())

subj = sys.argv[1] #reads subj number with s... from input of python file 
#subj = "S01912"

subj_ref = subj



#subj = "H21593" #reads subj number with s... from input of python file 

index_gz = ".gz"

root= '/mnt/munin2/Badea/Lab/mouse/mrtrix_ad_decode/'
#root= '/Users/ali/Desktop/Mar23/mrtrixc_ad_decode/'


orig_subj_path = '/mnt/munin2/Badea/ADdecode.01/Analysis/DWI/'
#orig_subj_path = '/Users/ali/Desktop/Mar23/mrtrixc_ad_decode/DWI/'

bvec_path_orig = orig_subj_path+subj_ref+'_bvecs_fix.txt' 
if not os.path.isfile(bvec_path_orig) : print('where is original bvec?')
nii_gz_path_orig =  orig_subj_path  + subj +'_subjspace_coreg.nii.gz'
if not os.path.isfile(nii_gz_path_orig) : print('where is original 4d?')
bval_path_orig = orig_subj_path + subj_ref +'_bvals_fix.txt'
if not os.path.isfile(bval_path_orig) : print('where is original bval?')
T1_orig =  orig_subj_path+subj+'_subjspace_dwi.nii.gz'
if not os.path.isfile(T1_orig) : print('where is original DWI?')
b0_orig =  orig_subj_path+subj+'_subjspace_b0.nii.gz'
if not os.path.isfile(b0_orig) : print('where is original b0?')




path_perm = root + 'perm_files/'
if not os.path.isdir(path_perm) : os.mkdir(path_perm)
bval_path = path_perm  + subj + '_bvals_RAS.txt'
old_bval = np.loadtxt(bval_path_orig)
new_bval = np.round(old_bval)
#new_bval.shape
np.savetxt(bval_path, new_bval, newline=" ", fmt='%f') # saving the RAS bvec

#bval_path = '/Users/ali/Downloads/N59066_bval.txt'



T1 =T1_orig
#os.system('/Applications/Convert3DGUI.app/Contents/bin/c3d ' + T1_orig +" -orient RAS -o "+T1) # RAS rotation T1

if not os.path.isdir(root +  'temp/' ) : os.mkdir(root +  'temp/' )
subj_path = root +  'temp/' + subj + '/'
if not os.path.isdir(subj_path) : os.mkdir(subj_path)

#diff=nib.load(nii_gz_path_orig) # read the data of this volumetrtic file as nib object
#diff_data=diff.get_fdata() #read data as array 

#for i in range(int(diff_data.shape[3])):
#    diff_data_3d=diff_data[:,:,:,i]
#    squuezed=squeeze_image(nib.Nifti1Image(diff_data_3d,diff.affine)) #squeeze the last dimension
#    vol_ith_path = subj_path + subj + '_'+str(i)+'.nii.gz'
#    nib.save(squuezed, vol_ith_path) 
#    vol_ith_RAS_path = subj_path + subj + '_'+str(i)+'_RAS.nii.gz'
#    os.system("/Applications/Convert3DGUI.app/Contents/bin/c3d "+vol_ith_path+" -orient RAS -o "+vol_ith_RAS_path)

#volS_RAS_path = subj_path + subj + '_' 
#nii_gz_path =  path_perm  + subj +'_RAS_coreg.nii.gz'
#os.system(f'/Users/ali/Downloads/ANTsR/install/bin/ImageMath 4 {nii_gz_path} TimeSeriesAssemble 1 0 {volS_RAS_path}*RAS.nii.gz') # concatenate volumes of fmri
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
T1_mif = subj_path+subj+'_T1.mif'+index_gz
os.system('mrconvert ' +T1+ ' '+T1_mif + ' -force' )

out_mif = subj_path + subj+'_subjspace_dwi.mif'+index_gz
os.system('mrconvert '+nii_gz_path+ ' ' +out_mif+' -fslgrad '+bvec_path+ ' '+ bval_path+' -bvalue_scaling 0 -force') #turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000 
    
#os.system('mrinfo '+out_mif+ ' -export_grad_fsl ' + '/Users/ali/Desktop/Feb23/mrtrix_pipeline/temp/N59141/test.bvecs /Users/ali/Desktop/Feb23/mrtrix_pipeline/temp/N59141/test.bvals -force'  ) #print the 4th dimension



#os.system('mrinfo '+out_mif) #info
#os.system('vi '+bval_path) #info
#os.system('cat '+bvec_path_PA) #info

#os.system('mrinfo -size '+out_mif+" | awk '{print $4}' ") #print the 4th dimension 
#os.system("awk '{print NF; exit}' "+bvec_path) #print the lenght of bvec : must match the previous number
#os.system("awk '{print NF; exit}' "+bval_path) #print the lenght of bval : must match the previous number
#os.system("mrview "+ out_mif) #viewer 
 
#preprocessing
#denoise:
    #####skip denoise for mouse so far
#output_denoise =   subj_path+subj+'_den.mif'
#os.system('dwidenoise '+out_mif + ' ' + output_denoise+' -force') #denoising
#compute residual to check if any resdiual is loaded on anatomy
#output_residual = subj_path+subj+'residual.mif'
#os.system('mrcalc '+out_mif + ' ' + output_denoise+ ' -subtract '+ output_residual+ ' -force') #compute residual
#os.system('mrview '+ output_denoise) #inspect residual
output_denoise = out_mif #####skip denoise

#making fa and Kurt:
    
dt_mif = path_perm+subj+'_dt.mif'+index_gz
fa_mif = path_perm+subj+'_fa.mif'+index_gz
dk_mif = path_perm+subj+'_dk.mif'+index_gz
mk_mif = path_perm+subj+'_mk.mif'+index_gz
md_mif = path_perm+subj+'_md.mif'+index_gz
ad_mif = path_perm+subj+'_ad.mif'+index_gz
rd_mif = path_perm+subj+'_rd.mif'+index_gz

#output_denoise = '/Users/ali/Desktop/Feb23/mrtrix_pipeline/temp/N59141/N59141_subjspace_dwi_copy.mif.gz'#

if np.unique(new_bval).shape[0] > 2 :
    os.system('dwi2tensor ' + output_denoise + ' ' + dt_mif + ' -dkt ' +  dk_mif +' -fslgrad ' +  bvec_path + ' ' + bval_path + ' -force'  )
    os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -adc '  + md_mif+' -ad '  + ad_mif + ' -rd '  + rd_mif   + ' -force' ) 

    #os.system('mrview '+ fa_mif) #inspect residual
else: 
    os.system('dwi2tensor ' + output_denoise + ' ' + dt_mif  +' -fslgrad ' +  bvec_path + ' ' + bval_path + ' -force'  )
    os.system('tensor2metric  -fa ' + fa_mif  + ' '+ dt_mif + ' -force' ) 
    os.system('tensor2metric  -rd ' + rd_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
    os.system('tensor2metric  -ad ' + ad_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
    os.system('tensor2metric  -adc ' + md_mif  + ' '+ dt_mif + ' -force' ) # if doesn't work take this out :(
    #os.system('mrview '+ fa_mif) #inspect residual

den_preproc_mif = output_denoise # already skipping preprocessing (always)
#os.system('mrview '+den_preproc_mif+' -overlay.load '+out_mif)


#createing mask after bias correction:
#den_unbiased_mif = subj_path+subj+'_den_preproc_unbiased.mif'
#bias_mif = subj_path+subj+'_bias.mif'
#os.system('dwibiascorrect ants '+den_preproc_mif+' '+den_unbiased_mif+ ' -bias '+ bias_mif + ' -force')
#cannot be done here go on on terminal after echoing and python it
den_unbiased_mif = den_preproc_mif  # bypassing

mask_mif  =  path_perm+subj+'_mask.mif'
os.system('dwi2mask '+den_unbiased_mif+  ' '+ mask_mif + ' -force')
#os.system('mrview '+fa_mif + ' -overlay.load '+ mask_mif )  
mask_mrtrix_nii = subj_path +subj+'_mask_mrtrix.nii.gz'

os.system('mrconvert ' +mask_mif+ ' '+mask_mrtrix_nii + ' -force' )


#making mask

mask_nii_gz = path_perm +subj+'_mask.nii.gz'
csf_nii_gz = subj_path +subj+'_csf.nii.gz'

os.system('ImageMath 3 ' + csf_nii_gz + ' ThresholdAtMean '+ b0_orig + ' 10')
os.system('ImageMath 3 ' + csf_nii_gz + ' MD '+ csf_nii_gz + ' 1')
os.system('ImageMath 3 ' + csf_nii_gz + ' ME '+ csf_nii_gz + ' 1')



os.system('ImageMath 3 ' + mask_nii_gz + ' - '+ mask_mrtrix_nii +  ' '+ csf_nii_gz)
os.system('ThresholdImage 3 ' + mask_nii_gz + ' '+ mask_nii_gz +  ' 0.0001 1 1 0')
os.system('ImageMath 3 ' + mask_nii_gz + ' ME '+ mask_nii_gz + ' 1')
os.system('ImageMath 3 ' + mask_nii_gz + ' MD '+ mask_nii_gz + ' 1')


os.system('mrconvert ' +mask_nii_gz+ ' '+mask_mif + ' -force' )
#os.system('mrview '+fa_mif + ' -overlay.load '+ mask_mif )  








########### making a mask out of labels

label_path_orig = orig_subj_path +subj+'_labels.nii.gz'
#label_path = path_perm +subj+'_labels.nii.gz'
#os.system("/Applications/Convert3DGUI.app/Contents/bin/c3d "+label_path_orig+" -orient RAS -o "+label_path)
label_path = label_path_orig
#mask_output = subj_path +subj+'_mask_of_label.nii.gz'
label_nii=nib.load(label_path)
#mask_labels_data = label_nii.get_fdata()
#mask_labels = np.unique(mask_labels_data)
#mask_labels=np.delete(mask_labels, 0)
#mask_of_label =label_nii.get_fdata()*0



path_atlas_legend = root+ 'IIT/IITmean_RPI_index.xlsx'
legend  = pd.read_excel(path_atlas_legend)
#index_csf = legend [ 'Subdivisions_7' ] == '8_CSF'
#index_wm = legend [ 'Subdivisions_7' ] == '7_whitematter'

#vol_index_csf = legend[index_csf]
#vol_index_csf = vol_index_csf['index2']

#mask_labels_no_csf= set(mask_labels)-set(vol_index_csf )
#mask_labels_no_csf = mask_labels
#for vol in mask_labels_no_csf: mask_of_label[  mask_labels_data == int(vol)] = int(1)
#mask_of_label= mask_of_label.astype(int)

#file_result= nib.Nifti1Image(mask_of_label, label_nii.affine, label_nii.header)
#nib.save(file_result,mask_output  )  
#mask_labels_mif   = subj_path +subj+'mask_of_label.mif'+index_gz
#os.system('mrconvert '+mask_output+ ' ' +mask_labels_mif+ ' -datatype uint16 -force')
#os.system('mrview '+fa_mif + ' -overlay.load '+ mask_labels_mif ) 


#new_bval_path = path_perm+subj+'_new_bvals.txt' 
#new_bvec_path = path_perm+subj+'_new_bvecs.txt' 
#os.system('dwigradcheck ' + out_mif +  ' -fslgrad '+bvec_path+ ' '+ bval_path +' -mask '+ mask_mif + ' -number 100000 -export_grad_fsl '+ new_bvec_path + ' '  + new_bval_path  +  ' -force' )
#bvec_temp=np.loadtxt(new_bvec_path)

    
#Estimating the Basis Functions:
wm_txt =   subj_path+subj+'_wm.txt' 
gm_txt =  subj_path+subj+'_gm.txt' 
csf_txt = subj_path+subj+'_csf.txt'
voxels_mif =  subj_path+subj+'_voxels.mif'+index_gz
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
os.system('mtnormalise ' +wmfod_mif+ ' '+wmfod_norm_mif+' -mask ' + mask_mif + '  -force')
#Viewing the normalise FODs:
#os.system('mrview ' +fa_mif+ ' -odf.load_sh '+wmfod_norm_mif )





gmwmSeed_coreg_mif  = mask_mif  

####read to creat streamlines
#Creating streamlines with tckgen: be carefull about number of threads on server
tracks_10M_tck  = subj_path +subj+'_tracks_10M.tck' 

#os.system('tckgen -act ' + fivett_coreg_mif + '  -backtrack -seed_gmwmi '+ gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.06 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')
#seconds1 = time.time()
os.system('echo tckgen -backtrack -seed_image '+ gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.06 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')

os.system('tckgen -backtrack -seed_image '+ gmwmSeed_coreg_mif + '  -maxlength 410 -cutoff 0.05 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')

#os.system('tckgen -backtrack -seed_image '+ gmwmSeed_coreg_mif + ' -maxlength 1000 -cutoff 0.3 -select 50k ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force')
#seconds2 = time.time()
#(seconds2 - seconds1)/360 # a million track in hippo takes 12.6 mins


#Extracting a subset of tracks:
smallerTracks = path_perm+subj+'_smallerTracks2mill.tck'
os.system('echo tckedit '+ tracks_10M_tck + ' -number 2000000 -minlength 0.1 ' + smallerTracks + ' -force')

os.system('tckedit '+ tracks_10M_tck + ' -number 2000000 -minlength 2 ' + smallerTracks + ' -force')
#os.system('mrview ' + den_unbiased_mif + ' -tractography.load '+ smallerTracks)
#os.system('mrview ' + den_unbiased_mif + ' -tractography.load '+ smallerTracks)




#Sifting the tracks with tcksift2: bc some wm tracks are over or underfitted
sift_mu_txt = subj_path+subj+'_sift_mu.txt'
sift_coeffs_txt = subj_path+subj+'_sift_coeffs.txt'
sift_1M_txt = subj_path+subj+'_sift_1M.txt'

os.system('echo tcksift2  -out_mu '+ sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif+ ' ' + sift_1M_txt  + ' -force')
os.system('tcksift2  -out_mu '+ sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif+ ' ' + sift_1M_txt  + ' -force')

#####connectome
##Running recon-all:

#os.system("SUBJECTS_DIR=`pwd`")
#sub_recon = subj_path+subj+'_recon3'
#os.system('recon-all -i '+ T1 +' -s '+ sub_recon +' -all -force')
# cant run here so do on command line


#Converting the labels:
#parcels_mif = subj_path+subj+'_parcels.mif'
#os.system('labelconvert '+ ' /Users/ali/sub-CON02_recon3/mri/aparc+aseg.mgz' + ' /Applications/freesurfer/7.3.2/FreeSurferColorLUT.txt ' +  '/Users/ali/opt/anaconda3/pkgs/mrtrix3-3.0.3-ha664bf1_0/share/mrtrix3/labelconvert/fs_default.txt '+ parcels_mif)



#Coregistering the parcellation:
#diff2struct_mrtrix_txt = subj_path+subj+'_diff2struct_mrtrix.txt'
#parcels_coreg_mif = subj_path+subj+'_parcels_coreg.mif'
#os.system('mrtransform '+parcels_mif + ' -interp nearest -linear ' + diff2struct_mrtrix_txt + ' -inverse -datatype uint32 ' + parcels_coreg_mif )



#convert subj labels to mif

labels_data = label_nii.get_fdata()
labels = np.unique(labels_data)
labels=np.delete(labels, 0)
label_nii_order = labels_data*0.0

#sum(legend['index2'] == labels)
for i in labels: 
    leg_index =  np.where(legend['index2'] == i )
    leg_index = leg_index [0][0]
    ordered_num = legend['index'][leg_index]
    label3d_index = np.where( labels_data == i )
    label_nii_order [ label3d_index]  = ordered_num


file_result= nib.Nifti1Image(label_nii_order, label_nii.affine, label_nii.header)
new_label  = path_perm +subj+'_new_label.nii.gz'
nib.save(file_result, new_label ) 

parcels_mif = subj_path+subj+'_parcels.mif'+index_gz
#new_label = label_path
os.system('mrconvert '+new_label+ ' ' +parcels_mif + ' -force' )


#os.system('mrview '+ fa_mif + ' -overlay.load '+ new_label) 



#Creating the connectome without coregistration:
### connectome folders :
    
conn_folder = root + 'connectome/'
if not os.path.isdir(conn_folder) : os.mkdir(conn_folder)
    
    

distances_csv = conn_folder +subj+'_distances.csv'
os.system('tck2connectome ' + smallerTracks + ' ' + parcels_mif+ ' ' + distances_csv + ' -zero_diagonal -symmetric -scale_length -stat_edge  mean' + ' -force')
mean_FA_per_streamline =  subj_path+subj+'_per_strmline_mean_FA.csv'
mean_FA_connectome =  conn_folder+subj+'_mean_FA_connectome.csv'
os.system('tcksample '+ smallerTracks+ ' '+ fa_mif + ' ' + mean_FA_per_streamline + ' -stat_tck mean ' + ' -force')
os.system('tck2connectome '+ smallerTracks + ' ' + parcels_mif + ' '+ mean_FA_connectome + ' -zero_diagonal -symmetric -scale_file ' + mean_FA_per_streamline + ' -stat_edge mean '+ ' -force')

  




parcels_csv = conn_folder+subj+'_conn_sift_node.csv'
assignments_parcels_csv = path_perm +subj+'_assignments_con_sift_node.csv'
os.system('tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in '+ sift_1M_txt+ ' '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv + ' -out_assignment ' + assignments_parcels_csv + ' -force')


parcels_csv_2 = conn_folder+subj+'_conn_plain.csv'
assignments_parcels_csv2 = path_perm +subj+'_assignments_con_plain.csv'
os.system('tck2connectome -symmetric -zero_diagonal '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv_2 + ' -out_assignment ' + assignments_parcels_csv2 + ' -force')

parcels_csv_3 = conn_folder+subj+'_conn_sift.csv'
assignments_parcels_csv3 = path_perm +subj+'_assignments_con_sift.csv'
os.system('tck2connectome -symmetric -zero_diagonal -tck_weights_in '+ sift_1M_txt+ ' '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv_3 + ' -out_assignment ' + assignments_parcels_csv3 + ' -force')


shutil.rmtree(subj_path )


#scale_invnodevol scale connectome by the inverse of size of each node
#tck_weights_in weight each connectivity by sift
#out assignment helo converting connectome to tracks

#Viewing the connectome in Matlab:

#connectome = importdata('sub-CON02_parcels.csv');
#imagesc(connectome, [0 1])

#Viewing the lookup labels:

