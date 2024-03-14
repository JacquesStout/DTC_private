import os, subprocess, sys, shutil, glob, re, fnmatch
import socket

from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
import numpy as np
from scipy.cluster.vq import kmeans, vq
import itertools
import nibabel as nib
import pandas as pd


def mkcdir(folderpaths, sftp=None):
    # creates new folder only if it doesnt already exists

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


"""
if socket.gethostname().split('.')[0] == 'santorini':
    root = '/Volumes/Data/Badea/Lab/'
    #root_proj = '/Volumes/Data/Badea/Lab/mouse/Jasien_mrtrix_pipeline/'
    root_proj = '/Volumes/Shared Folder/newJetStor/paros/paros_DB/Projects/Jasien/'
    data_path = '/Volumes/Data/Jasien/ADSB.01/Data/Anat/'
    #data_path_output = '/Volumes/Data/Badea/Lab/mouse/Jasien_mrtrix_pipeline/'
    data_path_output = '/Volumes/Shared Folder/newJetStor/paros/paros_DB/Projects/Jasien/'
else:
    root = '/mnt/munin2/Badea/Lab/'
    root_proj = '/mnt/munin2/Badea/Lab/mouse/Jasien_mrtrix_pipeline/'
    data_path = '/mnt/munin2/Jasien/ADSB.01/Data/Anat/'
    data_path_output = '/mnt/munin2/Badea/Lab/mouse/Jasien_mrtrix_pipeline/'
"""

if 'santorini' in socket.gethostname().split('.')[0]:
    work_path = '/Volumes/Data/Jasien/ADSB.01/Analysis/mrtrix_pipeline/work_dir'
    data_path = '/Volumes/Data/Jasien/ADSB.01/Data/Anat/'
    #data_path_output = '/Volumes/Data/Badea/Lab/mouse/Jasien_mrtrix_pipeline/'
    data_path_output = '/Volumes/Data/Jasien/ADSB.01/Analysis/mrtrix_pipeline'
    root_proj = '/Volumes/Data/Jasien/ADSB.01/Analysis'
    anat_folder_path = '/Volumes/Data/Badea/Lab/atlases/'
else:
    work_path = '/mnt/munin2/Jasien/ADSB.01/Analysis/mrtrix_pipeline/work_dir'
    data_path = '/mnt/munin2/Jasien/ADSB.01/Data/Anat/'
    #data_path_output = '/mnt/munin2/Badea/Lab/mouse/Jasien_mrtrix_pipeline/'
    data_path_output = '/mnt/munin2/Jasien/ADSB.01/Analysis/mrtrix_pipeline'
    root_proj = '/mnt/munin2/Jasien/ADSB.01/Analysis'
    anat_folder_path = '/mnt/munin2/Badea/Lab/atlases/'


dwi_manual_pe_scheme_txt = os.path.join(work_path, 'dwi_manual_pe_scheme.txt')
se_epi_manual_pe_scheme_txt = os.path.join(work_path, 'se_epi_manual_pe_scheme.txt')
bvecs_grad_scheme_txt = os.path.join(work_path, 'bvecs_grad_scheme.txt')

conn_output = os.path.join(root_proj, 'connectomes','tract_conn')
perm_input = os.path.join(data_path_output, 'perm_files/')

mkcdir(conn_output)

if socket.gethostname().split('.')[0] == 'santorini':
    fsl_config_path = '$FSLDIR/src/fsl-topup/flirtsch/b02b0.cnf'
else:
    fsl_config_path = '$FSLDIR/src/topup/flirtsch/b02b0.cnf'
# outputs

mkcdir(data_path_output)

# subjects to run
subjects = []
# subjects.append(sys.argv[1])
#subjects = ['J01277', 'J01402', 'J04472', 'J04129', 'J01257', 'J04300', 'J04086']
subjects = ['J01277', 'J01402', 'J04472', 'J04129', 'J01257', 'J04300', 'J04086','J01501','J01516','J04602','J01541']
#subjects = ['J01257', 'J01277', 'J01402', 'J04086', 'J04129', 'J04300', 'J04472','J01501','J01516','J01541','J04602']
#subjects = ['J01501','J01516','J04602','J01541']

subjects = ['J04472']
#subjects.append(sys.argv[1])
# subjects = ['ADRC0001']

index_gz = '.gz'
overwrite = False

cleanup = True

median_radius = 4
numpass = 7
binary_dilation = 1
full_name = False

verbose = True

SAMBA_path_results = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/connectomics/'

sifting = True

for subj in subjects:

    subj_folder = os.path.join(data_path, subj)
    if not os.path.exists(subj_folder):
        subj_folders = glob.glob(os.path.join(data_path,f'*{subj.replace("J","")}'))
        if np.size(subj_folders==1):
            subj_folder = subj_folders[0]
        else:
            txt = f'Issue identifying the correct subject path folder for subject {subj}'
            raise Exception(txt)
    allsubjs_out_folder = os.path.join(data_path_output, 'temp')
    subj_out_folder = os.path.join(allsubjs_out_folder, subj)
    scratch_path = os.path.join(subj_out_folder, 'scratch')
    perm_subj_output = conn_output
    #conn_folder_subj = os.path.join(perm_subj_output, subj)
    conn_folder_subj = perm_subj_output
    mkcdir([allsubjs_out_folder, subj_out_folder, perm_subj_output, conn_folder_subj])

    if sifting:
        # Sifting the tracks with tcksift2: bc some wm tracks are over or underfitted

        resampled_mif_path = os.path.join(perm_input, subj + '_coreg_resampled.mif')
        mask_mif_path = os.path.join(perm_input, subj + '_mask.mif')

        coreg_bvecs = os.path.join(perm_input, subj + '_coreg_bvecs.txt')
        coreg_bvals = os.path.join(perm_input, subj + '_coreg_bvals.txt')

        wm_txt = os.path.join(subj_out_folder, subj + '_wm.txt')
        gm_txt = os.path.join(subj_out_folder + subj + '_gm.txt')
        csf_txt = os.path.join(subj_out_folder + subj + '_csf.txt')
        voxels_mif = os.path.join(subj_out_folder, subj + '_voxels.mif' + index_gz)

        wmfod_mif = os.path.join(subj_out_folder, subj + '_wmfod.mif' + index_gz)
        gmfod_mif = os.path.join(subj_out_folder, subj + '_gmfod.mif' + index_gz)
        csffod_mif = os.path.join(subj_out_folder, subj + '_csffod.mif' + index_gz)

        wmfod_norm_mif = os.path.join(perm_input, subj + '_wmfod_norm.mif' + index_gz)
        gmfod_norm_mif = os.path.join(perm_input, subj + '_gmfod_norm.mif')
        csffod_norm_mif = os.path.join(perm_input, subj + '_csffod_norm.mif')

        smallerTracks = os.path.join(perm_input, subj + '_smallerTracks2mill.tck')

        if not os.path.exists(voxels_mif) or not os.path.exists(wm_txt) or not os.path.exists(
                gm_txt) or not os.path.exists(csf_txt) or overwrite:
            command = 'dwi2response dhollander ' + resampled_mif_path + ' ' + wm_txt + ' ' + gm_txt + ' ' + csf_txt + ' -voxels ' + voxels_mif + ' -mask ' + mask_mif_path + ' -scratch ' + subj_out_folder + ' -fslgrad ' + coreg_bvecs + ' ' + coreg_bvals + '  -force'
            print(command)
            os.system(command)

        # os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force' )
        if not os.path.exists(wmfod_mif) or not os.path.exists(gmfod_mif) or not os.path.exists(
                csffod_mif) or overwrite:
            command = 'dwi2fod msmt_csd ' + resampled_mif_path + ' -mask ' + mask_mif_path + ' ' + wm_txt + ' ' + wmfod_mif + ' ' + gm_txt + ' ' + gmfod_mif + ' ' + csf_txt + ' ' + csffod_mif + ' -force'
            print(command)
            os.system(command)

        # combine to single image to view them
        # Concatenating the FODs:
        vf_mif = os.path.join(subj_out_folder, subj + '_vf.mif')
        if not os.path.exists(vf_mif) or overwrite:
            command = 'mrconvert -coord 3 0 ' + wmfod_mif + ' -| mrcat ' + csffod_mif + ' ' + gmfod_mif + ' - ' + vf_mif + ' -force'
            print(command)
            os.system(command)

        # Normalizing the FODs:

        if not os.path.exists(wmfod_norm_mif) or not os.path.exists(gmfod_norm_mif) or not os.path.exists(
                csffod_norm_mif) or overwrite:
            command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' ' + gmfod_mif + ' ' + gmfod_norm_mif + ' ' + csffod_mif + ' ' + csffod_norm_mif + ' -mask ' + mask_mif_path + '  -force'
            print(command)
            os.system(command)

        sift_mu_txt = os.path.join(conn_folder_subj, subj + '_sift_mu.txt')
        sift_coeffs_txt = os.path.join(conn_folder_subj, subj + '_sift_coeffs.txt')
        sift_1M_txt = os.path.join(conn_folder_subj, subj + '_sift_1M.txt')

        label_path = ''

        fa_mif = os.path.join(perm_input, subj + '_fa.mif' + index_gz)

        if not os.path.exists(sift_1M_txt):
            command = f'tcksift2  -out_mu {sift_mu_txt} -out_coeffs {sift_coeffs_txt} {smallerTracks} {wmfod_norm_mif} {sift_1M_txt} -force'
            os.system(command)

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

    convert_S_t = True
    if convert_S_t:
        subj_temp = subj.replace('J', 'T')
    else:
        subj_temp = subj

    new_label = os.path.join(perm_subj_output, subj + '_new_labels.nii.gz')

    if not os.path.exists(new_label):
        label_path = os.path.join(SAMBA_path_results, subj_temp, subj_temp + '_IITmean_RPI_labels.nii.gz')
        label_nii = nib.load(label_path)
        labels_data = label_nii.get_fdata()
        labels = np.unique(labels_data)
        labels = np.delete(labels, 0)
        label_nii_order = labels_data * 0.0

        path_atlas_legend = os.path.join(anat_folder_path, 'IITmean_RPI', 'IITmean_RPI_index.xlsx')
        legend = pd.read_excel(path_atlas_legend)
        new_label = os.path.join(perm_subj_output, subj + '_new_labels.nii.gz')

        # index_csf = legend [ 'Subdivisions_7' ] == '8_CSF'
        # index_wm = legend [ 'Subdivisions_7' ] == '7_whitematter'
        # vol_index_csf = legend[index_csf]

        for i in labels:
            leg_index = np.where(legend['index2'] == i)
            leg_index = leg_index[0][0]
            ordered_num = legend['index'][leg_index]
            label3d_index = np.where(labels_data == i)
            label_nii_order[label3d_index] = ordered_num

        file_result = nib.Nifti1Image(label_nii_order, label_nii.affine, label_nii.header)
        new_label = os.path.join(perm_subj_output, subj + '_new_labels.nii.gz')
        nib.save(file_result, new_label)

    parcels_mif = os.path.join(subj_out_folder, subj + '_parcels.mif' + index_gz)
    # new_label = label_path

    if not os.path.exists(parcels_mif):
        os.system(f'mrconvert {new_label} {parcels_mif} -force')

    # os.system('mrview '+ fa_mif + ' -overlay.load '+ new_label)

    # Creating the connectome without coregistration:
    ### connectome folders :

    distances_csv = os.path.join(conn_folder_subj, subj + '_distances.csv')
    mean_FA_connectome = os.path.join(conn_folder_subj, subj + '_mean_FA_connectome.csv')
    assignments_parcels_csv = os.path.join(conn_folder_subj, subj + '_assignments_con_sift_node.csv')
    parcels_csv = os.path.join(conn_folder_subj, subj + '_conn_sift_node.csv')
    parcels_csv_2 = os.path.join(conn_folder_subj, subj + '_conn_plain.csv')
    assignments_parcels_csv2 = os.path.join(conn_folder_subj, subj + '_assignments_con_plain.csv')
    assignments_parcels_csv3 = os.path.join(conn_folder_subj, subj + '_assignments_con_sift.csv')
    parcels_csv_3 = os.path.join(conn_folder_subj, subj + '_conn_sift.csv')
    mean_FA_per_streamline = os.path.join(subj_out_folder, subj + '_per_strmline_mean_FA.csv')
    print(f'information outputted at {conn_folder_subj}')

    if not os.path.exists(distances_csv) or overwrite:
        os.system('tck2connectome ' + smallerTracks + ' ' + parcels_mif + ' ' + distances_csv +
                  ' -zero_diagonal -symmetric -scale_length -stat_edge  mean' + ' -force')

    if not os.path.exists(mean_FA_per_streamline) or overwrite:
        os.system('tcksample ' + smallerTracks + ' ' + fa_mif + ' ' + mean_FA_per_streamline +
                  ' -stat_tck mean ' + ' -force')

    if not os.path.exists(mean_FA_connectome) or overwrite:
        os.system('tck2connectome ' + smallerTracks + ' ' + parcels_mif + ' ' + mean_FA_connectome +
                  ' -zero_diagonal -symmetric -scale_file ' + mean_FA_per_streamline + ' -stat_edge mean ' + ' -force')

    if not os.path.exists(parcels_csv_2) or overwrite:
        os.system('tck2connectome -symmetric -zero_diagonal ' + smallerTracks + ' ' + parcels_mif + ' ' +
                  parcels_csv_2 + ' -out_assignment ' + assignments_parcels_csv2 + ' -force')

    if sifting:
        if not os.path.exists(assignments_parcels_csv) or overwrite:
            os.system('tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in ' + sift_1M_txt + ' '
                      + smallerTracks + ' ' + parcels_mif + ' ' + parcels_csv + ' -out_assignment ' +
                      assignments_parcels_csv + ' -force')
        if not os.path.exists(assignments_parcels_csv3) or overwrite:
            os.system('tck2connectome -symmetric -zero_diagonal -tck_weights_in ' + sift_1M_txt + ' ' + smallerTracks +
                      ' ' + parcels_mif + ' ' + parcels_csv_3 + ' -out_assignment ' + assignments_parcels_csv3 + ' -force')

