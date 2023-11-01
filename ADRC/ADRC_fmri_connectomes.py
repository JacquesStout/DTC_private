import os, socket, sys, glob
import numpy as np
import nibabel as nib
import pandas as pd
from nibabel.processing import resample_to_output


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


if socket.gethostname().split('.')[0]=='santorini':
    root = '/Volumes/Data/Badea/Lab/'
    root_proj = '/Volumes/Data/Badea/Lab/human/ADRC/'
    data_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
else:
    root = '/mnt/munin2/Badea/Lab/'
    root_proj = '/mnt/munin2/Badea/Lab/human/ADRC/'
    data_path = '/mnt/munin2/Badea/Lab/ADRC-20230511/'


list_folders_path = os.listdir(data_path)
list_of_subjs_long = [i for i in list_folders_path if 'ADRC' in i and not '.' in i]
list_of_subjs = sorted(list_of_subjs_long)


fmriprep_output = os.path.join(root_proj,'fmriprep_output')
conn_path = os.path.join(root_proj, 'connectomes')
func_conn_path = os.path.join(conn_path,'functional_conn')

SAMBA_path_results = '/Volumes/Data/Badea/Lab/mouse/VBM_23ADRC_IITmean_RPI-results/connectomics/'

slice_func = False #Do you want to split the functionnal time series into just the first three hundred points


subjects = []

for subj in subjects:
    subj_strip = subj.replace('J','')
    subj_path = os.path.join(fmriprep_output, f'sub-{subj_strip}')
    fmri_path = os.path.join(subj_path,'func',f'sub-{subj_strip}_task-restingstate_run-01_space-T1w_desc-preproc_bold.nii.gz')
    if not os.path.exists(fmri_path):
        txt = (f'Could not find the fmri for subject {subj_strip}')
        print(txt)
        continue

    flabel = os.path.join(conn_path, subj + '_new_labels_resampled.nii.gz')
    new_label = os.path.join(conn_path, subj + '_new_labels.nii.gz')

    subj_temp = f'T{subj_strip}'

    mkcdir(func_conn_path)
    fmri_nii=nib.load(fmri_path)

    time_serts_path = os.path.join(func_conn_path, f'time_serts_{subj}.csv')
    time_FC_path = os.path.join(func_conn_path,f'time_serFC_{subj}.csv')

    print(f'Running functionnal connectomes for subject {subj}')

    if not os.path.exists(flabel) or overwrite:
        if not os.path.exists(new_label) or overwrite:
            label_path = os.path.join(SAMBA_path_results, subj_temp, subj_temp + '_IITmean_RPI_labels.nii.gz')
            label_nii = nib.load(label_path)
            labels_data = label_nii.get_fdata()
            labels = np.unique(labels_data)
            labels = np.delete(labels, 0)
            label_nii_order = labels_data * 0.0

            path_atlas_legend = os.path.join(root, 'atlases', 'IITmean_RPI', 'IITmean_RPI_index.xlsx')
            legend = pd.read_excel(path_atlas_legend)
            new_label = os.path.join(conn_path, subj + '_new_labels.nii.gz')

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
            new_label = os.path.join(conn_path, subj + '_new_labels.nii.gz')
            nib.save(file_result, new_label)

        label_new_nii = nib.load(new_label)
        label_spacing = label_new_nii.header.get_zooms()[0:3]
        label_shape = label_new_nii.shape
        label_new_data = label_new_nii.get_fdata()
        fmri_spacing = fmri_nii.header.get_zooms()[0:3]
        fmri_shape = fmri_nii.shape
        #command = f'ResampleImage 3 {new_label} {flabel} {fmri_spacing[0]}x{fmri_spacing[1]}x{fmri_spacing[2]}x1'
        #os.system(command)

        target_nii = nib.load(fmri_path)

        resampled_source_image = resample_to_output(label_new_nii, np.diagonal(target_nii.affine)[:3])
        nib.save(resampled_source_image, flabel)

        """
        scaling_factors = [label_spacing[i] / fmri_spacing[i] for i in range(3)]
        resampled_image = zoom(label_new_data, scaling_factors, order=0, mode='nearest')
        resampled_img = nib.Nifti1Image(resampled_image, fmri_nii.affine)
        nib.save(resampled_img, flabel)
        """
        """
        ants_label = ants_imgread(new_label)
        ants_fmri = ants_imgread(fmri_path)
        ants_output = resample_image_to_target(ants_label,ants_fmri,interp ='genericLabel',imagetype=3)
        ants_imgwrite(ants_output, flabel)
        """

        # Run the command
        #ants_apply_transforms = apply_transforms_to_points(ants_apply_transforms_command)

    label_path= flabel
    label_nii=nib.load(label_path)
    #label_nii.shape
    data_label=label_nii.get_fdata()

    #atlas_idx = data_label

    """
    voxel_coords = nib.affines.apply_affine(nib.affines.inv(affine1), [x, y, z])
    voxel_coords = [int(round(coord)) for coord in voxel_coords]
    label_val = label_nii.get_fdata()[tuple(voxel_coords)]
    value_nifti2 = nifti2.get_fdata()[tuple(voxel_coords)]

    """


    #Creating a new label matrix that has the exact same dimensions as the fmri image.
    #Simple code, assumes that they're already aligned and have same voxel size and that voxel size is a good affine diagonal!!

    flabeltest = os.path.join(conn_path, subj + '_new_labels_resampled_test.nii.gz')
    if os.path.exists(flabeltest) and not overwrite:
        label_mask_in = nib.load(flabeltest).get_fdata()
    else:
        label_mask_in = label_mask_inplace(label_nii,fmri_nii)

    if check_label and (not os.path.exists(flabeltest) or overwrite):
        #Nice bit of code to check if the resampled version is still aligned with previous labels and fmri image
        nii_test = nib.Nifti1Image(label_mask_in, fmri_nii.affine)
        flabeltest = os.path.join(conn_path, subj + '_new_labels_resampled_test.nii.gz')
        nib.save(nii_test,flabeltest)

    sub_timeseries=fmri_nii.get_fdata()

    roi_list=np.unique(label_mask_in)
    roi_list = roi_list[1:]

    result=parcellated_matrix(sub_timeseries, label_mask_in, roi_list)

    if not os.path.exists(time_serts_path) or overwrite:
        if os.path.exists(time_serts_path):
            os.remove(time_serts_path)
        np.savetxt(time_serts_path, result, delimiter=',', fmt='%s')


    # if more than 298 time series limit the time series to 299 tp
    if slice_func:
        if sub_timeseries.shape[3] >298 : sub_timeseries = sub_timeseries[:,:,:,:299]


    resultFC=parcellated_FC_matrix(sub_timeseries, label_mask_in, roi_list)
    if not os.path.exists(resultFC) or overwrite:
        if os.path.exists(time_FC_path):
            os.remove(time_FC_path)
        np.savetxt(time_FC_path, resultFC, delimiter=',', fmt='%s')