import os, socket, sys, glob
import numpy as np
import nibabel as nib
import pandas as pd
from nibabel.processing import resample_to_output
from nilearn.input_data import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.connectome import ConnectivityMeasure


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


def label_mask_inplace(label_nii,target_nii):
    label_aff = label_nii.affine
    target_aff = target_nii.affine
    new_shape = target_nii.shape

    label_data = label_nii.get_fdata()
    new_label_mat = np.zeros(new_shape[:3])

    x_coord = target_aff[0, 3]
    y_coord = target_aff[1, 3]
    z_coord = target_aff[2, 3]

    x_label = np.round((x_coord - label_aff[0, 3]) / label_aff[0, 0])
    y_label = np.round((y_coord - label_aff[1, 3]) / label_aff[1, 1])
    z_label = np.round((z_coord - label_aff[2, 3]) / label_aff[2, 2])

    for x in np.arange(new_shape[0]):
        y_coord = target_aff[1, 3]
        for y in np.arange(new_shape[1]):
            z_coord = target_aff[2, 3]
            for z in np.arange(new_shape[2]):

                """
                x_coord = x * target_aff[0, 0] + target_aff[0, 3]
                y_coord = y * target_aff[1, 1] + target_aff[1, 3]
                z_coord = z * target_aff[2, 2] + target_aff[2, 3]

                x_label = (x_coord-label_aff[0, 3])/label_aff[0, 0]
                y_label = (y_coord-label_aff[1, 3])/label_aff[1, 1]
                z_label = (z_coord-label_aff[2, 3])/label_aff[2, 2]
                """

                if x_label>=0 and y_label>=0 and z_label>=0:
                    new_label_mat[x,y,z] = int(np.round(label_data[int(x_label),int(y_label),int(z_label)]))

                z_coord += target_aff[2, 2]
                z_label = np.round((z_coord - label_aff[2, 3]) / label_aff[2, 2])

            y_coord += target_aff[1, 1]
            y_label = np.round((y_coord - label_aff[1, 3]) / label_aff[1, 1])

        x_coord+= target_aff[0,0]
        x_label = np.round((x_coord - label_aff[0, 3]) / label_aff[0, 0])

    return(new_label_mat)




def parcellated_matrix(sub_timeseries, atlas_idx, roi_list):
    timeseries_dict = {}
    for i in roi_list:
        roi_mask = np.asarray(atlas_idx == i, dtype=bool)
        timeseries_dict[i] = sub_timeseries[roi_mask].mean(axis=0)
        #print (i)
    roi_labels = list(timeseries_dict.keys())
    sub_timeseries_mean = []
    for roi in roi_labels:
        sub_timeseries_mean.append(timeseries_dict[roi])
        #print(sum(sub_timeseries_mean[int(roi)]==0))
    #corr_matrix = np.corrcoef(sub_timeseries_mean)
    return sub_timeseries_mean


def parcellated_FC_matrix(sub_timeseries, atlas_idx, roi_list):
    timeseries_dict = {}
    for i in roi_list:
        roi_mask = np.asarray(atlas_idx == i, dtype=bool)
        timeseries_dict[i] = sub_timeseries[roi_mask].mean(axis=0)
        #print (i)
    roi_labels = list(timeseries_dict.keys())
    sub_timeseries_mean = []
    for roi in roi_labels:
        sub_timeseries_mean.append(timeseries_dict[roi])
        #print(sum(sub_timeseries_mean[int(roi)]==0))
    corr_matrix = np.corrcoef(sub_timeseries_mean)
    return corr_matrix


def round_label(label_path,label_outpath=None):
    if isinstance(label_path,str):
        label_nii = nib.load(label_path)
    else:
        label_nii = label_path

    label_val = label_nii.get_fdata()

    label_val_round = np.array([
        [
            [int(round(item)) if isinstance(item, float) else item for item in row]
            for row in plane
        ]
        for plane in label_val
    ])

    if label_outpath is None:
        if isinstance(label_path,str):
            label_outpath = label_path
        else:
            raise Exception('Need a nifti path')

    img_nii_new = nib.Nifti1Image(label_val_round, label_nii.affine, label_nii.header)
    nib.save(img_nii_new,label_outpath)


def all_integers(lst):
    return all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in lst)


if socket.gethostname().split('.')[0]=='santorini':
    root = '/Volumes/Data/Badea/Lab/'
    root_proj = '/Volumes/Data/Badea/Lab/human/ADRC/'
    data_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
else:
    root = '/mnt/munin2/Badea/Lab/'
    root_proj = '/mnt/munin2/Badea/Lab/human/ADRC/'
    data_path = '/mnt/munin2/Badea/Lab/ADRC-20230511/'

SAMBA_path_results = '/Volumes/Data/Badea/Lab/mouse/VBM_23ADRC_IITmean_RPI-results/connectomics/'

#list_folders_path = os.listdir(data_path)
list_folders_path = os.listdir(SAMBA_path_results)
list_of_subjs_long = [i for i in list_folders_path if 'ADRC' in i and not '.' in i]
list_of_subjs = sorted(list_of_subjs_long)


fmriprep_output = os.path.join(root_proj,'fmriprep_output')
conn_path = os.path.join(root_proj, 'connectomes')
func_conn_path = os.path.join(conn_path,'functional_conn')


slice_func = False #Do you want to split the functionnal time series into just the first three hundred points
check_label = True

overwrite=False
#subjects = ['ADRC0001']
subjects = list_of_subjs
subjects = ['ADRC0112','ADRC0113','ADRC0116','ADRC0117','ADRC0118','ADRC0119','ADRC0123','ADRC0127','ADRC0129','ADRC0130','ADRC0134','ADRC0136','ADRC0139','ADRC0147']

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

    subj_temp = f'{subj_strip}'

    mkcdir(func_conn_path)
    fmri_nii=nib.load(fmri_path)

    time_serts_path = os.path.join(func_conn_path, f'time_series_{subj}.csv')
    time_FC_path = os.path.join(func_conn_path,f'func_connectome_corr_{subj}.csv')
    time_FCvar_path = os.path.join(func_conn_path,f'func_connectome_covar_{subj}.csv')

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


        """
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

    label_nii=nib.load(new_label)

    if not all_integers(np.unique(label_nii.get_fdata())):
        round_label(new_label)
        label_nii=nib.load(new_label)

    masker = NiftiLabelsMasker(
        labels_img=label_nii,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        verbose=5,
    )

    # Extract the time series
    confounds, sample_mask = load_confounds(fmri_path, strategy=["motion", "wm_csf"], motion="basic")

    time_series = masker.fit_transform(fmri_nii, confounds=confounds, sample_mask=sample_mask)

    if not os.path.exists(time_serts_path) or overwrite:
        if os.path.exists(time_serts_path):
            os.remove(time_serts_path)
        np.savetxt(time_serts_path, time_series, delimiter=',', fmt='%s')

    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )

    covar_measure = ConnectivityMeasure(
        kind="covariance",
        standardize="zscore_sample",
    )

    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    covar_matrix = covar_measure.fit_transform([time_series])[0]

    np.fill_diagonal(correlation_matrix, 0)

    if not os.path.exists(time_FC_path) or overwrite:
        if os.path.exists(time_FC_path):
            os.remove(time_FC_path)
        np.savetxt(time_FC_path, correlation_matrix, delimiter=',', fmt='%s')

    if not os.path.exists(time_FCvar_path) or overwrite:
        if os.path.exists(time_FCvar_path):
            os.remove(time_FCvar_path)
        np.savetxt(time_FCvar_path, covar_matrix, delimiter=',', fmt='%s')
