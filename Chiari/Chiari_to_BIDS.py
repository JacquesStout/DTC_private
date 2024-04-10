
import json, os, shutil, glob
import numpy as np
import SimpleITK as sitk
#from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile

def mkcdir(folderpaths, sftp=None):
    #creates new folder only if it doesnt already exists
    import numpy as np
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

def get_bxh_method(bxh_path):
    if os.path.exists(bxh_path):
        with open(bxh_path, 'r') as file:
            for line in file:
                if '<psdname>' in line:
                    method_name = line.split('>')[1].split('<')[0]
        return(method_name)
    else:
        print(f'Could not find {bxh_path}')


subjects = ['J01501','J01516','J04602','J01541']

orig_path = '/Volumes/Data/Jasien/ADSB.01/Data/'
output_path = '/Users/jas/jacques/Jasien/Dataset_BIDS/Chiari_test'

for subj in subjects:
    subj = subj.replace('J','')
    subj_anat_folder = os.path.join(orig_path, 'Anat', subj)
    if not os.path.exists(subj_anat_folder):
        subj_anat_folders = glob.glob(os.path.join(orig_path,'Anat',f'*{subj}'))
        if np.size(subj_anat_folders==1):
            subj_anat_folder = subj_anat_folders[0]
        else:
            txt = f'Issue identifying the correct subject path folder for subject {subj}'
            raise Exception(txt)


    subj_folder = os.path.join(output_path, f'sub-{subj}')
    anat_folder = os.path.join(output_path, f'sub-{subj}/anat')
    func_folder = os.path.join(output_path, f'sub-{subj}/func')

    bxh_files = glob.glob(os.path.join(subj_anat_folder,'*bxh'))

    mkcdir([subj_folder, anat_folder, func_folder], None)

    for bxh_file in bxh_files:
        method = get_bxh_method(bxh_file)
        if method == 'MP-RAGE':
            t1_path_orig = bxh_file.replace('.bxh','.nii.gz')
            t1_json_path_orig = bxh_file.replace('.bxh','.json')
            break

    t1_nii_path = os.path.join(anat_folder,f'sub-{subj}_T1w.nii.gz')
    t1_json_path = os.path.join(anat_folder,f'sub-{subj}_T1w.json')

    if not os.path.exists(t1_nii_path):
        shutil.copy(t1_path_orig,t1_nii_path)

    # save dict in 'header.json'
    if not os.path.exists(t1_json_path):
        # read image
        if not os.path.exists(t1_json_path_orig):
            itk_image = sitk.ReadImage(t1_nii_path)

            # get metadata dict
            header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
            with open(t1_json_path, "w") as outfile:
                json.dump(header, outfile, indent=4)
        else:
            shutil.copy(t1_json_path_orig,t1_json_path)


    subj_func_folder = os.path.join(orig_path, 'Func', subj)
    if not os.path.exists(subj_func_folder):
        subj_func_folders = glob.glob(os.path.join(orig_path,'Func',f'*{subj}'))
        if np.size(subj_func_folders==1):
            subj_func_folder = subj_func_folders[0]
        else:
            txt = f'Issue identifying the correct subject path folder for subject {subj}'
            raise Exception(txt)

    func_path_orig = glob.glob(os.path.join(subj_func_folder,'*.nii.gz'))
    if np.size(func_path_orig)==1:
        func_path_orig = func_path_orig[0]
        func_json_orig = func_path_orig.replace('.nii.gz','.json')

    #os.path.join(orig_path,subj,'visit1/resting_state.nii.gz')  # change this with your file

    func_nii_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.nii.gz')
    func_json_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.json')

    if not os.path.exists(func_nii_path):
        shutil.copy(func_path_orig,func_nii_path)

    # save dict in 'header.json'
    if not os.path.exists(func_json_path):
        if not os.path.exists(func_json_orig):
            # read image
            itk_image = sitk.ReadImage(func_nii_path)

            # get metadata dict
            header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
            with open(func_json_path, "w") as outfile:
                json.dump(header, outfile, indent=4)
        else:
            shutil.copy(func_json_orig,func_json_path)
