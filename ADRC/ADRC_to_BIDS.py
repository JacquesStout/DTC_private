
import json, os, shutil
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

subj = 'ADRC0001'

orig_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
output_path = '/Users/jas/jacques/ADRC/ADRC_Dataset/'

subj_folder = os.path.join(output_path,f'sub-{subj}')
anat_folder = os.path.join(output_path,f'sub-{subj}/anat')
func_folder = os.path.join(output_path,f'sub-{subj}/func')

mkcdir([subj_folder,anat_folder,func_folder],None)

t1_path_orig = os.path.join(orig_path,subj,'visit1/T1.nii.gz')  # change this with your file

t1_nii_path = os.path.join(anat_folder,f'sub-{subj}_T1w.nii.gz')
t1_json_path = os.path.join(anat_folder,f'sub-{subj}_T1w.json')

if not os.path.exists(t1_nii_path):
    shutil.copy(t1_path_orig,t1_nii_path)

# save dict in 'header.json'
if not os.path.exists(t1_json_path):
    # read image
    itk_image = sitk.ReadImage(t1_nii_path)

    # get metadata dict
    header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
    with open(t1_json_path, "w") as outfile:
        json.dump(header, outfile, indent=4)


func_path_orig = os.path.join(orig_path,subj,'visit1/resting_state.nii.gz')  # change this with your file

func_nii_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.nii.gz')
func_json_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.json')

if not os.path.exists(func_nii_path):
    shutil.copy(func_path_orig,func_nii_path)

# save dict in 'header.json'
if not os.path.exists(func_json_path):
    # read image
    itk_image = sitk.ReadImage(func_nii_path)

    # get metadata dict
    header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
    with open(func_json_path, "w") as outfile:
        json.dump(header, outfile, indent=4)

