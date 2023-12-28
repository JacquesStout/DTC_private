
from DTC.nifti_handlers.transform_handler import img_transform_exec, header_superpose
from DTC.file_manager.computer_nav import make_temppath
import os, glob
import numpy as np


contrast = 'rd'
overwrite=False

input_folder = f'/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_ad_decode/{contrast}_nii_gz'
SAMBA_folder = '/Volumes/Data/Badea/Lab/mouse/ADDeccode_symlink_pool2'

subj_files_list = glob.glob(os.path.join(input_folder,f'*{contrast}.nii.gz'))
subj_list = [os.path.basename(subj_file)[:6] for subj_file in subj_files_list]

target_path = f'/Volumes/Data/Badea/Lab/mouse/ADDeccode_shortcut_pool/S02654_{contrast}.nii.gz'

print(subj_list)

for subj in subj_list:
    subj_mrtrix_path = os.path.join(input_folder,f'{subj}_{contrast}.nii.gz')
    subj_mrtrix_output_path = os.path.join(SAMBA_folder,f'{subj}_mrtrix{contrast}.nii.gz')
    temp_file = make_temppath(subj_mrtrix_path)
    if not os.path.exists(subj_mrtrix_output_path) or overwrite:
        img_transform_exec(subj_mrtrix_path, 'RPS', 'LAS', temp_file)
        header_superpose(target_path, temp_file, outpath=subj_mrtrix_output_path)
        os.remove(temp_file)


input_allsubj_folder = f'/Volumes/Data/Badea/Lab/human/AD_Decode/diffusion_prep_locale'
input_allsubj_folders = [subj_folder for subj_folder in glob.glob(os.path.join(input_allsubj_folder,'diffusion_prep_*/')) if np.size(glob.glob(os.path.join(subj_folder,f'*mrtrix{contrast}.nii.gz')))>0 ]
subj_list = [subj_folder.split('_')[-1][:6] for subj_folder in input_allsubj_folders]

target_path = f'/Volumes/Data/Badea/Lab/mouse/ADDeccode_shortcut_pool/S02654_{contrast}.nii.gz'

print(subj_list)

for subj in subj_list:
    subj_mrtrix_path = os.path.join(input_allsubj_folder,f'diffusion_prep_{subj}',f'{subj}_mrtrix{contrast}.nii.gz')
    subj_mrtrix_output_path = os.path.join(SAMBA_folder,f'{subj}_mrtrix{contrast}.nii.gz')
    temp_file = make_temppath(subj_mrtrix_path)
    if not os.path.exists(subj_mrtrix_output_path) or overwrite:
        img_transform_exec(subj_mrtrix_path, 'RPS', 'LAS', temp_file)
        header_superpose(target_path, temp_file, outpath=subj_mrtrix_output_path)
        os.remove(temp_file)