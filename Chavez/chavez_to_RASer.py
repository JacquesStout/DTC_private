from nifti_handlers.transform_handler import img_transform_exec, get_flip_bvecs
from file_manager.file_tools import mkcdir
import os, socket, glob

subjects_list = ['C_20220124_004', 'C_20220124_005', 'C_20220124_006', 'C_20220124_007']

computer_name = socket.gethostname()

if 'santorini' in computer_name:
    DWI_folder = "/Volumes/Data/Badea/Lab/RaulChavezValdez/"
    output_folder = "/Volumes/Data/Badea/Lab/RaulChavezValdez_RAS/"

if 'samos' in computer_name:
    DWI_folder = '/mnt/paros_MRI/jacques/APOE/DWI_allsubj/'
    output_folder = '/mnt/paros_MRI/jacques/APOE/DWI_allsubj_RAS/'

mkcdir(output_folder)
overwrite = True
flip_bvecs = True
current_vorder = 'ASR'
desired_vorder = 'RAS'

for subject in subjects_list:
    diff_file = os.path.join(DWI_folder,subject, f'{subject}_nii4D.nii.gz')
    mkcdir(os.path.join(output_folder,subject))
    outputRAS_file = os.path.join(output_folder,subject, f'{subject}_nii4D.nii.gz')
    transferred=0
    if not os.path.exists(outputRAS_file) or overwrite:
        if flip_bvecs:
            fbvecs = os.path.join(DWI_folder, subject, f'{subject}_bvecs.txt')
            outputbvecs_file = os.path.join(output_folder, subject, f'{subject}_bvecs.txt')
            get_flip_bvecs(fbvecs, current_vorder, desired_vorder, outputbvecs_file,writeformat='dsi')
        img_transform_exec(diff_file,'ASR','RAS',outputRAS_file)
        transferred=1
    if transferred:
        print(f'transferred subject {subject}')
    else:
        print(f'already transferred subject {subject}')
