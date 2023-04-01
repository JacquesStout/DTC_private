from DTC.nifti_handlers.transform_handler import img_transform_exec
from DTC.file_manager.file_tools import mkcdir
import os, socket, glob, shutil
import warnings
import numpy as np
import nibabel as nib
import sys
from DTC.file_manager.argument_tools import parse_arguments

"""
subjects_list = ["N58214", "N58215",
     "N58216", "N58217", "N58218", "N58219", "N58221", "N58222", "N58223", "N58224",
                "N58225", "N58226", "N58228",
                "N58229", "N58230", "N58231", "N58232", "N58633", "N58634", "N58635", "N58636", "N58649", "N58650",
                "N58651", "N58653", "N58654",
                'N58408', 'N58714', 'N58740', 'N58477', 'N58734', 'N58309', 'N58792', 'N58302',
                'N58784', 'N58706', 'N58361', 'N58355', 'N58712', 'N58790', 'N58606', 'N58350', 'N58608',
                'N58779', 'N58500', 'N58604', 'N58749', 'N58510', 'N58394', 'N58346', 'N58344', 'N58788', 'N58305',
                'N58514', 'N58794', 'N58733', 'N58655', 'N58735', 'N58310', 'N58400', 'N58708', 'N58780', 'N58512',
                'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58745', 'N58406', 'N58359', 'N58742', 'N58396',
                'N58613', 'N58732', 'N58516', 'N58402']
"""
#subjects_list = ["N58831","N59022","N59026","N59033","N59035","N59039","N59041","N59065","N59066","N59072","N59076","N59078","N59080","N59097","N59099","N59109","N59116","N59118","N59120"]
#subjects_list = ['N60188', 'N60190', 'N60192', 'N60194', 'N60198', 'N60219', 'N60221', 'N60223', 'N60225', 'N60229', 'N60231']
#removed 'N58398' could not find
computer_name = socket.gethostname()

if 'santorini' in computer_name:
    #DWI_folder = '/Volumes/Data/Badea/Lab/APOE/DWI_allsubj'
    DWI_folder = '/Volumes/dusom_abadea_nas1/munin_js/DWI_allsubj'
    labels_folder = '/Volumes/Data/Badea/Lab/APOE/DWI_allsubj'
    output_folder = '/Volumes/Data/Badea/Lab/APOE/DWI_allsubj_RAS'
    output_folder = '/Volumes/Data/Badea/Lab/APOE/oldFA_RAS'

if 'samos' in computer_name:
    DWI_folder = '/mnt/paros_MRI/jacques/APOE/DWI_allsubj/'
    labels_folder = '/mnt/paros_MRI/jacques/APOE/DWI_allsubj/'
    output_folder = '/mnt/paros_MRI/jacques/APOE/DWI_allsubj_RAS/'

#subjects_all = glob.glob(os.path.join(DWI_folder,'*coreg.nii.gz'))
subjects_all = glob.glob(os.path.join(DWI_folder,'*subjspace_fa.nii.gz'))
subjects_list = []
for subject in subjects_all:
    subject_name = os.path.basename(subject)
    subjects_list.append(subject_name[:6])
subjects_list.sort()
subjects_list = subjects_list[:]

#subjects_list = ['N60188', 'N60190', 'N60192', 'N60194', 'N60198', 'N60219', 'N60221', 'N60223', 'N60225', 'N60229', 'N60231']

removed_list = ['N57504']
for remove in removed_list:
    if remove in subjects_list:
        subjects_list.remove(remove)

mkcdir(output_folder)

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects_list)

subjects_list = subjects_list[firstsubj: lastsubj]

print(subjects_list)

for subject in subjects_list:
    print(f'Running subject {subject}')
    fa_file = os.path.join(DWI_folder,f'{subject}_subjspace_fa.nii.gz')
    fa_RAS_file = os.path.join(output_folder,f'{subject}_fa_RAS.nii.gz')

    if not os.path.exists(fa_RAS_file):
        if os.path.exists(fa_file):
            try:
                img_transform_exec(fa_file,'ARI','RAS',fa_RAS_file)
            except nib.filebasedimages.ImageFileError:
                print(f'Wrong file that is unreadable, erasing {fa_file}')
                os.remove(fa_file)
            transferred=1

"""
for subject in subjects_list:
    print(f'Running subject {subject}')
    coreg_file = os.path.join(DWI_folder,f'{subject}_subjspace_coreg.nii.gz')
    dwi_file = os.path.join(DWI_folder,f'{subject}_subjspace_dwi.nii.gz')
    labels_file = os.path.join(labels_folder,f'{subject}_labels.nii.gz')
    labels_lr_file = os.path.join(labels_folder,f'{subject}_labels_lr_ordered.nii.gz')
    mask_file = os.path.join(DWI_folder, f'{subject}_subjspace_mask.nii.gz')
    coreg_RAS_file = os.path.join(output_folder,f'{subject}_coreg_RAS.nii.gz')
    dwi_RAS_file = os.path.join(output_folder,f'{subject}_dwi_RAS.nii.gz')
    labels_RAS_file = os.path.join(output_folder,f'{subject}_labels_RAS.nii.gz')
    labels_RAS_lr_file = os.path.join(output_folder,f'{subject}_labels_lr_ordered_RAS.nii.gz')
    mask_RAS_file = os.path.join(output_folder, f'{subject}_RAS_mask.nii.gz')
    txt_files = glob.glob(os.path.join(DWI_folder,f'{subject}*txt'))
    transferred=0
    if np.size(txt_files)>0:
        for txt_file in txt_files:
            shutil.copy(txt_file, output_folder)
        transferred=1
    if not os.path.exists(coreg_RAS_file):
        if os.path.exists(coreg_file):
            try:
                img_transform_exec(coreg_file,'ARI','RAS',coreg_RAS_file)
            except nib.filebasedimages.ImageFileError:
                print(f'Wrong file that is unreadable, erasing {coreg_file}')
                os.remove(coreg_file)
            transferred=1
    if not os.path.exists(labels_RAS_file):
        if not os.path.exists(labels_file):
            txt = f'Label file {labels_file} not found, skip'
            warnings.warn(txt)
        else:
            img_transform_exec(labels_file,'ARI','RAS',labels_RAS_file)
        transferred=1
    if not os.path.exists(labels_RAS_lr_file): 
        if os.path.exists(labels_lr_file):
            img_transform_exec(labels_lr_file,'ARI','RAS',labels_RAS_lr_file)
        transferred=1
    if not os.path.exists(dwi_RAS_file):
        if os.path.exists(dwi_file):
            try:
                img_transform_exec(dwi_file,'ARI','RAS',dwi_RAS_file)
            except nib.filebasedimages.ImageFileError:
                print(f'Wrong file that is unreadable, erasing {dwi_file}')
                os.remove(dwi_file)
        else:
            print(f'cannot find file path {dwi_file}')
        transferred=1
    if not os.path.exists(mask_RAS_file):
        if not os.path.exists(mask_file):
            txt = f'Mask file {mask_file} not found, skip'
            warnings.warn(txt)
        else:
            img_transform_exec(mask_file,'ARI','RAS',mask_RAS_file)
        transferred=1
    if transferred:
        print(f'transferred subject {subject}')
    else:
        print(f'already transferred subject {subject}')
"""