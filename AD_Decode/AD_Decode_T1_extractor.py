import os, glob, shutil

import nibabel as nib

#
from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile, glob_remote
import subprocess

#folders = glob.glob('/Users/jas/jacques/CS_Project/CS_Data_all/Bruker_data/20230419_165630_211001_21_1_DEV_18abb11_DEV_1_1/12*')

data_path = '/Volumes/Data/Badea/ADdecode.01/Data/Anat/'

outpath = '/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/'

subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'*/'))]


fid_size_lim_mb = 0
dir_min_lim = 1

verbose = False

read_all_methods = True

#info_type => 'Show_niipath' or 'Show_methodinfo'
info_type = 'Show_methodinfo'

method_tofind = '3D Ax T1 MPRAGE'

list_toview = []
list_notfound = []

for subject in subjects:
    subject_path = os.path.join(data_path, subject)

    method_paths = glob.glob(os.path.join(subject_path,'*.bxh'))
    method_paths.sort()

    found = 0
    scanno_num = ''
    t1_path = ''

    for methodpath in method_paths:
        scanno = os.path.basename(methodpath)
        nifti_path = methodpath.replace('.bxh','.nii.gz')
        with open(methodpath, 'r') as file:
            for line in file:
                if line.startswith('    <description>'):
                    method_name = line.split('>')[1].split('<')[0].strip()


        if method_name == method_tofind:
            #print(f'Found T1 for subject {subject}')
            found = 1
            scanno_num = scanno
            t1_path = nifti_path
            break

    subj = f'S{subject[-5:]}'
    if found==1:
        subj = f'S{subject[-5:]}'
        t1_outpath = os.path.join(outpath,f'{subj}_T1.nii.gz')
        list_toview.append(subj)
        if not os.path.exists(t1_outpath):
            shutil.copy(t1_path, t1_outpath)
            print(f'Copied the T1 for subject {subj}')
        else:
            print(f'Already copied the T1 for subject {subj}')
    if found==0:
        print(f'Did not find T1 for subject {subj}')
        list_notfound.append(subj)

#print(','.join(list_toview))
print(list_notfound)