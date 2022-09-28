import os, shutil
from DTC.file_manager.file_tools import mkcdir
from DTC.nifti_handlers.transform_handler import affine_superpose
import numpy as np
from DTC.file_manager.computer_nav import ants_loadmat
import nibabel as nib
import copy
from DTC.file_manager.computer_nav import save_nifti_remote

def angle_to_matrix(angle, axis, degrees=True):
    if degrees:
        angle = angle * (np.pi/180)
    matrix = np.eye(3)
    if axis == 0:
        matrix[1,1] = np.cos(angle)
        matrix[1,2] = - np.sin(angle)
        matrix[2,1] = np.sin(angle)
        matrix[2,2] = np.cos(angle)
    if axis == 1:
        matrix[0,0] = np.cos(angle)
        matrix[0,2] = np.sin(angle)
        matrix[2,0] = - np.sin(angle)
        matrix[2,2] = np.cos(angle)
    if axis == 2:
        matrix[0,0] = np.cos(angle)
        matrix[0,1] = - np.sin(angle)
        matrix[1,0] = np.sin(angle)
        matrix[1,1] = np.cos(angle)
    return(matrix)


transforms_folder = '/Users/jas/jacques/Daniel_test/Transforms'

subjects = ['sub22040413', 'sub22040411', 'sub2204041', 'sub22040410', 'sub2204042', 'sub2204043', 'sub2204044',
            'sub2204045', 'sub2204046', 'sub2204047', 'sub2204048', 'sub2204049', 'sub2205091', 'sub22050910',
            'sub22050911', 'sub22050912', 'sub22050913', 'sub22050914', 'sub2205094', 'sub2205097', 'sub2205098',
            'sub2206061', 'sub22060610', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub2206062',
            'sub2206063', 'sub2206064', 'sub2206065', 'sub2206066', 'sub2206067', 'sub2206068', 'sub2206069']
subjects = ['sub2204048']
overwrite = True
verbose = True
remote = False
recenter = False
copytype = "truecopy"
ext = '.nii.gz'
native_ref = ''

orig_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS/'
nii_temp_dir = '/Users/jas/jacques/Daniel_test/temp_nii'
nii_MDT = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_MDT'

MDT_baseimages = '/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/preprocess/'

mkcdir(nii_temp_dir)
mkcdir(nii_MDT)

ref_t1 = '/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_test_i7/median_images/MDT_T1.nii.gz'

subj_rot = {'sub2204048': [180,0,90]}

for subj in subjects:

    print(f'running move for subject {subj}')
    trans = os.path.join(transforms_folder, f"{subj}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(transforms_folder, f"{subj}_rigid.mat")
    affine_orig = os.path.join(transforms_folder, f"{subj}_affine.mat")
    runno_to_MDT = os.path.join(transforms_folder, f'{subj}_to_MDT_warp.nii.gz')
    MDT_to_subject = os.path.join(transforms_folder, f"MDT_to_{subj}_warp.nii.gz")

    subj_dwi = os.path.join(orig_dir, subj.replace('b', 'b-'), 'ses-1', 'anat',
                            f'{subj.replace("b", "b-")}_ses-1_T1w{ext}')
    subj_func = os.path.join(orig_dir, subj.replace('b', 'b-'), 'ses-1', 'func',
                             f'{subj.replace("b", "b-")}_ses-1_bold{ext}')

    SAMBA_init = subj_dwi
    SAMBA_reorient = os.path.join(nii_temp_dir, f'{subj}_reorient_rotmat{ext}')
    func_reorient = os.path.join(nii_temp_dir, f'{subj}_func_reorient{ext}')

    #'LPS', 'ALS'

    if not os.path.exists(SAMBA_reorient) or overwrite:
        oldnifti = nib.load(subj_dwi)
        newnifti = copy.deepcopy(oldnifti)
        old_affine = copy.deepcopy(oldnifti.affine)
        new_affine = newnifti.affine

        matrix = np.zeros([4,4,3])
        for i in np.arange(3):
            matrix[:,:,i] = np.eye(4)
            matrix[:3,:3,i] = angle_to_matrix(subj_rot[subj][i], i)

        matrix_transform = np.matmul(np.matmul(matrix[:,:,0],matrix[:,:,1]),matrix[:,:,2])

        new_affine[:3,3] = new_affine[:3,3] + matrix_transform[:3,3]
        new_affine[:3,:3] = np.matmul(matrix_transform[:3,:3],new_affine[:3,:3])
        save_nifti_remote(newnifti,SAMBA_reorient,sftp=None)
        print(f'Saved {SAMBA_reorient}')


    target_base = os.path.join(MDT_baseimages, f'{subj}_T1_masked.nii.gz')
    SAMBA_preprocess = os.path.join(nii_temp_dir, f'{subj}_preprocess{ext}')
    overwrite = True
    if not os.path.exists(SAMBA_preprocess) or overwrite:
        shutil.copy(target_base, SAMBA_preprocess)

    SAMBA_preprocess_test_posttrans = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans{ext}')
    SAMBA_preprocess_test_posttrans_2 = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans_2{ext}')
    SAMBA_preprocess_test_posttrans_3 = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans_3{ext}')

    SAMBA_preprocess_test_rigid = os.path.join(nii_temp_dir, f'{subj}_postrigid{ext}')
    SAMBA_preprocess_test_rigid_affine = os.path.join(nii_temp_dir, f'{subj}_postrigid_affine{ext}')
    SAMBA_preprocess_test_postwarp = os.path.join(nii_MDT, f'{subj}_postwarp{ext}')
    if native_ref == '':
        native_ref = SAMBA_preprocess
    if not os.path.exists(SAMBA_preprocess_test_postwarp) or overwrite:

        print(ants_loadmat(trans))
        print(ants_loadmat(rigid))
        print(ants_loadmat(affine_orig))

        reverse_test = False
        if reverse_test:
            reg_subj = os.path.join(
                '/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_test_i7/MDT_images',
                f'{subj}_T1_to_MDT.nii.gz')
            subj_T1_reversetest = os.path.join(nii_temp_dir, f'{subj}_reversed_subj{ext}')
            cmd = f"antsApplyTransforms -v 1 -d 3 -i {reg_subj} -o {subj_T1_reversetest} -r {target_base} -n MultiLabel -t [{trans},1] [{rigid},1] [{affine_orig},1] {MDT_to_subject}"
            os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 -d 3  -i {SAMBA_preprocess} -r {SAMBA_preprocess}  -n Linear  -o {SAMBA_preprocess_test_posttrans}'
        os.system(cmd)
        # shutil.copy(SAMBA_preprocess, SAMBA_preprocess_test_posttrans)
        affine_superpose(SAMBA_preprocess, SAMBA_preprocess_test_posttrans, outpath=SAMBA_preprocess_test_posttrans_2)
        # shutil.copy(SAMBA_preprocess_test_posttrans, SAMBA_preprocess_test_posttrans_2)

        cmd = f'antsApplyTransforms -v 1 -d 3  -i {SAMBA_preprocess_test_posttrans_2} -r {ref_t1}  -n Linear  -o {SAMBA_preprocess_test_posttrans_3} -t {trans}'
        os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_posttrans_3} -o {SAMBA_preprocess_test_rigid} ' \
            f'-r {ref_t1} -n Linear -t [{rigid},0]'
        os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_rigid} -o {SAMBA_preprocess_test_rigid_affine} ' \
            f'-r {SAMBA_preprocess_test_posttrans_2} -n Linear -t [{affine_orig},0]'
        os.system(cmd)

        # shutil.copy(SAMBA_preprocess_test_rigid, SAMBA_preprocess_test_rigid_affine)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_rigid_affine} -o {SAMBA_preprocess_test_postwarp} ' \
            f'-r {ref_t1} -n Linear -t {runno_to_MDT}'
        os.system(cmd)
