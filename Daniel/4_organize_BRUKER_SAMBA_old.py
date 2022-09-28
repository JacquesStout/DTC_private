import os, shutil
from DTC.file_manager.file_tools import mkcdir
from DTC.nifti_handlers.transform_handler import img_transform_exec, get_affine_transform, rigid_reg
import nibabel as nib
import numpy as np
from dipy.io.image import save_nifti
import pandas as pd

transforms_folder = '/Users/jas/jacques/Daniel_test/Transforms'

subjects = ['sub22040413', 'sub22040411', 'sub2204041', 'sub22040410', 'sub2204042', 'sub2204043', 'sub2204044',
            'sub2204045', 'sub2204046', 'sub2204047', 'sub2204048', 'sub2204049', 'sub2205091', 'sub22050910',
            'sub22050911', 'sub22050912', 'sub22050913', 'sub22050914', 'sub2205094', 'sub2205097', 'sub2205098',
            'sub2206061', 'sub22060610', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub2206062',
            'sub2206063', 'sub2206064', 'sub2206065', 'sub2206066', 'sub2206067', 'sub2206068', 'sub2206069']

"""
subjects = ['sub22040413', 'sub22040411', 'sub22040401', 'sub22040410', 'sub22040402', 'sub22040403', 'sub22040404',
            'sub22040405', 'sub22040406', 'sub22040407', 'sub22040408', 'sub22040409', 'sub22050901', 'sub22050910',
            'sub22050911', 'sub22050912', 'sub22050913', 'sub22050914', 'sub22050904', 'sub22050907', 'sub22050908',
            'sub22060601', 'sub22060610', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub22060602',
            'sub22060603', 'sub22060604', 'sub22060605', 'sub22060606', 'sub22060607', 'sub22060608', 'sub22060609']
"""

overwrite = False
verbose = True
recenter = False
remote = False
toclean = True

if not remote:
    sftp = None

copytype = "truecopy"
ext = '.nii.gz'
native_ref = ''

orig_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS/'
# new_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_SAMBAD/'
new_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_SAMBAD_2/'

nii_temp_dir = '/Users/jas/jacques/Daniel_test/temp_nii'
# nii_MDT = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_MDT'

# MDT_baseimages = '/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/preprocess/'
MDT_baseimages = '/Volumes/Data/Badea/Lab/mouse/VBM_18APOERAREset02_invivoAPOE1-work/preprocess'

mkcdir(nii_temp_dir)
mkcdir(new_dir)

ref_t1 = '/Volumes/Data/Badea/Lab/mouse/VBM_18APOERAREset02_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_test_i7/median_images/MDT_T1.nii.gz'

transf_level = 'torigid'
# transf_level = 'towarp'

csv_summary_path = '/Users/jas/jacques/Daniel_test/FMRI_mastersheet.xlsx'
csv_summary = pd.read_excel(csv_summary_path)

for subj in subjects:

    print(f'running move for subject {subj}')
    trans = os.path.join(transforms_folder, f"{subj}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(transforms_folder, f"{subj}_rigid.mat")
    affine_orig = os.path.join(transforms_folder, f"{subj}_affine.mat")
    runno_to_MDT = os.path.join(transforms_folder, f'{subj}_to_MDT_warp.nii.gz')
    MDT_to_subject = os.path.join(transforms_folder, f"MDT_to_{subj}_warp.nii.gz")

    subj_anat = os.path.join(orig_dir, subj.replace('b', 'b-'), 'ses-1', 'anat',
                             f'{subj.replace("b", "b-")}_ses-1_T1w{ext}')
    subj_func = os.path.join(orig_dir, subj.replace('b', 'b-'), 'ses-1', 'func',
                             f'{subj.replace("b", "b-")}_ses-1_bold{ext}')

    folder_ID = os.path.join(new_dir, subj.replace('b', 'b-'))
    folder_ses = os.path.join(folder_ID, 'ses-1')
    folder_anat = os.path.join(folder_ses, 'anat')
    folder_func = os.path.join(folder_ses, 'func')
    mkcdir([folder_ID, folder_ses, folder_anat, folder_func], sftp)
    subj_anat_sambad = os.path.join(folder_anat, f'{subj.replace("b", "b-")}_ses-1_T1w{ext}')
    subj_func_sambad = os.path.join(folder_func, f'{subj.replace("b", "b-")}_ses-1_bold.nii.gz')

    if not os.path.exists(subj_anat_sambad) or not os.path.exists(subj_func_sambad):
        func_reorient = os.path.join(nii_temp_dir, f'{subj}_func_reorient{ext}')
        func_reorient_affined = os.path.join(nii_temp_dir, f'{subj}_func_reorient_affined{ext}')
        target_base = os.path.join(MDT_baseimages, f'{subj}_T1_masked.nii.gz')
        SAMBA_preprocess = os.path.join(nii_temp_dir, f'{subj}_preprocess{ext}')

        print(f'Reorienting the file {subj_anat}')

        if not os.path.exists(func_reorient) or overwrite:
            img_transform_exec(subj_func, 'LPI', 'ALS', output_path=func_reorient, verbose=True)

        target_base = os.path.join(MDT_baseimages, f'{subj}_T1_masked.nii.gz')
        if not os.path.exists(SAMBA_preprocess) or overwrite:
            shutil.copy(target_base, SAMBA_preprocess)

        if not os.path.exists(func_reorient_affined) or overwrite:
            affine = nib.load(subj_anat).affine
            affine_recentered = nib.load(SAMBA_preprocess).affine

            transform = get_affine_transform(affine, affine_recentered)

            affine_func = nib.load(func_reorient).affine
            affine_func_new = np.eye(4)
            affine_func_new[:3, :3] = np.dot(affine_func[:3, :3], transform[:3, :3])
            affine_func_new[:3, 3] = [0, 0, 0]
            affine_func_new[1, :] = - affine_func_new[1, :]
            affine_func_new[:3, 3] = affine_func[:3, 3] - transform[:3, 3]

            func_reorient_affined_nii = nib.Nifti1Image(nib.load(func_reorient).get_data(), affine_func_new)
            nib.save(func_reorient_affined_nii, func_reorient_affined)
            print(f'Saved {func_reorient_affined}')

        target = nib.load(SAMBA_preprocess)
        target_data = target.get_fdata()
        target_grid2world = target.affine

        moving = nib.load(func_reorient_affined)
        moving_data = moving.get_fdata()
        moving_grid2world = moving.affine

        from dipy.align.imaffine import transform_centers_of_mass

        target = nib.load(SAMBA_preprocess)
        target_data = target.get_fdata()
        target_grid2world = target.affine

        moving = nib.load(func_reorient_affined)
        moving_data = moving.get_fdata()
        moving_grid2world = moving.affine

        """
        func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_masscenter{ext}')
        if not os.path.exists(func_reorient_recentered) or overwrite:
            c_of_mass = transform_centers_of_mass(target_data, target_grid2world,
                                                  moving_data[:, :, :, 0], moving_grid2world)
            transformed = c_of_mass.transform(moving_data[:, :, :, 0])
            new_nii = nib.Nifti1Image(transformed, target_grid2world)
            nib.save(new_nii, func_reorient_recentered)
            print(f'Saved {func_reorient_recentered}')

        func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_trans{ext}')
        if not os.path.exists(func_reorient_recentered) or overwrite:
            moved, trans_affine = trans_reg(target_data, target_grid2world,
                                            moving_data[..., 0], moving_grid2world)
            save_nifti(func_reorient_recentered, moved, target_grid2world)
            print(f'Saved {func_reorient_recentered}')

        """

        func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_rigid{ext}')
        if not os.path.exists(func_reorient_recentered) or overwrite:
            moved, trans_affine = rigid_reg(target_data, target_grid2world,
                                            moving_data[..., 0], moving_grid2world)
            save_nifti(func_reorient_recentered, moved, target_grid2world)
            print(f'Saved {func_reorient_recentered}')

        if transf_level == 'torigid':
            if not os.path.exists(subj_anat_sambad) or overwrite:
                cmd = f"antsApplyTransforms -v 1 -d 3 -i {SAMBA_preprocess} -o {subj_anat_sambad} -r {SAMBA_preprocess} -n MultiLabel -t [{rigid},0] [{trans},0]"
                os.system(cmd)

            if not os.path.exists(subj_func_sambad) or overwrite:
                cmd = f"antsApplyTransforms -v 1 -d 3 -i {func_reorient_recentered} -o {subj_func_sambad} -r {SAMBA_preprocess} -n MultiLabel -t [{rigid},0] [{trans},0]"
                os.system(cmd)

        if transf_level == 'tofullwarp':
            if not os.path.exists(subj_anat_sambad) or overwrite:
                cmd = f"antsApplyTransforms -v 1 -d 3 -i {SAMBA_preprocess} -o {subj_anat_sambad} -r {SAMBA_preprocess} -n MultiLabel -t {runno_to_MDT} [{affine_orig},0] [{rigid},0] [{trans},0]"
                os.system(cmd)
            if not os.path.exists(subj_func_sambad) or overwrite:
                cmd = f"antsApplyTransforms -v 1 -d 3 -i {func_reorient_recentered} -o {subj_func_sambad} -r {SAMBA_preprocess} -n MultiLabel -t {runno_to_MDT} [{affine_orig},0] [{rigid},0] [{trans},0]"
                os.system(cmd)

        if toclean:
            tempfiles = [func_reorient, func_reorient_affined, SAMBA_preprocess, func_reorient_recentered]
            for tmpfile in tempfiles:
                os.remove(tmpfile)
    else:
        print(f'already wrote {subj_anat_sambad} and {subj_func_sambad}')