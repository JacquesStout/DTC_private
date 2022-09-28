import os
from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile
from DTC.nifti_handlers.transform_handler import recenter_nii_save_test, affine_superpose, img_transform_exec, recenter_to_eye, get_affine_transform, get_flip_affine
import nibabel as nib
import numpy as np
from dipy.viz import regtools

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
import copy

transforms_folder = '/Users/jas/jacques/Daniel_test/Transforms'

subjects = ['sub22040413','sub22040411','sub2204041','sub22040410','sub2204042','sub2204043','sub2204044','sub2204045','sub2204046','sub2204047','sub2204048','sub2204049','sub2205091','sub22050910','sub22050911','sub22050912','sub22050913','sub22050914','sub2205094','sub2205097','sub2205098','sub2206061','sub22060610','sub22060611','sub22060612','sub22060613','sub22060614','sub2206062','sub2206063','sub2206064','sub2206065','sub2206066','sub2206067','sub2206068','sub2206069']
subjects = ['sub2204041']
overwrite= True
verbose=True
remote=False
recenter=False
copytype = "truecopy"
ext='.nii.gz'
native_ref=''

orig_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_masked/'
nii_temp_dir = '/Users/jas/jacques/Daniel_test/temp_nii'
nii_MDT = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_MDT'

MDT_baseimages = '/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/preprocess/base_images/'

mkcdir(nii_temp_dir)
mkcdir(nii_MDT)


for subj in subjects:

    print(f'running move for subject {subj}')
    trans = os.path.join(transforms_folder, f"{subj}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(transforms_folder, f"{subj}_rigid.mat")
    affine_orig = os.path.join(transforms_folder, f"{subj}_affine.mat")
    runno_to_MDT = os.path.join(transforms_folder, f'{subj}_to_MDT_warp.nii.gz')
    subj_dwi = os.path.join(orig_dir, subj.replace('b','b-'),'ses-1','anat',f'{subj.replace("b","b-")}_ses-1_T1w{ext}')
    subj_func = os.path.join(orig_dir, subj.replace('b','b-'),'ses-1','func',f'{subj.replace("b","b-")}_ses-1_bold{ext}')

    SAMBA_init = subj_dwi
    SAMBA_reorient = os.path.join(nii_temp_dir, f'{subj}_reorient{ext}')
    func_reorient = os.path.join(nii_temp_dir, f'{subj}_func_reorient{ext}')

    SAMBA_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_reorient_recentered{ext}')
    func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_recentered{ext}')

    SAMBA_preprocess = os.path.join(nii_temp_dir, f'{subj}_preprocess{ext}')

    #SAMBA_preprocess_2 = os.path.join(DWI_save, f'{subj}_{contrast}_preprocess_2{ext}')
    #SAMBA_preprocess_recentered_1 = os.path.join(DWI_save, f'{subj}_{contrast}_masked_recenter_1{ext}')
    #SAMBA_preprocess_recentered_2 = os.path.join(DWI_save, f'{subj}_{contrast}_masked_recenter_2{ext}')

    print(f'Reorienting the file {subj_dwi}')
    #img_transform_exec(subj_dwi, 'LPS', 'ALS', output_path=SAMBA_reorient, recenter_eye=False)


    ######### affine flipping
    """
    from DTC.file_manager.computer_nav import save_nifti_remote

    affine_niipath = os.path.join(nii_temp_dir,f'{subj}_affine_reorient{ext}')

    subj_nii = nib.load(subj_dwi)
    subj_affine = subj_nii.affine
    flip_affine = get_flip_affine('LPS', 'ALS')

    subj_newnii = copy.deepcopy(subj_nii)
    new_affine = subj_newnii.affine
    new_affine[:3, :3] = np.matmul(flip_affine[:3,:3], subj_affine[:3,:3])
    save_nifti_remote(subj_newnii, affine_niipath, sftp=None)
    """
    #######




    img_transform_exec(subj_dwi, 'LPS', 'ALS', output_path=SAMBA_reorient, verbose=True)
    img_transform_exec(subj_func, 'LPS', 'ALS', output_path=func_reorient, verbose=True)


    recenter_to_eye(SAMBA_reorient, SAMBA_reorient_recentered,False)
    recenter_to_eye(func_reorient, func_reorient_recentered,False)

    affine = nib.load(SAMBA_reorient).affine
    affine_recentered = nib.load(SAMBA_reorient_recentered).affine
    
    transform=get_affine_transform(affine,affine_recentered)
    
    
    #test_new=np.eye(4)
    #test_new[:3,:3] = np.dot(affine[:3,:3],transform[:3,:3])
    #test_new[:3,3] = [0,0,0]
    #test_new[:3,3] = affine[:3,3]-transform[:3,3]
    
    affine_func = nib.load(func_reorient).affine
    affine_func_new = np.eye(4)
    affine_func_new[:3,:3] = np.dot(affine_func[:3,:3],transform[:3,:3])
    affine_func_new[:3,3] = [0,0,0]
    affine_func_new[:3,3] = affine_func[:3,3]-transform[:3,3]
    
    func_reorient_recentered_nii = nib.Nifti1Image(nib.load(func_reorient).get_data(), affine_func_new, nib.load(func_reorient).header)
    #nib.save(func_reorient_recentered_nii, func_reorient_recentered)
    
    ###apply same to func affine!


    from dipy.align.imaffine import transform_centers_of_mass
    target_base = os.path.join(MDT_baseimages,f'{subj}_T1_masked.nii.gz')
    target = nib.load(target_base)
    target_data = target.get_fdata()
    target_grid2world = target.affine

    moving = nib.load(SAMBA_reorient)
    moving_data = moving.get_fdata()
    moving_grid2world = moving.affine

    c_of_mass = transform_centers_of_mass(target_data, target_grid2world,
                                          moving_data, moving_grid2world)

    regtools.overlay_slices(target_data, moving_data, None, 0,
                            "Static", "Moving")
    transformed = c_of_mass.transform(moving_data)
    regtools.overlay_slices(target_data, transformed, None, 0,
                            "Static", "Transformed")
    new_nii=nib.Nifti1Image(transformed, moving_grid2world, moving.header)
    nib.save(new_nii, SAMBA_preprocess)


    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(target_data, moving_data, transform, params0,
                                  target_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)
    transformed = translation.transform(moving_data)
    #regtools.overlay_slices(target_data, transformed, None, 0,
    #                        "Static", "Transformed")
    new_nii=nib.Nifti1Image(transformed, moving_grid2world, moving.header)
    nib.save(new_nii, SAMBA_preprocess)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(target_data, moving_data, transform, params0,
                            target_grid2world, moving_grid2world,
                            starting_affine=starting_affine)
    transformed = rigid.transform(moving_data)
    regtools.overlay_slices(target_data, transformed, None, 0,
                            "Static", "Transformed")
    new_nii=nib.Nifti1Image(transformed, moving_grid2world, moving.header)
    nib.save(new_nii, SAMBA_preprocess)

    from dipy.viz import regtools

    print(f'Reoriented file is {SAMBA_preprocess}')




    if recenter and (not os.path.exists(SAMBA_preprocess) or overwrite):
        """
        header_superpose(SAMBA_preprocess_ref, SAMBA_init, outpath=SAMBA_preprocess, verbose=False)
        img_transform_exec(SAMBA_preprocess, 'RAS', 'LPS', output_path=SAMBA_preprocess_2)
        recenter_nii_save(SAMBA_preprocess_2, SAMBA_preprocess_recentered_1, verbose=True)
        recenter_nii_save(SAMBA_preprocess_recentered_1,SAMBA_preprocess_2)
        SAMBA_init = SAMBA_preprocess_2
        """
        recenter_nii_save_test(SAMBA_reorient, SAMBA_preprocess)
        SAMBA_init = SAMBA_preprocess

    SAMBA_preprocess_test_posttrans = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans{ext}')
    SAMBA_preprocess_test_posttrans_2 = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans_2{ext}')
    SAMBA_preprocess_test_posttrans_3 = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans_3{ext}')

    SAMBA_preprocess_test_rigid = os.path.join(nii_temp_dir, f'{subj}_postrigid{ext}')
    SAMBA_preprocess_test_rigid_affine = os.path.join(nii_temp_dir, f'{subj}_postrigid_affine{ext}')
    SAMBA_preprocess_test_postwarp = os.path.join(nii_MDT, f'{subj}_postwarp{ext}')
    if native_ref == '':
        native_ref = SAMBA_init
    if not os.path.exists(SAMBA_preprocess_test_postwarp) or overwrite:
        cmd = f'antsApplyTransforms -v 1 -d 3  -i {SAMBA_init} -r {SAMBA_init}  -n Linear  -o {SAMBA_preprocess_test_posttrans}'
        os.system(cmd)

        affine_superpose(SAMBA_init, SAMBA_preprocess_test_posttrans, outpath=SAMBA_preprocess_test_posttrans_2)

        cmd = f'antsApplyTransforms -v 1 -d 3  -i {SAMBA_preprocess_test_posttrans_2} -r {SAMBA_preprocess_test_posttrans_2}  -n Linear  -o {SAMBA_preprocess_test_posttrans_3} -t {trans}'
        os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_posttrans_3} -o {SAMBA_preprocess_test_rigid} ' \
            f'-r {SAMBA_preprocess_test_posttrans_2} -n Linear -t [{rigid},0]'
        os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_rigid} -o {SAMBA_preprocess_test_rigid_affine} ' \
            f'-r {SAMBA_preprocess_test_posttrans_2} -n Linear -t [{affine_orig},0]'
        os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_rigid_affine} -o {SAMBA_preprocess_test_postwarp} ' \
            f'-r {SAMBA_preprocess_test_posttrans_2} -n Linear -t {runno_to_MDT}'
        os.system(cmd)
