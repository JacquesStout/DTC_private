import os, shutil
from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile
from DTC.nifti_handlers.transform_handler import recenter_nii_save_test, affine_superpose, img_transform_exec, \
    recenter_to_eye, get_affine_transform, get_flip_affine, affine_superpose
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
from DTC.file_manager.computer_nav import true_loadmat, ants_loadmat


def affine_reg(static, static_grid2world,
               moving, moving_grid2world):


    #https://gist.github.com/Garyfallidis/42dd1ab04371272050221275c6ab9bd6

    c_of_mass = transform_centers_of_mass(static,
                                          static_grid2world,
                                          moving,
                                          moving_grid2world)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [500, 100, 10]

    sigmas = [3.0, 1.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)
    transformed = rigid.transform(moving)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    transformed = affine.transform(moving)

    return transformed, affine.affine


def trans_reg(static, static_grid2world,
               moving, moving_grid2world):

    c_of_mass = transform_centers_of_mass(static,
                                          static_grid2world,
                                          moving,
                                          moving_grid2world)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [500, 100, 10]

    sigmas = [3.0, 1.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)

    return transformed, translation.affine


def rigid_reg(static, static_grid2world,
               moving, moving_grid2world):

    c_of_mass = transform_centers_of_mass(static,
                                          static_grid2world,
                                          moving,
                                          moving_grid2world)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [500, 100, 10]

    sigmas = [3.0, 1.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)
    transformed = rigid.transform(moving)

    return transformed, rigid.affine


transforms_folder = '/Users/jas/jacques/Daniel_test/Transforms'

subjects = ['sub22040413', 'sub22040411', 'sub2204041', 'sub22040410', 'sub2204042', 'sub2204043', 'sub2204044',
            'sub2204045', 'sub2204046', 'sub2204047', 'sub2204048', 'sub2204049', 'sub2205091', 'sub22050910',
            'sub22050911', 'sub22050912', 'sub22050913', 'sub22050914', 'sub2205094', 'sub2205097', 'sub2205098',
            'sub2206061', 'sub22060610', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub2206062',
            'sub2206063', 'sub2206064', 'sub2206065', 'sub2206066', 'sub2206067', 'sub2206068', 'sub2206069']
subjects = ['sub2204048']
overwrite = False
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
    SAMBA_reorient = os.path.join(nii_temp_dir, f'{subj}_reorient{ext}')
    func_reorient = os.path.join(nii_temp_dir, f'{subj}_func_reorient{ext}')



    SAMBA_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_reorient_recentered{ext}')
    func_reorient_affined = os.path.join(nii_temp_dir, f'{subj}_func_reorient_affined{ext}')

    SAMBA_preprocess = os.path.join(nii_temp_dir, f'{subj}_preprocess{ext}')

    print(f'Reorienting the file {subj_dwi}')


    if not os.path.exists(SAMBA_reorient) or overwrite:
        img_transform_exec(subj_dwi, 'LPS', 'ALS', output_path=SAMBA_reorient, verbose=True)
    if not os.path.exists(func_reorient) or overwrite:
        img_transform_exec(subj_func, 'LPI', 'ALS', output_path=func_reorient, verbose=True)

    target_base = os.path.join(MDT_baseimages,f'{subj}_T1_masked.nii.gz')

    if not os.path.exists(SAMBA_reorient_recentered) or overwrite:
        affine_superpose(target_base, SAMBA_reorient, SAMBA_reorient_recentered, verbose=True)

    #

    if not os.path.exists(func_reorient_affined) or overwrite:
        affine = nib.load(SAMBA_reorient).affine
        affine_recentered = nib.load(target_base).affine

        transform = get_affine_transform(affine, affine_recentered)

        affine_func = nib.load(func_reorient).affine
        affine_func_new = np.eye(4)
        affine_func_new[:3, :3] = np.dot(affine_func[:3, :3], transform[:3, :3])
        affine_func_new[:3, 3] = [0, 0, 0]
        #affine_func_new[2,:] = affine_func_new[2,:]
        affine_func_new[1,:] = - affine_func_new[1,:]
        affine_func_new[:3, 3] = affine_func[:3, 3] - transform[:3, 3]

        #func_reorient_affined_nii = nib.Nifti1Image(nib.load(func_reorient).get_data(), affine_func_new, nib.load(func_reorient).header)
        func_reorient_affined_nii = nib.Nifti1Image(nib.load(func_reorient).get_data(), affine_func_new)
        nib.save(func_reorient_affined_nii, func_reorient_affined)
        print(f'Saved {func_reorient_affined}')


    target_base = os.path.join(MDT_baseimages,f'{subj}_T1_masked.nii.gz')
    SAMBA_preprocess = os.path.join(nii_temp_dir, f'{subj}_preprocess{ext}')
    if not os.path.exists(SAMBA_preprocess) or overwrite:
        shutil.copy(target_base,SAMBA_preprocess)

    target = nib.load(SAMBA_preprocess)
    target_data = target.get_fdata()
    target_grid2world = target.affine

    moving = nib.load(func_reorient_affined)
    moving_data = moving.get_fdata()
    moving_grid2world = moving.affine

    """
    moved, trans_affine = affine_reg(target_data, target_grid2world,
                                     moving_data[...,0], moving_grid2world)

    from dipy.io.image import save_nifti

    func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_targetgrid{ext}')
    save_nifti(func_reorient_recentered, moved, target_grid2world)

    func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_movinggrid{ext}')
    save_nifti(func_reorient_recentered, moved, moving_grid2world)
    """

    """
    for i in range(data.shape[-1]):
        print('Volume index %d' % (i,))
        if not gtab.b0s_mask[i]:
    
            print('Affine registration started')
            moving = data[..., i]
            moved, trans_affine = affine_reg(static, static_grid2world,
                                             moving, moving_grid2world)
            reg_affines.append(trans_affine)
            print('Affine registration finished')
        else:
            moved = data[..., i]
            trans_affine = affine
    
        data_corr[..., i] = moved

    gtab_corr = reorient_bvecs(gtab, reg_affines)
    
    save_nifti(fcorr_dwi, data_corr, affine)
    
    np.savetxt(fcorr_bval, bvals)
    np.savetxt(fcorr_bvec, gtab_corr.bvecs)
    """


    from dipy.align.imaffine import transform_centers_of_mass

    target = nib.load(SAMBA_preprocess)
    target_data = target.get_fdata()
    target_grid2world = target.affine

    moving = nib.load(func_reorient_affined)
    moving_data = moving.get_fdata()
    moving_grid2world = moving.affine


    func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_masscenter{ext}')
    if not os.path.exists(func_reorient_recentered) or overwrite:
        c_of_mass = transform_centers_of_mass(target_data, target_grid2world,
                                              moving_data[:, :, :, 0], moving_grid2world)
        # regtools.overlay_slices(target_data, moving_data, None, 0,
        #                        "Static", "Moving")
        transformed = c_of_mass.transform(moving_data[:, :, :, 0])
        new_nii = nib.Nifti1Image(transformed, target_grid2world)
        nib.save(new_nii, func_reorient_recentered)
        print(f'Saved {func_reorient_recentered}')

    from dipy.io.image import save_nifti

    func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_trans{ext}')
    if not os.path.exists(func_reorient_recentered) or overwrite:
        moved, trans_affine = trans_reg(target_data, target_grid2world,
                                        moving_data[..., 0], moving_grid2world)
        save_nifti(func_reorient_recentered, moved, target_grid2world)
        print(f'Saved {func_reorient_recentered}')

    func_reorient_recentered = os.path.join(nii_temp_dir, f'{subj}_func_reorient_rigid{ext}')
    if not os.path.exists(func_reorient_recentered) or overwrite:
        moved, trans_affine = rigid_reg(target_data, target_grid2world,
                                        moving_data[..., 0], moving_grid2world)
        save_nifti(func_reorient_recentered, moved, target_grid2world)
        print(f'Saved {func_reorient_recentered}')

    """
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
    """


    SAMBA_preprocess_test_posttrans = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans{ext}')
    SAMBA_preprocess_test_posttrans_2 = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans_2{ext}')
    SAMBA_preprocess_test_posttrans_3 = os.path.join(nii_temp_dir, f'{subj}_masked_posttrans_3{ext}')

    SAMBA_preprocess_test_rigid = os.path.join(nii_temp_dir, f'{subj}_postrigid{ext}')
    SAMBA_preprocess_test_rigid_affine = os.path.join(nii_temp_dir, f'{subj}_postrigid_affine{ext}')
    SAMBA_preprocess_test_postwarp = os.path.join(nii_MDT, f'{subj}_postwarp{ext}')
    SAMBA_preprocess_test_postwarp_oneline = os.path.join(nii_MDT, f'{subj}_postwarp_oneline{ext}')

    if not os.path.exists(SAMBA_preprocess_test_postwarp_oneline) or overwrite:
        cmd = f"antsApplyTransforms -v 1 -d 3 -i {SAMBA_preprocess} -o {SAMBA_preprocess_test_postwarp_oneline} -r {target_base} -n MultiLabel -t {runno_to_MDT} [{affine_orig},0] [{rigid},0] [{trans},0]"
        os.system(cmd)

    func_postwarp = os.path.join(nii_MDT, f'{subj}_func_postwarp{ext}')
    if not os.path.exists(func_postwarp) or overwrite:
        cmd = f"antsApplyTransforms -v 1 -d 3 -i {func_reorient_recentered} -o {func_postwarp} -r {target_base} -n MultiLabel -t {runno_to_MDT} [{affine_orig},0] [{rigid},0] [{trans},0]"
        os.system(cmd)

    func_postrigid = os.path.join(nii_MDT, f'{subj}_func_postrigid{ext}')
    if not os.path.exists(func_postrigid) or overwrite:
        cmd = f"antsApplyTransforms -v 1 -d 3 -i {func_reorient_recentered} -o {func_postrigid} -r {target_base} -n MultiLabel -t [{rigid},0] [{trans},0]"
        os.system(cmd)

    if native_ref == '':
        native_ref = SAMBA_preprocess
    if not os.path.exists(SAMBA_preprocess_test_postwarp) or overwrite:

        print(ants_loadmat(trans))
        print(ants_loadmat(rigid))
        print(ants_loadmat(affine_orig))

        reverse_test = False
        if reverse_test:
            reg_subj = os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_test_i7/MDT_images',f'{subj}_T1_to_MDT.nii.gz')
            subj_T1_reversetest = os.path.join(nii_temp_dir, f'{subj}_reversed_subj{ext}')
            cmd = f"antsApplyTransforms -v 1 -d 3 -i {reg_subj} -o {subj_T1_reversetest} -r {target_base} -n MultiLabel -t [{trans},1] [{rigid},1] [{affine_orig},1] {MDT_to_subject}"
            os.system(cmd)


        cmd = f'antsApplyTransforms -v 1 -d 3  -i {SAMBA_preprocess} -r {SAMBA_preprocess}  -n Linear  -o {SAMBA_preprocess_test_posttrans}'
        os.system(cmd)
        #shutil.copy(SAMBA_preprocess, SAMBA_preprocess_test_posttrans)
        affine_superpose(SAMBA_preprocess, SAMBA_preprocess_test_posttrans, outpath=SAMBA_preprocess_test_posttrans_2)
        #shutil.copy(SAMBA_preprocess_test_posttrans, SAMBA_preprocess_test_posttrans_2)

        cmd = f'antsApplyTransforms -v 1 -d 3  -i {SAMBA_preprocess_test_posttrans_2} -r {ref_t1}  -n Linear  -o {SAMBA_preprocess_test_posttrans_3} -t {trans}'
        os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_posttrans_3} -o {SAMBA_preprocess_test_rigid} ' \
            f'-r {ref_t1} -n Linear -t [{rigid},0]'
        os.system(cmd)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_rigid} -o {SAMBA_preprocess_test_rigid_affine} ' \
            f'-r {SAMBA_preprocess_test_posttrans_2} -n Linear -t [{affine_orig},0]'
        os.system(cmd)

        #shutil.copy(SAMBA_preprocess_test_rigid, SAMBA_preprocess_test_rigid_affine)

        cmd = f'antsApplyTransforms -v 1 --float -d 3 -i {SAMBA_preprocess_test_rigid_affine} -o {SAMBA_preprocess_test_postwarp} ' \
            f'-r {ref_t1} -n Linear -t {runno_to_MDT}'
        os.system(cmd)

    """
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
    
    """