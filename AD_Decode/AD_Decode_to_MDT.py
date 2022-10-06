import os
from DTC.nifti_handlers.transform_handler import get_affine_transform, get_flip_affine, header_superpose, \
    recenter_nii_affine, \
    convert_ants_vals_to_affine, read_affine_txt, recenter_nii_save, add_translation, recenter_nii_save_test, \
    affine_superpose, get_affine_transform_test, recenter_affine_test
import nibabel as nib
import numpy as np
from dipy.tracking.streamline import deform_streamlines, transform_streamlines
from DTC.tract_manager.streamline_nocheck import load_trk, unload_trk
from DTC.tract_manager.tract_save import save_trk_header
from scipy.io import loadmat
import socket
from DTC.tract_manager.tract_handler import gettrkpath
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
import glob, warnings
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote
import time
from DTC.tract_manager.tract_handler import transform_streamwarp

def _to_streamlines_coordinates(inds, lin_T, offset):
    """Applies a mapping from streamline coordinates to voxel_coordinates,
    raises an error for negative voxel values."""
    inds = inds - offset
    streamline = np.dot(inds, np.linalg.inv(lin_T))
    # streamline = streamline - [1,1,1]
    return streamline


def _to_voxel_coordinates_notint(streamline, lin_T, offset):
    """Applies a mapping from streamline coordinates to voxel_coordinates,
    raises an error for negative voxel values."""
    inds = np.dot(streamline, lin_T)
    inds += offset
    if inds.min().round(decimals=6) < 0:
        raise IndexError('streamline has points that map to negative voxel'
                         ' indices')
    return inds


ext = ".nii.gz"
computer_name = socket.gethostname()

remote=True
project='AD_Decode'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)

outpath = '/mnt/paros_WORK/jacques/AD_Decode'
if project == "AD_Decode":
    path_TRK = os.path.join(inpath, 'TRK_MPCA_fixed')
    path_TRK_output = os.path.join(inpath, 'TRK_MPCA_MDT')
    path_DWI = os.path.join(inpath, 'DWI')
    path_DWI_MDT = os.path.join(inpath, 'DWI_MDT')
    path_transforms = os.path.join(inpath, 'Transforms')
    ref = "md"

    path_trk_tempdir = os.path.join(outpath, 'TRK_transition')
    path_TRK_output = os.path.join(outpath, 'TRK_MDT_real_testtemp')
    DWI_save = os.path.join(outpath, 'NII_trans_save')

    mkcdir([DWI_save, path_DWI_MDT, path_TRK_output, path_trk_tempdir],sftp=sftp)

    # Get the values from DTC_launcher_ADDecode. Should probalby create a single parameter file for each project one day
    stepsize = 2
    ratio = 1
    trkroi = ["wholebrain"]
    str_identifier = get_str_identifier(stepsize, ratio, trkroi)
    SAMBA_MDT = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz'

subjects_all = glob.glob(os.path.join(path_DWI, '*subjspace_dwi*.nii.gz'))
subjects = []
for subject in subjects_all:
    subject_name = os.path.basename(subject)
    subjects.append(subject_name[:6])

# removed_list = ['S02804', 'S02817', 'S02898', 'S02871', 'S02877','S03045', 'S02939', 'S02840']

"""
subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361',
            'S02363', 'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451',
            'S02469', 'S02473', 'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654',
            'S02666', 'S02670', 'S02686', 'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753',
            'S02765', 'S02771', 'S02781', 'S02802', 'S02804', 'S02813', 'S02812', 'S02817', 'S02840', 'S02842',
            'S02871', 'S02877', 'S02898', 'S02926', 'S02938', 'S02939', 'S02954', 'S02967', 'S02987', 'S03010',
            'S03017', 'S03028', 'S03033', 'S03034', 'S03045', 'S03048', 'S03069', 'S03225', 'S03265', 'S03293',
            'S03308', 'S03321', 'S03343', "S03350", "S03378", "S03391", "S03394"]

#removed_list = ['S02771',"S03343", "S03350", "S03378", "S03391", "S03394","S03225", "S03293", "S03308", "S02842", "S02804"]
removed_list = ["S02654"]
"""
subjects = ['S02266']
subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361',
            'S02363', 'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451',
            'S02469', 'S02473', 'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654',
            'S02666', 'S02670', 'S02686', 'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753',
            'S02765', 'S02771', 'S02781', 'S02802', 'S02804', 'S02813', 'S02812', 'S02817', 'S02840', 'S02842',
            'S02871', 'S02877', 'S02898', 'S02926', 'S02938', 'S02939', 'S02954', 'S02967', 'S02987', 'S03010',
            'S03017', 'S03028', 'S03033', 'S03034', 'S03045', 'S03048', 'S03069', 'S03225', 'S03265', 'S03293',
            'S03308', 'S03321', 'S03343', "S03350", "S03378", "S03391", "S03394"]
removed_list = ['S02654']
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

# subjects.reverse()
subjects = subjects[:]
print(subjects)

overwrite = False
cleanup = False
verbose = True
save_temp_trk_files = True
recenter = 1
contrast = 'fa'
prune = True
nii_test_files = False

native_ref = ''

for subj in subjects:
    trans = os.path.join(path_transforms, f"{subj}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(path_transforms, f"{subj}_rigid.mat")
    affine_orig = os.path.join(path_transforms, f"{subj}_affine.mat")
    affine = os.path.join(path_transforms, f"{subj}_affine.txt")
    runno_to_MDT = os.path.join(path_transforms, f'{subj}_to_MDT_warp.nii.gz')
    subj_dwi_path = os.path.join(path_DWI, f'{subj}_subjspace_{contrast}{ext}')

    if nii_test_files:
        mkcdir([DWI_save, path_DWI_MDT], sftp)
        SAMBA_preprocess_ref = os.path.join(path_DWI, f'{subj}_labels{ext}')
        SAMBA_coreg_ref = os.path.join(path_DWI, f'{subj}_{contrast}{ext}')
        SAMBA_init = os.path.join(path_DWI, f'{subj}_{contrast}{ext}')
        SAMBA_init = subj_dwi_path
        SAMBA_preprocess = os.path.join(DWI_save, f'{subj}_{contrast}_preprocess{ext}')

        if recenter and (not os.path.exists(SAMBA_preprocess) or overwrite):
            """
            header_superpose(SAMBA_preprocess_ref, SAMBA_init, outpath=SAMBA_preprocess, verbose=False)
            img_transform_exec(SAMBA_preprocess, 'RAS', 'LPS', output_path=SAMBA_preprocess_2)
            recenter_nii_save(SAMBA_preprocess_2, SAMBA_preprocess_recentered_1, verbose=True)
            recenter_nii_save(SAMBA_preprocess_recentered_1,SAMBA_preprocess_2)
            SAMBA_init = SAMBA_preprocess_2
            """
            recenter_nii_save_test(SAMBA_init, SAMBA_preprocess)
            SAMBA_init = SAMBA_preprocess

        SAMBA_preprocess_test_posttrans = os.path.join(DWI_save, f'{subj}_{contrast}_masked_posttrans{ext}')
        SAMBA_preprocess_test_posttrans_2 = os.path.join(DWI_save, f'{subj}_{contrast}_masked_posttrans_2{ext}')
        SAMBA_preprocess_test_posttrans_3 = os.path.join(DWI_save, f'{subj}_{contrast}_masked_posttrans_3{ext}')

        SAMBA_preprocess_test_rigid = os.path.join(DWI_save, f'{subj}_{contrast}_postrigid{ext}')
        SAMBA_preprocess_test_rigid_affine = os.path.join(DWI_save, f'{subj}_{contrast}_postrigid_affine{ext}')
        SAMBA_preprocess_test_postwarp = os.path.join(path_DWI_MDT, f'{subj}_{contrast}_postwarp{ext}')
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

    trk_preprocess_posttrans = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_posttrans.trk')
    trk_preprocess_postrigid = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_postrigid.trk')
    trk_preprocess_postrigid_affine = os.path.join(path_trk_tempdir,
                                                   f'{subj}{str_identifier}_preprocess_postrigid_affine.trk')

    subj_trk, trkexists = gettrkpath(path_TRK, subj, str_identifier, pruned=prune, verbose=False, sftp=sftp)
    # trk_MDT_space = os.path.join(path_TRK_output, f'{subj}_MDT.trk')
    trk_MDT_space = os.path.join(path_TRK_output, os.path.basename(subj_trk))
    MDT_exists = checkfile_exists_remote(trk_MDT_space, sftp)

    if not trkexists:
        warnings.warn(f'Could not find the trk for subj {subj} at {subj_trk}, will continue with next subject')
        continue

    if not MDT_exists or overwrite:

        print(f'Beginning the moving of subject {subj} to {trk_MDT_space}')
        tstart = time.perf_counter()

        subj_dwi_path = os.path.join(path_DWI, f'{subj}_subjspace_dwi{ext}')
        subj_dwi_data, subj_affine, _, _, _ = load_nifti_remote(subj_dwi_path, sftp)

        print('Loaded nifti')
        t1 = time.perf_counter()

        SAMBA_input_real_file = os.path.join(path_DWI, f'{subj}_dwi{ext}')

        check_dif_ratio(path_TRK, subj, str_identifier, ratio)

        _, exists = check_files([trans, rigid, runno_to_MDT], sftp)
        if np.any(exists == 0):
            raise Exception('missing transform file')
        _, exists = check_files([affine, affine_orig], sftp)
        if np.any(exists == 0):
            raise Exception('missing transform file')

        subj_affine_new = recenter_affine_test(np.shape(subj_dwi_data), subj_affine)

        subj_centered = np.copy(subj_affine)
        subj_centered[:3, 3] = [0, 0, 0]
        subj_dwi_c = nib.Nifti1Image(subj_dwi_data, subj_centered)
        subj_dwi_c_path = os.path.join(DWI_save, f'{subj}_subjspace_centered{ext}')
        # nib.save(subj_dwi_c, subj_dwi_c_path)

        subj_centered_rotated = np.copy(subj_affine_new)
        subj_centered_rotated[:3, 3] = [0, 0, 0]
        subj_dwi_cr = nib.Nifti1Image(subj_dwi_data, subj_centered_rotated)
        subj_dwi_cr_path = os.path.join(DWI_save, f'{subj}_subjspace_centered_rotated{ext}')
        # nib.save(subj_dwi_cr, subj_dwi_cr_path)

        streamlines_data = load_trk_remote(subj_trk, 'same', sftp)
        header = streamlines_data.space_attributes
        streamlines = streamlines_data.streamlines

        ttrk = time.perf_counter()
        print(f'Loaded trk, took {ttrk - t1:0.2f} seconds')
        transform_centered_affine = np.eye(4)
        transform_centered_affine[:3, 3] = - subj_affine[:3, 3] + subj_centered[:3, 3]
        subj_centered_streamlines = transform_streamlines(streamlines, transform_centered_affine,
                                                          in_place=False)
        trk_centered = os.path.join(path_trk_tempdir, f'{subj}_centered.trk')

        if (not checkfile_exists_remote(trk_centered, sftp) or overwrite) and save_temp_trk_files:
            save_trk_header(filepath=trk_centered, streamlines=subj_centered_streamlines, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp)

        nii_shape = np.array(np.shape(subj_dwi_data))[0:3]

        preprocess_transform = np.eye(4)
        transform_centered_affine = - subj_affine[:3, 3] + subj_centered[:3, 3]
        recentering_translation = recenter_affine_test(nii_shape, subj_affine)
        preprocess_transform[:3, 3] = recentering_translation[:3, 3]
        subj_torecenter_transform_affine = get_affine_transform_test(subj_affine, subj_affine_new)
        preprocess_transform[:3, :3] = np.linalg.inv(subj_torecenter_transform_affine[:3, :3])
        streamlines_prepro = transform_streamlines(subj_centered_streamlines, preprocess_transform,
                                                   in_place=False)
        trk_preprocess_path = os.path.join(path_trk_tempdir, f'{subj}_preprocess.trk')
        tras = time.perf_counter()

        if (not checkfile_exists_remote(trk_preprocess_path, sftp) or overwrite) and save_temp_trk_files:
            save_trk_header(filepath=trk_preprocess_path, streamlines=streamlines_prepro, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp)

        mat_struct = loadmat_remote(trans, sftp)
        var_name = list(mat_struct.keys())[0]
        later_trans_mat = mat_struct[var_name]
        new_transmat = np.eye(4)
        vox_dim = [1, 1, -1]
        # new_transmat[:3, 3] = np.squeeze(later_trans_mat[3:6]) * vox_dim
        new_transmat[:3, 3] = np.squeeze(np.matmul(subj_affine[:3, :3], later_trans_mat[
                                                                        3:6]))  # should be the AFFINE of the current image, to make sure the slight difference in orientation is ACCOUNTED FOR!!!!!!!!!!
        new_transmat[:3, 3] = np.squeeze(np.matmul(subj_torecenter_transform_affine[:3, :3], later_trans_mat[3:6]))
        new_transmat[:3, 3] = np.squeeze(later_trans_mat[3:6])
        new_transmat[2, 3] = - new_transmat[2, 3]
        print(new_transmat)
        streamlines_posttrans = transform_streamlines(streamlines_prepro, (new_transmat))
        del streamlines_prepro
        ttrans = time.perf_counter()
        print(f'added translation, took {ttrans - tras:0.2f} seconds')

        if (not checkfile_exists_remote(trk_preprocess_posttrans, sftp) or overwrite) and save_temp_trk_files:
            save_trk_header(filepath=trk_preprocess_posttrans, streamlines=streamlines_posttrans, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp)

        rigid_struct = loadmat_remote(rigid, sftp)
        var_name = list(rigid_struct.keys())[0]
        rigid_ants = rigid_struct[var_name]
        rigid_mat = convert_ants_vals_to_affine(rigid_ants)

        # streamlines_posttrans, header_posttrans = unload_trk(trk_preprocess_posttrans)
        streamlines_postrigid = transform_streamlines(streamlines_posttrans, np.linalg.inv(rigid_mat))
        del streamlines_posttrans
        trig = time.perf_counter()
        print(f'added rigid transform, took {trig - ttrans:0.2f} seconds')

        if (not checkfile_exists_remote(trk_preprocess_postrigid, sftp) or overwrite) and save_temp_trk_files:
            save_trk_header(filepath=trk_preprocess_postrigid, streamlines=streamlines_postrigid, header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp)

        """
        if checkfile_exists_remote(affine, sftp):
            affine_mat_s = read_affine_txt(affine)
        else:
            cmd = f'ConvertTransformFile 3 {affine_orig} {affine} --matrix'
            os.system(cmd)
            affine_mat_s = read_affine_txt(affine)
        """

        affine_struct = loadmat_remote(affine_orig, sftp)
        var_name = list(affine_struct.keys())[0]
        affine_ants = affine_struct[var_name]
        affine_mat = convert_ants_vals_to_affine(affine_ants)

        streamlines_postrigidaffine = transform_streamlines(streamlines_postrigid, np.linalg.inv(affine_mat))
        del streamlines_postrigid
        taf = time.perf_counter()
        print(f'added affine transform, took {taf - trig:0.2f} seconds')
        if (not checkfile_exists_remote(trk_preprocess_postrigid_affine, sftp) or overwrite) and save_temp_trk_files:
            save_trk_header(filepath=trk_preprocess_postrigid_affine, streamlines=streamlines_postrigidaffine,
                            header=header,
                            affine=np.eye(4), verbose=verbose, sftp=sftp)


        warp_data, warp_affine, _, _, _ = load_nifti_remote(runno_to_MDT, sftp)
        mni_streamlines = transform_streamwarp(streamlines_postrigidaffine, SAMBA_MDT, warp_data)
        del streamlines_postrigidaffine
        twarp = time.perf_counter()
        print(f'added warp transform, took {twarp - taf:0.2f} seconds')

        save_trk_header(filepath=trk_MDT_space, streamlines=mni_streamlines, header=header,
                affine=np.eye(4), verbose=verbose, sftp=sftp)

        tf = time.perf_counter()
        del mni_streamlines

        print(f'Saved MDT image to {trk_MDT_space}, took {tf - twarp:0.2f} seconds, subject {subj} run took {tf - tstart:0.2f} seconds total')
        print('\n')
    else:
        print(f'{trk_MDT_space} already exists')
