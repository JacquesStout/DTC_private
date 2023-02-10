import os, sys
from DTC.nifti_handlers.transform_handler import affine_superpose
import numpy as np
from DTC.tract_manager.DTC_manager import get_str_identifier
import glob
import shutil
from DTC.nifti_handlers.transform_handler import img_transform_exec
from DTC.file_manager.file_tools import check_files
from DTC.tract_manager.DTC_manager import check_dif_ratio
from dipy.tracking.streamline import transform_streamlines
from DTC.tract_manager.tract_handler import gettrkpath
from DTC.nifti_handlers.transform_handler import convert_ants_vals_to_affine
from DTC.tract_manager.tract_save import save_trk_header
from DTC.tract_manager.tract_handler import transform_streamwarp
from DTC.file_manager.computer_nav import get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote, \
    checkfile_exists_remote, glob_remote
from DTC.file_manager.file_tools import mkcdir, getfromfile
import time
from DTC.file_manager.argument_tools import parse_arguments
from datetime import datetime


"""
from dipy.io.utils import get_reference_info, is_header_compatible
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from scipy.ndimage import map_coordinates
import logging
from DTC.visualization_tools.tract_visualize import setup_view
"""
from dipy.viz import window, actor


def _to_streamlines_coordinates(inds, lin_T, offset):
    """Applies a mapping from streamline coordinates to voxel_coordinates,
    raises an error for negative voxel values."""
    inds = inds - offset
    streamline = np.dot(inds, np.linalg.inv(lin_T))
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


remote=True
project='APOE'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)

#subjects = ['N58214', 'N58215', 'N58216', 'N58217', 'N58218', 'N58219', 'N58221', 'N58222', 'N58223', 'N58224', 'N58225', 'N58226', 'N58228', 'N58229', 'N58230', 'N58231', 'N58232', 'N58302', 'N58303', 'N58305', 'N58309', 'N58310', 'N58344', 'N58346', 'N58350', 'N58355', 'N58359', 'N58361', 'N58394', 'N58396', 'N58398', 'N58400', 'N58402', 'N58404', 'N58406', 'N58408', 'N58477', 'N58500', 'N58510', 'N58512', 'N58514', 'N58516', 'N58604', 'N58606', 'N58608', 'N58611', 'N58613', 'N58633', 'N58634', 'N58635', 'N58636', 'N58649', 'N58650', 'N58651', 'N58653', 'N58654', 'N58655', 'N58706', 'N58708', 'N58712', 'N58714', 'N58732', 'N58733', 'N58734', 'N58735', 'N58740', 'N58742', 'N58745', 'N58747', 'N58749', 'N58751', 'N58779', 'N58780', 'N58784', 'N58788', 'N58790', 'N58792', 'N58794', 'N58813', 'N58815', 'N58819', 'N58821', 'N58829', 'N58831', 'N58851', 'N58853', 'N58855', 'N58857', 'N58859', 'N58861', 'N58877', 'N58879', 'N58881', 'N58883', 'N58885', 'N58887', 'N58889', 'N58906', 'N58909', 'N58913', 'N58917', 'N58919', 'N58935', 'N58941', 'N58952', 'N58995', 'N58997', 'N58999', 'N59003', 'N59010', 'N59022', 'N59026', 'N59033', 'N59035', 'N59039', 'N59041', 'N59065', 'N59066', 'N59072', 'N59076', 'N59078', 'N59080', 'N59097', 'N59099', 'N59109', 'N59116', 'N59118', 'N59120']

outpath = '/mnt/paros_WORK/jacques/APOE'
if project == "APOE":
    path_DWI = os.path.join(inpath, 'DWI_RAS')
    path_DWI_MDT = os.path.join(inpath, 'DWI_MDT')
    path_transforms = os.path.join(inpath, 'Transforms')
    path_TRK = os.path.join(inpath, 'TRK_RAS')
    path_trk_tempdir = os.path.join(outpath, 'TRK_transition')
    path_TRK_output = os.path.join(outpath, 'TRK_MDT')
    ref = "md"
    DWI_save = os.path.join(outpath, 'NII_trans_save')
    mkcdir([DWI_save, path_DWI_MDT, path_TRK_output, path_trk_tempdir],sftp=sftp)
    # Get the values from DTC_launcher_ADDecode. Should probalby create a single parameter file for each project one day
    stepsize = 2
    ratio = 1
    trkroi = ["wholebrain"]
    str_identifier = get_str_identifier(stepsize, ratio, trkroi)
    SAMBA_MDT = '/Volumes/Data/Badea/Lab/mouse/VBM_20APOE01_chass_symmetric3_allAPOE-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_NoNameYet_n32_i6/median_images/MDT_dwi.nii.gz'
    #SAMBA_MDT = '/home/alex/MDT_dwi.nii.gz'

subjects_all = glob_remote(os.path.join(path_DWI,'*subjspace*coreg*.nii.gz'),sftp)
subjects = []
for subject in subjects_all:
    subject_name = os.path.basename(subject)
    subjects.append(subject_name[:6])

removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects)

subjects = subjects[firstsubj:lastsubj]
print(subjects)

overwrite = False
cleanup = False
verbose = True
save_temp_files = True
recenter = 1
# contrast = 'dwi'
contrast = 'subjspace_coreg_RAS'
prune = True
nii_test_files = False
ext = ".nii.gz"
RASTI = True
native_ref = ''
save_temp_trk_files = False

for subj in subjects:
    trans = os.path.join(path_transforms, f"{subj}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(path_transforms, f"{subj}_rigid.mat")
    affine_orig = os.path.join(path_transforms, f"{subj}_affine.mat")
    affine = os.path.join(path_transforms, f"{subj}_affine.txt")
    runno_to_MDT = os.path.join(path_transforms, f'{subj}_to_MDT_warp.nii.gz')
    # subj_dwi = os.path.join(path_DWI, f'{subj}_{contrast}_masked{ext}')
    subj_dwi = os.path.join(path_DWI, f'{subj}_{contrast}{ext}')


    if nii_test_files:
        SAMBA_init = subj_dwi
        SAMBA_preprocess = os.path.join(DWI_save, f'{subj}_{contrast}_preprocess{ext}')

        if recenter and (not os.path.exists(SAMBA_preprocess) or overwrite):
            if RASTI:
                # img_transform_exec(SAMBA_init,'RAS','LAS',output_path = SAMBA_LAS)
                img_transform_exec(SAMBA_init, 'RAS', 'ALS', output_path=SAMBA_preprocess, recenter=True)

                # affine_superpose(dwi_masked_path, SAMBA_init, outpath=SAMBA_preprocess, transpose=None)
            else:
                shutil.copy(SAMBA_init, SAMBA_preprocess)
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


    trk_preprocess = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess.trk')
    trk_preprocess_posttrans = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_posttrans.trk')
    trk_preprocess_postrigid = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_postrigid.trk')
    # trk_preprocess_postrigid_affine = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_postrigid_affine.trk')
    trk_preprocess_postrigid_affine = os.path.join(path_trk_tempdir,
                                                   f'{subj}{str_identifier}_preprocess_postrigid_affine.trk')
    trk_MDT_space = os.path.join(path_TRK_output, f'{subj}_MDT.trk')
    MDT_exists = checkfile_exists_remote(trk_MDT_space, sftp)

    if not MDT_exists or overwrite:

        print(f'Beginning the moving of subject {subj} to {trk_MDT_space}')
        tstart = time.perf_counter()

        _, exists = check_files([trans, rigid, runno_to_MDT], sftp)

        nii_data, subj_affine, _, _, _ = load_nifti_remote(subj_dwi, sftp)
        # subj_affine_new = subj_affine

        print('Loaded nifti')
        t1 = time.perf_counter()
        check_dif_ratio(path_TRK, subj, str_identifier, ratio, sftp)
        subj_trk, trkexists = gettrkpath(path_TRK, subj, str_identifier, pruned=prune, verbose=False, sftp=sftp)
        if not trkexists:
            txt = f'Could not find TRK file for subject {subj}'
            raise Exception(txt)
        _, exists = check_files([trans, rigid, runno_to_MDT], sftp)
        if np.any(exists == 0):
            raise Exception('missing transform file')
        _, exists = check_files([affine, affine_orig], sftp)
        if np.any(exists == 0):
            raise Exception('missing transform file')
        # streamlines_prepro, header = unload_trk(subj_trk)
        streamlines_data = load_trk_remote(subj_trk, 'same', sftp)
        ttrk = time.perf_counter()
        print(f'Loaded trk, took {ttrk - t1:0.2f} seconds')

        header = streamlines_data.space_attributes
        streamlines_orig = streamlines_data.streamlines
        del streamlines_data
        if RASTI:
            affine_RAS_ALS = np.eye(4)

            affine_RAS_ALS[0, 1] = -1
            affine_RAS_ALS[1, 0] = 1
            affine_RAS_ALS[0, 0] = 0
            affine_RAS_ALS[1, 1] = 0

            affine_RAS_ALS[0, 3] = (header[1][0] * header[0][0, 0] * 0.5) - 0.045 + subj_affine[0, 3]
            affine_RAS_ALS[1, 3] = (header[1][1] * header[0][1, 1] * 0.5) - 0.045 + subj_affine[1, 3]
            affine_RAS_ALS[2, 3] = (header[1][2] * header[0][2, 2] * 0.5) - 0.045 + subj_affine[2, 3]

            streamlines_prepro = transform_streamlines(streamlines_orig, np.linalg.inv(affine_RAS_ALS))
            if (not checkfile_exists_remote(trk_preprocess, sftp) or overwrite) and save_temp_trk_files:
                save_trk_header(filepath=trk_preprocess, streamlines=streamlines_prepro, header=header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp)
            tras = time.perf_counter()
            print(f'RAStified, took {tras - ttrk:0.2f} seconds')

        else:
            streamlines_prepro = np.copy(streamlines_orig)
            tras = time.perf_counter()

        del streamlines_orig

        # streamlines_prepro = streamlines_data.streamlines
        # from DTC.tract_manager.streamline_nocheck import unload_trk
        # streamlines_prepro, header_prepro = unload_trk(trk_preprocess)
        mat_struct = loadmat_remote(trans, sftp)
        var_name = list(mat_struct.keys())[0]
        later_trans_mat = mat_struct[var_name]
        new_transmat = np.eye(4)
        vox_dim = [1, 1, -1]
        # new_transmat[:3, 3] = np.squeeze(later_trans_mat[3:6]) * vox_dim
        new_transmat[:3, 3] = np.squeeze(np.matmul(subj_affine[:3, :3], later_trans_mat[
                                                                        3:6]))  # should be the AFFINE of the current image, to make sure the slight difference in orientation is ACCOUNTED FOR!!!!!!!!!!
        new_transmat[2, 3] = 0
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
        if os.path.exists(affine):
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

        print(f'Saved MDT image to {trk_MDT_space}, took {tf - twarp:0.2f} seconds, subject {subj} run took {tf - tstart:0.2f} seconds total\n')
        current_time = datetime.now()
        print("Current Time =", current_time)
        print('\n')

    else:
        print(f'{trk_MDT_space} already exists')
