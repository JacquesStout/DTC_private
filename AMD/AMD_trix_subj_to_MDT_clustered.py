import os
from DTC.nifti_handlers.transform_handler import get_affine_transform, get_flip_affine, header_superpose, \
    recenter_nii_affine, \
    convert_ants_vals_to_affine, read_affine_txt, recenter_nii_save, add_translation, recenter_nii_save_test, \
    affine_superpose, get_affine_transform_test

import numpy as np
import nibabel as nib
from dipy.tracking.streamline import deform_streamlines, transform_streamlines

from DTC.tract_manager.tract_save import save_trk_header

import socket
from DTC.tract_manager.tract_handler import gettrkpath
from DTC.tract_manager.DTC_manager import get_str_identifier
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from DTC.tract_manager.DTC_manager import check_dif_ratio
from DTC.file_manager.computer_nav import get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote, \
    checkfile_exists_remote
import random
from DTC.tract_manager.tract_handler import transform_streamwarp
import time
from DTC.file_manager.argument_tools import parse_arguments
import sys
from ants.core.ants_transform_io import create_ants_transform, write_transform, read_transform
from DTC.nifti_handlers.transform_handler import rigid_reg, translation_reg

project = 'AMD'

subj = sys.argv[1]
#subj = 'H26578'
# subjects = ['H26578']
# subjects = ["H26660"]
# subjects = ["H29410"]
# random.shuffle(subjects)

# temporarily removing "H29056" to recalculate it
ext = ".nii.gz"
computer_name = socket.gethostname()

username = None
passwd = None

sftp = None

"""
server = getremotehome('remotename')
if server not in computer_name:
    remote=True
else:
    remote=False
if remote:
    username, passwd = getfromfile('/Users/jas/remote_connect.rtf')
inpath, outpath, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
"""

inpath = '/mnt/munin2/Badea/Lab/human/AMD_project_23'
outpath = '/mnt/munin2/Badea/Lab/human/AMD_project_23'

if project == "AMD":
    path_TRK = os.path.join(inpath, 'TRK_trix')
    path_DWI = os.path.join(inpath, 'DWI')
    ref = "fa"
    path_trk_tempdir = os.path.join(outpath, 'TRK_mrtrix_transition')
    path_TRK_output = os.path.join(outpath, 'TRK_mrtrix_MDT_farun')

    path_DWI_temp = os.path.join(outpath, 'NII_toMDT_temp')
    path_DWI_output = os.path.join(outpath, 'DWI_trix_MDT_farun')
    # Get the values from DTC_launcher_ADDecode. Should probably create a single parameter file for each project one day

    path_transforms = os.path.join(inpath, 'Transforms_farun')

mkcdir([outpath, path_trk_tempdir, path_TRK_output, path_DWI_output, path_transforms], sftp)

stepsize = 2
ratio = 1
trkroi = ["wholebrain"]
str_identifier = get_str_identifier(stepsize, ratio, trkroi)
prune = False
overwrite = False
cleanup = False
verbose = True
recenter = True

save_temp_trk_files = False

nii_to_MDT = False
trk_to_MDT = True

contrasts = ['fa', 'md', 'rd', 'ad']
contrasts = ['fa']
# contrast = 'fa'
native_ref = ''


#SAMBA_MDT = '/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_fa/faMDT_Control_n72_i6/median_images/MDT_dwi.nii.gz'
SAMBA_MDT = os.path.join(path_transforms, 'MDT_dwi.nii.gz')
SAMBA_inits = '/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/preprocess/base_images/'
SAMBA_inits = os.path.join(inpath, 'DWI_inits')

if save_temp_trk_files:
    mkcdir(path_trk_tempdir, sftp)


print(f'running move for subject {subj}')

#trans = os.path.join(f'/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/preprocess/base_images/',f"{subj}_0DerivedInitialMovingTranslation.mat")
trans = os.path.join(path_transforms, f"{subj}_0DerivedInitialMovingTranslation.mat")
rigid = os.path.join(path_transforms, f"{subj}_rigid.mat")
#rigid = f'/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/{subj}_rigid.mat'
#affine_orig = f'/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/{subj}_affine.mat'
#affine_orig = f'/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/{subj}_affine.mat'
affine_orig = os.path.join(path_transforms, f"{subj}_affine.mat")
#affine = os.path.join(path_transforms, f"{subj}_affine.txt")
runno_to_MDT = os.path.join(path_transforms, f'{subj}_to_MDT_warp.nii.gz')
#runno_to_MDT = f'/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_fa/faMDT_Control_n72_i6/reg_diffeo/{subj}_to_MDT_warp.nii.gz'
subj_dwi = os.path.join(path_DWI, f'{subj}_subjspace_dwi{ext}')
SAMBA_ref = os.path.join(path_transforms, f'reference_file_c_isotropi.nii.gz')

if nii_to_MDT:
    mkcdir(path_DWI_output)
    if 'translation' in locals():
        del translation

    for contrast in contrasts:

        SAMBA_postwarp = os.path.join(path_DWI_output, f'{subj}_{contrast}_to_MDT{ext}')

        if not os.path.exists(SAMBA_postwarp) or overwrite:

            SAMBA_subj = f'/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd/{contrast}/{subj}_{contrast}.nii.gz'
            SAMBA_init_test_t = f'/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd/contrasts_init/{subj}_{contrast}_init_transition.nii.gz'
            SAMBA_init_test = f'/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd/contrasts_init/{subj}_{contrast}_init.nii.gz'
            SAMBA_init = os.path.join(SAMBA_inits, f'{subj}_{contrast}_masked.nii.gz')

            voxel_size = [1, 1, 1]
            new_voxel_size = [1, 1, 1]

            from dipy.align.reslice import reslice

            new_affine = np.eye(4)
            new_affine[2, 2] = 2
            new_affine[0, 0] = -1
            new_affine[1, 1] = -1

            new_affine[0, 3] = -127
            new_affine[1, 3] = -127
            new_affine[2, 3] = -67

            data2, affine2 = reslice(nib.load(SAMBA_subj).get_fdata(), new_affine, voxel_size, new_voxel_size)

            SAMBA_init_nii = nib.Nifti1Image(data2, affine2)
            nib.save(SAMBA_init_nii, SAMBA_init_test_t)

            if 'translation' not in locals():
                transformed, _, translation = translation_reg(nib.load(SAMBA_init).get_fdata(),
                                                              nib.load(SAMBA_init).affine,
                                                              nib.load(SAMBA_init_test_t).get_fdata(),
                                                              nib.load(SAMBA_init_test_t).affine, return_trans=True)
            else:
                transformed = translation.transform(nib.load(SAMBA_init_test_t).get_fdata())

            new_affine[:3, :3] = np.eye(3)

            SAMBA_init_nii = nib.Nifti1Image(transformed, new_affine)
            nib.save(SAMBA_init_nii, SAMBA_init_test)

            SAMBA_ref = os.path.join(SAMBA_inits, f'reference_file_c_isotropi.nii.gz')

            command_all = f"antsApplyTransforms -v 1 --float -d 3  -i {SAMBA_init_test} -o {SAMBA_postwarp} -r {SAMBA_ref} -n Linear -t {runno_to_MDT} [{affine_orig},0]  [{rigid},0]";
            os.system(command_all)

            if os.path.exists(SAMBA_postwarp):
                os.remove(SAMBA_init_test_t)

        else:
            print(f'Already wrote {SAMBA_postwarp}')

trk_preprocess_posttrans = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_posttrans.trk')
trk_preprocess_postrigid = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_postrigid.trk')
trk_preprocess_postrigid_affine = os.path.join(path_trk_tempdir,
                                               f'{subj}{str_identifier}_preprocess_postrigid_affine.trk')
trk_MDT_space = os.path.join(path_TRK_output, f'{subj}_MDT.trk')

final_img_exists = checkfile_exists_remote(trk_MDT_space, sftp)

if trk_to_MDT and (not final_img_exists or overwrite):

    subj_trk, trkexists = gettrkpath(path_TRK, subj, '_smallerTracks2mill', pruned=prune, verbose=False, sftp=sftp)

    _, exists = check_files([trans, rigid, runno_to_MDT], sftp)
    print('reaching point 1')

    # subj_dwi = os.path.join(path_DWI, f'{subj}_coreg_diff{ext}')
    subj_dwi = os.path.join(path_DWI, f'{subj}_subjspace_dwi{ext}')
    _, subj_affine, _, _, _ = load_nifti_remote(subj_dwi, sftp)

    SAMBA_init = os.path.join(SAMBA_inits, f'{subj}_dwi_masked{ext}')
    _, init_affine, _, _, _ = load_nifti_remote(SAMBA_init, sftp)

    affine_subj_to_prepro = get_affine_transform(subj_affine, np.eye(4))

    from ants.registration import resample_image_to_target
    from ants import image_write, image_read

    filepath_resampled_path = os.path.join(outpath, f'{subj}_subj_resampled{ext}')
    resampled = resample_image_to_target(image_read(subj_dwi), image_read(SAMBA_init))
    image_write(resampled, filepath_resampled_path)

    filepath_resampled_reoriented = os.path.join(outpath, f'{subj}_subj_resampled_reoriented{ext}')
    filepath_resampled_reoriented_rigid = os.path.join(outpath, f'{subj}_subj_resampled_rigidreoriented{ext}')
    affine_subj_to_init = get_affine_transform(subj_affine, init_affine)
    affine_subj_to_init[:3, 3] = [0, 0, 0]
    params = np.array(np.squeeze(np.reshape(affine_subj_to_init[:3, :3], [9, 1])).tolist() + [0.0, 0.0, 0.0])

    ants_transform_rigid = create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3,
                                                 parameters=params)

    affine_rigid_path = os.path.join(outpath, f'{subj}_subjtoinitaffinerigid.mat')
    if not os.path.exists(affine_rigid_path):
        write_transform(ants_transform_rigid, affine_rigid_path)

    cmd = f"antsApplyTransforms -v 1 --float -d 3  -i {filepath_resampled_path} -o {filepath_resampled_reoriented} -r {SAMBA_ref} -n Linear -t [{affine_rigid_path},0]";
    os.system(cmd)

    SAMBA_init = os.path.join(SAMBA_inits, f'{subj}_dwi_masked.nii.gz')
    SAMBA_init_nii = nib.load(SAMBA_init)
    SAMBA_init_data = SAMBA_init_nii.get_fdata()

    resampled_aff = nib.load(filepath_resampled_path).affine
    resampled_hdr = nib.load(filepath_resampled_path).header

    transformed, rigid_affine = rigid_reg(SAMBA_init_data, SAMBA_init_nii.affine, resampled.numpy(), resampled_aff)

    new_nii = nib.Nifti1Image(transformed, resampled_aff, resampled_hdr)
    # nib.save(new_nii,filepath_resampled_reoriented_rigid)

    if os.path.exists(filepath_resampled_reoriented):
        os.remove(filepath_resampled_reoriented)

    if os.path.exists(filepath_resampled_path):
        os.remove(filepath_resampled_path)

    print('reaching point 2')
    check_dif_ratio(path_TRK, subj, str_identifier, ratio, sftp)
    subj_trk, trkexists = gettrkpath(path_TRK, subj, '_smallerTracks2mill', pruned=prune, verbose=False, sftp=sftp)
    if not trkexists:
        txt = f'Could not find TRK file for subject {subj}'
        raise Exception(txt)
    _, exists = check_files([trans, rigid, runno_to_MDT], sftp)
    if np.any(exists == 0):
        raise Exception('missing transform file')
    #_, exists = check_files([affine_orig], sftp)
    if not os.path.exists(affine_orig):
        raise Exception('missing transform file')
    streamlines_data = load_trk_remote(subj_trk, 'same', sftp)
    streamlines_subj = streamlines_data.streamlines
    header = streamlines_data.space_attributes

    rigid_affine_apply = np.copy(rigid_affine)
    rigid_affine_apply[:3, :3] = np.linalg.inv(rigid_affine[:3, :3])
    rigid_affine_apply[:, 3] = -rigid_affine[:, 3]
    streamlines_posttrans = transform_streamlines(streamlines_subj, rigid_affine_apply)
    print('reaching point 3')
    if (not checkfile_exists_remote(trk_preprocess_posttrans, sftp) or overwrite) and save_temp_trk_files:
        save_trk_header(filepath=trk_preprocess_posttrans, streamlines=streamlines_posttrans, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp)

    rigid_struct = loadmat_remote(rigid, sftp)
    var_name = list(rigid_struct.keys())[0]
    rigid_ants = rigid_struct[var_name]
    rigid_mat = convert_ants_vals_to_affine(rigid_ants)

    # streamlines_posttrans, header_posttrans = unload_trk(trk_preprocess_posttrans)
    streamlines_postrigid = transform_streamlines(streamlines_posttrans, np.linalg.inv(rigid_mat))
    print('reaching point 4')
    if (not checkfile_exists_remote(trk_preprocess_postrigid, sftp) or overwrite) and save_temp_trk_files:
        save_trk_header(filepath=trk_preprocess_postrigid, streamlines=streamlines_postrigid, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp)

    # if os.path.exists(affine):
    #    affine_mat_s = read_affine_txt(affine)
    # else:
    #    cmd = f'ConvertTransformFile 3 {affine_orig} {affine} --matrix'
    #    os.system(cmd)
    #    affine_mat_s = read_affine_txt(affine)

    affine_struct = loadmat_remote(affine_orig, sftp)
    var_name = list(affine_struct.keys())[0]
    affine_ants = affine_struct[var_name]
    affine_mat = convert_ants_vals_to_affine(affine_ants)

    streamlines_postrigidaffine = transform_streamlines(streamlines_postrigid, np.linalg.inv(affine_mat))
    print('reaching point 5')
    if (not checkfile_exists_remote(trk_preprocess_postrigid_affine, sftp) or overwrite) and save_temp_trk_files:
        save_trk_header(filepath=trk_preprocess_postrigid_affine, streamlines=streamlines_postrigidaffine,
                        header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp)
        print('did this')
    print('finished')
    # streamlines_postrigidaffine, header_postrigidaffine = unload_trk(trk_preprocess_postrigid_affine)

    warp_data, warp_affine, _, _, _ = load_nifti_remote(runno_to_MDT, None)

    taf = time.perf_counter()

    mni_streamlines = transform_streamwarp(streamlines_postrigidaffine, SAMBA_MDT, warp_data)

    del streamlines_postrigidaffine

    twarp = time.perf_counter()
    print(f'added warp transform, took {twarp - taf:0.2f} seconds')

    save_trk_header(filepath=trk_MDT_space, streamlines=mni_streamlines, header=header,
                    affine=np.eye(4), verbose=verbose, sftp=sftp)
    tsave = time.perf_counter()
    print(f'Saved to {trk_MDT_space}, took {tsave - twarp:0.2f} seconds')

    tf = time.perf_counter()
    del mni_streamlines

    os.remove(subj_trk)

elif not final_img_exists or overwrite:
    print(f'{trk_MDT_space} already exists')
