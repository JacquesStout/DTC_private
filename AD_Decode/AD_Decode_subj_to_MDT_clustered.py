import os
from DTC.nifti_handlers.transform_handler import get_affine_transform, get_flip_affine, header_superpose, \
    recenter_nii_affine, \
    convert_ants_vals_to_affine, read_affine_txt, recenter_nii_save, add_translation, recenter_nii_save_test, \
    affine_superpose, get_affine_transform_test
import numpy as np
import nibabel as nib
from dipy.tracking.streamline import deform_streamlines, transform_streamlines

from DTC.tract_manager.tract_save import save_trk_header, convert_tck_to_trk

import socket
from DTC.tract_manager.tract_handler import gettrkpath
from DTC.tract_manager.DTC_manager import get_str_identifier
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from DTC.tract_manager.DTC_manager import check_dif_ratio
from DTC.file_manager.computer_nav import get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote, \
    checkfile_exists_remote, pickledump_remote, remote_pickle
import random
from DTC.tract_manager.tract_handler import transform_streamwarp
import time
from DTC.file_manager.argument_tools import parse_arguments
import sys, shutil
from ants.core.ants_transform_io import create_ants_transform, write_transform, read_transform
from DTC.nifti_handlers.transform_handler import rigid_reg, translation_reg
from DTC.tract_manager.tract_handler import reducetractnumber
from DTC.nifti_handlers.transform_handler import img_transform_exec
from dipy.io.streamline import load_tractogram, save_tractogram
from scipy.io import loadmat


def get_rotation_matrix(angle, axis):
    """Compute the rotation matrix for a 3D rotation.

    Args:
        angle (float): The rotation angle in radians.
        axis (int): The directional axis of rotation. 0 for x-axis, 1 for y-axis, and 2 for z-axis.

    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    if axis == 0:  # x-axis rotation
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cos_theta, -sin_theta],
                                    [0, sin_theta, cos_theta]])
    elif axis == 1:  # y-axis rotation
        rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                                    [0, 1, 0],
                                    [-sin_theta, 0, cos_theta]])
    elif axis == 2:  # z-axis rotation
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                    [sin_theta, cos_theta, 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Valid values are 0, 1, or 2.")

    return rotation_matrix


def load_matrix_in_any_format(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    elif ext == '.mat':
        # .mat are actually dictionnary. This function support .mat from
        # antsRegistration that encode a 4x4 transformation matrix.
        transfo_dict = loadmat(filepath)
        lps2ras = np.diag([-1, -1, 1])

        rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
        trans = transfo_dict['AffineTransform_float_3_3'][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))

    return data


project = 'AD_Decode'

test_mode = False

#subj = 'S02670'

if len(sys.argv[1])>6:
    subj_trk = sys.argv[1]
    subj = os.path.basename(subj_trk).split('_')[0]
    if not os.path.exists(subj_trk):
        txt = f'Could not find {subj_trk}'
        raise Exception(txt)
else:
    subj = sys.argv[1]

if test_mode:
    erase=True
    save_temp_trk_files = True
    save_temp_nii_files = True
    overwrite = True
    ratio = 100
    mainpath = '/Volumes/Data/Badea/Lab/'

else:
    erase = False
    save_temp_trk_files = False
    save_temp_nii_files = False
    overwrite = False
    ratio = 1
    mainpath = '/mnt/munin2/Badea/Lab/'

if 'santorini' in socket.gethostname():
    mainpath = '/Volumes/Data/Badea/Lab/'
else:
    mainpath = '/mnt/munin2/Badea/Lab/'

# temporarily removing "H29056" to recalculate it
ext = ".nii.gz"

username = None
passwd = None

sftp = None

inpath = os.path.join(mainpath, 'human/AD_Decode_trk_transfer')
outpath = os.path.join(mainpath, 'human/AD_Decode_trk_transfer')

path_TRK = os.path.join(inpath, 'TRK')
path_DWI = os.path.join(inpath, 'DWI')
path_DWI = os.path.join(mainpath,'../ADdecode.01/Analysis/DWI')

ref = "fa"
path_trk_tempdir = os.path.join(outpath, 'TRK_transition')
path_TRK_output = os.path.join(outpath, 'TRK_MDT')

path_DWI_temp = os.path.join(outpath, 'NII_toMDT_temp')
path_DWI_output = os.path.join(outpath, 'DWI_MDT')
# Get the values from DTC_launcher_ADDecode. Should probably create a single parameter file for each project one day

path_transforms = os.path.join(inpath, 'Transforms_farun')

mkcdir([outpath, path_TRK_output, path_DWI_output, path_transforms, path_DWI_temp], sftp)

stepsize = 2

trkroi = ["wholebrain"]
prune = True
cleanup = False
verbose = True
recenter = True

del_orig_files = True


contrasts = ['fa', 'md', 'rd', 'ad']
contrasts = ['fa']
# contrast = 'fa'
native_ref = ''


#SAMBA_MDT = '/Volumes/dusom_abadea_nas1/munin_js/VBM_backups/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_fa/faMDT_Control_n72_i6/median_images/MDT_dwi.nii.gz'
SAMBA_MDT = os.path.join(path_transforms, 'MDT_dwi.nii.gz')

print(f'running move for subject {subj}')

SAMBA_work_folder = os.path.join(mainpath, 'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work')

SAMBA_inits = os.path.join(SAMBA_work_folder,'preprocess', 'base_images')
#SAMBA_inits = '/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_perm_files/all_fa_SAMBA_init/'

final_template_run = os.path.join(SAMBA_work_folder, 'dwi', 'SyN_0p5_3_0p5_fa', 'faMDT_NoNameYet_n37_i6')
trans = os.path.join(SAMBA_work_folder, "preprocess", "base_images", "translation_xforms",
                     f"{subj}_0DerivedInitialMovingTranslation.mat")
rigid = os.path.join(SAMBA_work_folder, "dwi", f"{subj}_rigid.mat")
affine_orig = os.path.join(SAMBA_work_folder, "dwi", f"{subj}_affine.mat")
runno_to_MDT = os.path.join(final_template_run, "reg_diffeo", f"{subj}_to_MDT_warp.nii.gz")
MDT_to_runno = os.path.join(final_template_run, "reg_diffeo", f'MDT_to_{subj}_warp.nii.gz')
MDT_median_img = os.path.join(final_template_run, "median_images", f'MDT_fa.nii.gz')

#SAMBA_ref = os.path.join(SAMBA_inits, f'reference_file_c_isotropi.nii.gz')

SAMBA_ref = os.path.join(SAMBA_inits, f'reference_image_native_S01912.nii.gz')

MDT_median_img = os.path.join(final_template_run, "median_images", f'MDT_fa.nii.gz')
MDT_median_mif = os.path.join(final_template_run, "median_images", f'MDT_fa.mif')

nii_to_MDT = False
trk_to_MDT = True

if save_temp_trk_files:
    mkcdir(path_trk_tempdir, sftp)

if save_temp_trk_files:
    mkcdir(path_trk_tempdir, sftp)

SAMBA_subj_toRAS = False

str_identifier = get_str_identifier(stepsize, ratio, trkroi, type='mrtrix')

transform_to_init_mat_path = os.path.join(path_trk_tempdir, f'{subj}_subj_to_init.mat')

if nii_to_MDT or not os.path.exists(transform_to_init_mat_path):
    mkcdir(path_DWI_output)
    if 'rigid_orient' in locals():
        del rigid_orient

    for contrast in contrasts:

        SAMBA_postwarp = os.path.join(path_DWI_output, f'{subj}_{contrast}_to_MDT{ext}')

        SAMBA_subj = os.path.join(path_DWI, f'{subj}_subjspace_{contrast}.nii.gz')

        if SAMBA_subj_toRAS:
            SAMBA_subj_RAS = os.path.join(path_DWI_temp, f'{subj}_subjspace_{contrast}_RAS.nii.gz')
            img_transform_exec(SAMBA_subj, 'ARI', 'RAS', output_path=SAMBA_subj_RAS)
            SAMBA_subj = SAMBA_subj_RAS

        SAMBA_init_test_t = os.path.join(path_DWI_temp, f'{subj}_{contrast}_init_transition.nii.gz')
        SAMBA_init_test = os.path.join(path_DWI_temp, f'{subj}_{contrast}_init.nii.gz')
        SAMBA_init = os.path.join(SAMBA_inits, f'{subj}_fa_masked.nii.gz')

        rigid_reorient_path = os.path.join(path_DWI_temp, f'{subj}_subj_reorient.mat')

        import ants

        # fixed_image = ants.image_read(SAMBA_init)
        fixed_image = ants.image_read(SAMBA_init)
        moving_image = ants.image_read(SAMBA_subj)

        # Perform trans registration
        transform_rigid = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')

        # Apply the transformation to the moving image
        registered_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                                 transformlist=transform_rigid['fwdtransforms'])

        shutil.copy(transform_rigid['fwdtransforms'][0], transform_to_init_mat_path)

        # Save the registered image
        ants.image_write(registered_image, SAMBA_init_test)
        print(f'wrote {SAMBA_init_test}')
        overwrite = False
        print(f'overwrite is {overwrite}')
        """
        # Perform rigid registration
        transform_rigid = ants.registration(fixed=fixed_image, moving=ants.image_read(SAMBA_init_test_t), type_of_transform='Rigid')

        # Apply the transformation to the moving image
        registered_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                                 transformlist=transform_rigid['fwdtransforms'])

        # Save the registered image
        ants.image_write(registered_image, SAMBA_init_test)
        print(f'wrote {SAMBA_init_test}')
        """

        if not os.path.exists(SAMBA_postwarp) or overwrite:
            if not save_temp_nii_files:
                command_all = f"antsApplyTransforms -v 1 --float -d 3  -i {SAMBA_init_test} -o {SAMBA_postwarp} -r {SAMBA_ref} -n Linear -t {runno_to_MDT} [{affine_orig},0]  [{rigid},0]";
                os.system(command_all)
            else:
                rigid_temp_path = os.path.join(path_DWI_temp, f'{subj}_{contrast}_postrigid{ext}')
                affine_temp_path = os.path.join(path_DWI_temp, f'{subj}_{contrast}_postaffine{ext}')

                command_all = f"antsApplyTransforms -v 1 --float -d 3  -i {SAMBA_init_test} -o {rigid_temp_path} -r {SAMBA_ref} -n Linear -t [{rigid},0]";
                os.system(command_all)

                command_all = f"antsApplyTransforms -v 1 --float -d 3  -i {rigid_temp_path} -o {affine_temp_path} -r {SAMBA_ref} -n Linear -t [{affine_orig},0]";
                os.system(command_all)

                command_all = f"antsApplyTransforms -v 1 --float -d 3  -i {affine_temp_path} -o {SAMBA_postwarp} -r {SAMBA_ref} -n Linear -t {runno_to_MDT}";
                os.system(command_all)
                print(f'Wrote SAMBA postwarp to {SAMBA_postwarp}')
        else:
            print(f'Already wrote {SAMBA_postwarp}')

        if os.path.exists(SAMBA_postwarp) and os.path.exists(SAMBA_init_test_t) and erase:
            os.remove(SAMBA_init_test_t)
            # os.remove(SAMBA_init_test

############

"""
if ratio == 1:
    str_identifier = '_smallerTracks2mill'
elif ratio == 10:
    str_identifier = '_smallerTracks100thousand'
elif ratio == 100:
    str_identifier = '_smallerTracks10thousand'
else:
    raise Exception('unrecognizable ratio value')
"""

str_identifier = get_str_identifier(stepsize, ratio, trkroi, type='mrtrix')

prune = False

trk_preprocess_init = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_trk_init.trk')
trk_preprocess_posttrans = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_posttrans.trk')
trk_preprocess_postrigid = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_preprocess_postrigid.trk')
trk_preprocess_postrigid_affine = os.path.join(path_trk_tempdir,
                                               f'{subj}{str_identifier}_preprocess_postrigid_affine.trk')
trk_MDT_space = os.path.join(path_TRK_output, f'{subj}_MDT{str_identifier}.trk')

warp_path = MDT_to_runno
# warp_path = runno_to_MDT

# final_img_exists = checkfile_exists_remote(trk_MDT_space)
final_img_exists = checkfile_exists_remote(trk_MDT_space, sftp)
print('point 0')
print(trk_MDT_space)

if test_mode:
    overwrite = True

if trk_to_MDT and (not final_img_exists or overwrite):

    """
    if ratio == 1:
        str_identifier = '_smallerTracks2mill'
    elif ratio == 10:
        str_identifier = '_smallerTracks100thousand'
    elif ratio == 100:
        str_identifier = '_smallerTracks10thousand'
    else:
        raise Exception('unrecognizable ratio value')
    """
    prune = False

    if 'subj_trk' not in locals():
        subj_trk, trkexists = gettrkpath(path_TRK, subj, str_identifier, pruned=prune, verbose=False, sftp=sftp)
    else:
        trkexists = os.path.exists(subj_trk)

    print('point 1')
    print(subj_trk)

    if not trkexists:
        subj_trk, trkexists = gettrkpath(path_TRK, subj, '_smallerTracks2mill', pruned=prune, verbose=False, sftp=sftp)
        smallertrkpath = os.path.join(path_TRK, subj + str_identifier + '.trk')
        if trkexists and ratio != 1:
            reducetractnumber(subj_trk, smallertrkpath, getdata=False, ratio=ratio,
                              return_affine=False, verbose=False)
            subj_trk = smallertrkpath
        else:
            txt = f'Could not find TRK file for subject {subj}'
            raise Exception(txt)

    try:
        streamlines_data = load_trk_remote(subj_trk, 'same', sftp)
    except:
        txt = f'Could not load file found {subj_trk}'
        raise Exception(txt)

    streamlines_subj = streamlines_data.streamlines
    header = streamlines_data.space_attributes
    affine_subj_trk = streamlines_data.affine
    """
    affine_reorient[3, 3] = 1
    streamlines_postreorient = transform_streamlines(streamlines_subj, affine_reorient)

    trk_preprocess_reoriented = os.path.join(path_trk_tempdir, f'{subj}{str_identifier}_trk_init_transition.trk')

    if (not checkfile_exists_remote(trk_preprocess_reoriented, sftp) or overwrite) and save_temp_trk_files:
        save_trk_header(filepath=trk_preprocess_reoriented, streamlines=streamlines_postreorient, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp)
    """

    transform_to_init = transform_to_init_mat_path
    toinit_reorient_struct = loadmat_remote(transform_to_init, sftp)

    var_name = list(toinit_reorient_struct.keys())[0]
    toinit_ants = toinit_reorient_struct[var_name]
    # toinit_mat = convert_ants_vals_to_affine(toinit_ants)
    # toinit_mat = convert_ants_vals_to_affine(toinit_ants)
    toinit_mat = load_matrix_in_any_format(transform_to_init_mat_path)

    toinit_trans_mat = np.eye(4)
    toinit_trans_mat[:, 3] = toinit_mat[:, 3]
    # toinit_trans_mat[2,3] = toinit_trans_mat[2,3]+4
    streamlines_init_1 = transform_streamlines(streamlines_subj, np.linalg.inv(toinit_trans_mat))

    # streamlines_init = transform_streamlines(streamlines_postreorient, np.linalg.inv(toinit_mat))
    toinit_rot_mat = np.eye(4)
    toinit_rot_mat[:3, :3] = toinit_mat[:3, :3]
    streamlines_init = transform_streamlines(streamlines_init_1, np.linalg.inv(toinit_rot_mat))

    """
    toinit_inv_mat = np.linalg.inv(toinit_mat)
    toinit_trans_mat = np.eye(4)
    toinit_trans_mat[:,3] = toinit_inv_mat[:,3]
    streamlines_init_1 = transform_streamlines(streamlines_subj, (toinit_trans_mat))

    toinit_rot_mat = np.eye(4)
    toinit_rot_mat[:3,:3] = toinit_inv_mat[:3,:3]
    streamlines_init = transform_streamlines(streamlines_init_1, (toinit_rot_mat))
    trk_preprocess_init = '/Volumes/Data/Badea/Lab/human/AD_Decode_trk_transfer/TRK_transition/S01912_smallerTracks20000_trk_init_test.trk'
    """

    if (not checkfile_exists_remote(trk_preprocess_init, sftp) or overwrite) and save_temp_trk_files:
        save_trk_header(filepath=trk_preprocess_init, streamlines=streamlines_init, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp)

    print('reaching point 2')

    rigid_struct = loadmat_remote(rigid, sftp)
    var_name = list(rigid_struct.keys())[0]
    rigid_ants = rigid_struct[var_name]
    rigid_mat = convert_ants_vals_to_affine(rigid_ants)

    # streamlines_posttrans, header_posttrans = unload_trk(trk_preprocess_posttrans)
    streamlines_postrigid = transform_streamlines(streamlines_init, np.linalg.inv(rigid_mat))
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
    if (not checkfile_exists_remote(trk_preprocess_postrigid_affine, sftp) or overwrite):
        save_trk_header(filepath=trk_preprocess_postrigid_affine, streamlines=streamlines_postrigidaffine,
                        header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp)
        print('did this')
    print('finished')
    # streamlines_postrigidaffine, header_postrigidaffine = unload_trk(trk_preprocess_postrigid_affine)

    # warp_data, warp_affine, _, _, _ = load_nifti_remote(warp_path, None)

    taf = time.perf_counter()

    # mni_streamlines = transform_streamwarp(streamlines_postrigidaffine, MDT_median_img, warp_data)

    # first convert nii to fa
    # MDT_median_img = os.path.join(final_template_run, "median_images", f'MDT_fa.nii.gz')
    # MDT_median_mif = os.path.join(final_template_run, "median_images", f'MDT_fa.mif')

    if not os.path.exists(MDT_median_mif) or overwrite:
        command = f'mrconvert  {MDT_median_img} {MDT_median_mif} -force'
        os.system(command)

    inv_identity_warp_basepath = os.path.join(path_trk_tempdir, f'{subj}_inv_identity_warp')
    inv_mrtrix_warp_basepath = os.path.join(path_trk_tempdir, f'{subj}_inv_mrtrix_warp')
    inv_warp_corrected_path = os.path.join(path_trk_tempdir, f'{subj}_inv_mrtrix_warp_corrected.mif')

    # make identity warp
    command = f'warpinit {MDT_median_mif} {inv_identity_warp_basepath}[].nii -force'
    if not os.path.exists(inv_identity_warp_basepath) or overwrite:
        os.system(command)

    # fill into the identities
    for i in np.arange(3):
        command = f'antsApplyTransforms -d 3 -e 0 -i {inv_identity_warp_basepath}{i}.nii -o {inv_mrtrix_warp_basepath}{i}.nii -r {MDT_median_img} -t {warp_path} --default-value 2147483647'
        if not os.path.exists(f'{inv_mrtrix_warp_basepath}{i}.nii') or overwrite:
            os.system(command)

    # combine the 3 filled warps
    command = f'warpcorrect {inv_mrtrix_warp_basepath}[].nii {inv_warp_corrected_path} -marker 2147483647 -force'
    os.system(command)

    # apply the transform
    cc_trk = load_tractogram(trk_preprocess_postrigid_affine, 'same', bbox_valid_check=False)
    tck_preprocess_postrigid_affine = trk_preprocess_postrigid_affine.replace('.trk', '.tck')
    save_tractogram(cc_trk, tck_preprocess_postrigid_affine, bbox_valid_check=False)

    if not save_temp_trk_files:
        os.remove(trk_preprocess_postrigid_affine)

    tck_MDT_space = trk_MDT_space.replace('.trk', '.tck')
    command = f'tcktransform {tck_preprocess_postrigid_affine} {inv_warp_corrected_path} {tck_MDT_space} -force'
    os.system(command)

    os.remove(tck_preprocess_postrigid_affine)
    convert_tck_to_trk(tck_MDT_space, trk_MDT_space, MDT_median_img)

    os.remove(tck_MDT_space)
    """
    vox_size = nib.load(f'/mnt/munin2/Badea/Lab/human/AD_Decode_trk_transfer/DWI_MDT/{subj}_fa_to_MDT.nii.gz').header.get_zooms()[0]
    target_isocenter = np.diag(np.array([-vox_size, vox_size, vox_size, 1]))

    from dipy.tracking.streamline import deform_streamlines
    mni_streamlines = deform_streamlines(streamlines_postrigidaffine, deform_field=warp_data,
    stream_to_current_grid=target_isocenter,
    current_grid_to_world=affine_subj_trk, stream_to_ref_grid=target_isocenter,
    ref_grid_to_world=np.eye(4))
    """
    
    if not save_temp_trk_files:
        #os.remove(trk_preprocess_postrigid_affine)
        os.remove(inv_warp_corrected_path)
        for i in np.arange(3):
            os.remove(f'{inv_identity_warp_basepath}{i}.nii')
            os.remove(f'{inv_mrtrix_warp_basepath}{i}.nii')
    del streamlines_postrigidaffine

    """
    twarp = time.perf_counter()
    print(f'added warp transform, took {twarp - taf:0.2f} seconds')
    trk_MDT_space = os.path.join(path_TRK_output, f'{subj}_MDT{str_identifier}_6.trk')
    save_trk_header(filepath=trk_MDT_space, streamlines=mni_streamlines, header=header,
                    affine=np.eye(4), verbose=verbose, sftp=sftp)
    """
    tsave = time.perf_counter()

    print(f'Saved to {trk_MDT_space}, took {tsave - taf:0.2f} seconds')

    tf = time.perf_counter()

    if del_orig_files:
        os.remove(subj_trk)

elif final_img_exists:
    print(f'{trk_MDT_space} already exists')
