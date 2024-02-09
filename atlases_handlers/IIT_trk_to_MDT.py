import os,glob,ants,shutil
from DTC.nifti_handlers.transform_handler import img_transform_exec, header_superpose, label_rounder
from DTC.file_manager.file_tools import buildlink, mkcdir
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import create_backport_labels, create_MDT_labels, atlas_to_MDT_transfer
from DTC.file_manager.computer_nav import load_matrix_in_any_format, load_nifti_remote, load_trk_remote, loadmat_remote
import numpy as np
from dipy.tracking.streamline import deform_streamlines, transform_streamlines
from DTC.tract_manager.tract_save import save_trk_header, convert_tck_to_trk
from DTC.nifti_handlers.transform_handler import header_superpose, convert_ants_vals_to_affine

input_folder_IIT = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_bundles_masks'
input_folder_trk_IIT = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_tracks/'
SAMBA_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun'
output_folder_MDT = os.path.join(f'{SAMBA_folder}-work', 'dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/labels_MDT/atlas_wm_trks')
atlas_folder = '/Volumes/Data/Badea/Lab/atlases'

final_template_run = os.path.join(f'{SAMBA_folder}-work','dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6')

overwrite=False

transition_folder = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_trk_transfer'

mkcdir([transition_folder,output_folder_MDT])

input_files_IIT = glob.glob(os.path.join(input_folder_trk_IIT,'*.trk'))

orient_orig = 'LPI'
orient_init = 'RPI'

label_name = 'IITmean'

target_atlas_path = os.path.join(atlas_folder,f'{label_name}_{orient_init}/{label_name}_{orient_init}_fa.nii.gz')

label_orient_name = '_'.join([label_name,orient_init])

label=True

contrast = 'fa'

transform_mats_path = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/LPI_to_RPI_transform.mat'

if not os.path.exists(transform_mats_path):
    IITMDT_file_orig_path = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IITcv_t1.nii.gz'
    IITMDT_file_RPI_path = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IITcv_t1_RPI.nii.gz'
    file_test_path = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IITcv_t1_RPI_test.nii.gz'

    if not os.path.exists(IITMDT_file_RPI_path):
        init_orient_prehead = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IITcv_t1_preheader.nii.gz'
        if not os.path.exists(init_orient_prehead):
            img_transform_exec(IITMDT_file_orig_path, orient_orig, orient_init, output_path=init_orient_prehead,
                               recenter_test=False)
        header_superpose(target_atlas_path, init_orient_prehead, outpath=init_orient, verbose=False)

    fixed_image = ants.image_read(IITMDT_file_RPI_path)
    moving_image = ants.image_read(IITMDT_file_orig_path)

    # Perform trans registration
    transform_rigid = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')

    # Apply the transformation to the moving image
    registered_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                             transformlist=transform_rigid['fwdtransforms'])

    shutil.copy(transform_rigid['fwdtransforms'][0], transform_mats_path)

    # Save the registered image
    ants.image_write(registered_image, file_test_path)

save_temp_trk_files = True

MDT_median_img = os.path.join(final_template_run, "median_images", f'MDT_fa.nii.gz')
MDT_median_mif = os.path.join(final_template_run, "median_images", f'MDT_fa.mif')

if not os.path.exists(MDT_median_mif) or overwrite:
    command = f'mrconvert  {MDT_median_img} {MDT_median_mif} -force'
    os.system(command)

for input_file_path in input_files_IIT:

    output_file_path = os.path.join(output_folder_MDT,os.path.basename(input_file_path))

    if not os.path.exists(output_file_path) or overwrite:

        print(f'Running for label trks of {os.path.basename(input_file_path).split(".")[0]}')

        output_transition_test = os.path.join(transition_folder,os.path.basename(input_file_path).replace('.trk','_test.trk'))
        output_transition_test_tck = output_transition_test.replace('.trk','.tck')
        output_transition_postwarp_tck = os.path.join(transition_folder,os.path.basename(input_file_path).replace('.trk','_postwarp.tck'))
        output_transition_postwarp_trk = os.path.join(transition_folder,os.path.basename(input_file_path).replace('.trk','_postwarp.trk'))


        basename = os.path.basename(input_file_path)
        trk_base = load_trk_remote(input_file_path,'same')
        streamlines = trk_base.streamlines
        header = trk_base.space_attributes

        if orient_orig!=orient_init:

            toinit_reorient_struct = loadmat_remote(transform_mats_path, None)

            var_name = list(toinit_reorient_struct.keys())[0]
            toinit_ants = toinit_reorient_struct[var_name]

            toinit_mat = load_matrix_in_any_format(transform_mats_path)

            toinit_trans_mat = np.eye(4)
            toinit_trans_mat[:, 3] = toinit_mat[:, 3]
            streamlines_trans_1 = transform_streamlines(streamlines, np.linalg.inv(toinit_trans_mat))

            toinit_rot_mat = np.eye(4)
            toinit_rot_mat[:3, :3] = toinit_mat[:3, :3]
            streamlines_reorient = transform_streamlines(streamlines_trans_1, np.linalg.inv(toinit_rot_mat))

            if not (os.path.exists(output_transition_test) or overwrite) and save_temp_trk_files:
                save_trk_header(filepath=output_transition_test, streamlines=streamlines_reorient, header=header,
            affine=np.eye(4), verbose=False, sftp=None)

        from dipy.io.streamline import load_tractogram, save_tractogram

        MDT_to_atlas = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"MDT_to_{label_orient_name}_warp.nii.gz")

        inv_identity_warp_basepath = os.path.join(transition_folder, f'{label_orient_name}_inv_identity_warp')
        inv_mrtrix_warp_basepath = os.path.join(transition_folder, f'{label_orient_name}_inv_mrtrix_warp')
        inv_warp_corrected_path = os.path.join(transition_folder, f'{label_orient_name}_inv_mrtrix_warp_corrected.mif')

        # make identity warp
        command = f'warpinit {MDT_median_mif} {inv_identity_warp_basepath}[].nii -force'
        if not os.path.exists(inv_identity_warp_basepath) or overwrite:
            os.system(command)

        # fill into the identities
        for i in np.arange(3):
            command = f'antsApplyTransforms -d 3 -e 0 -i {inv_identity_warp_basepath}{i}.nii -o {inv_mrtrix_warp_basepath}{i}.nii -r {MDT_median_img} -t {MDT_to_atlas} --default-value 2147483647'
            if not os.path.exists(f'{inv_mrtrix_warp_basepath}{i}.nii') or overwrite:
                os.system(command)

        # combine the 3 filled warps
        if not os.path.exists(inv_warp_corrected_path):
            command = f'warpcorrect {inv_mrtrix_warp_basepath}[].nii {inv_warp_corrected_path} -marker 2147483647 -force'
            os.system(command)

        from DTC.tract_manager.tract_save import convert_trk_to_tck

        if not os.path.exists(output_transition_test_tck):
            convert_trk_to_tck(output_transition_test, output_transition_test_tck)

        if not os.path.exists(output_transition_postwarp_tck):
            command = f'tcktransform {output_transition_test_tck} {inv_warp_corrected_path} {output_transition_postwarp_tck} -force'
            os.system(command)

        #os.remove(tck_preprocess_postrigid_affine)
        if not os.path.exists(output_transition_postwarp_trk):
            convert_tck_to_trk(output_transition_postwarp_tck, output_transition_postwarp_trk, MDT_median_img)

        #os.remove(tck_MDT_space)

        if not save_temp_trk_files:
            # os.remove(trk_preprocess_postrigid_affine)
            os.remove(inv_warp_corrected_path)
            for i in np.arange(3):
                os.remove(f'{inv_identity_warp_basepath}{i}.nii')
                os.remove(f'{inv_mrtrix_warp_basepath}{i}.nii')


        #MDT_ref = os.path.join(final_template_run, 'median_images',f"MDT_dwi.nii.gz")
        #MDT_to_atlas_affine = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"MDT_*_to_{label_orient_name}_affine.mat")
        #atlas_to_MDT = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"{label_orient_name}_to_MDT_warp.nii.gz")
        #MDT_to_atlas = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"MDT_to_{label_orient_name}_RPI_warp.nii.gz")
        #atlas_to_MDT_transfer(init_orient, output_file_path, MDT_ref, MDT_to_atlas_affine, atlas_to_MDT)

        MDT_to_atlas_affine = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"MDT_{contrast}_to_{label_orient_name}_affine.mat")

        trk_postwarp = load_trk_remote(output_transition_postwarp_trk,'same')
        streamlines_postwarp = trk_postwarp.streamlines
        header = trk_base.space_attributes

        affine_struct = loadmat_remote(MDT_to_atlas_affine, None)
        var_name = list(affine_struct.keys())[0]
        affine_ants = affine_struct[var_name]
        affine_mat = convert_ants_vals_to_affine(affine_ants)

        streamlines_MDT = transform_streamlines(streamlines_postwarp, affine_mat)

        if not os.path.exists(output_file_path):
            save_trk_header(filepath=output_file_path, streamlines=streamlines_MDT,
                            header=header,
                            affine=np.eye(4), verbose=False, sftp=None)
            print('did this')
        print('finished')