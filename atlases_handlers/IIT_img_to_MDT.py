import os,glob
from DTC.nifti_handlers.transform_handler import img_transform_exec, header_superpose
from DTC.file_manager.file_tools import buildlink, mkcdir
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import create_backport_labels, create_MDT_labels, atlas_to_MDT_transfer

input_folder_IIT = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_bundles_masks'
SAMBA_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun'
output_folder_MDT = os.path.join(f'{SAMBA_folder}-work', 'dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/labels_MDT/atlas_wm_MDT_masks')
atlas_folder = '/Volumes/Data/Badea/Lab/atlases'

final_template_run = os.path.join(f'{SAMBA_folder}-work','dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6')

overwrite=False

transition_folder = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_bundles_masks_transtion_temp'

mkcdir(transition_folder)

input_files_IIT = glob.glob(os.path.join(input_folder_IIT,'*.nii.gz'))

orient_orig = 'LPI'
orient_init = 'RPI'

label_name = 'IITmean'

target_atlas_path = os.path.join(atlas_folder,f'{label_name}_{orient_init}/{label_name}_{orient_init}_fa.nii.gz')

label_orient_name = '_'.join([label_name,orient_init])

for input_file_path in input_files_IIT:

    output_file_path = os.path.join(output_folder_MDT,os.path.basename(input_file_path))

    if not os.path.exists(output_file_path) or overwrite:

        basename = os.path.basename(input_file_path)

        if orient_orig!=orient_init:

            init_orient_prehead = os.path.join(transition_folder,basename.replace('.nii.gz',f'_{orient_init}_preheadermod.nii.gz'))
            init_orient = os.path.join(transition_folder,basename.replace('.nii.gz',f'_{orient_init}.nii.gz'))
            if not os.path.exists(init_orient):
                if not os.path.exists(init_orient):
                    img_transform_exec(input_file_path, orient_orig, orient_init, output_path=init_orient_prehead, recenter_test=False)
                header_superpose(target_atlas_path, init_orient_prehead, outpath=init_orient, verbose=False)


        MDT_ref = os.path.join(final_template_run, 'median_images',f"MDT_dwi.nii.gz")
        MDT_to_atlas_affine = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"MDT_*_to_{label_orient_name}_affine.mat")
        atlas_to_MDT = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"{label_orient_name}_to_MDT_warp.nii.gz")

        atlas_to_MDT_transfer(init_orient, output_file_path, MDT_ref, MDT_to_atlas_affine, atlas_to_MDT)