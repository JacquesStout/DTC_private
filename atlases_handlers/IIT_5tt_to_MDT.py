import os,glob, shutil
from DTC.nifti_handlers.transform_handler import img_transform_exec, header_superpose
from DTC.file_manager.file_tools import buildlink, mkcdir
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import create_backport_labels, create_MDT_labels, atlas_to_MDT_transfer
import nibabel as nib
import numpy as np
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import create_backport_labels


input_folder_IIT = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_bundles_masks'
SAMBA_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun'
atlas_folder = '/Volumes/Data/Badea/Lab/atlases'

final_template_run = os.path.join(f'{SAMBA_folder}-work','dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6')

overwrite=False

transition_folder = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_bundles_masks_transtion_temp'

mkcdir(transition_folder)

input_file_path = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IIT_fornix_fixed_5tt_file_for_ACT_tractography.nii.gz'

orient_orig = 'LPI'
orient_init = 'RPI'


subjects_need5ttgen = ['S02224', 'S02230', 'S02266', 'S02289', 'S02361', 'S02469', 'S02490', 'S02523', 'S02745', 'S03321', 'S03867', 'S04738', 'S04696', 'S01620', 'S01621']

removed_subjs = ['S02230','S02490','S02523', 'S02745']

subjects_need5ttgen = [subject for subject in subjects_need5ttgen if subject not in removed_subjs]

mainpath = '/Volumes/Data/Badea/Lab/mouse'
label_name = 'IITmean'
label_name_samba = f'IITmean_{orient_init}'
project_name = 'VBM_21ADDecode03_IITmean_RPI_fullrun'
prep_folder = '/Volumes/Data/Badea/Lab/mouse/ADDeccode_symlink_pool2'
prep_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/'
atlas_labels = '/Volumes/Data/Badea/Lab/atlas/IITmean_RPI/IITmean_RPI_labels.nii.gz'
headfile = '/Volumes/Data/Badea/Lab/samba_startup_cache/jas297_SAMBA_ADDecode.headfile'
identifier_SAMBA_folder = 'faMDT_NoName'

target_atlas_path = os.path.join(atlas_folder,f'{label_name}_{orient_init}/{label_name}_{orient_init}_fa.nii.gz')

label_orient_name = '_'.join([label_name,orient_init])

for subj in subjects_need5ttgen:

    out_dir_base = os.path.join(mainpath, f"{project_name}-results","connectomics")
    output_folder_MDT = os.path.join(out_dir_base,subj)
    final_labels = os.path.join(output_folder_MDT,f"{subj}_5tt_nocoreg.nii.gz")
    final_labels_actprepare = os.path.join(prep_folder,f"{subj}_5tt_nocoreg.nii.gz")

    #output_file_path = os.path.join(output_folder_MDT,os.path.basename(input_file_path))

    if not os.path.exists(final_labels_actprepare) or overwrite:

        basename = os.path.basename(input_file_path)

        if orient_orig!=orient_init:

            init_orient_prehead = os.path.join(transition_folder,basename.replace('.nii.gz',f'_{orient_init}_preheadermod.nii.gz'))
            init_orient = os.path.join(transition_folder,basename.replace('.nii.gz',f'_{orient_init}.nii.gz'))
            if not os.path.exists(init_orient) or overwrite:
                if not os.path.exists(init_orient) or overwrite:
                    img_transform_exec(input_file_path, orient_orig, orient_init, output_path=init_orient_prehead, recenter_test=False)
                header_superpose(target_atlas_path, init_orient_prehead, outpath=init_orient, verbose=False)


        MDT_ref = os.path.join(final_template_run, 'median_images',f"MDT_dwi.nii.gz")
        MDT_to_atlas_affine = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"MDT_*_to_{label_orient_name}_affine.mat")
        atlas_to_MDT = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"{label_orient_name}_to_MDT_warp.nii.gz")

        img4d = nib.load(init_orient)
        data = img4d.get_fdata()
        split_volumes = np.split(data, data.shape[-1], axis=-1)
        for i, volume in enumerate(split_volumes):
            new_img = nib.Nifti1Image(volume, img4d.affine, img4d.header)
            output_split_name = os.path.join(transition_folder,f'{basename.replace(".nii.gz","")}_volume_{i + 1}.nii.gz')
            if not os.path.exists(output_split_name) or overwrite:
                nib.save(new_img, output_split_name)

        for i in np.arange(np.shape(data)[3]):
            output_split_name = os.path.join(transition_folder,f'{basename.replace(".nii.gz","")}_volume_{i + 1}.nii.gz')
            output_split_MDT_name = os.path.join(output_folder_MDT,f'{basename.replace(".nii.gz","")}_volume_{i + 1}.nii.gz')
            #atlas_to_MDT_transfer(output_split_name, output_split_MDT_name, MDT_ref, MDT_to_atlas_affine, atlas_to_MDT, dimension = 3)
            if not os.path.exists(output_split_MDT_name) or overwrite:
                create_backport_labels(subj, mainpath, project_name, prep_folder, output_split_name, label_name = label_name_samba,
                                       headfile=headfile, overwrite=overwrite, verbose=True, final_labels = output_split_MDT_name,
                                       identifier=identifier_SAMBA_folder, shorten = False)
        split_paths = []
        for i in np.arange(np.shape(data)[3]):
            output_split_MDT_name = os.path.join(output_folder_MDT,
                                                 f'{basename.replace(".nii.gz", "")}_volume_{i + 1}.nii.gz')
            split_paths.append(output_split_MDT_name)

        nib_MDT_default = nib.load(split_paths[0])
        header = nib_MDT_default.header
        affine = nib_MDT_default.affine

        combined_array = np.zeros(list(nib_MDT_default.shape)+[np.shape(data)[3]])
        for i,path in enumerate(split_paths):
            combined_array[:,:,:,i] = nib.load(path).get_fdata()

        recombined_img = nib.Nifti1Image(combined_array, affine, header)

        nib.save(recombined_img, final_labels)

        shutil.copy(final_labels,final_labels_actprepare)

