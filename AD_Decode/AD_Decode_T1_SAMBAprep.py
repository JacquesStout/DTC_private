
from DTC.nifti_handlers.transform_handler import img_transform_exec, header_superpose
from DTC.file_manager.computer_nav import make_temppath
import os, glob
import numpy as np
import ants
from ants.utils.iMath import multiply_images

contrast = 'T1'
overwrite=False

#input_folder = f'/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_ad_decode/{contrast}_nii_gz'
input_folder = f'/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/'
SAMBA_folder = '/Volumes/Data/Badea/Lab/mouse/ADDeccode_symlink_pool2'
temp_folder = f'/Volumes/Data/Badea/ADdecode.01/Analysis/DWI_temp/'

subj_files_list = glob.glob(os.path.join(input_folder,f'*{contrast}.nii.gz'))
subj_list = [os.path.basename(subj_file)[:6] for subj_file in subj_files_list]

target_path = f'/Volumes/Data/Badea/Lab/mouse/ADDeccode_shortcut_pool/S02654_{contrast}.nii.gz'
target_path = f'/Volumes/Data/Badea/Lab/mouse/ADDeccode_shortcut_pool/S02654_fa.nii.gz'

base_image_folder = f'/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/preprocess/base_images/'

overwrite=False

for subj in subj_list:

    subjspace_T1 = os.path.join(input_folder,f'{subj}_T1.nii.gz')

    subjspace_dwi = os.path.join(input_folder,f'{subj}_subjspace_dwi.nii.gz')

    subjspace_mask = os.path.join(input_folder,f'{subj}_subjspace_mask.nii.gz')

    SAMBA_mask = os.path.join(base_image_folder,f'{subj}_mask.nii.gz')

    T1_reggedtodwi = os.path.join(temp_folder,f'{subj}_T1_regged.nii.gz')

    T1_transform_cut_path = os.path.join(temp_folder,f'T1_to_dwi_')
    T1_transform_path = os.path.join(temp_folder,f'T1_to_dwi_0GenericAffine.mat')

    baseimage_T1unmasked_path = os.path.join(temp_folder, f'{subj}_T1.nii.gz')
    baseimage_T1masked_path = os.path.join(base_image_folder, f'{subj}_T1_masked.nii.gz')


    if not os.path.exists(baseimage_T1masked_path) or overwrite:
        # fixed_image = ants.image_read(SAMBA_init)

        fa_subj_path = f'/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/{subj}_subjspace_fa.nii.gz'
        fa_init_path = f'/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/preprocess/base_images/{subj}_fa_masked.nii.gz'
        fa_test_path = f'/Volumes/Data/Badea/ADdecode.01/Analysis/DWI_MDT_test/{subj}_fa_init_test.nii.gz'

        try:
            fixed_image = ants.image_read(fa_init_path)
            moving_image = ants.image_read(fa_subj_path)
        except ValueError:
            print(f'Could not find subject {subj}, continue')
            continue
        if not os.path.exists(T1_transform_path):
            command = f'antsRegistration -v 1 -d 3 -m Mattes[{subjspace_dwi}, {subjspace_T1}] -t affine[0.1] -c [3000x3000x0x0, 1e-8, 20] -s 4x2x1x0.5vox -f 6x4x2x1 -u 1 -z 1 -l 1 -x {subjspace_mask} -o {T1_transform_cut_path}'
            os.system(command)

        if not os.path.exists(T1_reggedtodwi):
            command = f'antsApplyTransforms -d 3 -e 0 -i {subjspace_T1} -r {subjspace_dwi} -u float -o {T1_reggedtodwi} -t {T1_transform_path}'
            os.system(command)

        if not os.path.exists(baseimage_T1unmasked_path):
            # Perform trans registration
            transform_rigid = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')

            # Apply the transformation to the moving image
            registered_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                                     transformlist=transform_rigid['fwdtransforms'])

            # Save the registered image
            #ants.image_write(registered_image, fa_test_path)

            t1_image = ants.image_read(T1_reggedtodwi)

            registered_image_t1 = ants.apply_transforms(fixed=fixed_image, moving=t1_image,
                                                        transformlist=transform_rigid['fwdtransforms'])

            ants.image_write(registered_image_t1, baseimage_T1unmasked_path)

        unmasked_T1 = ants.image_read(baseimage_T1unmasked_path)
        mask = ants.image_read(SAMBA_mask)

        #masked_T1 = ants.image_math(unmasked_T1, "multiply_images", mask)
        masked_T1 = multiply_images(unmasked_T1,mask)

        ants.image_write(masked_T1, baseimage_T1masked_path)

        print(f'Finished for subject {subj}')
    else:
        print(f'Already wrote the init file of {subj} for {contrast}')
