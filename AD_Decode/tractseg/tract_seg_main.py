import os,shutil,glob,ants
from scipy.io import loadmat
import numpy as np
import sys

#DWI_folder = '/mnt/badea/ADdecode.01/Analysis/DWI/'
#tractseg_folder = '/home/alex/TractSeg_project/'


def ants_bvec_rotation(inputbvecs,transform_mat,outputbvecs):
    trans = loadmat(transform_mat)
    matrix = trans['AffineTransform_float_3_3'][:9].reshape((3,3))

    #loading bvecs
    bvecs = np.genfromtxt(inputbvecs)
    #heuristic to have bvecs shape 3xN
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T

    # Rotating bvecs
    newbvecs = np.dot(matrix, bvecs)

    # saving bvecs
    np.savetxt(outputbvecs, newbvecs,fmt='%.6f')


DWI_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/'
#DWI_folder = '/mnt/badea/ADdecode.01/Analysis/DWI/'
tractseg_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/'
#tractseg_folder = '/home/alex/TractSeg_project/'
template_MNI = '/Users/jas/Downloads/MNI_FA_template.nii.gz'
#template_MNI = '~/anaconda3/lib/python3.11/site-packages/tractseg/resources/MNI_FA_template.nii.gz'

tractseg_inputs_folder = os.path.join(tractseg_folder,'TractSeg_inputs')
tractseg_outputs_folder = os.path.join(tractseg_folder,'TractSeg_outputs')

if not os.path.exists(tractseg_inputs_folder): os.mkdir(tractseg_inputs_folder)
if not os.path.exists(tractseg_outputs_folder): os.mkdir(tractseg_outputs_folder)

overwrite=False

subject = 'S00775'
files_subj = glob.glob(os.path.join(DWI_folder,'*subjspace_coreg.nii.gz'))
subjects = [os.path.basename(file_path)[:6] for file_path in files_subj]
print(subjects)
FA_type = 'calc_FA'

testmode=False
verbose = True

"""
subjects = ['S03265', 'S03293', 'S03308', 'S03321', 'S03343', 'S03350', 'S03378', 'S03391',
'S03394', 'S03847', 'S03866', 'S03867', 
'S03889', 'S03890', 'S03896', 'S00775', 'S04491', 'S04493', 'S01412', 'S04526', 'S01470', 'S01619', 'S01621', 'S04696', 'S04738']
"""

subjects_fixedangle = ['S01412','S02266','S02410','S02421','S02954','S03225','S03350','S03394','S03896','S04491','S04493','S04696','S04738']
subjects_toinvestigate = ['S01621']
"""
subjects_f = []
for subject in subjects:
	if subject not in subjects_toremove:
		subjects_f.append(subject)
"""
subjects = ['S04491','S04493','S04696','S04738']

ref_subj = 'S02390'

for subject in subjects:
    overwrite=False

    print(f'Starting for subject {subject}')

    diff_subjspace_path = os.path.join(DWI_folder,f'{subject}_subjspace_coreg.nii.gz')
    bvals_path = os.path.join(DWI_folder,f'{subject}_bvals.txt')
    bvecs_path = os.path.join(DWI_folder,f'{subject}_bvecs.txt')

    bvals_checked_path = os.path.join(tractseg_inputs_folder,f'{subject}_checked_bvals.txt')
    bvecs_checked_path = os.path.join(tractseg_inputs_folder,f'{subject}_checked_bvecs.txt')

    if not os.path.exists(bvecs_checked_path) or not os.path.exists(bvals_checked_path) or overwrite:
        diff_subjspace_mif_path = os.path.join(tractseg_inputs_folder,f'{subject}_subjspace_coreg.mif')
        if not os.path.exists(diff_subjspace_mif_path) or overwrite:
            cmd = f'mrconvert {diff_subjspace_path} {diff_subjspace_mif_path}'
            if testmode or verbose: print(cmd)
            if not testmode: os.system(cmd)
        cmd = f'dwigradcheck {diff_subjspace_mif_path} -fslgrad {bvecs_path} {bvals_path} -number 10000 -export_grad_fsl {bvecs_checked_path} {bvals_checked_path}'
        if testmode or verbose: print(cmd)
        if not testmode: os.system(cmd)

    

    FA_subj_path = os.path.join(tractseg_inputs_folder,f'{subject}_subjspace_fa.nii.gz')
    if not os.path.exists(FA_subj_path):
        if FA_type == 'tensor2metric':
            dt_mif = os.path.join(tractseg_inputs_folder,f'{subject}_subjspace_dt.mif')
            fa_mif = os.path.join(tractseg_inputs_folder,f'{subject}_subjspace_fa.mif')
            os.system('dwi2tensor ' + diff_subjspace_path + ' ' + dt_mif + ' -fslgrad ' + bvecs_checked_path + ' ' + bvals_checked_path + ' -force')
            os.system('tensor2metric  -fa ' + fa_mif + ' ' + dt_mif + ' -force')
            os.system(f'mrconvert {fa_mif} {FA_subj_path}')
        elif FA_type == 'calc_FA':
            mask_subjspace_path = os.path.join(DWI_folder,f'{subject}_subjspace_mask.nii.gz')
            cmd = f"calc_FA -i {diff_subjspace_path} -o {FA_subj_path} --bvals {bvals_checked_path} --bvecs {bvecs_checked_path} --brain_mask {mask_subjspace_path}"
            if testmode or verbose: print(cmd)
            if not testmode: os.system(cmd)
        else:
            print(f'Unrecognized argument')


    tractseg_outputs_subj_folder = os.path.join(tractseg_outputs_folder,f'{subject}')
    if not os.path.exists(tractseg_outputs_subj_folder): os.mkdir(tractseg_outputs_subj_folder)


    FA_MNI_path = os.path.join(tractseg_outputs_subj_folder,f'{subject}_MNI_fa.nii.gz')
    subj_to_MNI_mat = os.path.join(tractseg_outputs_subj_folder,f'{subject}_FA_to_MNI.mat')
    init_mat_path = os.path.join(tractseg_outputs_folder,ref_subj,f'{ref_subj}_FA_to_MNI.mat')
    diff_MNI_path = os.path.join(tractseg_outputs_subj_folder, f'{subject}_MNI_coreg.nii.gz')

    if subject not in subjects_fixedangle:
        if not os.path.exists(FA_MNI_path) or not os.path.exists(subj_to_MNI_mat) or overwrite:
            cmd = f'flirt -ref {template_MNI} -in {FA_subj_path} -out {FA_MNI_path} -omat {subj_to_MNI_mat} -dof 6 -cost mutualinfo -searchcost mutualinfo'
            if testmode or verbose: print(cmd)
            if not testmode: os.system(cmd)
    else:
        if not os.path.exists(FA_MNI_path) or not os.path.exists(subj_to_MNI_mat) or overwrite:
            #cmd = f'flirt -ref {template_MNI} -in {FA_subj_path} -out {FA_MNI_path} -omat {subj_to_MNI_mat} -dof 6 -cost mutualinfo -searchcost mutualinfo -searchrx -90 -80'
            cmd = f'flirt -ref {template_MNI} -in {FA_subj_path} -out {FA_MNI_path} -omat {subj_to_MNI_mat} -dof 6 -cost mutualinfo -searchcost mutualinfo -searchrx -90 -80 -init {init_mat_path}'
            if testmode or verbose: print(cmd)
            if not testmode: os.system(cmd)

    if not os.path.exists(diff_MNI_path) or overwrite:
        cmd = f'flirt -ref {template_MNI} -in {diff_subjspace_path} -out {diff_MNI_path} -applyxfm -init {subj_to_MNI_mat} -dof 6'
        if testmode or verbose: print(cmd)
        if not testmode: os.system(cmd)

        """
        if not os.path.exists(FA_MNI_path) or not os.path.exists(subj_to_MNI_mat):
            fixed_image = ants.image_read(template_MNI)
            moving_image = ants.image_read(FA_subj_path)

            # Perform trans registration
            transform_rigid = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')

            # Apply the transformation to the moving image
            registered_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                                     transformlist=transform_rigid['fwdtransforms'])

            shutil.copy(transform_rigid['fwdtransforms'][0], subj_to_MNI_mat)

            ants.image_write(registered_image, FA_MNI_path)

        if not os.path.exists(diff_MNI_path):
            fixed_image = ants.image_read(template_MNI)

            subj_coreg_ants = ants.image_read(diff_subjspace_path)
            registered_image_coreg = ants.apply_transforms(fixed=fixed_image, moving=subj_coreg_ants,
                                                     transformlist=subj_to_MNI_mat,imagetype =3)
            ants.image_write(registered_image_coreg, diff_MNI_path)
        """

    bvals_MNI_path = os.path.join(tractseg_outputs_subj_folder,f'{subject}_bvals_MNI.txt')
    bvecs_MNI_path = os.path.join(tractseg_outputs_subj_folder,f'{subject}_bvecs_MNI.txt')

    if not os.path.exists(bvals_MNI_path) or overwrite:
        if testmode:
            print(f'cp {bvals_checked_path} {bvals_MNI_path}')
        else:
            shutil.copy(bvals_checked_path,bvals_MNI_path)

    if not os.path.exists(bvecs_MNI_path) or overwrite:
        cmd = f'rotate_bvecs -i {bvecs_checked_path} -t {subj_to_MNI_mat} -o {bvecs_MNI_path}'
        if testmode: print(cmd)
        else: os.system(cmd)


    """
	
    peaks_path = os.path.join(tractseg_outputs_subj_folder,f'peaks.nii.gz')
    if not os.path.exists(peaks_path) or overwrite:
        cmd = f'TractSeg -i {diff_MNI_path} -o {tractseg_outputs_subj_folder} --bvals {bvals_MNI_path} --bvecs {bvecs_MNI_path} --raw_diffusion_input'
        if testmode: print(cmd)
        else: os.system(cmd)

    endings_folder = os.path.join(tractseg_outputs_subj_folder,'endings_segmentations')
    if not os.path.exists(endings_folder) or overwrite:
        cmd = f'TractSeg -i {peaks_path} -o {tractseg_outputs_subj_folder} --output_type endings_segmentation'
        if testmode: print(cmd)
        else: os.system(cmd)


    TOM_folder = os.path.join(tractseg_outputs_subj_folder,'TOM')
    if not os.path.exists(TOM_folder) or overwrite:
        cmd = f'TractSeg -i {peaks_path} -o {tractseg_outputs_subj_folder} --output_type TOM'
        if testmode: print(cmd)
        else: os.system(cmd)

    TOM_trackings_folder = os.path.join(tractseg_outputs_subj_folder,'TOM_trackings')
    if not os.path.exists(TOM_trackings_folder) or overwrite:
        cmd = f'Tracking -i {peaks_path} -o {tractseg_outputs_subj_folder} --nr_fibers 5000'
        if testmode: print(cmd)
        else: os.system(cmd)


    csv_path = os.path.join(tractseg_outputs_subj_folder,f'Tractometry_{subject}.csv')
    if not os.path.exists(csv_path) or overwrite:
        cmd = f'Tractometry -i {TOM_trackings_folder} -o {csv_path} -e {endings_folder} -s {FA_MNI_path}'
        if testmode: print(cmd)
        else: os.system(cmd)
 	"""