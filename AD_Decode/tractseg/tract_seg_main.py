import os,shutil,glob

DWI_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/DWI/'
tractseg_folder = '/Users/jas/jacques/TractSeg_testzone/'
tractseg_inputs_folder = os.path.join(tractseg_folder,'TractSeg_inputs')
tractseg_outputs_folder = os.path.join(tractseg_folder,'TractSeg_outputs')

if not os.path.exists(tractseg_inputs_folder): os.mkdir(tractseg_inputs_folder)
if not os.path.exists(tractseg_outputs_folder): os.mkdir(tractseg_outputs_folder)

overwrite=False

subject = 'S00775'
files_subj = glob.glob(os.path.join(DWI_folder,'*subjspace_coreg.nii.gz'))
subjects = [os.path.basename(file_path)[:6] for file_path in files_subj]

FA_type = 'tensor2metric'
FA_type = 'calc_FA'

for subject in subjects:

    diff_subjspace_path = os.path.join(DWI_folder,f'{subject}_subjspace_coreg.nii.gz')
    bvals_path = os.path.join(DWI_folder,f'{subject}_bvals.txt')
    bvecs_path = os.path.join(DWI_folder,f'{subject}_bvecs.txt')

    bvals_checked_path = os.path.join(DWI_folder,f'{subject}_checked_bvals.txt')
    bvecs_checked_path = os.path.join(DWI_folder,f'{subject}_checked_bvecs.txt')

    if not os.path.exists(bvecs_checked_path) or not os.path.exists(bvals_checked_path) or overwrite:
        diff_subjspace_mif_path = os.path.join(tractseg_inputs_folder,f'{subject}_subjspace_coreg.mif')
        if not os.path.exists(diff_subjspace_mif_path) or overwrite:
            cmd = f'mrconvert {diff_subjspace_path} {diff_subjspace_mif_path}'
            os.system(cmd)
        os.system(
            f'dwigradcheck {diff_subjspace_mif_path} -fslgrad {bvecs_path} {bvals_path} -number 10000 -export_grad_fsl {bvecs_checked_path} {bvals_checked_path}')


    template_MNI = '~/anaconda3/lib/python3.11/site-packages/tractseg/resources/MNI_FA_template.nii.gz'

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
            cmd = f"calc_FA -i {diff_subjspace_path} -o {FA_subj_path} --bvals {bvals_path} --bvecs  {bvecs_path} --brain_mask {mask_subjspace_path}"
            os.system(cmd)
        else:
            print(f'Unrecognized argument')

    tractseg_outputs_subj_folder = os.path.join(tractseg_outputs_folder,f'{subject}')
    if not os.path.exists(tractseg_outputs_subj_folder): os.mkdir(tractseg_outputs_subj_folder)

    FA_MNI_path = os.path.join(tractseg_outputs_subj_folder,f'{subject}_MNI_fa.nii.gz')
    subj_to_MNI_mat = os.path.join(tractseg_outputs_subj_folder,f'{subject}_FA_to_MNI.mat')
    if not os.path.exists(FA_MNI_path):
        cmd = f'flirt -ref {template_MNI} -in {FA_subj_path} -out {FA_MNI_path} -omat {subj_to_MNI_mat} -dof 6 -cost mutualinfo -searchcost mutualinfo'
        os.system(cmd)

    diff_MNI_path = os.path.join(tractseg_outputs_subj_folder,f'{subject}_MNI_coreg.nii.gz')
    if not os.path.exists(diff_MNI_path):
        cmd = f'flirt -ref {template_MNI} -in {diff_subjspace_path} -out {diff_MNI_path} -applyxfm -init {subj_to_MNI_mat} -dof 6'
        os.system(cmd)


    bvals_MNI_path = os.path.join(tractseg_outputs_subj_folder,f'{subject}_bvals_MNI.txt')
    bvecs_MNI_path = os.path.join(tractseg_outputs_subj_folder,f'{subject}_bvecs_MNI.txt')
    if not os.path.exists(bvals_MNI_path):  shutil.copy(bvals_checked_path,bvals_MNI_path)
    if not os.path.exists(bvecs_MNI_path): os.system(f'rotate_bvecs -i {bvecs_checked_path} -t {subj_to_MNI_mat} -o {bvecs_MNI_path}')

    peaks_path = os.path.join(tractseg_outputs_subj_folder,f'peaks.nii.gz')
    if not os.path.exists(peaks_path) or overwrite:
        cmd = f'TractSeg -i {diff_MNI_path} -o {tractseg_outputs_subj_folder} --bvals {bvals_MNI_path} --bvecs {bvecs_MNI_path} --raw_diffusion_input'
        os.system(cmd)

    endings_folder = os.path.join(tractseg_outputs_subj_folder,'endings_segmentation')
    if not os.path.exists(endings_folder) or overwrite:
        cmd = f'TractSeg -i {peaks_path} -o {tractseg_outputs_subj_folder} --output_type endings_segmentation'
        os.system(cmd)

    TOM_folder = os.path.join(tractseg_outputs_subj_folder,'TOM')
    if not os.path.exists(TOM_folder) or overwrite:
        cmd = f'TractSeg -i {peaks_path} -o {tractseg_outputs_subj_folder} --output_type TOM'
        os.system(cmd)

    TOM_trackings_folder = os.path.join(tractseg_outputs_subj_folder,'TOM_trackings')
    if not os.path.exists(TOM_trackings_folder) or overwrite:
        cmd = f'Tracking -i {peaks_path} -o {tractseg_outputs_subj_folder} --nr_fibers 5000'
        os.system(cmd)

    csv_path = os.path.join(tractseg_outputs_subj_folder,f'Tractometry_{subject}.csv')
    if not os.path.exists(csv_path) or overwrite:
        cmd = f'Tractometry -i {TOM_trackings_folder} -o {csv_path} -e {endings_folder} -s {FA_MNI_path}'
        os.system(cmd)