import os
from DTC.nifti_handlers.transform_handler import header_superpose
from DTC.nifti_handlers.transform_handler import img_transform_exec


from DTC.diff_handlers.bvec_handler import read_bvals
fbvals = '/Volumes/Data/Badea/Lab/mouse/Vitek_series_altvals/diffusion_prep_locale/diffusion_prep_V11_8_13/V11_8_13_bvals.txt'
fbvecs = '/Volumes/Data/Badea/Lab/mouse/Vitek_series_altvals/diffusion_prep_locale/diffusion_prep_V11_8_13/V11_8_13_bvecs.txt'
bvals, bvecs = read_bvals(fbvals,fbvecs, sftp = None)



inpath = '/Volumes/Data/Badea/Lab/jacques/temp_subj_ADDecode/'
subject = 'S02654'
files_to_reorient = [os.path.join(inpath, f'{subject}_fa.nii.gz'),os.path.join(inpath, f'{subject}_labels_lr_ordered.nii.gz')]

for subj in files_to_reorient:
    SAMBA_init = f'/Volumes/Data/Badea/Lab/mouse/VBM_20APOE01_chass_symmetric3_allAPOE-work/preprocess/base_images/{subj}_dwi_masked.nii.gz'
    outpath_1 = os.path.join(outdir, f'{subj}_dwi_masked_ARI.nii.gz')
    outpath_2 =  os.path.join(outdir, f'{subj}_subjspace_dwi.nii.gz')
    ref_orig = f'/Volumes/dusom_civm-atlas/18.abb.11/research/diffusion{subj}dsi_studio/nii4D_{subj}.nii'
    ref_to_use = os.path.join(outdir,f'nii4D_{subj}.nii')

    img_transform_exec(SAMBA_init, 'ALS',  'ARI', outpath_1)
    header_superpose(ref_orig, outpath_1, outpath_2)


