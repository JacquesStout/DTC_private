
import subprocess
import os
from DTC.file_manager.computer_nav import checkfile_exists_all

# Define the input diffusion image and output tractogram file paths

orig_folder = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/'
work_dir = '/Users/jas/jacques/CS_Project/trk_dir'

overwrite=False

nominal_bval = 2127

CS_vals = ['4','5.9998','8','10.0009','11.9985']
CS_val = CS_vals[2]

diff_name = f'Bruker_diffusion_test_15_0.045_af_{CS_val}x__TV_and_L1_wavelet_0.01_0.01_bart_recon.nii.gz'
diff_path = os.path.join(orig_folder, diff_name)

diff_path_mif = os.path.join(work_dir, diff_name.replace('.nii.gz','.mif'))

dwi_path = os.path.join(work_dir, diff_name.replace('.nii','_dwi.nii'))
dwi_path_mif = dwi_path.replace('.nii.gz','.mif')

tractogram_file = os.path.join(orig_folder, 'Bruker_diffusion_test.tck')

response_txt = os.path.join(work_dir, 'mrtrix_response.txt')
gm_txt = os.path.join(work_dir, 'mrtrix_gm.txt')
wm_txt = os.path.join(work_dir, 'mrtrix_wm.txt')
csf_txt = os.path.join(work_dir, 'mrtrix_csf.txt')

# Use the mrconvert command to convert the input DWI image to the MRtrix format

if not os.path.exists(diff_path_mif) or overwrite:
    subprocess.run(['mrconvert', diff_path, diff_path_mif])

#check bval / bvecs

bval_name = '15_1_fully_sampled_DWI_16p25_120_21directions_Brukerdirs.bval'
bvec_name = '15_1_fully_sampled_DWI_16p25_120_21directions_Brukerdirs.bvec'
bval_path_orig = os.path.join(orig_folder, bval_name)
bvec_path_orig = os.path.join(orig_folder, bvec_name)
bval_path = os.path.join(work_dir, bval_name)
bvec_path = os.path.join(work_dir, bvec_name)

wm_mif = os.path.join(orig_folder, diff_path_mif.replace('.mif','_wm.mif'))
gm_mif = os.path.join(orig_folder, diff_path_mif.replace('.mif','_gm.mif'))
csf_mif = os.path.join(orig_folder, diff_path_mif.replace('.mif','_csf.mif'))

if not os.path.exists(bval_path) or not os.path.exists(bvec_path) or overwrite:

    if not os.path.isfile(bval_path_orig): print('Missing original bval')
    if not os.path.isfile(bvec_path_orig): print('Missing original bvec')

    os.system('dwigradcheck ' + diff_path_mif + ' -fslgrad ' + bvec_path_orig + ' ' + bval_path_orig +
              ' -number 50000 -export_grad_fsl ' + bvec_path + ' ' + bval_path + ' -force')

if not os.path.exists(dwi_path) or overwrite:
    #cmd = f'select_dwi_vols {diff_path} {bval_path} {dwi_path} {nominal_bval}  -m'
    subprocess.run(['select_dwi_vols', diff_path, bval_path, dwi_path, str(nominal_bval), '-m'])

if not os.path.exists(dwi_path_mif) or overwrite:
    subprocess.run(['mrconvert', dwi_path, dwi_path_mif, '-force'])



# Use the dwi2response command to estimate the response function of the DWI data
#if not os.path.exists(response_txt) or not os.path.exists(gm_txt) or not os.path.exists(wm_txt):
voxels_mif = diff_path_mif.replace('.mif','_voxels.mif')
if not checkfile_exists_all([csf_txt, gm_txt, wm_txt]) or overwrite:
    subprocess.run(['dwi2response', 'dhollander', diff_path_mif, wm_txt, gm_txt, csf_txt, '-fslgrad', bvec_path, bval_path, '-voxels', voxels_mif, '-force'])

overwrite=True
# Use the dwi2fod command to estimate the fiber orientation distribution (FOD) from the DWI data
predicted_mif = diff_path_mif.replace('.mif','_predict.mif')
if not checkfile_exists_all([wm_mif,gm_mif,csf_mif]) or overwrite:
    subprocess.run(['dwi2fod','-force', '-fslgrad', bvec_path, bval_path, '-predicted_signal', predicted_mif, 'msmt_csd', diff_path_mif, wm_txt, wm_mif, gm_txt, gm_mif, csf_txt, csf_mif])

# Use the tckgen command to generate the tractogram file from the FOD image
#if not checkfile_exists_all([wm_mif,gm_mif,csf_mif]) or overwrite:
#    subprocess.run(['tckgen', '-act', '5tt.mif', '-backtrack', '-crop_at_gmwmi', '-maxlength', '250', '-select', '100000', '-seed_image', 'wm.mif', 'fod.mif', tractogram_file])