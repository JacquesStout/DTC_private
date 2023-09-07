import numpy as np
import os, shutil

def generate_bvec_gradscheme(bvecs_orig, bvals_orig, bvecs_new, bvals_new, gradscheme_path):

    if isinstance(bvecs_orig,str):
        bvecs_orig = np.loadtxt(bvecs_orig)
        bvals_orig = np.loadtxt(bvals_orig)
        bvecs_new = np.loadtxt(bvecs_new)
        bvals_new = np.loadtxt(bvals_new)

    bvecs_transpose = []

    bvecs_orig = np.round(bvecs_orig, decimals=2)
    bvecs_new = np.round(bvecs_new, decimals=2)
    bvals_orig = np.round(bvals_orig, decimals=2)
    bvals_new = np.round(bvals_new, decimals=2)

    if np.any(bvecs_orig!=bvecs_new):
        num_bvecs = np.shape(bvecs_orig)[1]
        for i in range(3):
            if all(bvecs_orig[i][j] == -bvecs_new[(i + 1) % 3][j] for j in range(num_bvecs)):
                bvecs_transpose.append(f'-{chr(97 + i)}')
            elif all(bvecs_orig[i][j] == bvecs_new[(i + 1) % 3][j] for j in range(num_bvecs)):
                bvecs_transpose.append(f'{chr(97 + i)}')
    else:
        bvecs_transpose = ['1, 2, 3']
    if np.any(bvals_orig!=bvals_new):
        print('must investigate')
        raise Exception
    gradscheme_txt = f'bvecs: {bvecs_transpose}'

    with open(gradscheme_path, 'w') as file:
        file.write(gradscheme_txt)


def save_new_bvecs(bvecs_orig, gradscheme_path, bvecs_new_path):

    if isinstance(bvecs_orig, str):
        bvecs_orig=np.loadtxt(bvecs_orig)
    #bvecs_new = [row[:] for row in bvecs_orig]  # Create a copy of the original matrix

    with open(gradscheme_path, 'r') as file:
        bvecs_transpose = file.read()
        bvecs_transpose = bvecs_transpose.split("'")[1].split(',')
        bvecs_transpose = np.array(list(map(int, bvecs_transpose)))
    #print(bvecs_transpose)

    bvecs_new = np.vstack((bvecs_orig[abs(bvecs_transpose[0])-1]*np.sign(bvecs_transpose[0]),
                           bvecs_orig[abs(bvecs_transpose[1])-1]*np.sign(bvecs_transpose[1]),
                           bvecs_orig[abs(bvecs_transpose[2])-1]*np.sign(bvecs_transpose[2])))

    np.savetxt(bvecs_new_path, bvecs_new, fmt='%.2f')


munin=False
if munin:
    gunniespath = "~/wuconnectomes/gunnies"
    mainpath = "/mnt/munin6/Badea/ADdecode.01/"
    outpath = "/mnt/munin6/Badea/Lab/human/AD_Decode/diffusion_prep_locale/"
    SAMBA_inputs_folder = "/mnt/munin6/Badea/Lab/mouse/ADDeccode_symlink_pool/"
    shortcuts_all_folder = "/mnt/munin6/Badea/Lab/human/ADDeccode_symlink_pool_allfiles/"

else:
    gunniespath = "/Users/alex/bass/gitfolder/wuconnectomes/gunnies/"
    mainpath="/Volumes/Data/Badea/ADdecode.01/"
    outpath = "/Volumes/Data/Badea/Lab/human/AD_Decode/diffusion_prep_locale/"
    #bonusshortcutfolder = "/Volumes/Data/Badea/Lab/mouse/ADDeccode_symlink_pool/"
    SAMBA_inputs_folder = None
    shortcuts_all_folder = None


subjects = ['00775', '04491', '04493', '01412', '04526', '01470']
subjects = ['00775']
proc_subjn="S"
proc_name ="diffusion_prep_"+proc_subjn

overwrite=False

for subject in subjects:
    subj_path = os.path.join(outpath, proc_name+subject)
    subj = proc_subjn + subject
    bvecs_check = os.path.join(subj_path, f'{subj}_bvecs_checked.txt')
    bvals_check = os.path.join(subj_path, f'{subj}_bvals_checked.txt')

    bvecs_grad_scheme_txt = os.path.join(subj_path, 'bvecs_grad_scheme.txt')

    bvecs_fixed = os.path.join(subj_path, f'{subj}_bvecs_fix.txt')
    bvals_fixed = os.path.join(subj_path, f'{subj}_bvals_fix.txt')

    coreg_preproced = os.path.join(subj_path, f'Reg_MPCA_{subj}_nii4D.nii.gz')

    mask_nii = os.path.join(subj_path, f'{subj}_mask.nii.gz')
    mask_mif = os.path.join(subj_path, f'{subj}_mask.mif')

    dt_mif = os.path.join(subj_path, subj + '_dt.mif')
    fa_mif = os.path.join(subj_path, subj + '_fa.mif')
    dk_mif = os.path.join(subj_path, subj + '_dk.mif')
    mk_mif = os.path.join(subj_path, subj + '_mk.mif')
    md_mif = os.path.join(subj_path, subj + '_md.mif')
    ad_mif = os.path.join(subj_path, subj + '_ad.mif')
    rd_mif = os.path.join(subj_path, subj + '_rd.mif')

    fa_nii = os.path.join(subj_path, subj + '_mrtrixfa.nii.gz')
    ad_nii = os.path.join(subj_path, subj + '_mrtrixad.nii.gz')
    rd_nii = os.path.join(subj_path, subj + '_mrtrixrd.nii.gz')
    md_nii = os.path.join(subj_path, subj + '_mrtrixmd.nii.gz')


    if not os.path.exists(mask_mif) or overwrite:
        command = f'mrconvert {mask_nii} {mask_mif}'
        os.system(command)

    if not os.path.exists(bvecs_check) or overwrite:
        os.system(
            f'dwigradcheck ' + coreg_preproced + ' -fslgrad ' + bvecs_fixed + ' ' + bvals_fixed + ' -mask ' + mask_mif + ' -number 100000 -export_grad_fsl ' + bvecs_check + ' ' + bvals_check + ' -force')

        generate_bvec_gradscheme(bvecs_fixed, bvals_fixed, bvecs_check, bvals_check, bvecs_grad_scheme_txt)
    else:
        bvec_orig = np.loadtxt(bvecs_fixed)
        bval_orig = np.loadtxt(bvals_fixed)

        save_new_bvecs(bvec_orig, bvecs_grad_scheme_txt, bvecs_check)
        shutil.copy(bvals_fixed, bvals_check)

    bvals = np.loadtxt(bvals_fixed)
    if np.unique(bvals_fixed).shape[0] > 2 :
        os.system \
            ('dwi2tensor ' + coreg_preproced + ' ' + dt_mif + ' -dkt ' +  dk_mif +' -fslgrad ' + bvecs_check + ' ' + bvals_check + ' -force')
        os.system(
            'tensor2metric  -fa ' + fa_mif + ' ' + dt_mif + ' -adc ' + md_mif + ' -ad ' + ad_mif + ' -rd ' + rd_mif + ' -force')

        # os.system('mrview '+ fa_mif) #inspect residual
    else:
        os.system(
            'dwi2tensor ' + coreg_preproced + ' ' + dt_mif + ' -fslgrad ' + bvecs_check + ' ' + bvals_check + ' -force')
        os.system('tensor2metric  -fa ' + fa_mif + ' ' + dt_mif + ' -force')
        os.system('tensor2metric  -rd ' + rd_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
        os.system('tensor2metric  -ad ' + ad_mif + ' ' + dt_mif + ' -force')  # if doesn't work take this out :(
        os.system('tensor2metric  -adc ' + md_mif + ' ' + dt_mif + ' -force')

    if not os.path.exists(fa_nii):
        command = f'mrconvert {fa_mif} {fa_nii}'