import numpy as np
import os, shutil, re, fnmatch, glob, sys

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

    bvecs_transpose = np.zeros(3)
    if np.any(bvecs_orig!=bvecs_new):
        num_bvecs = np.shape(bvecs_orig)[1]
        for i in range(3):
            #if all(bvecs_orig[i][j] == -bvecs_new[(i + 1) % 3][j] for j in range(num_bvecs)):
            #    bvecs_transpose.append(f'-{chr(97 + i)}')
            #elif all(bvecs_orig[i][j] == bvecs_new[(i + 1) % 3][j] for j in range(num_bvecs)):
            #    bvecs_transpose.append(f'{chr(97 + i)}')
            for j in range(3):
                if np.all(np.abs(bvecs_orig[:][i] - bvecs_new[:][j]) < 2e-2):
                    bvecs_transpose[i] = j + 1
                if np.all(np.abs(bvecs_orig[:][i] + bvecs_new[:][j]) < 2e-2):
                    bvecs_transpose[i] = -(j + 1)
    else:
        bvecs_transpose = [1, 2, 3]
    bvecs_transpose = list(bvecs_transpose)
    if np.any(bvals_orig!=bvals_new):
        bvals_transpose = bvals_new/bvals_orig
        bvals_transpose = [1 if np.isnan(x) else x for x in bvals_transpose]
    else:
        bvals_transpose = list(np.ones(np.size(bvals_orig)))
    gradscheme_txt = f'bvecs: {bvecs_transpose}\n' + f'bvals: {bvals_transpose}'
    with open(gradscheme_path, 'w') as file:
        file.write(gradscheme_txt)


def save_new_bvecs(bvecs_orig, gradscheme_path, bvecs_new_path):

    if isinstance(bvecs_orig, str):
        bvecs_orig=np.loadtxt(bvecs_orig)
    #bvecs_new = [row[:] for row in bvecs_orig]  # Create a copy of the original matrix

    with open(gradscheme_path, 'r') as file:
        for line in file:
            if line.startswith('bvecs'):
                bvecs_transpose = line.split("[")[1].split(']')[0].split(',')
                #bvecs_transpose = np.array(list(map(int, bvecs_transpose)))
                bvecs_transpose = [int(float(x)) for x in bvecs_transpose]
    #print(bvecs_transpose)

    bvecs_new = np.vstack((bvecs_orig[abs(bvecs_transpose[0])-1]*np.sign(bvecs_transpose[0]),
                           bvecs_orig[abs(bvecs_transpose[1])-1]*np.sign(bvecs_transpose[1]),
                           bvecs_orig[abs(bvecs_transpose[2])-1]*np.sign(bvecs_transpose[2])))

    np.savetxt(bvecs_new_path, bvecs_new, fmt='%.2f')


def save_new_bvals(bvals_orig, gradscheme_path,bvals_newpath):
    if isinstance(bvals_orig, str):
        bvecs_orig=np.loadtxt(bvals_orig)
    #bvecs_new = [row[:] for row in bvecs_orig]  # Create a copy of the original matrix

    with open(gradscheme_path, 'r') as file:
        for line in file:
            if line.startswith('bvals'):
                bvals_transpose = line.split("[")[1].split(']')[0].split(',')
                bvals_transpose = [(float(x)) for x in bvals_transpose]

    bvals_new = bvals_orig * list(bvals_transpose)
    np.savetxt(bvals_newpath, bvals_new, fmt='%.2f')


def regexify(string):
    newstring = ('^'+string+'$').replace('*','.*')
    return newstring


def glob_remote(path, sftp=None):
    match_files = []
    if sftp is not None:
        if '*' in path:
            pathdir, pathname = os.path.split(path)
            pathname = regexify(pathname)
            allfiles = sftp.listdir(pathdir)
            for file in allfiles:
                if re.search(pathname, file) is not None:
                    match_files.append(os.path.join(pathdir,file))
        elif '.' not in path:
            allfiles = sftp.listdir(path)
            for filepath in allfiles:
                match_files.append(os.path.join(path, filepath))
            return match_files
        else:
            dirpath = os.path.dirname(path)
            try:
                sftp.stat(dirpath)
            except:
                return match_files
            allfiles = sftp.listdir(dirpath)
            #if '*' in path:
            #    for filepath in allfiles:
            #            match_files.append(os.path.join(dirpath,filepath))
            #else:
            for filepath in allfiles:
                if fnmatch.fnmatch(os.path.basename(filepath), os.path.basename(path)):
                    match_files.append(os.path.join(dirpath, filepath))
    else:
        if '.' not in path:
            match_files = glob.glob(path)
        else:
            dirpath = os.path.dirname(path)
            if not os.path.exists(dirpath):
                return(match_files)
            else:
                allfiles = glob.glob(os.path.join(dirpath,'*'))
                for filepath in allfiles:
                    if fnmatch.fnmatch(os.path.basename(filepath), os.path.basename(path)):
                        match_files.append(os.path.join(dirpath, filepath))
    return(match_files)


def checkallfiles(paths, sftp=None):
    existing = True
    for path in paths:
        match_files = glob_remote(path, sftp)
        if not match_files:
            existing= False
    return existing


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


#subjects = ['00775', '04491', '04493', '01412', '04526', '01470']
subjects = []
#subjects.append(sys.argv[1])
subjects = ['00775']
proc_subjn="S"
proc_name ="diffusion_prep_"+proc_subjn

overwrite=False

cleanup = True

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

    exists_all_nii = checkallfiles([dt_mif, fa_nii, ad_nii, rd_nii, md_nii])

    if not exists_all_nii:
        if not os.path.exists(mask_mif) or overwrite:
            command = f'mrconvert {mask_nii} {mask_mif}'
            os.system(command)


        if not os.path.exists(bvecs_fixed) or overwrite:
            if not os.path.exists(bvecs_grad_scheme_txt):
                os.system(
                    f'dwigradcheck ' + coreg_preproced + ' -fslgrad ' + bvecs_fixed + ' ' + bvals_fixed + ' -mask ' + mask_mif + ' -number 100000 -export_grad_fsl ' + bvecs_check + ' ' + bvals_check + ' -force')
                generate_bvec_gradscheme(bvecs_fixed, bvals_fixed, bvecs_check, bvals_check, bvecs_grad_scheme_txt)
            else:
                bvec_orig = np.loadtxt(bvecs_fixed)
                bval_orig = np.loadtxt(bvals_fixed)

                save_new_bvecs(bvec_orig, bvecs_grad_scheme_txt, bvecs_check)
                save_new_bvals(bval_orig, bvecs_grad_scheme_txt, bvals_check)

        bvals = np.loadtxt(bvals_fixed)
        exists_all_mif = checkallfiles([dt_mif, fa_mif, rd_mif, ad_mif, md_mif])

        if not exists_all_mif:
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
            os.system(command)

        if not os.path.exists(rd_nii):
            command = f'mrconvert {rd_mif} {rd_nii}'
            os.system(command)

        if not os.path.exists(ad_nii):
            command = f'mrconvert {ad_mif} {ad_nii}'
            os.system(command)

        if not os.path.exists(md_nii):
            command = f'mrconvert {md_mif} {md_nii}'
            os.system(command)

        if cleanup:
            os.remove(fa_mif)
            os.remove(rd_mif)
            os.remove(ad_mif)
            os.remove(md_mif)