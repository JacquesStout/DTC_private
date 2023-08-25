import os, glob, socket, re, fnmatch, subprocess
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti

def regexify(string):
    newstring = ('^'+string+'$').replace('*','.*')
    return newstring


def nifti_corrupt_check(nifti_path, subj=None):

    if subj is None:
        subj = nifti_path
    try:
        nii = nib.load(nifti_path)
    except:
        return 0

    try:
        nii.header.get_zooms()
        nii.affine
        data = nii.get_fdata()
    except:
        return 0

    data[np.isnan(data)] = 0

    if np.mean(data)>0:
        return 1
    else:
        return 0

def mif_corrupt_check(mif_path):

    command = (f'mrinfo {mif_path}')

    completed_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check the return code to determine if the command had an error
    if completed_process.returncode == 0:
        return 1
    else:
        return 0

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


if socket.gethostname().split('.')[0]=='santorini':
    root = '/Volumes/Data/Badea/Lab/mouse/ADRC_jacques_pipeline/'
    data_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
    data_path_output = '/Volumes/Data/Badea/Lab/mouse/ADRC_jacques_pipeline/'
else:
    root = '/mnt/munin2/Badea/Lab/mouse/ADRC_jacques_pipeline/'
    data_path = '/mnt/munin2/Badea/Lab/ADRC-20230511/'
    data_path_output = '/mnt/munin2/Badea/Lab/mouse/ADRC_jacques_pipeline/'

dwi_manual_pe_scheme_txt = os.path.join(root, 'dwi_manual_pe_scheme.txt')
perm_output = os.path.join(root, 'perm_files/')

se_epi_manual_pe_scheme_txt = os.path.join(root, 'se_epi_manual_pe_scheme.txt')
if socket.gethostname().split('.')[0]=='santorini':
    fsl_config_path = '$FSLDIR/src/fsl-topup/flirtsch/b02b0.cnf'
else:
    fsl_config_path = '$FSLDIR/src/topup/flirtsch/b02b0.cnf'
#outputs


data_folder_path ='/Volumes/Data/Badea/Lab/ADRC-20230511/'
#list_folders_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
list_folders_path = os.listdir(data_folder_path)
directories = [item for item in list_folders_path if os.path.isdir(os.path.join(data_folder_path, item))]
list_of_subjs_long = [i for i in directories if 'ADRC' in i]
subjects = list_of_subjs_long


index_gz = '.gz'
overwrite = False

cleanup = True

verbose = True

for subj in subjects:

    subj_folder = os.path.join(data_path, subj,'visit1')
    allsubjs_out_folder = os.path.join(data_path_output, 'temp')
    subj_out_folder = os.path.join(allsubjs_out_folder, subj)
    scratch_path = os.path.join(subj_out_folder, 'scratch')
    perm_subj_output = perm_output
    #perm_subj_output = os.path.join(perm_output, subj)

    bvec_path_AP = os.path.join(subj_out_folder, subj + '_bvecs.txt')
    bval_path_AP = os.path.join(subj_out_folder, subj + '_bvals.txt')
    bvec_path_PA = os.path.join(subj_out_folder, subj+'_bvecs_rvrs.txt')
    bval_path_PA = os.path.join(subj_out_folder, subj+'_bvals_rvrs.txt')

    DTI_forward_nii_path = os.path.join(subj_folder, 'HCP_DTI.nii.gz')
    DTI_reverse_nii_path = os.path.join(subj_folder, 'HCP_DTI_reverse_phase.nii.gz')

    if not os.path.exists(DTI_forward_nii_path):
        print(f'Missing {DTI_forward_nii_path} for subject {subj}')
        continue
    if not os.path.exists(DTI_reverse_nii_path):
        print(f'Missing {DTI_reverse_nii_path} for subject {subj}')
        continue

    DTI_forward_nii_path = os.path.join(subj_folder,'HCP_DTI.nii.gz')

    resampled_nii_path = os.path.join(subj_out_folder, subj + '_coreg_resampled.nii.gz')
    resampled_mif_path = os.path.join(perm_subj_output, subj + '_coreg_resampled.mif')

    dwi_nii_gz = os.path.join(perm_subj_output, subj + '_dwi.nii.gz')

    mask_nii_path = os.path.join(subj_out_folder, subj + '_mask.nii.gz')
    mask_mif_path = os.path.join(perm_subj_output, subj + '_mask.mif')

    coreg_bvecs = os.path.join(perm_subj_output, subj + '_coreg_bvecs.txt')
    coreg_bvals = os.path.join(perm_subj_output, subj + '_coreg_bvals.txt')


    if not os.path.exists(resampled_mif_path) or not os.path.exists(dwi_nii_gz) or not os.path.exists(mask_mif_path) \
            or not os.path.exists(coreg_bvecs):
        print(f'Initial unwarping and resampling not finished for subject {subj}')
        continue
    else:
        if not mif_corrupt_check(resampled_mif_path):
            print(f'Resampled coreg is corrupted for subject {subj}')
            continue
        if not nifti_corrupt_check(dwi_nii_gz):
            print(f'Resampled dwi is corrupted for subject {subj}')
            continue
        if not mif_corrupt_check(mask_mif_path):
            print(f'Resampled mask is corrupted for subject {subj}')
            continue

    dt_mif = os.path.join(perm_subj_output, subj + '_dt.mif' + index_gz)
    fa_mif = os.path.join(perm_subj_output, subj + '_fa.mif' + index_gz)
    dk_mif = os.path.join(perm_subj_output, subj + '_dk.mif' + index_gz)
    mk_mif = os.path.join(perm_subj_output, subj + '_mk.mif' + index_gz)
    md_mif = os.path.join(perm_subj_output, subj + '_md.mif' + index_gz)
    ad_mif = os.path.join(perm_subj_output, subj + '_ad.mif' + index_gz)
    rd_mif = os.path.join(perm_subj_output, subj + '_rd.mif' + index_gz)

    fa_nii = os.path.join(perm_subj_output, subj + '_fa.nii' + index_gz)


    if not checkallfiles([fa_nii, dt_mif, fa_mif, dk_mif, md_mif, ad_mif, rd_mif]):
        print(f'Obtaining fas and other stats was unfinished for {subj}')
        continue
    else:
        if not nifti_corrupt_check(fa_nii):
            print(f'Fa nifti is corrupted for subject {subj}')
            continue
        if not mif_corrupt_check(dt_mif):
            print(f'DT mif is corrupted for subject {subj}')
            continue
        if not mif_corrupt_check(ad_mif):
            print(f'AD mif is corrupted for subject {subj}')
            continue
        #### Could use a checker for the mif, not done for now

    smallerTracks = os.path.join(perm_subj_output, subj + '_smallerTracks2mill.tck')

    """
    if not os.path.exists(smallerTracks):
        print(f'Unfinished process for trks for subj {subj} ')
        continue
    else:
        #Later will add tck checkers if necessary, right now do not want to mess with ongoing process
        print(f'Process complete for subj {subj}')
    """