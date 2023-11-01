import os, socket, shutil, glob, json
import numpy as np
import SimpleITK as sitk


def mkcdir(folderpaths, sftp=None):
    #creates new folder only if it doesnt already exists
    import numpy as np
    if sftp is None:
        if np.size(folderpaths) == 1:
            if not os.path.exists(folderpaths):
                os.mkdir(folderpaths)
        else:
            for folderpath in folderpaths:
                if not os.path.exists(folderpath):
                    os.mkdir(folderpath)
    else:
        if np.size(folderpaths) == 1:
            try:
                sftp.chdir(folderpaths)
            except:
                sftp.mkdir(folderpaths)
        else:
            for folderpath in folderpaths:
                try:
                    sftp.chdir(folderpath)
                except:
                    sftp.mkdir(folderpath)


def get_bxh_method(bxh_path):
    if os.path.exists(bxh_path):
        with open(bxh_path, 'r') as file:
            for line in file:
                if '<psdname>' in line:
                    method_name = line.split('>')[1].split('<')[0]
        return(method_name)
    else:
        print(f'Could not find {bxh_path}')


if socket.gethostname().split('.')[0]=='santorini':
    root = '/Volumes/Data/Badea/Lab/'
    root_proj = '/Volumes/Data/Badea/Lab/human/ADRC/'
    data_path = '/Volumes/Data/Badea/Lab/ADRC-20230511/'
else:
    root = '/mnt/munin2/Badea/Lab/'
    root_proj = '/mnt/munin2/Badea/Lab/human/ADRC/'
    data_path = '/mnt/munin2/Badea/Lab/ADRC-20230511/'


list_folders_path = os.listdir(data_path)
list_of_subjs_long = [i for i in list_folders_path if 'ADRC' in i and not '.' in i]
subjects = sorted(list_of_subjs_long)

output_BIDS = os.path.join(root_proj,'ADRC_BIDS')
fmriprep_output = os.path.join(root_proj,'fmriprep_output')

work_dir = os.path.join(root_proj,'work_dir')

run_fmriprep = False

for subj in subjects:
    subj_folder_orig = os.path.join(data_path, subj,'visit1')

    subj_folder = os.path.join(output_BIDS, f'sub-{subj}')
    anat_folder = os.path.join(output_BIDS, f'sub-{subj}/anat')
    func_folder = os.path.join(output_BIDS, f'sub-{subj}/func')

    bxh_files = glob.glob(os.path.join(subj_folder_orig,'*bxh'))

    """
    for bxh_file in bxh_files:
        method = get_bxh_method(bxh_file)
        if method == 'MP-RAGE':
            t1_path_orig = bxh_file.replace('.bxh','.nii.gz')
            t1_json_path_orig = bxh_file.replace('.bxh','.json')
            break
    """
    t1_path_orig = os.path.join(subj_folder_orig,'T1.nii.gz')
    t1_json_path_orig = t1_path_orig.replace('.nii.gz', '.json')

    t1_nii_path = os.path.join(anat_folder,f'sub-{subj}_T1w.nii.gz')
    t1_json_path = os.path.join(anat_folder,f'sub-{subj}_T1w.json')

    if not os.path.exists(t1_nii_path):
        if not os.path.exists(t1_path_orig):
            print(f'Could not find anatomical image for subject {subj}')
            continue
        else:
            mkcdir([subj_folder, anat_folder], None)
            shutil.copy(t1_path_orig,t1_nii_path)

    # save dict in 'header.json'
    if not os.path.exists(t1_json_path):
        # read image
        if not os.path.exists(t1_json_path_orig):
            itk_image = sitk.ReadImage(t1_nii_path)

            # get metadata dict
            header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
            with open(t1_json_path, "w") as outfile:
                json.dump(header, outfile, indent=4)
        else:
            shutil.copy(t1_json_path_orig,t1_json_path)


    func_path_orig = glob.glob(os.path.join(subj_folder_orig,'resting_state.nii.gz'))
    if np.size(func_path_orig)==1:
        func_path_orig = func_path_orig[0]
        func_json_orig = func_path_orig.replace('.nii.gz','.json')

    #os.path.join(data_path,subj,'visit1/resting_state.nii.gz')  # change this with your file

    func_nii_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.nii.gz')
    func_json_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.json')

    if not os.path.exists(func_nii_path):
        mkcdir([subj_folder, func_folder], None)
        shutil.copy(func_path_orig,func_nii_path)

    # save dict in 'header.json'
    if not os.path.exists(func_json_path):
        if not os.path.exists(func_json_orig):
            # read image
            itk_image = sitk.ReadImage(func_nii_path)

            # get metadata dict
            header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
            with open(func_json_path, "w") as outfile:
                json.dump(header, outfile, indent=4)
        else:
            shutil.copy(func_json_orig,func_json_path)


for subj in subjects:

    if run_fmriprep:
        command = f'qsub singularity run --cleanenv {os.path.join(root,"fmriprep.simg")} {output_BIDS} {fmriprep_output} ' \
            f'participant --participant-label {subj} -w {work_dir} --nthreads 20 ' \
            f'--fs-license-file {os.path.join(root,"license.txt")} --output-spaces T1w'
        os.system(command)
        try:
            work_subj_dir = os.path.join(glob.glob(os.path.join(work_dir,'fmriprep*'))[0],f'*{subj}')
        except IndexError:
            print(f'Could not find work subject directory for ')



