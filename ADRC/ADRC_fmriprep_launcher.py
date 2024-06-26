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
subjects = subjects[92:]
output_BIDS = os.path.join(root_proj,'ADRC_BIDS')
fmriprep_output = os.path.join(root_proj,'fmriprep_output')

work_dir = os.path.join(root_proj,'work_dir')

run_fmriprep = False

overwrite=False

datatype_fix = True

for subj in subjects:
    print(f'Starting subject {subj}')
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
    t1_bxh_orig = t1_path_orig.replace('.nii.gz','.bxh')

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
    if not os.path.exists(t1_json_path) or True:
        # read image
        if not os.path.exists(t1_json_path_orig) or overwrite:

            itk_image = sitk.ReadImage(t1_nii_path)

            # get metadata dict
            header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
            with open(t1_json_path_orig, "w") as outfile:
                json.dump(header, outfile, indent=4)
            """
            command = f'bxh2json --input {t1_bxh_orig}'
            os.system(command)
            """

        shutil.copy(t1_json_path_orig,t1_json_path)
        if datatype_fix:
            with open(t1_json_path, 'r') as file:
                lines = file.readlines()

            datatype_index = next((i for i, line in enumerate(lines) if line.strip().startswith('"datatype":')), None)
            if datatype_index is not None:
                lines[datatype_index] = '    "datatype": "anat",' + '\n'
                with open(t1_json_path, 'w') as file:
                    file.writelines(lines)
            else:
                print('Datatype not specified, fix presumably unnecessary')

    func_nii_orig = glob.glob(os.path.join(subj_folder_orig,'resting_state.nii.gz'))
    if np.size(func_nii_orig)==1:
        func_nii_orig = func_nii_orig[0]
        func_json_orig = func_nii_orig.replace('.nii.gz','.json')
        func_bxh_orig = func_nii_orig.replace(".nii.gz", '.bxh')

    #os.path.join(data_path,subj,'visit1/resting_state.nii.gz')  # change this with your file

    func_nii_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.nii.gz')
    func_json_path = os.path.join(func_folder,f'sub-{subj}_task-restingstate_run-01_bold.json')

    if not os.path.exists(func_nii_path):
        mkcdir([subj_folder, func_folder], None)
        shutil.copy(func_nii_orig,func_nii_path)

    # save dict in 'header.json'
    if not os.path.exists(func_json_path) or True:
        """
        if not os.path.exists(func_json_orig):
            # read image
            itk_image = sitk.ReadImage(func_nii_path)

            # get metadata dict
            header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
            with open(func_json_path, "w") as outfile:
                json.dump(header, outfile, indent=4)
        else:
            shutil.copy(func_json_orig,func_json_path)
        """
        if not os.path.exists(func_json_orig):
            func_bxh_orig_2 = func_bxh_orig.replace('.bxh','_fixed.bxh')
            func_json_orig_2 = func_json_orig.replace('.json','_fixed.json')
            new_line = '      <datapoints label="acquisitiontimeindex">1 12 2 13 3 14 4 15 5 16 6 17 7 18 8 19 9 20 10 21 ' \
                       '11 1 12 2 13 3 14 4 15 5 16 6 17 7 18 8 19 9 20 10 21 11 1 12 2 13 3 14 4 15 5 16 6 17 7 18 8 19 9 ' \
                       '20 10 21</datapoints>'

            with open(func_bxh_orig, 'r') as file:
                lines = file.readlines()

            # Find the index of the line starting with '<Tada>'
            z_index = next((i for i, line in enumerate(lines) if line.strip().startswith('<dimension type="z">')), None)
            spacing_index = next((i for i, line in enumerate(lines[z_index:]) if line.strip().startswith('<spacing>')),
                                 None) + z_index
            new_line_index = next(
                (i for i, line in enumerate(lines) if line.strip().startswith('<datapoints label="acquisitiontimeindex">')),
                None)

            if new_line_index is None and spacing_index is not None:
                # Insert the new line right below the '<Tada>' line
                lines.insert(spacing_index + 1, new_line + '\n')

                # Open the file for writing and overwrite its contents
                with open(func_bxh_orig_2, 'w') as file:
                    file.writelines(lines)

                command = f'bxh2json --input {func_bxh_orig_2}'
                os.system(command)
                shutil.move(func_json_orig_2, func_json_orig)

            elif spacing_index is None:
                print("'<spacing>' line not found in the file.")
            else:
                command = f'bxh2json --input {func_bxh_orig}'
                os.system(command)

        shutil.copy(func_json_orig, func_json_path)
        if datatype_fix:
            with open(func_json_path, 'r') as file:
                lines = file.readlines()

run_fmriprep = True
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



