import os, glob, shutil

subjects = ['sub-01277', 'sub-01402', 'sub-04472', 'sub-04129', 'sub-01257', 'sub-04300', 'sub-04086', 'sub-04248']

data_BIDS_path = '/Users/jas/jacques/Jasien/Dataset_BIDS/Chiari'

task_name = 'task-restingstate'

for subj in subjects:
    anat_path= os.path.join(data_BIDS_path,subj,'anat')
    func_path = os.path.join(data_BIDS_path,subj,'func')

    bxh_files = glob.glob(os.path.join(anat_path,'*.bxh'))
    for bxh_file in bxh_files:
        print(f'Removing {bxh_file}')
        os.remove(bxh_file)

    func_path = os.path.join(data_BIDS_path, subj, 'func')
    bxh_files = glob.glob(os.path.join(func_path,'*.bxh'))
    for bxh_file in bxh_files:
        print(f'Removing {bxh_file}')
        os.remove(bxh_file)

    func_files = glob.glob(os.path.join(func_path,'*.nii.gz')) + (glob.glob(os.path.join(func_path,'*.json')))
    for func_file in func_files:
        if 'task' not in func_file:
            func_name = os.path.basename(func_file)
            func_name_new = func_name.replace(subj, f'{subj}_task-restingstate')
            print(f'{os.path.join(func_path,func_name)}, {os.path.join(func_path,func_name_new)}')
            shutil.move(os.path.join(func_path,func_name), os.path.join(func_path,func_name_new))

    func_files = glob.glob(os.path.join(func_path,'*.nii.gz')) + (glob.glob(os.path.join(func_path,'*.json')))
    for func_file in func_files:
        if 'run-0' not in func_file:
            func_name = os.path.basename(func_file)
            func_name_new = func_name.replace('run-', 'run-0')
            print(f'{os.path.join(func_path,func_name)}, {os.path.join(func_path,func_name_new)}')
            shutil.move(os.path.join(func_path,func_name), os.path.join(func_path,func_name_new))




