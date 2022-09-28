import os, glob, shutil

orig_folder = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS/'
new_folder = '/Users/jas/jacques/Daniel_test/RARE_folder'
folder_subjects = glob.glob(os.path.join(orig_folder,'*/'))

for folder_subj in folder_subjects:
    anat_file = glob.glob(os.path.join(folder_subj,'ses-1','anat','*w.nii.gz'))
    try:
        new_path = os.path.join(new_folder,os.path.basename(anat_file[0]))
    except:
        print('hi')
    shutil.copy(anat_file[0], new_path)
    print(f'copied {anat_file[0]} to {new_path}')

