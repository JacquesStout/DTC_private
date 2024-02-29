import os
import pandas as pd
import numpy as np
import glob, warnings

tractseg_outputs_subj_folder = '/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/TractSeg_outputs'
png_path = os.path.join(tractseg_outputs_subj_folder,'tract_analysis.png')

subjects_txt_path = '/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/AD_DECODE_subjects.txt'

testmode = True

check_csvs = True

missing_subj = []
missing_df_subj = []

if check_csvs:
    df_subj = pd.read_csv(subjects_txt_path, sep=" ", comment="#")
    subjects = df_subj["subject_id"].astype(str)

    for subject in subjects:
        subj_csv_path = os.path.join(tractseg_outputs_subj_folder,subject,f'Tractometry_{subject}.csv')
        if not os.path.exists(subj_csv_path):
            missing_subj.append(subject)

    all_folders = glob.glob(os.path.join(tractseg_outputs_subj_folder,'*/'))
    for subject_folder in all_folders:
        subject = subject_folder[-7:-1]
        if subject not in list(subjects):
            missing_df_subj.append(subject)

if np.size(missing_df_subj)>0:
    print(missing_df_subj)
    warnings.warn('The subjects listed above were not found in metadata file')

if np.size(missing_subj)>0:
    print(missing_subj)
    raise Exception('Missing subjects')

cmd = f'plot_tractometry_results -i {subjects_txt_path} -o {png_path} --mc'
if testmode:
    print(cmd)
else:
    os.system(cmd)
