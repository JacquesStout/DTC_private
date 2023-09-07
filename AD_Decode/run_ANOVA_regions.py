import numpy as np
import nibabel as nib
from scipy import stats
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from DTC.file_manager.file_tools import mkcdir, check_files


def get_group(subject, excel_path, subj_column = 'RUNNO', group_column = 'Risk'):
    df = pd.read_excel(excel_path)
    columns_subj = []
    for column_name in df.columns:
        if subj_column in column_name:
            columns_subj.append(column_name)

    columns_group = []
    for column_name in df.columns:
        if group_column in column_name:
            columns_group.append(column_name)

    #subjects = database[columns_subj[0]]

    #strip down subject
    df[columns_subj[0]] = df[columns_subj[0]].str.split('_').str[1]

    try:
        result = df[df[columns_subj[0]] == subject][columns_group[0]].values[0]
    except IndexError:
        result = None

    return result

def filter_strings_with_substrings(first_list, second_list):
    filtered_list = []

    for string1 in first_list:
        for string2 in second_list:
            if string2 in string1:
                filtered_list.append(string1)
                break  # Once a match is found, no need to check further

    return filtered_list

def standardize_database(db, subjects):
    all_subj = db.columns.values
    all_subj_trunc = [subj.split('_')[0] for subj in all_subj]
    for i,subj in enumerate(subjects):
        indices = [i for i, x in enumerate(all_subj_trunc) if x == subj]
        if np.size(indices)>1:
            test1 = np.all(abs(db[all_subj[indices[0]]]-db[all_subj[indices[1]]])<1e-1)
            if not test1:
                print(f'{subj} has 2 dissimilar instances')
            if np.size(indices)>2:
                test2 = np.all(abs(db[all_subj[indices[1]]]-db[all_subj[indices[2]]])<1e-1)
                if not test2:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices)>3:
                test3 = np.all(abs(db[all_subj[indices[2]]]-db[all_subj[indices[3]]])<1e-1)
                if not test3:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices)>4:
                test4 = np.all(abs(db[all_subj[indices[3]]]-db[all_subj[indices[4]]])<1e-1)
                if not test4:
                    print(f'{subj} has 2 dissimilar instances')
            for drop_indice in np.arange(1,np.size(indices)):
                db = db.drop(all_subj[indices[drop_indice]], axis=1)
            db = db.rename(columns={all_subj[indices[0]]: all_subj[indices[0]].split('_')[0]})

    col_ordered = ['ROI']+subjects
    col_ordered = [item for item in col_ordered if item in list(db.columns.values)]

    db = db.drop(columns=[col for col in db if col not in col_ordered])
    db = db[col_ordered + [c for c in db.columns if c not in col_ordered]]
    return db

ROI_legends = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx'
excel_path = '/Users/jas/jacques/AD_Decode_excels/AD_DecodeList_2023test.xlsx'

index1_to_2, _, index2_to_struct, _ = atlas_converter(ROI_legends)
index1_to_2.pop(0, None)
index_to_struct = {}
for key in index1_to_2.keys():
    index_to_struct[key] = index2_to_struct[index1_to_2[key]]

QSM_folder = '/Users/jas/jacques/AD_Decode/QSM_MDT/smoothed_1_5_masked/'
QSM_files = glob.glob(os.path.join(QSM_folder,'*.nii.gz'))
QSM_files.sort()

subj_val_base = {}
#subjects = [os.path.basename(file).split('_')[0].split('S')[1] for file in QSM_files]
subjects = [os.path.basename(file).split('_')[0] for file in QSM_files]

for subj in subjects:
    subj_stripped = subj.split('S')[1]
    subj_val_base[subj] = get_group(subj_stripped, excel_path, subj_column = 'BIAC_RUNNO')

subj_val = {key:val for key, val in subj_val_base.items() if val is not None}
subjects = list(subj_val.keys())

#values_list = list(np.unique(list(subj_val.values())))
#values_list.sort()

#QSM_files = filter_strings_with_substrings(QSM_files,subjects)
stat_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/stats_by_region/labels/pre_rigid_native_space/IITmean_RPI/stats/studywide_label_statistics/'

stat_types = ['b0', 'dwi', 'fa', 'QSM', 'volume']
#stat_types = ['dwi', 'fa', 'QSM', 'volume']
#ROIs = [17, 53, 18, 54, 1031, 2031, 1023, 2023]
for stat_type in stat_types:
    stat_file = os.path.join(stat_folder, f'studywide_stats_for_{stat_type}.txt')
    stat_db = pd.read_csv(stat_file, sep='\t')

    ROIs = list(stat_db['ROI'][stat_db['ROI']!=0])

    group_vals = pd.DataFrame(subj_val, index=[2])

    stat_db = standardize_database(stat_db, subjects)

    stats_folder_results = '/Users/jas/jacques/AD_Decode/ANOVA_results'
    stat_type_folder = os.path.join(stats_folder_results,stat_type)
    mkcdir([stats_folder_results, stat_type_folder])

    for ROI in ROIs:
        stat_path = os.path.join\
            (stat_type_folder,f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-","_")}.png')
        stat_ROI = stat_db[stat_db['ROI']==ROI]
        stat_ROI = stat_ROI.drop(columns='ROI')
        stat_ROI = pd.concat([stat_ROI, group_vals])
        stat_ROI = stat_ROI.transpose()
        stat_ROI.columns = [stat_type, 'Groups']
        ax = sns.boxplot(x='Groups', y=stat_type, data=stat_ROI, color='#99c2a2')
        ax = sns.swarmplot(x='Groups', y=stat_type, data=stat_ROI, color='#7d0013')
        plt.title(f'ROI {index_to_struct[ROI]}')
        plt.savefig(stat_path)
        plt.close()
        #for subject in subjects:


"""
#VBM test code


design_matrix = np.zeros([np.size(subjects),np.size(values_list)])

for i,subj in enumerate(subjects):
    design_matrix[i,values_list.index(subj_val[subj])] = 1

nifti_data = [nib.load(file).get_fdata() for file in QSM_files]


nifti_data = np.array(nifti_data)
design_matrix = design_matrix.T  # Transpose the design matrix
n_voxels = np.prod(nifti_data.shape[1:])  # Calculate the number of voxels
n_samples = nifti_data.shape[0]
n_conditions = design_matrix.shape[0]
n_samples_per_condition = n_samples // n_conditions
0
# Reshape NIfTI data for voxel-wise analysis
nifti_data_reshaped = nifti_data.reshape((n_samples, n_voxels))

# Prepare a list to store ANOVA results for each voxel
anova_results = []

# Perform ANOVA at each voxel
for voxel_idx in range(n_voxels):
    voxel_data = nifti_data_reshaped[:, voxel_idx]
    voxel_data_per_condition = voxel_data.reshape((n_conditions, n_samples_per_condition))
    f_statistic, p_value = stats.f_oneway(*voxel_data_per_condition)
    anova_results.append((f_statistic, p_value))

# Reshape the results back to the 3D NIfTI image dimensions
anova_results = np.array(anova_results).reshape(nifti_data.shape[1:])
"""