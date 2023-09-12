import numpy as np
import nibabel as nib
from scipy import stats
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from DTC.file_manager.file_tools import mkcdir, check_files
import shutil

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

def degree_connectivity(connectome, method = 'sum'):
    if method=='sum':
        return np.sum(connectome,1)


root = '/Volumes/Data/Badea/Lab'

ROI_legends = os.path.join(root,'atlases/IITmean_RPI/IITmean_RPI_index.xlsx')

excel_path = '/Users/jas/jacques/Jasien/Jasien_list.xlsx'

index1_to_2, _, index2_to_struct, _ = atlas_converter(ROI_legends)
index1_to_2.pop(0, None)
index2_to_1 = {v: k for k, v in index1_to_2.items()}

index_to_struct = {}
for key in index1_to_2.keys():
    index_to_struct[key] = index2_to_struct[index1_to_2[key]]

subj_val_base = {}
#subjects = [os.path.basename(file).split('_')[0].split('S')[1] for file in QSM_files]
#subjects = [os.path.basename(file).split('_')[0] for file in QSM_files]
subjects = ['T04086', 'T04129', 'T04248', 'T04300', 'T01257', 'T01277', 'T04472', 'T01402']


group_column = 'Ambulatory'
#group_column = 'Genotype'
p_value_sig = 0.05

for subj in subjects:
    subj_J = subj.replace('T','J')
    subj_val_base[subj] = get_group(subj_J, excel_path, subj_column = 'RUNNO', group_column = group_column)

subj_val = {key:val for key, val in subj_val_base.items() if val is not None}
subjects = list(subj_val.keys())

#values_list = list(np.unique(list(subj_val.values())))
#values_list.sort()


ROIs = [17, 53, 18, 54, 1031, 2031, 1023, 2023]
ROIs_comb = [(17,53), (18,54), (1031,2031), (1023,2023)]
#ROIs = list(index_to_struct.keys())


region_stats = True #Make the region comparison based on SAMBA stat files
connectome_stats = True #Make the degree of connectivity comparison based on connectomes


stat_folder = os.path.join(root, 'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/stats_by_region/labels/pre_rigid_native_space/IITmean_RPI/stats/studywide_label_statistics/')
#stat_types = ['b0', 'dwi', 'fa', 'volume'] b0 does not seem to be working yet
#stat_types = ['b0', 'dwi', 'fa', 'QSM', 'volume']
stat_types = ['fa', 'rd', 'md', 'ad', 'volume']
#QSM_files = filter_strings_with_substrings(QSM_files,subjects)

connectome_folder = os.path.join(root, 'mouse/Jasien_mrtrix_pipeline/connectomes')


if region_stats:
    for stat_type in stat_types:
        stat_file = os.path.join(stat_folder, f'studywide_stats_for_{stat_type}.txt')
        stat_db = pd.read_csv(stat_file, sep='\t')

        #ROIs = list(stat_db['ROI'][stat_db['ROI']!=0])

        group_vals = pd.DataFrame(subj_val, index=[2])

        stat_db = standardize_database(stat_db, subjects)

        stats_folder_results = f'/Users/jas/jacques/Jasien/ANOVA_results_all_{group_column}'
        stats_folder_results_sig = f'/Users/jas/jacques/Jasien/ANOVA_results_sig_{group_column}'
        stat_type_folder = os.path.join(stats_folder_results,stat_type)
        stat_type_folder_sig = os.path.join(stats_folder_results_sig,stat_type)
        mkcdir([stats_folder_results, stat_type_folder])

        for ROI in ROIs:
            stat_path = os.path.join\
                (stat_type_folder,f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-","_")}.png')
            stat_ROI = stat_db[stat_db['ROI']==ROI]
            stat_ROI = stat_ROI.drop(columns='ROI')
            stat_ROI = pd.concat([stat_ROI, group_vals])
            stat_ROI = stat_ROI.transpose()
            stat_ROI.columns = [stat_type, 'Groups']
            stat_ROI = stat_ROI.dropna()

            if stat_ROI[stat_type].isnull().all():
                print(f'Could not find stat type {stat_type}')
                continue
            ax = sns.boxplot(x='Groups', y=stat_type, data=stat_ROI, color='#99c2a2')
            ax = sns.swarmplot(x='Groups', y=stat_type, data=stat_ROI, color='#7d0013')
            #plt.xticks([0, 1], ['A', 'B'])

            plt.title(f'ROI {index_to_struct[ROI]}')

            grouped_data = [group[stat_type] for _, group in stat_ROI.dropna().groupby('Groups')]
            f_statistic, p_value = stats.f_oneway(*grouped_data)

            if group_column != 'Genotype':
                plt.text(0.1, 0.9, f'pvalue = {"{:.2f}".format(p_value)}', transform=plt.gca().transAxes, fontsize=12, color='red')

            plt.savefig(stat_path)
            if p_value<p_value_sig:
                mkcdir([stats_folder_results_sig, stat_type_folder_sig])
                stat_path_sig = os.path.join \
                    (stat_type_folder_sig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                shutil.copy(stat_path, stat_path_sig)


            plt.close()
            #for subject in subjects:

if connectome_stats:

    #stat_file = os.path.join(stat_folder, f'studywide_stats_for_fa.txt')
    #stat_db = pd.read_csv(stat_file, sep='\t')
    #stat_db = standardize_database(stat_db, subjects)

    stat_type = 'DConn'

    stats_folder_results = f'/Users/jas/jacques/Jasien/ANOVA_results_all_{group_column}'
    stats_folder_results_sig = f'/Users/jas/jacques/Jasien/ANOVA_results_sig_{group_column}'
    stat_type_folder = os.path.join(stats_folder_results, stat_type)
    stat_type_folder_sig = os.path.join(stats_folder_results_sig, stat_type)
    mkcdir([stats_folder_results, stat_type_folder])

    group_vals = pd.DataFrame(subj_val, index=[2])

    connectome_db = pd.DataFrame(columns=['ROI'])
    connectome_db['ROI'] = list(index1_to_2.keys())

    for subj in subj_val_base.keys():
        subj_j = subj.replace('T','J')
        connectome_path = os.path.join(connectome_folder,subj_j,f'{subj_j}_distances.csv')
        if not os.path.exists(connectome_path):
            continue
        connectome = np.genfromtxt(connectome_path, delimiter=',')

        degree_c = degree_connectivity(connectome)

        connectome_db[subj] = degree_c

    for ROI in ROIs:
        stat_ROI = connectome_db[connectome_db['ROI'] == ROI]
        stat_ROI = stat_ROI.drop(columns='ROI')
        stat_ROI = pd.concat([stat_ROI, group_vals])
        stat_ROI = stat_ROI.transpose()
        stat_ROI.columns = [stat_type, 'Groups']
        stat_ROI = stat_ROI.dropna()

        stat_path = os.path.join \
            (stat_type_folder, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')

        ax = sns.boxplot(x='Groups', y=stat_type, data=stat_ROI, color='#99c2a2')
        ax = sns.swarmplot(x='Groups', y=stat_type, data=stat_ROI, color='#7d0013')
        plt.title(f'ROI {index_to_struct[ROI]}')

        grouped_data = [group[stat_type] for _, group in stat_ROI.dropna().groupby('Groups')]
        f_statistic, p_value = stats.f_oneway(*grouped_data)

        if group_column != 'Genotype':
            plt.text(0.1, 0.9, f'pvalue = {"{:.2f}".format(p_value)}', transform=plt.gca().transAxes, fontsize=12,
                     color='red')

        plt.savefig(stat_path)
        if p_value < p_value_sig:
            mkcdir([stats_folder_results_sig, stat_type_folder_sig])
            stat_path_sig = os.path.join \
                (stat_type_folder_sig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
            shutil.copy(stat_path, stat_path_sig)

        plt.close()

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