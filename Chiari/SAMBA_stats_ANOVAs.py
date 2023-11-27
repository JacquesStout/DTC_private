import numpy as np
import nibabel as nib
from scipy import stats
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from DTC.file_manager.file_tools import mkcdir, check_files
import shutil, socket
import networkx as nx
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


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
                found=0
                for indice in indices:
                    if 'master' in all_subj[indice]:
                        indices = [indice]
                        found=1
                if not found:
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
    if method=='mean':
        return np.mean(connectome,1)


def clustering_coefficient(matrix):
    # Local clustering coefficient for each node
    local_clustering = np.sum(matrix @ matrix @ matrix, axis=1) / 2
    # Average clustering coefficient for the entire network
    return np.mean(local_clustering / (np.sum(matrix, axis=1) * (np.sum(matrix, axis=1) - 1)))


def eigenvector_centrality(matrix):
    _, vectors = eigsh(matrix, k=1, which='LM')
    centrality = np.abs(vectors.flatten())
    return centrality / np.sum(centrality)


root = '/Volumes/Data/Badea/Lab'

ROI_legends = os.path.join(root,'atlases/IITmean_RPI/IITmean_RPI_index.xlsx')

if 'santorini' in socket.gethostname().split('.')[0]:
    excel_path = '/Users/jas/jacques/Jasien/Jasien_list.xlsx'
    output_path = '/Users/jas/jacques/Jasien/ANOVA_results'
if socket.gethostname().split('.')[0] == 'lefkada':
    excel_path = '/Users/alex/jacques/Jasien/Jasien_list.xlsx'
    output_path =  '/Users/alex/jacques/Jasien/ANOVA_results'

index1_to_2, _, index2_to_struct, _ = atlas_converter(ROI_legends)
index1_to_2.pop(0, None)
index2_to_1 = {v: k for k, v in index1_to_2.items()}

index_to_struct = {}
for key in index1_to_2.keys():
    index_to_struct[key] = index2_to_struct[index1_to_2[key]]

subj_val_base = {}
#subjects = [os.path.basename(file).split('_')[0].split('S')[1] for file in QSM_files]
#subjects = [os.path.basename(file).split('_')[0] for file in QSM_files]
#subjects = ['T04086', 'T04129', 'T04248', 'T04300', 'T01257', 'T01277', 'T04472', 'T01402']
subjects_orig = ['T04086', 'T04129', 'T04300', 'T01257', 'T01277', 'T04472', 'T01402','T01501','T01516','T04602','T01541']

p_value_sig = 0.05

#values_list = list(np.unique(list(subj_val.values())))
#values_list.sort()

ROIs_type = 'subset'

if ROIs_type == 'full':
    ROIs = list(index_to_struct.keys())
if ROIs_type == 'subset':
    ROIs = [17, 53, 18, 54, 1031, 2031, 1023, 2023, 1016, 2016]
    #ROIs_comb = [(17,53), (18,54), (1031,2031), (1023,2023)]

output_path = os.path.join(output_path,ROIs_type)
mkcdir(output_path)

region_stats = True #Make the region comparison based on SAMBA stat files
pairwise_stats = False #Make the degree of connectivity comparison based on connectomes


stat_folder = os.path.join(root, 'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/stats_by_region/labels/pre_rigid_native_space/IITmean_RPI/stats/studywide_label_statistics/')
VBM_folder = os.path.join(root, 'mouse','VBM_21ADDecode03_IITmean_RPI_fullrun-work/')

#reg_stat_types = ['b0', 'dwi', 'fa', 'volume'] b0 does not seem to be working yet
#reg_stat_types = ['b0', 'dwi', 'fa', 'QSM', 'volume']
reg_stat_types = ['fa', 'rd', 'md', 'ad', 'volume', 'volume_prop']
reg_stat_types = ['volume_prop', 'StConn_DC','fa', 'rd', 'md', 'ad', 'volume']
reg_stat_types = ['FConn_eigenC','FConn_cluster','StConn_eigenC','Stconn_cluster','FConn_DC','StConn_DC']


pair_stat_types = ['Struct','Func']
#reg_stat_types = ['rd', 'md', 'ad']
#reg_stat_types = ['volume']
#QSM_files = filter_strings_with_substrings(QSM_files,subjects)

#connectome_folder = os.path.join(root, 'mouse/Jasien_mrtrix_pipeline/connectomes')
connectome_folder = os.path.join(root, 'human/Jasien/connectomes')
func_conn_path = os.path.join(connectome_folder,'functional_conn')

group_column = 'Ambulatory'
#group_column = 'Genotype'
#group_column = 'Lesion'

group_columns = ['Ambulatory', 'Genotype', 'Lesion']
#group_columns = ['Ambulatory']


if region_stats:

    for group_column in group_columns:
        print(f'Running it for group {group_column}')
        for subj in subjects_orig:
            subj_J = subj.replace('T','J')
            subj_val_base[subj] = get_group(subj_J, excel_path, subj_column = 'RUNNO', group_column = group_column)

        subj_val = {key:val for key, val in subj_val_base.items() if val is not None}
        subjects = list(subj_val.keys())

        stats_folder_results = os.path.join(output_path, f'ANOVA_results_all_{group_column}')
        stats_folder_results_sig = os.path.join(output_path, f'ANOVA_results_sig_{group_column}')
        stats_folder_results_fsig = os.path.join(output_path, f'ANOVA_results_fsig_{group_column}')

        for stat_type in reg_stat_types:

            print(f'Running the statistic {stat_type}')

            if 'prop' in stat_type:
                stat_type_db = stat_type.split('_')[0]
            else:
                stat_type_db = stat_type

            stat_file = os.path.join(stat_folder, f'studywide_stats_for_{stat_type_db}.txt')

            #ROIs = list(stat_db['ROI'][stat_db['ROI']!=0])

            group_vals = pd.DataFrame(subj_val, index=[2])


            stat_type_folder = os.path.join(stats_folder_results,stat_type)
            stat_type_folder_sig = os.path.join(stats_folder_results_sig,stat_type)
            stat_type_folder_fsig = os.path.join(stats_folder_results_fsig,stat_type)
            mkcdir([stats_folder_results, stat_type_folder])

            if 'DConn' in stat_type or 'FConn' in stat_type:
                stat_db = pd.DataFrame(columns=['ROI'])
                stat_db['ROI'] = list(index1_to_2.keys())

                for subj in subj_val_base.keys():
                    subj_j = subj.replace('T', 'J')

                    struct_conn_path = os.path.join(connectome_folder, subj_j, f'{subj_j}_distances.csv')
                    #fconn_path = os.path.join(func_conn_path, f'time_serts_{subj}_nartest.csv')
                    #fconnFC_path = os.path.join(func_conn_path, f'time_serFC_{subj}_nartest.csv')
                    fconn_path = os.path.join(func_conn_path, f'time_serFC_{subj_j}_nartest.csv')

                    if 'StConn' in stat_type:
                        connectome_path = struct_conn_path
                        row_method = 'sum'

                    if 'FConn' in stat_type:
                        connectome_path = fconn_path
                        row_method = 'mean'

                    connectome = np.genfromtxt(connectome_path, delimiter=',')

                    if 'DC' in stat_type:
                        stat_node = degree_connectivity(connectome, method='sum')
                    if 'cluster' in stat_type:
                        #G = nx.Graph(connectome)
                        #clustering_coeff = nx.clustering(G)
                        stat_node = clustering_coefficient(connectome)
                    if 'eigenC' in stat_type:
                        #G = nx.Graph(connectome)
                        #eigen_centervals = nx.eigenvector_centrality(G)
                        stat_node = eigenvector_centrality(connectome)
                    if not os.path.exists(connectome_path):
                        continue
                    try:
                        stat_db[subj] = stat_node
                    except:
                        print('hi')
            else:
                stat_db = pd.read_csv(stat_file, sep='\t')
                stat_db = standardize_database(stat_db, subjects)

            if 'prop' in stat_type:
                total_volumes = {'ROI': 'all'}
                for subject in subjects:
                    mask_path = os.path.join(VBM_folder,'preprocess','base_images',f'{subject}_mask.nii.gz')
                    brain_mask_data = nib.load(mask_path).get_fdata()
                    total_volumes[subject] = np.sum(brain_mask_data>0)

                #stat_db = stat_db.append(total_volumes, ignore_index=True)
                stat_db.loc[np.shape(stat_db)[0]] = total_volumes  # adding a row
                stat_db = stat_db.sort_index()
                last_row = (stat_db.iloc[-1, 1:])
                proportional_vals = stat_db.iloc[0:, 1:].div(last_row.iloc[:], axis=1)*100.0
                stat_db.iloc[0:, 1:] = proportional_vals.astype(float) #For some reason, div output is object and causes errors, forcing tit to be float
                #stat_db.iloc[0:, 1:] = stat_db.iloc[0:, 1:].div(last_row.iloc[:], axis=1)*100.0

                #stat_db.loc[len(stat_db)] = total_volumes

            if group_column == 'Genotype':
                group_vals = group_vals.replace(
                    {'APOE24': 'APOE4', 'APOE34': 'APOE4', 'APOE33': 'APOE3'}, regex=True)

            if group_column == 'Ambulatory':
                group_vals = group_vals.replace(
                    {'FA': 'A'}, regex=True)

            stat_ROIs = {}
            p_values = []

            for ROI in ROIs:

                stat_ROI = stat_db[stat_db['ROI']==ROI]
                stat_ROI = stat_ROI.drop(columns='ROI')
                stat_ROI = pd.concat([stat_ROI, group_vals])
                stat_ROI = stat_ROI.transpose()
                stat_ROI.columns = [stat_type, 'Groups']
                stat_ROI = stat_ROI.dropna()

                #if stat_ROI.isnull().all():
                #    print(f'Could not find stat type {stat_type}')
                #    break

                grouped_data = [group[stat_type] for _, group in stat_ROI.dropna().groupby('Groups')]
                f_statistic, p_value = stats.f_oneway(*grouped_data)

                stat_ROIs[ROI] = stat_ROI

                p_values.append(p_value)

            _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

            for i,ROI in enumerate(ROIs):
                stat_path = os.path.join\
                    (stat_type_folder,f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-","_")}.png')

                stat_ROI = stat_ROIs[ROI]
                p_value = p_values[i]
                corrected_p_value = corrected_p_values[i]

                ax = sns.boxplot(x='Groups', y=stat_type, data=stat_ROI, color='#99c2a2')
                ax = sns.swarmplot(x='Groups', y=stat_type, data=stat_ROI, color='#7d0013')
                #plt.xticks([0, 1], ['A', 'B'])

                plt.title(f'ROI {index_to_struct[ROI]}')

                p_value_text = plt.text(0.1, 0.9, f'pvalue = {"{:.2f}".format(p_value)}', transform=plt.gca().transAxes, fontsize=12, color='red')

                plt.savefig(stat_path)
                if p_value<p_value_sig:
                    mkcdir([stats_folder_results_sig, stat_type_folder_sig])
                    stat_path_sig = os.path.join \
                        (stat_type_folder_sig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                    shutil.copy(stat_path, stat_path_sig)

                if corrected_p_value<p_value_sig:
                    mkcdir([stats_folder_results_fsig, stat_type_folder_fsig])
                    p_value_text.remove()
                    fp_value_text = plt.text(0.1, 0.9, f'fpvalue = {"{:.2f}".format(corrected_p_value)}', transform=plt.gca().transAxes, fontsize=12, color='red')
                    stat_path_fsig = os.path.join \
                        (stat_type_folder_fsig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                    plt.savefig(stat_path_fsig)
                    #if not os.path.exists(stat_path_fsig):
                    #    shutil.copy(stat_path, stat_path_sig)


                plt.close()
                #for subject in subjects:

if pairwise_stats:

    for group_column in group_columns:
        print(f'Running it for group {group_column}')
        for subj in subjects_orig:
            subj_J = subj.replace('T','J')
            subj_val_base[subj] = get_group(subj_J, excel_path, subj_column = 'RUNNO', group_column = group_column)

        subj_val = {key:val for key, val in subj_val_base.items() if val is not None}
        subjects = list(subj_val.keys())

        stats_folder_results = os.path.join(output_path, f'ANOVA_results_all_{group_column}')
        stats_folder_results_sig = os.path.join(output_path, f'ANOVA_results_sig_{group_column}')
        stats_folder_results_fsig = os.path.join(output_path, f'ANOVA_results_fsig_{group_column}')

        for stat_type in pair_stat_types:

            print(f'Running the statistic {stat_type}')

            if 'prop' in stat_type:
                stat_type_db = stat_type.split('_')[0]
            else:
                stat_type_db = stat_type

            stat_file = os.path.join(stat_folder, f'studywide_stats_for_{stat_type_db}.txt')

            #ROIs = list(stat_db['ROI'][stat_db['ROI']!=0])

            group_vals = pd.DataFrame(subj_val, index=[2])


            stat_type_folder = os.path.join(stats_folder_results,stat_type)
            stat_type_folder_sig = os.path.join(stats_folder_results_sig,stat_type)
            stat_type_folder_fsig = os.path.join(stats_folder_results_fsig,stat_type)
            mkcdir([stats_folder_results, stat_type_folder])


            # Generate pairwise combinations and filter out the ones with the same element
            values = list(index1_to_2.keys())
            values = list(map(str,values))
            combinations_list = []
            for combination in combinations(values, 2):
                combinations_list.append('_'.join(combination))
                combinations_list.append('_'.join(reversed(combination)))

            stat_db = pd.DataFrame({'ROI': combinations_list})
            stat_db['ROI'] = combinations_list

            if 'DConn' in stat_type or 'FConn' in stat_type:

                for subj in subj_val_base.keys():
                    subj_j = subj.replace('T', 'J')

                    struct_conn_path = os.path.join(connectome_folder, subj_j, f'{subj_j}_distances.csv')
                    #fconn_path = os.path.join(func_conn_path, f'time_serts_{subj}_nartest.csv')
                    #fconnFC_path = os.path.join(func_conn_path, f'time_serFC_{subj}_nartest.csv')
                    fconn_path = os.path.join(func_conn_path, f'time_serFC_{subj_j}_nartest.csv')

                    if 'DConn'==stat_type:
                        connectome_path = struct_conn_path
                        connectome = np.genfromtxt(connectome_path, delimiter=',')
                        degree_c = degree_connectivity(connectome,method='sum')
                    if 'FConn'==stat_type:
                        connectome_path = fconn_path
                        connectome = np.genfromtxt(connectome_path, delimiter=',')
                        degree_c = degree_connectivity(connectome,method='mean')
                    if 'FConn_cluster'==stat_type:
                        connectome_path = fconn_path
                        connectome = np.genfromtxt(connectome_path, delimiter=',')
                        G = nx.Graph(connectome)
                        clustering_coeff = nx.clustering(G)
                        degree_c = clustering_coeff
                    if not os.path.exists(connectome_path):
                        continue
                    try:
                        stat_db[subj] = degree_c
                    except:
                        print('hi')

            stat_ROIs = {}
            p_values = []

            for ROI in ROIs:

                stat_ROI = stat_db[stat_db['ROI']==ROI]
                stat_ROI = stat_ROI.drop(columns='ROI')
                stat_ROI = pd.concat([stat_ROI, group_vals])
                stat_ROI = stat_ROI.transpose()
                stat_ROI.columns = [stat_type, 'Groups']
                stat_ROI = stat_ROI.dropna()

                #if stat_ROI.isnull().all():
                #    print(f'Could not find stat type {stat_type}')
                #    break

                grouped_data = [group[stat_type] for _, group in stat_ROI.dropna().groupby('Groups')]
                f_statistic, p_value = stats.f_oneway(*grouped_data)

                stat_ROIs[ROI] = stat_ROI

                p_values.append(p_value)

            _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

            for i,ROI in enumerate(ROIs):
                stat_path = os.path.join\
                    (stat_type_folder,f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-","_")}.png')

                stat_ROI = stat_ROIs[ROI]
                p_value = p_values[i]
                corrected_p_value = corrected_p_values[i]

                ax = sns.boxplot(x='Groups', y=stat_type, data=stat_ROI, color='#99c2a2')
                ax = sns.swarmplot(x='Groups', y=stat_type, data=stat_ROI, color='#7d0013')
                #plt.xticks([0, 1], ['A', 'B'])

                plt.title(f'ROI {index_to_struct[ROI]}')

                p_value_text = plt.text(0.1, 0.9, f'pvalue = {"{:.2f}".format(p_value)}', transform=plt.gca().transAxes, fontsize=12, color='red')

                plt.savefig(stat_path)
                if p_value<p_value_sig:
                    mkcdir([stats_folder_results_sig, stat_type_folder_sig])
                    stat_path_sig = os.path.join \
                        (stat_type_folder_sig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                    shutil.copy(stat_path, stat_path_sig)

                if corrected_p_value<p_value_sig:
                    mkcdir([stats_folder_results_fsig, stat_type_folder_fsig])
                    p_value_text.remove()
                    fp_value_text = plt.text(0.1, 0.9, f'fpvalue = {"{:.2f}".format(corrected_p_value)}', transform=plt.gca().transAxes, fontsize=12, color='red')
                    stat_path_fsig = os.path.join \
                        (stat_type_folder_fsig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                    plt.savefig(stat_path_fsig)
                    #if not os.path.exists(stat_path_fsig):
                    #    shutil.copy(stat_path, stat_path_sig)


                plt.close()
                #for subject in subjects: