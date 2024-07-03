import numpy as np
import nibabel as nib
from scipy import stats
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from DTC.file_manager.file_tools import mkcdir, check_files
import socket, time, shutil
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from plotnine import ggplot


def get_group(subject, data_path, subj_column='RUNNO', group_column='Risk'):
    ext = os.path.splitext(data_path)[1]
    if ext == '.xlsx':
        df = pd.read_excel(data_path)
    elif ext == '.csv':
        df = pd.read_csv(data_path)
    else:
        txt = f'Unrecognized file type for {data_path}'
        raise Exception(txt)
    columns_subj = []
    for column_name in df.columns:
        if subj_column in column_name:
            columns_subj.append(column_name)

    columns_group = []
    for column_name in df.columns:
        if group_column in column_name:
            columns_group.append(column_name)

    # subjects = database[columns_subj[0]]

    # strip down subject

    # df[columns_subj[0]] = df[columns_subj[0]].str.split('_').str[1]
    df[columns_subj[0]] = df[columns_subj[0]].astype(str)

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
    db = db.dropna(axis=1, how='all')
    all_subj = db.columns.values
    all_subj_trunc = [subj.split('_')[0] for subj in all_subj]
    for i, subj in enumerate(subjects):
        indices = [i for i, x in enumerate(all_subj_trunc) if x == subj]
        if np.size(indices) > 1:
            test1 = np.all(abs(db[all_subj[indices[0]]] - db[all_subj[indices[1]]]) < 1e-1)
            if not test1:
                print(f'{subj} has 2 dissimilar instances')
            if np.size(indices) > 2:
                test2 = np.all(abs(db[all_subj[indices[1]]] - db[all_subj[indices[2]]]) < 1e-1)
                if not test2:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices) > 3:
                test3 = np.all(abs(db[all_subj[indices[2]]] - db[all_subj[indices[3]]]) < 1e-1)
                if not test3:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices) > 4:
                test4 = np.all(abs(db[all_subj[indices[3]]] - db[all_subj[indices[4]]]) < 1e-1)
                if not test4:
                    print(f'{subj} has 2 dissimilar instances')
            for drop_indice in np.arange(1, np.size(indices)):
                db = db.drop(all_subj[indices[drop_indice]], axis=1)
            db = db.rename(columns={all_subj[indices[0]]: all_subj[indices[0]].split('_')[0]})

    col_ordered = ['ROI'] + subjects
    col_ordered = [item for item in col_ordered if item in list(db.columns.values)]

    db = db.drop(columns=[col for col in db if col not in col_ordered])
    db = db[col_ordered + [c for c in db.columns if c not in col_ordered]]
    return db


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


def get_large_folders(folder_list, min_size_bytes):
    large_folders = []
    folder_sizes = []
    for folder_path in folder_list:
        folder_size = get_folder_size(folder_path)
        if folder_size > min_size_bytes:
            large_folders.append(folder_path)
            folder_sizes.append(folder_size)
    return large_folders, folder_sizes


def file_recent(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        return False

    # Get the file's creation timestamp
    file_stat = os.stat(file_path)
    creation_time = file_stat.st_ctime

    # Get the current timestamp
    current_time = time.time()

    # Calculate the time difference in seconds
    time_difference = current_time - creation_time

    # Check if the file was created less than a day ago (86400 seconds in a day)
    return time_difference < 86400


def degree_connectivity(connectome, method='sum'):
    if method == 'sum':
        return np.sum(connectome, 1)


def round_low_ten(number):
    return (number // 10) * 10


ROI_legends = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx'
VBM_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/'
stat_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/stats_by_region/labels/pre_rigid_native_space/IITmean_RPI/stats/studywide_label_statistics/'
root = '/Volumes/Data/Badea/Lab'

if socket.gethostname().split('.')[0] == 'santorini':
    excel_path = '/Users/jas/jacques/AD_Decode/AD_DECODE_data2.csv'
    output_path = '/Users/jas/jacques/AD_Decode/ANOVA_results/'
# if socket.gethostname().split('.')[0] == 'lefkada':

# QSM_folder = '/Users/jas/jacques/AD_Decode/QSM_MDT/smoothed_1_5_masked/'
# QSM_files = glob.glob(os.path.join(QSM_folder,'*.nii.gz'))
# QSM_files.sort()
# subjects = [os.path.basename(file).split('_')[0].split('S')[1] for file in QSM_files]
# subj_val_base = {}
# subjects = [os.path.basename(file).split('_')[0] for file in QSM_files]

index1_to_2, _, index2_to_struct, _ = atlas_converter(ROI_legends)
index1_to_2.pop(0, None)
index_to_struct = {}
for key in index1_to_2.keys():
    index_to_struct[key] = index2_to_struct[index1_to_2[key]]

# df_meta = pd.read_excel(excel_path)

subj_column = 'MRI_Exam'
group_column = 'Risk'

covariate_column = 'age'
group_columns = ['Risk', 'genotype']

datapath = '/Volumes/Data/Badea/ADdecode.01/Data/Anat/'
connectome_folder = os.path.join(root, 'mouse/mrtrix_ad_decode/connectome')
subject_text = os.path.join(output_path, 'subjects.txt')

if file_recent(subject_text):
    subjects = []
    with open(subject_text, "r") as file:
        for line in file:
            subject = line.strip()  # Remove newline character
            subjects.append(subject)
else:
    subjects_folders_path = glob.glob(os.path.join(datapath, '*/'))
    min_size_bytes = 250 * 1024 * 1024
    subjects_folders_path_true, _ = get_large_folders(subjects_folders_path, min_size_bytes)
    subjects_folders = [os.path.split(subject_folder)[0].split('/')[-1] for subject_folder in
                        subjects_folders_path_true]
    subjects = ['S' + subj.split('_')[1] for subj in subjects_folders]

    with open(subject_text, "w") as file:
        for subject in subjects:
            file.write(subject + "\n")

# ROIs = [17, 53, 18, 54, 1031, 2031, 1023, 2023]
ROIs_comb = [(17, 53), (18, 54), (1031, 2031), (1023, 2023)]
ROIs = list(index_to_struct.keys())
# ROIs = [17]


stat_types = ['fa', 'md', 'ad', 'QSM', 'volume', 'volume_prop', 'DConn']
stat_types = ['mrtrixfa', 'mrtrixmd', 'mrtrixad', 'QSM', 'volume', 'volume_prop', 'DConn']
#stat_types = ['mrtrixfa']
# stat_types = ['dwi', 'fa', 'QSM', 'volume']
# ROIs = [17, 53, 18, 54, 1031, 2031, 1023, 2023]

region_stats = True
connectome_stats = True
geno_strip = True
risk_strip = True

p_value_sig = 0.05

subj_val_base = {}
subj_cov_base = {}

p_xval = -1
p_yval = 1

p_xval = 0.3
p_yval = 0.9

plottype = 'box'
plottype = 'violin'

for group_column in group_columns:

    for subj in subjects:
        subj_stripped = subj.split('S0')[1]
        # subj = df_meta[subj_column][ind]
        # subj_val_base[subj] = get_group(subj_stripped, excel_path, subj_column = 'BIAC_RUNNO', group_column = group_column)
        subj_val_base[subj] = get_group(subj_stripped, excel_path, subj_column=subj_column, group_column=group_column)

    for subj in subjects:
        subj_stripped = subj.split('S0')[1]
        # subj = df_meta[subj_column][ind]
        # subj_val_base[subj] = get_group(subj_stripped, excel_path, subj_column = 'BIAC_RUNNO', group_column = group_column)
        subj_cov_base[subj] = get_group(subj_stripped, excel_path, subj_column=subj_column,
                                        group_column=covariate_column)

    subj_val = {key: val for key, val in subj_val_base.items() if val is not None}
    subj_cov = {key: int(val) for key, val in subj_cov_base.items() if val is not None}
    subjects = list(subj_val.keys())

    stats_folder_results = os.path.join(output_path, f'ANCOVA_results_all_{group_column}')
    stats_folder_results_sig = os.path.join(output_path, f'ANCOVA_results_sig_{group_column}')
    stats_folder_results_fsig = os.path.join(output_path, f'ANCOVA_results_fsig_{group_column}')

    print(f'Running the stats analysis for {group_column}')

    for stat_type in stat_types:

        if 'prop' in stat_type:
            stat_type_db = stat_type.split('_')[0]
        else:
            stat_type_db = stat_type

        stat_file = os.path.join(stat_folder, f'studywide_stats_for_{stat_type_db}.txt')

        print(f'Running the statistic {stat_type}')

        # ROIs = list(stat_db['ROI'][stat_db['ROI']!=0])

        group_vals = pd.DataFrame(subj_val, index=[2])
        cov_vals = pd.DataFrame(subj_cov, index=[2])

        stat_type_folder = os.path.join(stats_folder_results, stat_type)
        stat_type_folder_sig = os.path.join(stats_folder_results_sig, stat_type)
        stat_type_folder_fsig = os.path.join(stats_folder_results_fsig, stat_type)
        mkcdir([stats_folder_results, stat_type_folder])

        if 'DConn' not in stat_type:
            stat_db = pd.read_csv(stat_file, sep='\t')
            stat_db = standardize_database(stat_db, subjects)

        if 'prop' in stat_type:
            total_volumes = {'ROI': 'all'}
            for subject in subjects:
                mask_path = os.path.join(VBM_folder, 'preprocess', 'base_images', f'{subject}_mask.nii.gz')
                brain_mask_data = nib.load(mask_path).get_fdata()
                total_volumes[subject] = np.sum(brain_mask_data > 0)

            stat_db = stat_db.append(total_volumes, ignore_index=True)

            last_row = stat_db.iloc[-1, 1:]
            stat_db.iloc[1:, 1:] = stat_db.iloc[1:, 1:].div(last_row.iloc[:], axis=1) * 100

            # stat_db.loc[len(stat_db)] = total_volumes

        if 'DConn' in stat_type:
            stat_db = pd.DataFrame(columns=['ROI'])
            stat_db['ROI'] = list(index1_to_2.keys())

            for subj in subj_val_base.keys():
                # subj_j = subj.replace('T','J')
                connectome_path = os.path.join(connectome_folder, f'{subj}_distances.csv')
                if not os.path.exists(connectome_path):
                    continue
                connectome = np.genfromtxt(connectome_path, delimiter=',')

                degree_c = degree_connectivity(connectome)

                stat_db[subj] = degree_c

        if 'CBF' in stat_type:
            medians = stat_db.median()
            filtered_subj = stat_db.columns[medians<1]
            stat_db.drop(columns=filtered_subj, inplace = True)

        if group_column is 'genotype' and geno_strip:
            group_vals = group_vals.replace({'APOE33': 'APOE3', 'APOE34': 'APOE4', 'APOE44': 'APOE4', 'APOE23': 'APOE3'}, regex=True)
        if group_column is 'Risk' and risk_strip:
            group_vals = group_vals[(group_vals != 'MCI') & (group_vals != 'AD')]
        stat_ROIs = {}
        ancovas = {}
        p_values_stat = []
        p_values_covar = []
        for ROI in ROIs:
            stat_ROI = stat_db[stat_db['ROI'] == ROI]
            stat_ROI = stat_ROI.drop(columns='ROI')
            stat_ROI = pd.concat([stat_ROI, group_vals, cov_vals])
            stat_ROI = stat_ROI.transpose()
            stat_ROI.columns = [stat_type, group_column, covariate_column]
            stat_ROI = stat_ROI.dropna()
            stat_ROI[covariate_column] = stat_ROI[covariate_column].astype({covariate_column: 'int64'})
            stat_ROI[stat_type] = stat_ROI[stat_type].astype({stat_type: 'float64'})

            grouped_data = [group[stat_type] for _, group in stat_ROI.dropna().groupby(group_column)]
            # f_statistic, p_value = stats.f_oneway(*grouped_data)
            anc = pg.ancova(data=stat_ROI, dv=stat_type, covar=covariate_column, between=group_column)

            stat_ROIs[ROI] = stat_ROI
            ancovas[ROI] = anc
            p_values_stat.append(anc['p-unc'][0])
            p_values_covar.append(anc['p-unc'][1])

        _, p_values_stat_corr, _, _ = multipletests(p_values_stat, method='fdr_bh')
        _, p_values_covar_corr, _, _ = multipletests(p_values_covar, method='fdr_bh')

        for i, ROI in enumerate(ROIs):
            stat_path = os.path.join \
                (stat_type_folder, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
            stat_lm_path = os.path.join \
                (stat_type_folder, f'ANOVA_lm_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')

            stat_ROI = stat_ROIs[ROI]
            p_value_stat = p_values_stat[i]
            p_value_stat_corr = p_values_stat_corr[i]

            p_value_covar = p_values_covar[i]
            p_value_covar_corr = p_values_covar_corr[i]

            list_covars = list(stat_ROI[covariate_column])

            xaxis_custom = np.arange(round_low_ten(np.min(list_covars)), np.max(list_covars), 10)

            mylm = sns.lmplot(data=stat_ROI, x=covariate_column, y=stat_type, hue=group_column)
            mylm.fig.suptitle(f'ROI {index_to_struct[ROI]}')

            plt.savefig(stat_lm_path)
            plt.close()

            """
            #ax= sns.scatterplot(data=stat_ROI, x=covariate_column, y=stat_type, hue=stat_ROI[group_column].tolist(), ax=axs[ax_val])

            sns.lmplot(
                data=stat_ROI,
                x=covariate_column,
                y=stat_type,
                row =group_column,
                hue=stat_ROI[group_column].tolist(),
                ci=None,  # Set confidence intervals for the regression lines
                palette='colorblind',  # Change the color palette as needed
            )
            """

            myplt = plt.figure(figsize=(6, 8))

            ncols = 1
            if ncols==1:
                axs = []
                axs.append(myplt.subplots(ncols=1))
            else:
                axs = myplt.subplots(ncols=ncols)
            ax_val = 0

            #ax1= sns.scatterplot(data=stat_ROI, x=covariate_column, y=stat_type, hue=stat_ROI[group_column].tolist(), ax=axs[ax_val])
            #ax_val+=1

            if plottype == 'box':
                ax2 = sns.boxplot(x=group_column, y=stat_type, data=stat_ROI, hue=stat_ROI[group_column].tolist(), ax=axs[ax_val], fill=False)
            if plottype == 'violin':
                ax2 = sns.violinplot(x=group_column, y=stat_type, data=stat_ROI, hue=stat_ROI[group_column].tolist(), ax=axs[ax_val], fill=False)
            ax2 = sns.swarmplot(x=group_column, y=stat_type, data=stat_ROI, hue=stat_ROI[group_column].tolist(), ax=axs[ax_val])
            ax2.legend([], [], frameon=False)
            ax_val+=1

            #sns.boxplot(data=stat_ROI, x=group_column, y=covariate_column, hue=stat_ROI[group_column].tolist(), ax=axs[ax_val])

            #ax = sns.boxplot(x='Groups', y=stat_type, data=stat_ROI, color='#99c2a2')
            #ax = sns.swarmplot(x='Groups', y=stat_type, data=stat_ROI, color='#7d0013')

            #plt.xticks([0, 1], ['A', 'B'])

            plt.title(f'ROI {index_to_struct[ROI]}')

            p_value_stat_text = plt.text(p_xval, p_yval, f'{group_column} pvalue = {"{:.2f}".format(p_value_stat)}', transform=plt.gca().transAxes, fontsize=12, color='red')
            p_value_covar_text = plt.text(p_xval, p_yval-0.05, f'{covariate_column} pvalue = {"{:.2f}".format(p_value_covar)}', transform=plt.gca().transAxes, fontsize=12, color='blue')

            #plt.show()

            plt.savefig(stat_path)

            if p_value_stat < p_value_sig:
                mkcdir([stats_folder_results_sig, stat_type_folder_sig])
                stat_path_sig = os.path.join \
                    (stat_type_folder_sig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                stat_lm_path_sig = os.path.join \
                    (stat_type_folder_sig, f'ANOVA_lm_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                plt.savefig(stat_path_sig)
                shutil.copy(stat_lm_path, stat_lm_path_sig)

            if p_value_stat_corr < p_value_sig:
                mkcdir([stats_folder_results_fsig, stat_type_folder_fsig])
                p_value_stat_text.remove()
                p_value_covar_text.remove()
                fp_value_text = plt.text(p_xval, p_yval, f'{group_column} fpvalue = {"{:.2f}".format(p_value_stat_corr)}',
                                         transform=plt.gca().transAxes, fontsize=12, color='red')
                fp_value_text = plt.text(p_xval, p_yval+0.05, f'{covariate_column} fpvalue = {"{:.2f}".format(p_value_covar_corr)}',
                                         transform=plt.gca().transAxes, fontsize=12, color='blue')
                stat_path_fsig = os.path.join \
                    (stat_type_folder_fsig, f'ANOVA_boxplot_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                stat_lm_path_fsig = os.path.join(stat_type_folder_fsig, f'ANOVA_lm_{stat_type}_{index_to_struct[ROI].replace("-", "_")}.png')
                plt.savefig(stat_path_fsig)
                shutil.copy(stat_lm_path, stat_lm_path_fsig)
                # if not os.path.exists(stat_path_fsig):
                #    shutil.copy(stat_path, stat_path_sig)

            plt.close()

            # for subject in subjects: