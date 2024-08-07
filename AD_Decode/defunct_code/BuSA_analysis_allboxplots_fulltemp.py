#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:35 2023

@author: ali
"""
import os, glob
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import socket
from DTC.file_manager.file_tools import mkcdir, check_files
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, remote_pickle, checkfile_exists_all, write_parameters_to_ini, \
    read_parameters_from_ini
from skfda.representation import FDataGrid
import skfda


def outlier_removal(values, qsep=3):
    iqr = abs(np.quantile(values, 0.25) - np.quantile(values, 0.75))
    median = np.quantile(values, 0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values


remote = False

if len(sys.argv) < 2:
    #project = 'V0_9_10template_100_6_interhe_majority'
    project = 'V0_9_reg_superiorfrontalleft_precentralleft_6'
else:
    project = sys.argv[1]

loc = 'munin'

if 'santorini' in socket.gethostname().split('.')[0]:
    root = f'/Users/jas/jacques/AD_Decode/BuSA_analysis/{project}'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
else:
    root = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/'
    master = '/Users/ali/Desktop/Dec23/BuSA/AD_DECODE_data_stripped.csv'

if loc == 'kos':
    root = f'/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'
elif loc == 'munin':
    root = f'/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/{project}'

if remote:
    from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
    from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths
    from DTC.file_manager.computer_nav import glob_remote, load_df_remote

    username, passwd = getfromfile(os.path.join(os.environ['HOME'], 'remote_connect.rtf'))
    root = f'/mnt/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'
    inpath, _, _, sftp = get_mainpaths(remote, project='AD_Decode', username=username, password=passwd)
else:
    sftp = None

project_headfile_folder = '/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/'
project_summary_file = os.path.join(project_headfile_folder,project+'.ini')

if not os.path.exists(project_summary_file):
    txt = f'Could not find configuration file at {project_summary_file}'
    raise Exception(txt)
else:
    params = read_parameters_from_ini(project_summary_file)


template_subjects = params['template_subjects']
added_subjects = params['added_subjects']
num_bundles = int(params['num_bundles'])
removed_list = params['removed_list']

full_subjects_list = template_subjects + added_subjects

for remove in removed_list:
    if remove in full_subjects_list:
        full_subjects_list.remove(remove)

master_df = pd.read_csv(master)

length_cut = 40
if length_cut is None:
    length_str = ''
else:
    length_str = f'_Lengthcut{length_cut}'

figures_path = os.path.join(root, f'Figures{length_str}')
excel_path = os.path.join(root, f'Excels{length_str}')
stats_path = os.path.join(root, 'stats')

mkcdir([stats_path, figures_path, excel_path])

siding = False

if not siding:
    bundles = [f'_bundle_{bundle_id}.xlsx' for bundle_id in np.arange(num_bundles)]

num_groups = 5

tr_list = [0] + [np.quantile(master_df['age'], (i + 1) * (1 / num_groups)) for i in np.arange(num_groups)] + [
    1 + np.max(master_df['age'])]

# sds = np.zeros([num_groups,num_bundles*2])
# means = np.zeros([num_groups,num_bundles*2])

test = False

if not siding:
    side_spec = ''
    sides = ['all']
else:
    side_spec = 'Side_'

if test:
    bundles = [f'_left_bundle_0.xlsx']

if not siding:
    columns_list = [f'ID_{bundle.split("_")[2].split(".")[0]}' for bundle in bundles]
else:
    columns_list = [f'{side_spec}{bundle.split("_")[1]}_ID_{bundle.split("_")[3].split(".")[0]}' for bundle in bundles]

datameans = pd.DataFrame(columns=(['Age Group'] + columns_list))
datasds = pd.DataFrame(columns=(['Age Group'] + columns_list))

verbose = True

ref = 'mrtrixfa'

figures_box_path = os.path.join(figures_path, f'boxsquares_age_{ref}')
figures_agegrouping_path = os.path.join(figures_path, f'agegrouping_mean_{ref}')

mkcdir([figures_box_path, figures_agegrouping_path])

# meanbundle.split('_')[1]s = {}
# sds = {}

testmode = False
verbose = True

allvars = {}
allmeans = {}

savedatagrid_figs = True
basis = skfda.representation.basis.MonomialBasis(n_basis=10)

if siding:
    total_num_bundles = np.size(bundles)/2
else:
    total_num_bundles = np.size(bundles)


for bundle in bundles:

    if not siding:
        side = ''
        side_str = ''
    else:
        if 'left' in bundle:
            side = 'left'
        if 'right' in bundle:
            side = 'right'
        side_str = '_' + side

    bundle_num = bundle.split('bundle_')[1].split('.')[0]

    fig_bundle_path = os.path.join(figures_box_path, f'bundle{side_str}_{bundle_num}_boxsquaremodel.png')
    """
    this_bundle_subjs = [i for i in all_subj_bundles if bundle in i]
    this_bundle_subjs = sorted(this_bundle_subjs)
    if testmode:
        this_bundle_subjs = this_bundle_subjs[:3]
    """
    # bundle_df = pd.DataFrame()
    for subj in full_subjects_list:

        subj_bundle_path = os.path.join(stats_path, subj+bundle)
        if remote:
            load_df_remote(subj_bundle_path, sftp)
        else:
            temp = pd.read_excel(subj_bundle_path)

        # temp = pd.DataFrame()
        temp['Subject'] = subj[2:6]
        index = master_df["MRI_Exam"] == int(subj[2:6])
        try:
            temp['age'] = master_df[index]['age'].iloc[0]
            # temp['sex'] = master_df[index]['sex']
        except:
            print(f'Subject {subj[:6]} is missing from excel database')
            continue
        # temp = temp.to_numpy()
        if 'bundle_df' not in locals():
            bundle_df = temp
        else:
            bundle_df = pd.concat([bundle_df, temp])

        bundle_df[f'average{ref}'] = np.mean(bundle_df[column_names],1)
        column_indices = [bundle_df.columns.get_loc(column_names[i]) for i in np.arange(np.size(column_names))]
        #bundle_df = pd.DataFrame()
        num_streamlines = np.shape(bundle_df)[0]
        streamlines_fas = [bundle_df.iloc[i,column_indices].to_list() for i in np.arange(num_streamlines)]

        Y= FDataGrid(streamlines_fas)
        Y_basis = Y.to_basis(basis)
        var = skfda.exploratory.stats.var(Y_basis)
        mean = skfda.exploratory.stats.mean(Y_basis)

        if savedatagrid_figs:
            from skfda.exploratory.visualization import Boxplot
            figs_datagridcomparison = os.path.join(figures_path, 'datagridtostream_comparison')
            mkcdir(figs_datagridcomparison)
            figs_bundle_subj_path = os.path.join(figs_datagridcomparison, f'subj_{subj[2:6]}_side_{side}_bundle_{bundle_num}')
            Y_small= FDataGrid(streamlines_fas[:5])
            Y_basis_small = Y_small.to_basis(basis)
            fig, ax = plt.subplots()
            Y_Box = Boxplot(Y_small)
            #Y_Box.plot()
            Y_basis_small.plot(axes=ax)
            Y_small.scatter(axes=ax)
            plt.savefig(figs_bundle_subj_path)
            if verbose:
                print(f'Saved figure at {figs_bundle_subj_path}')
            plt.close()

        allvars[subj[2:6], side, bundle_num] = var
        allmeans[subj[2:6], side, bundle_num] = mean

    bundle_df = bundle_df.dropna()

    if length_cut is not None:
        bundle_df = bundle_df[bundle_df['Length'] >= int(length_cut)]

    column_names = []
    for i in range(1, 50):
        column_names.append("point_" + str(i) + f"_{ref}")
    bundle_df[f'average{ref}'] = np.mean(bundle_df[column_names], 1)

    bundle_df_reduced = bundle_df[[f"average{ref}", "age"]]
    #    bundle_df_reduced = bundle_df_reduced.iloc[ 0:2000]

    # bundle_df_reduced.boxplot(column='averageFA',by='age')
    # sns.boxplot(x="age", y="averageFA", data=bundle_df_reduced)
    sns.lmplot(x="age", y=f"average{ref}", data=bundle_df_reduced, x_estimator=np.mean, order=2)
    plt.title(f'Boxplot for bundle {int(bundle_num) + 1}', y=0.9)

    plt.savefig(fig_bundle_path)
    if verbose:
        print(f'Saved figure of bundles at {fig_bundle_path}')

    list_df = []
    for i in np.arange(num_groups):
        list_df.append(
            bundle_df_reduced[(bundle_df_reduced['age'] > tr_list[i]) & (bundle_df_reduced['age'] <= tr_list[i + 1])][
                f'average{ref}'])
    # group1 = bundle_df_reduced[bundle_df_reduced['age']<=tr1]['averageFA']
    # group2 = bundle_df_reduced[(bundle_df_reduced['age']>tr1) & (bundle_df_reduced['age']<=tr2)]['averageFA']
    # group3 = bundle_df_reduced[bundle_df_reduced['age']>tr2]['averageFA']

    for i, group in enumerate(list_df):
        group = outlier_removal(group, qsep=1.5)
        # means[int(i),int(bundle_num)] = np.mean(group)
        # sds[int(i),int(bundle_num)] = np.std(group)
        datameans.loc[i, 'Age Group'] = f'Age Group {i}'
        datameans.loc[i, f'{side_spec}{side_str}ID_{bundle_num}'] = np.mean(group)
        datasds.loc[i, 'Age Group'] = f'Age Group {i}'
        datasds.loc[i, f'{side_spec}{side}_ID_{bundle_num}'] = np.std(group)

    del (bundle_df)

    plt.close()

    if verbose:
        print(f'Finished bundle {bundle_num} side {side}')

    # bundle_df['agecat'] = bundle_df['age'] > np.median(bundle_df['age'])
    # bundle_df['agecat'] = np.multiply(bundle_df['agecat'], 1)
    # model = LinearRegression()

    # bundle_df['sex']

# datameans.plot.scatter(x=index)

x_labels = [f'Age {int(tr_list[i])}-{int(tr_list[i + 1])}' for i in np.arange(num_groups)]

# datameans = pd.DataFrame(index= np.arange(12),columns=x_labels)
# datameans.iloc[:,:] = means.transpose()

bundle_labels = [f'Bundle {str(num + 1)}' for num in np.arange(np.shape(datameans)[1])]

# Create 12 different plots with different colors
for i, bundle_name in enumerate(columns_list):
    fig_plot_path = os.path.join(figures_agegrouping_path, f'{bundle_name}_AgeGrouping_Mean{ref}.png')
    plt.figure()
    plt.plot(x_labels, datameans.loc[:, bundle_name], marker='o', color=plt.cm.viridis(i / datameans.shape[0]),
             label=bundle_name)
    plt.title(f'Change in values for {bundle_name}')
    plt.xlabel('X-axis Groups')
    plt.ylabel('Change in Values')
    plt.legend()
    plt.savefig(fig_plot_path)
    if verbose:
        print(f'Saved figure at {fig_plot_path}')
    plt.close()

for i, bundle_name in enumerate(columns_list):
    fig_plot_path = os.path.join(figures_agegrouping_path, f'{bundle_name}_AgeGrouping_Sd{ref}.png')
    plt.figure()
    plt.plot(x_labels, datasds.loc[:, bundle_name], marker='o', color=plt.cm.viridis(i / datameans.shape[0]),
             label=bundle_name)
    plt.title(f'Change in values for {bundle_name}')
    plt.xlabel('X-axis Groups')
    plt.ylabel('Change in Values')
    plt.legend()
    plt.savefig(fig_plot_path)
    if verbose:
        print(f'Saved figure at {fig_plot_path}')
    plt.close()


metadf = master_df[['MRI_Exam','age','sex','Risk','genotype']]


if siding:

    distance_var_FAbundle_array = np.zeros([np.size(full_subjects_list), int(total_num_bundles)])
    distance_var_FAbundle_dic = {}
    distance_mean_FAbundle_dic = {}

    for i,subj in enumerate(full_subjects_list):
        for bundle_id in np.arange(total_num_bundles):
            left_side_vars = allvars[subj[2:6],'left',str(int(bundle_id))]
            right_side_vars = allvars[subj[2:6],'right',str(int(bundle_id))]

            left_side_means = allmeans[subj[2:6],'left',str(int(bundle_id))]
            right_side_means = allmeans[subj[2:6],'right',str(int(bundle_id))]

            norm = skfda.misc.metrics.LpNorm(2)
            distance_var_FAbundle_dic[f'{subj[2:6]}',f'bundle_{int(bundle_id)}'] = norm(right_side_vars - left_side_vars)[0]
            distance_mean_FAbundle_dic[f'{subj[2:6]}',f'bundle_{int(bundle_id)}'] = norm(right_side_means - left_side_means)[0]
            #distance_var_FAbundle_array[i, int(bundle_id)] = norm(right_side_coefs - left_side_coefs)[0]
            if save_fig:
                for side in ['left','right']:
                    figs_plotvar_path = os.path.join(grid_var_path, f'grid_var_{contrast}_{subj[2:6]}_{side}_bundle_{int(bundle_id)}')
                    var = allvars[subj[2:6], side, str(int(bundle_id))]
                    var.plot()
                    plt.savefig(figs_plotvar_path)
                    plt.close()

    subs = set(key[0] for key in distance_var_FAbundle_dic.keys())
    column_vars = set(f'{key[1]}_var' for key in distance_var_FAbundle_dic.keys())
    column_means = set(f'{key[1]}_mean' for key in distance_var_FAbundle_dic.keys())

    subjects_list = [subj[2:6] for subj in full_subjects_list]

    #distances_var_FAbundle_df = pd.DataFrame(columns=list(['MRI_Exam']+columns))
    distances_var_FAbundle_df = pd.DataFrame(columns=(['MRI_Exam']+list(column_vars)+list(column_means)))

    for key, value in distance_var_FAbundle_dic.items():
        sub_value, col_name = key
        distances_var_FAbundle_df.at[sub_value, f'{col_name}_var'] = value
        distances_var_FAbundle_df.at[sub_value, 'MRI_Exam'] = sub_value

    for key, value in distance_mean_FAbundle_dic.items():
        sub_value, col_name = key
        distances_var_FAbundle_df.at[sub_value, f'{col_name}_mean'] = value
        distances_var_FAbundle_df.at[sub_value, 'MRI_Exam'] = sub_value

    distances_var_FAbundle_df = distances_var_FAbundle_df.reset_index(drop=True)

    column_indices = [bundle_df.columns.get_loc(column_names[i]) for i in np.arange(np.size(column_names))]
    #metadf['MRI_Exam'] = metadf['MRI_Exam'].astype(str)

    #Converts the columns to str to avoid warning
    metadf.iloc[:,metadf.columns.get_loc('MRI_Exam')] = metadf.iloc[:,metadf.columns.get_loc('MRI_Exam')].astype(str)
    distances_var_FAbundle_df.iloc[:,distances_var_FAbundle_df.columns.get_loc('MRI_Exam')] = distances_var_FAbundle_df.iloc[:,distances_var_FAbundle_df.columns.get_loc('MRI_Exam')].astype(str)
    #distances_var_FAbundle_df['MRI_Exam'] = distances_var_FAbundle_df['MRI_Exam'].astype(str)
    merged_df = pd.merge(metadf, distances_var_FAbundle_df, on='MRI_Exam', how='inner')

    merged_df.to_excel(os.path.join(excel_path,'distances_var_FAbundle.xlsx'))
