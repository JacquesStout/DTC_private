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
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from statsmodels.stats.api import anova_lm


from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, remote_pickle, checkfile_exists_all, write_parameters_to_ini, \
    read_parameters_from_ini

def outlier_removal(values, qsep=3):
    iqr = abs(np.quantile(values, 0.25) - np.quantile(values, 0.75))
    median = np.quantile(values, 0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values


def models_pvalue(df, model = None,order=1):
    if model == 'anova_n':
        # model1 = f'average{ref} ~ np.power(age, 2) + age'
        # formula_age_gen = f'average{ref} ~ genotype + np.power(age, 2) + age'

        bundle_df_used_2 = pd.get_dummies(bundle_df_used, columns=['genotype'], drop_first=True)
        bundle_df_used_2['genotype_APOE4'] = bundle_df_used_2['genotype_APOE4'].replace({False: 0, True: 1})
        if order==2:
            bundle_df_used_2['age2'] = bundle_df_used_2.age * bundle_df_used_2.age
            X1 = bundle_df_used_2[['age', 'age2']]  # Independent variables for func1
            X2 = bundle_df_used_2[['age', 'age2', 'genotype_APOE4']]
        if order==1:
            X1 = bundle_df_used_2[['age']]  # Independent variables for func1
            X2 = bundle_df_used_2[['age','genotype_APOE4']]

        y = bundle_df_used_2[f'average{ref}']  # Dependent variable

        # Add a constant term to the independent variables (similar to intercept in lm)
        X1 = sm.add_constant(X1)
        X2 = sm.add_constant(X2)

        # Fit the linear regression models
        model1 = sm.OLS(y, X1).fit()
        model2 = sm.OLS(y, X2).fit()
        anova_results = anova_lm(model1, model2)

        # Extract p-value from ANOVA table
        p_value = anova_results.iloc[1, 5]
        return(p_value)

    if model == 'anova':
        if order == 2:
            formula1 = f'average{ref} ~ genotype + age + np.power(age, 2)'
            formula2 = f'average{ref} ~ genotype + age + np.power(age, 2) + np.power(age, 2):genotype +age:genotype'
        if order == 1:
            formula1 = f'average{ref} ~ genotype + age'
            formula2 = f'average{ref} ~ genotype + age + age:genotype'

        model1 = ols(formula1, data=bundle_df_used).fit()
        model2 = ols(formula2, data=bundle_df_used).fit()

        # Perform ANOVA test
        anova_table = sm.stats.anova_lm(model1, model2)

        # Extract p-value from ANOVA table
        p_value = anova_table.iloc[1, 5]
        return(p_value)


    if model == 'anova_mixed':
        if order == 2:
            formula1 = f'average{ref} ~ genotype + age + np.power(age, 2)'
            formula2 = f'average{ref} ~ genotype + age + np.power(age, 2) + np.power(age, 2):genotype +age:genotype'
        if order == 1:
            formula1 = f'average{ref} ~ genotype + age'
            formula2 = f'average{ref} ~ genotype + age + age:genotype'

        model1 = smf.mixedlm(formula1, data=bundle_df_used, groups=bundle_df_used['Subject']).fit(method='powell')
        model2 = smf.mixedlm(formula2, data=bundle_df_used, groups=bundle_df_used['Subject']).fit(method='powell')

        residuals = model1.resid
        squared_residuals = residuals ** 2
        #model1.ssr = np.sum(squared_residuals)
        model1.ssr = np.sum(np.power(model1.fittedvalues - np.mean(bundle_df_used['averagemrtrixfa']), 2))

        residuals = model2.resid
        squared_residuals = residuals ** 2
        #model2.ssr = np.sum(squared_residuals)
        model2.ssr = np.sum(np.power(model2.fittedvalues - np.mean(bundle_df_used['averagemrtrixfa']), 2))

        # Perform ANOVA test
        anova_table = sm.stats.anova_lm(model1, model2)

        # Extract p-value from ANOVA table
        p_value = anova_table.iloc[1, 5]
        return(p_value)

    if model == 'anova_backup':
        formula = f'average{ref} ~ genotype + age + np.power(age, 2) + np.power(age, 2):genotype +age:genotype'

        model = ols(formula, data=bundle_df_used).fit()

        # Perform ANOVA test
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Extract p-value from ANOVA table
        p_value = anova_table.loc['genotype', 'PR(>F)']
        return(p_value)

    if model == 't-test':
        gen_group1 = bundle_df_used[bundle_df_used['genotype'] == 'APOE3'][f'average{ref}']
        gen_group2 = bundle_df_used[bundle_df_used['genotype'] == 'APOE4'][f'average{ref}']
        t_stat, p_value = ttest_ind(gen_group1, gen_group2)
        return(p_value)


remote = False

if len(sys.argv) < 2:
    #project = 'V0_9_10template_100_6_interhe_majority'
    #project = 'V0_9_reg_superiorfrontalleft_precentralleft_6'
    #project = 'V0_9_reg_precuneusright_thalamusproper_left_split_1.ini'
    project = 'V0_9_reg_precuneusleft_precuneus_right_split_3'
else:
    project = sys.argv[1]
    if os.path.exists(project):
        project_summary_file = project
        project = os.path.basename(project).split('.ini')[0]

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
if 'project_summary_file' not in locals():
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

length_cut = None
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
geno_simplify = True

#models = ['anova','anova_mixed'] #['anova','t-test']
models = ['anova']

if testmode:
    save_fig = False
else:
    save_fig = True

order=1

for bundle in bundles:

    print(f'Beginning for bundle {bundle}')

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

    if testmode:
        full_subjects_list = full_subjects_list[:10]

    """
    this_bundle_subjs = [i for i in all_subj_bundles if bundle in i]
    this_bundle_subjs = sorted(this_bundle_subjs)
    if testmode:
        this_bundle_subjs = this_bundle_subjs[:3]
    """
    # bundle_df = pd.DataFrame()
    for subj in full_subjects_list:

        subj_bundle_path = os.path.join(stats_path, subj+bundle)
        try:
            if remote:
                load_df_remote(subj_bundle_path, sftp)
            else:
                temp = pd.read_excel(subj_bundle_path)
        except:
            print(f'Not including subject {subj}')
            continue

        # temp = pd.DataFrame()
        temp['Subject'] = subj[2:6]
        index = master_df["MRI_Exam"] == int(subj[2:6])
        try:
            temp['age'] = master_df[index]['age'].iloc[0]
            temp['genotype'] = master_df[index]['genotype'].iloc[0]
            # temp['sex'] = master_df[index]['sex']
        except:
            print(f'Subject {subj[:6]} is missing from excel database')
            continue
        # temp = temp.to_numpy()
        if 'bundle_df' not in locals():
            bundle_df = temp
        else:
            bundle_df = pd.concat([bundle_df, temp])

    bundle_df = bundle_df.dropna()

    if length_cut is not None:
        bundle_df = bundle_df[bundle_df['Length'] >= int(length_cut)]

    column_names = []
    for i in range(1, 50):
        column_names.append("point_" + str(i) + f"_{ref}")
    bundle_df[f'average{ref}'] = np.mean(bundle_df[column_names], 1)

    bundle_df_reduced = bundle_df[[f"average{ref}", "age","genotype","Subject"]]
    bundle_df_tiny = bundle_df.groupby('Subject').agg({'age': 'first', 'genotype': 'first', f'average{ref}': 'mean'}).reset_index()

    # bundle_df_reduced.boxplot(column='averageFA',by='age')
    # sns.boxplot(x="age", y="averageFA", data=bundle_df_reduced)
    if geno_simplify:
        bundle_df_reduced = bundle_df_reduced.replace(
            {'APOE23': 'APOE3', 'APOE34': 'APOE4', 'APOE33': 'APOE3','APOE44':'APOE4'}, regex=True)

        bundle_df_tiny = bundle_df_tiny.replace(
            {'APOE23': 'APOE3', 'APOE34': 'APOE4', 'APOE33': 'APOE3', 'APOE44': 'APOE4'}, regex=True)

    snsgraph = sns.lmplot(x="age", y=f"average{ref}", data=bundle_df_reduced, x_estimator=np.mean, order=2, hue='genotype')
    snsgraph.set_axis_labels('Age (years)', 'Average FA')

    x_loc = 0.18
    y_loc = 0.20

    for model in models:

        bundle_df_used = bundle_df_reduced

        p_value = models_pvalue(bundle_df_used,model=model,order=2)
        #plt.text(x_loc, y_loc, f'quadratic {model} p-value: {"{:.3g}".format(p_value)}', transform=plt.gcf().transFigure,
        #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),size=8)
        plt.text(x_loc, y_loc, f'quadratic p-value: {"{:.3g}".format(p_value)}', transform=plt.gcf().transFigure,size=8)
        y_loc += 0.06

    for model in models:
        bundle_df_used = bundle_df_reduced

        p_value = models_pvalue(bundle_df_used,model=model,order=1)
        plt.text(x_loc, y_loc, f'Linear p-value: {"{:.3g}".format(p_value)}', transform=plt.gcf().transFigure,size=8)
        y_loc += 0.06

    plt.title(f'Boxplot for bundle {int(bundle_num) + 1}', y=0.95,size= 10)

    if save_fig:
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
