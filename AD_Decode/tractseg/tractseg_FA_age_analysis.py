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

"""
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    loadmat_remote, pickledump_remote, remote_pickle, checkfile_exists_all, write_parameters_to_ini, \
    read_parameters_from_ini
"""


def outlier_removal(values, qsep=3):
    iqr = abs(np.quantile(values, 0.25) - np.quantile(values, 0.75))
    median = np.quantile(values, 0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values


def models_pvalue(df, model = None,order=1):
    if model == 'anova_n':
        # model1 = f'average{ref} ~ np.power(age, 2) + age'
        # formula_age_gen = f'average{ref} ~ genotype + np.power(age, 2) + age'

        bundle_df_used_2 = pd.get_dummies(df, columns=['genotype'], drop_first=True)
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

        model1 = ols(formula1, data=df).fit()
        model2 = ols(formula2, data=df).fit()

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

        model1 = smf.mixedlm(formula1, data=df, groups=df['Subject']).fit(method='powell')
        model2 = smf.mixedlm(formula2, data=df, groups=df['Subject']).fit(method='powell')

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


project = ''
root = f'/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/TractSeg_analysis'
metadata_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx'

if metadata_path.split('.')[1]=='csv':
    master_df = pd.read_csv(metadata_path)
elif metadata_path.split('.')[1]=='xlsx':
    master_df = pd.read_excel(metadata_path)
else:
    txt = f'Unidentifiable data file path {metadata_path}'
    raise Exception(txt)

master_df = master_df.dropna(subset=['MRI_Exam'])
full_subjects_list = list(master_df['MRI_Exam'].astype(int).astype(str))
for i in range(len(full_subjects_list)):
    if len(full_subjects_list[i]) < 4:
        full_subjects_list[i] = '0' + full_subjects_list[i]

full_subjects_list = ['S0' + s for s in full_subjects_list]


figures_path = os.path.join(root, f'Figures')
excel_path = os.path.join(root, f'Excels')
stats_path = os.path.join(root, 'stats')

#mkcdir([root,stats_path, figures_path, excel_path])

siding = False

num_groups = 5

tr_list = [0] + [np.quantile(master_df['age'], (i + 1) * (1 / num_groups)) for i in np.arange(num_groups)]

"""
if not siding:
    columns_list = [f'ID_{bundle.split("_")[2].split(".")[0]}' for bundle in bundles]
else:
    columns_list = [f'{side_spec}{bundle.split("_")[1]}_ID_{bundle.split("_")[3].split(".")[0]}' for bundle in bundles]
    
datameans = pd.DataFrame(columns=(['Age Group'] + columns_list))
datasds = pd.DataFrame(columns=(['Age Group'] + columns_list))
"""
regions_list = ['SLF_I_left','SLF_I_right','SLF_II_left','SLF_II_right','STR_left','STR_right','CC']
datameans = pd.DataFrame(columns=(['Age Group'] + regions_list))
datasds = pd.DataFrame(columns=(['Age Group'] + regions_list))

verbose = True

ref = 'mrtrixfa'

figures_box_path = os.path.join(figures_path, f'boxsquares_age_{ref}')
figures_agegrouping_path = os.path.join(figures_path, f'agegrouping_mean_{ref}')

mkcdir([figures_box_path, figures_agegrouping_path])

verbose = True
geno_simplify = True
testmode=False

save_fig = True

#models = ['anova','anova_mixed'] #['anova','t-test']
models = ['anova']

order=1

while 'S000' in full_subjects_list:
    full_subjects_list.remove('S000')

if testmode:
    full_subjects_list = full_subjects_list[:10]

for region in regions_list:

    print(f'Beginning for region {region}')

    if not siding:
        side = ''
        side_str = ''
    else:
        if 'left' in region:
            side = 'left'
        if 'right' in region:
            side = 'right'
        side_str = '_' + side

    fig_bundle_path = os.path.join(figures_box_path, f'region_{region}_boxsquaremodel.png')


    # bundle_df = pd.DataFrame()
    for subj in full_subjects_list:

        subj_stats_path = os.path.join(stats_path, f'Tractometry_{subj}.csv')
        if not os.path.exists(subj_stats_path):
            continue
        else:
            temp_df = pd.read_csv(subj_stats_path,sep=';')

        index = master_df["MRI_Exam"] == int(subj[2:6])
        try:
            temp_df['age'] = master_df[index]['age'].iloc[0]
            temp_df['genotype'] = master_df[index]['genotype'].iloc[0]
            temp_df['Subject'] = subj
            # temp['sex'] = master_df[index]['sex']
        except:
            print(f'Subject {subj[:6]} is missing from excel database')
            continue
        # temp = temp.to_numpy()
        if 'region_df' not in locals():
            region_df = temp_df
        else:
            region_df = pd.concat([region_df, temp_df])

        region_df = region_df.dropna()

    column_names = []

    region_df_reduced = region_df[[region, "age","genotype","Subject"]]
    #bundle_df_tiny = region_df.groupby('Subject').agg({'age': 'first', 'genotype': 'first', f'average{ref}': 'mean'}).reset_index()

    # bundle_df_reduced.boxplot(column='averageFA',by='age')
    # sns.boxplot(x="age", y="averageFA", data=bundle_df_reduced)
    if geno_simplify:
        region_df_reduced = region_df_reduced.replace(
            {'APOE23': 'APOE3', 'APOE34': 'APOE4', 'APOE33': 'APOE3','APOE44':'APOE4'}, regex=True)

        #bundle_df_tiny = bundle_df_tiny.replace(
        #    {'APOE23': 'APOE3', 'APOE34': 'APOE4', 'APOE33': 'APOE3', 'APOE44': 'APOE4'}, regex=True)

    snsgraph = sns.lmplot(x="age", y=region, data=region_df_reduced, x_estimator=np.mean, order=2, hue='genotype')
    snsgraph.set_axis_labels('Age (years)', 'Average FA')

    x_loc = 0.18
    y_loc = 0.20

    for model in models:
        region_df_used = region_df_reduced

        p_value = models_pvalue(region_df_used,model=model,order=2)
        #plt.text(x_loc, y_loc, f'quadratic {model} p-value: {"{:.3g}".format(p_value)}', transform=plt.gcf().transFigure,
        #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),size=8)
        plt.text(x_loc, y_loc, f'quadratic p-value: {"{:.3g}".format(p_value)}', transform=plt.gcf().transFigure,size=8)
        y_loc += 0.06

    for model in models:
        region_df_used = region_df_reduced

        p_value = models_pvalue(region_df_used,model=model,order=1)
        plt.text(x_loc, y_loc, f'Linear p-value: {"{:.3g}".format(p_value)}', transform=plt.gcf().transFigure,size=8)
        y_loc += 0.06

    plt.title(f'Boxplot for region {region}', y=0.95,size= 10)

    if save_fig:
        plt.savefig(fig_bundle_path)
        if verbose:
            print(f'Saved figure of bundles at {fig_bundle_path}')

    list_df = []
    for i in np.arange(num_groups):
        list_df.append(
            region_df_reduced[(region_df_reduced['age'] > tr_list[i]) & (region_df_reduced['age'] <= tr_list[i + 1])][
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
