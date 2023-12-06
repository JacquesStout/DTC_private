#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:35 2023
#varation
@author: ali
"""
import os , glob
import sys, math
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import matplotlib    
import matplotlib.pyplot as plt  
import seaborn as sns
import socket

import pickle
#import scipy.special.kl_div
import numpy as np
from skfda.representation import FDataGrid
from skfda.exploratory import stats
import skfda
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from DTC.file_manager.computer_nav import glob_remote, load_df_remote, save_fig_remote, save_df_remote, \
    checkfile_exists_remote, get_mainpaths

def mkcdir(folderpaths, sftp=None):
    #creates new folder only if it doesnt already exists

    if sftp is None:
        if np.size(folderpaths) == 1:
            if not os.path.exists(folderpaths):
                os.mkdir(folderpaths)
        else:
            for folderpath in folderpaths:
                if not os.path.exists(folderpath):
                    os.mkdir(folderpath)
    else:
        if np.size(folderpaths) == 1:
            try:
                sftp.chdir(folderpaths[0])
            except:
                sftp.mkdir(folderpaths[0])
        else:
            for folderpath in folderpaths:
                try:
                    sftp.chdir(folderpath)
                except:
                    sftp.mkdir(folderpath)


def outlier_removal(values, qsep=3):

    iqr = abs(np.quantile(values,0.25) - np.quantile(values,0.75))
    median = np.quantile(values,0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values


project = '202311_10template_test01'
kos = True


if 'santorini' in socket.gethostname().split('.')[0]:
    root = f'/Users/jas/jacques/AD_Decode/BuSA_analysis/{project}'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
else:
    root = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/'
    master = '/Users/ali/Desktop/Dec23/BuSA/AD_DECODE_data_stripped.csv'

stats_path = os.path.join(root,'stats')

#Note, it takes too long right now to download the files from kos, so only saving files to kos for now by defining stats_path based on local directory
if kos:
    root = f'/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'

remote=False
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
    root = f'/mnt/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'
    inpath, _, _, sftp = get_mainpaths(remote,project = 'AD_Decode', username=username,password=passwd)
else:
    sftp = None

figures_path = os.path.join(root,'Figures')
excel_path = os.path.join(root,'Excels')
mkcdir([excel_path,figures_path],sftp)


master_df = pd.read_csv(master)

all_subj_bundles = os.listdir(stats_path)

ref_subj = 'S02224' 
bundles = [i for i in all_subj_bundles if ref_subj in i]
bundles = [ i[6:] for i in bundles]

num_groups = 6

tr_list = [0] + [np.quantile(master_df['age'],(i+1)*(1/num_groups)) for i in np.arange(num_groups)] + [1+ np.max(master_df['age'])]

allvars_fdata = {}
allmeans_fdata = {}

bundle_var = {}
bundle_mean = {}

verbose=True
save_fig = True
test = False

savedatagrid_figs = False

if test:
    bundles = [f'_left_bundle_0.xlsx',f'_right_bundle_0.xlsx']

basis = skfda.representation.basis.MonomialBasis(n_basis=10)

total_num_bundles = np.size(bundles)/2

for bundle in bundles:

    if 'left' in bundle:
        side = 'left'
    if 'right' in bundle:
        side = 'right'
    bundle_num = bundle.split('bundle_')[1].split('.')[0]

    this_bundle_subjs = [i for i in all_subj_bundles if bundle in i]
    this_bundle_subjs = sorted(this_bundle_subjs)
    if test:
        this_bundle_subjs = [this_bundle_subjs[0]]

    column_names = []
    for i in range(0,50):
        column_names.append("point_"+str(i)+"_fa")

    
    for subj in this_bundle_subjs:
        temp = pd.read_excel(os.path.join(stats_path,subj))
        #temp = pd.DataFrame()
        temp['Subject']=subj[2:6]
        index = master_df["MRI_Exam"] == int(subj[2:6])
        try:
            temp['age'] = master_df[index]['age'].iloc[0]
            #temp['sex'] = master_df[index]['sex']
        except:
            print(f'Subject {subj} is missing')
        #temp = temp.to_numpy()
        bundle_df = temp
            
        bundle_df['averageFA'] = np.mean(bundle_df[column_names],1)
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
            save_fig_remote(figs_bundle_subj_path,sftp=sftp)
            if verbose:
                print(f'Saved figure at {figs_bundle_subj_path}')
            plt.close()

        #var_coefs[subj[2:6], bundle_num] = var.coefficients
        allvars_fdata[subj[2:6], side, bundle_num] = var
        allmeans_fdata[subj[2:6], side, bundle_num] = mean
        bundle_mean[subj[2:6], side, bundle_num] = np.mean(streamlines_fas)
        bundle_var[subj[2:6], side, bundle_num] = np.var(streamlines_fas)

    if verbose:
        print(f'Finished bundle {bundle_num} side {side}')

#pickle.dump(var_coefs, os.path.join(pickle_path, 'var_coefs.py'))


metadf = master_df[['MRI_Exam','age','sex','Risk','genotype']]
metadf.loc[:,'Risk'] = metadf.loc[:,'Risk'].astype(str)
metadf.loc[:, 'Risk'] = metadf.loc[:, 'Risk'].replace('nan', 'None')

#nanindex = metadf[metadf['Risk'].isna()].index

distance_var_FAbundle_array = np.zeros([np.size(this_bundle_subjs),int(total_num_bundles)])
distance_var_FAbundle_dic = {}
distance_mean_FAbundle_dic = {}

for i,subj in enumerate(this_bundle_subjs):
    for bundle_id in np.arange(total_num_bundles):
        left_side_vars = allvars_fdata[subj[2:6],'left',str(int(bundle_id))]
        right_side_vars = allvars_fdata[subj[2:6],'right',str(int(bundle_id))]

        left_side_means = allmeans_fdata[subj[2:6],'left',str(int(bundle_id))]
        right_side_means = allmeans_fdata[subj[2:6],'right',str(int(bundle_id))]

        norm = skfda.misc.metrics.LpNorm(2)
        distance_var_FAbundle_dic[f'{subj[2:6]}',f'bundle_{int(bundle_id)}'] = norm(right_side_vars - left_side_vars)[0]
        distance_mean_FAbundle_dic[f'{subj[2:6]}',f'bundle_{int(bundle_id)}'] = norm(right_side_means - left_side_means)[0]

        #distance_var_FAbundle_array[i, int(bundle_id)] = norm(right_side_coefs - left_side_coefs)[0]
        if save_fig:
            for side in ['left','right']:
                figs_plotvar_path = os.path.join(figures_path, f'grid_var_FA_{subj[2:6]}_{side}_bundle_{int(bundle_id)}')
                var = allvars_fdata[subj[2:6], side, str(int(bundle_id))]
                var.plot()
                #plt.savefig(figs_plotvar_path)
                save_fig_remote(figs_plotvar_path, sftp=sftp)
                plt.close()

subs = set(key[0] for key in distance_var_FAbundle_dic.keys())
column_div_vars = set(f'{key[1]}_divergence_var' for key in distance_var_FAbundle_dic.keys())
column_div_means = set(f'{key[1]}_divergence_mean' for key in distance_var_FAbundle_dic.keys())

column_vars = set(f'side_{key[1]}_bundle_{key[2]}_var' for key in bundle_var.keys())
column_means = set(f'side_{key[1]}_bundle_{key[2]}_mean' for key in bundle_mean.keys())

subjects_list = [subj[2:6] for subj in this_bundle_subjs]
columns_list = ['MRI_Exam']+list(column_div_vars)+list(column_div_means) +list(column_means)+list(column_vars)

#distances_var_FAbundle_df = pd.DataFrame(columns=list(['MRI_Exam']+columns))
distances_var_FAbundle_df = pd.DataFrame(columns=(columns_list))

for key, value in distance_var_FAbundle_dic.items():
    sub_value, col_name = key
    distances_var_FAbundle_df.at[sub_value, f'{col_name}_divergence_var'] = value
    distances_var_FAbundle_df.at[sub_value, 'MRI_Exam'] = sub_value

for key, value in distance_mean_FAbundle_dic.items():
    sub_value, col_name = key
    distances_var_FAbundle_df.at[sub_value, f'{col_name}_divergence_mean'] = value
    #distances_var_FAbundle_df.at[sub_value, 'MRI_Exam'] = sub_value

for key, value in bundle_mean.items():
    sub_value, col_name_1,col_name_2 = key
    col_name = f'side_{col_name_1}_bundle_{col_name_2}'
    distances_var_FAbundle_df.at[sub_value, f'{col_name}_mean'] = value
    #distances_var_FAbundle_df.at[sub_value, 'MRI_Exam'] = sub_value

for key, value in bundle_var.items():
    sub_value, col_name_1,col_name_2 = key
    col_name = f'side_{col_name_1}_bundle_{col_name_2}'
    distances_var_FAbundle_df.at[sub_value, f'{col_name}_var'] = value
    #distances_var_FAbundle_df.at[sub_value, 'MRI_Exam'] = sub_value

distances_var_FAbundle_df = distances_var_FAbundle_df.reset_index(drop=True)

column_indices = [bundle_df.columns.get_loc(column_names[i]) for i in np.arange(np.size(column_names))]
#metadf['MRI_Exam'] = metadf['MRI_Exam'].astype(str)

#Converts the columns to str to avoid warning
metadf.iloc[:,metadf.columns.get_loc('MRI_Exam')] = metadf.iloc[:,metadf.columns.get_loc('MRI_Exam')].astype(str)
distances_var_FAbundle_df.iloc[:,distances_var_FAbundle_df.columns.get_loc('MRI_Exam')] = distances_var_FAbundle_df.iloc[:,distances_var_FAbundle_df.columns.get_loc('MRI_Exam')].astype(str)
#distances_var_FAbundle_df['MRI_Exam'] = distances_var_FAbundle_df['MRI_Exam'].astype(str)
merged_df = pd.merge(metadf, distances_var_FAbundle_df, on='MRI_Exam', how='inner')

distance_excel_path = os.path.join(excel_path,'distances_var_FAbundle.xlsx')
#merged_df.to_excel(os.path.join(excel_path,'distances_var_FAbundle.xlsx'))
save_df_remote(merged_df,distance_excel_path, sftp)

