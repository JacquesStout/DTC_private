#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:35 2023
#varation
@author: ali
"""
import os , glob
import sys
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


if 'santorini' in socket.gethostname().split('.')[0]:
    root = '/Users/jas/Downloads/Busa_analysis/AD_decode_bundles/'
    #master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3_zscores.xlsx'
    figures_path = '/Users/jas/jacques/AD_Decode/BuSA_analysis/Figures'
    excel_path = '/Users/jas/jacques/AD_Decode/BuSA_analysis/Excels'
else:
    root = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/stats/'
    #master = '/Users/ali/Desktop/Dec23/BuSA/AD_DECODE_data_stripped.csv'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3_zscores.xlsx'
    excel_path = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/Excels'
    figures_path = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/Figures'


if len(sys.argv)<2:
    project = 'V_1_0_10template_100_6_interhe_majority'
else:
    project = sys.argv[1]

loc = 'munin'

if loc=='kos':
    root = f'/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'
elif loc=='munin':
    root = f'/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/{project}'

#master_df = pd.read_csv(master)
master_df = pd.read_excel(master)

length_cut = None
if length_cut is None:
    length_str = ''
else:
    length_str = f'_Lengthcut{length_cut}'

figures_path = os.path.join(root,f'Figures{length_str}')
excel_path = os.path.join(root,f'Excels{length_str}')
stats_path = os.path.join(root,'stats')

mkcdir(excel_path)

all_subj_bundles = os.listdir(stats_path)


list1 = ['_0','_1','_2','_3','_4','_5']
list2 = ['_0','_1','_2']

pattern_lvl = '_1'

if pattern_lvl=='_1':
    dm_patterns = [x for x in list1]
elif pattern_lvl == '_2':
    dm_patterns = [(x + y) for x in list1 for y in list2]
elif pattern_lvl == '_3':
    dm_patterns = [(x + y + z) for x in list1 for y in list2 for z in list2]
elif pattern_lvl == '_4':
    dm_patterns = [(x + y + z + k) for x in list1 for y in list2 for z in list2 for k in list2]
elif pattern_lvl == '_4':
    dm_patterns = [x for x in list1] + [(x + y) for x in list1 for y in list2] + [(x + y + z) for x in list1 for y in
                                                                                  list2 for z in list2] + [
                      (x + y + z + k) for x in list1 for y in list2 for z in list2 for k in list2]
    pattern_lvl = ''
else:
    raise Exception('Unrecognized')


ref_subj = 'S02224'

bundles_left = [f'_bundle_left{dm_pattern}.xlsx' for dm_pattern in dm_patterns]
bundles_right = [f'_bundle_right{dm_pattern}.xlsx' for dm_pattern in dm_patterns]
bundles = bundles_left + bundles_right

num_groups = 6

tr_list = [np.min(master_df['age'])] + [np.quantile(master_df['age'],(i+1)*(1/num_groups)) for i in np.arange(num_groups)] #+ [1+ np.max(master_df['age'])]

#bundle = bundles[0] #
#sds = np.zeros([num_groups,12])
#means = np.zeros([num_groups,12])
#pickle_path = '/Users/ali/Desktop/Dec23/BuSA/variances/'

allvars = {}
allmeans = {}

contrast = 'mrtrixfa'

removed_subj = ['S01621','S04696','S04491','S03890','S03048','S03017','S02987','S02967','S02670']

grid_var_path = os.path.join(figures_path,f'grid_var_{contrast}')

mkcdir(grid_var_path)

verbose=True
save_fig = True
test = False

savedatagrid_figs = True

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

    for subj_r in removed_subj:
        for bundle in this_bundle_subjs:
            if subj_r in bundle:
                this_bundle_subjs.remove(bundle)
                break

    column_names = []
    for i in range(0,50):
        column_names.append("point_"+str(i)+"_mrtrixfa")

    allvars[bundle_num]= {}
    allmeans[bundle_num]= {}
    for subj in this_bundle_subjs:
        bundle_df = pd.read_excel(os.path.join(stats_path, subj))
        #temp = pd.DataFrame()
        bundle_df['Subject']=subj[2:6]
        index = master_df["MRI_Exam"] == f'S0{(subj[2:6])}'
        try:
            bundle_df['age'] = master_df[index]['age'].iloc[0]
            #temp['sex'] = master_df[index]['sex']
        except:
            print(f'Subject {subj} is missing from master data')
        #temp = temp.to_numpy()

        if length_cut is not None:
            bundle_df = bundle_df[bundle_df['Length'] >= int(length_cut)]
        #bundle_df.loc[:,bundle_df.columns.get_loc(f'average{contrast}')] = np.mean(bundle_df[column_names],1)
        bundle_df[f'average{contrast}'] = np.mean(bundle_df[column_names],1)
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
        allvars[bundle_num] = {
            subj[2:6]: {side+'_mean': FDataBasis.coefficients, 'column2': 'A'},
            'row2': {'column1': 2, 'column2': 'B'},
            'row3': {'column1': 3, 'column2': 'C'},
            'row4': {'column1': 4, 'column2': 'D'}
        }
        #var_coefs[subj[2:6], bundle_num] = var.coefficients
        allvars[bundle_num][subj[2:6], side+ bundle_num] = var
        allmeans[bundle_num][subj[2:6], side+ bundle_num] = mean

    if verbose:
        print(f'Finished bundle {bundle_num} side {side}')
    path_excel_bundle_norm_summary = os.path.join(figures_path,f'sum_{bundle.split("_")[1]}')


    'S02670_bundle_left_1.xlsx'

data = {
    'row1': {'column1': 1, 'column2': 'A'},
    'row2': {'column1': 2, 'column2': 'B'},
    'row3': {'column1': 3, 'column2': 'C'},
    'row4': {'column1': 4, 'column2': 'D'}
}


'S02670_bundle_left_1.xlsx'
metadf = master_df[['MRI_Exam','age','sex','Risk','genotype']]

distance_var_FAbundle_array = np.zeros([np.size(this_bundle_subjs),int(total_num_bundles)])
distance_var_FAbundle_dic = {}
distance_mean_FAbundle_dic = {}

for i,subj in enumerate(this_bundle_subjs):
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

subjects_list = [subj[2:6] for subj in this_bundle_subjs]

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
