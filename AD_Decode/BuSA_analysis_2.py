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

import numpy as np
from skfda.representation import FDataGrid
from skfda.exploratory import stats
import skfda


def outlier_removal(values, qsep=3):

    iqr = abs(np.quantile(values,0.25) - np.quantile(values,0.75))
    median = np.quantile(values,0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values


if 'santorini' in socket.gethostname().split('.')[0]:
    root = '/Users/jas/Downloads/Busa_analysis/AD_decode_bundles/'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
else:
    root = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/'
    master = '/Users/ali/Desktop/Dec23/BuSA/AD_DECODE_data_stripped.csv'


master_df = pd.read_csv(master)

all_subj_bundles = os.listdir(root)

ref_subj = 'S02224' 
bundles = [i for i in all_subj_bundles if ref_subj in i]
bundles = [ i[6:] for i in bundles]

num_groups = 6

tr_list = [0] + [np.quantile(master_df['age'],(i+1)*(1/num_groups)) for i in np.arange(num_groups)] + [1+ np.max(master_df['age'])]
#tr1 = np.quantile(master_df['age'], 0.3)
#tr2 = np.quantile(master_df['age'], 0.66)
#tr3 = np.quantile(master_df['age'], 1)


#bundle = bundles[0] #
#sds = np.zeros([num_groups,12])
#means = np.zeros([num_groups,12])
pickle_path = '/Users/ali/Desktop/Dec23/BuSA/variances/'
figures_path = '/Users/jas/jacques/AD_Decode/BuSA_analysis/'

var_coefs = {}
verbose=True

bundles = [f'_left_bundle_0.xlsx',f'_right_bundle_0.xlsx']

basis = skfda.representation.basis.MonomialBasis(n_basis=10)

total_num_bundles = np.size(bundles)/2

for bundle in bundles:

    if 'left' in bundle:
        side = 'left'
    if 'right' in bundle:
        side = 'right'
    bundle_num = bundle.split('bundle_')[1].split('.')[0]

    fig_bundle_path = os.path.join(figures_path,f'bundle_{side}_{bundle_num}_boxsquaremodel.png')
    this_bundle_subjs = [i for i in all_subj_bundles if bundle in i]
    this_bundle_subjs = sorted(this_bundle_subjs)
    #this_bundle_subjs = [this_bundle_subjs[0]]

    column_names = []
    for i in range(10,40):
        column_names.append("point_"+str(i)+"_fa")

    
    for subj in this_bundle_subjs:
        temp =  pd.read_excel(root + subj)
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
        #var_coefs[subj[2:6], bundle_num] = var.coefficients
        var_coefs[subj[2:6], side, bundle_num] = var

    if verbose:
        print(f'Finished bundle {bundle_num} side {side}')

#pickle.dump(var_coefs, os.path.join(pickle_path, 'var_coefs.py'))


metadf = master_df[['MRI_Exam','age']]

distance_var_FAbundle_array = np.zeros([np.size(this_bundle_subjs),int(total_num_bundles)])
distance_var_FAbundle_dic = {}

for i,subj in enumerate(this_bundle_subjs):
    for bundle_id in np.arange(total_num_bundles):
        left_side_coefs = var_coefs[subj[2:6],'left',str(int(bundle_id))]
        right_side_coefs = var_coefs[subj[2:6],'right',str(int(bundle_id))]
        norm = skfda.misc.metrics.LpNorm(2)
        distance_var_FAbundle_dic[f'{subj[2:6]}',f'bundle_{int(bundle_id)}'] = norm(right_side_coefs - left_side_coefs)[0]
        #distance_var_FAbundle_array[i, int(bundle_id)] = norm(right_side_coefs - left_side_coefs)[0]


subs = set(key[0] for key in distance_var_FAbundle_dic.keys())
columns = set(key[1] for key in distance_var_FAbundle_dic.keys())

subjects_list = [subj[2:6] for subj in this_bundle_subjs]

#distances_var_FAbundle_df = pd.DataFrame(columns=list(['MRI_Exam']+columns))
distances_var_FAbundle_df = pd.DataFrame(columns=(['MRI_Exam']+list(columns)))

for key, value in distance_var_FAbundle_dic.items():
    sub_value, col_name = key
    distances_var_FAbundle_df.at[sub_value, col_name] = value
    distances_var_FAbundle_df.at[sub_value, 'MRI_Exam'] = sub_value

distances_var_FAbundle_df = distances_var_FAbundle_df.reset_index(drop=True)

metadf['MRI_Exam'] = metadf['MRI_Exam'].astype(str)
distances_var_FAbundle_df['MRI_Exam'] = distances_var_FAbundle_df['MRI_Exam'].astype(str)
merged_df = pd.merge(metadf, distances_var_FAbundle_df, on='MRI_Exam', how='outer')


"""
for subj in this_bundle_subjs:
    test_array
    test_array = np.array(list(var_coefs[list(var_coefs.keys())[0]])[0])  
    test_array = np.array(list(var_coefs[list(var_coefs.keys())[0]])[0])     
    skfda.misc.metrics.LpNorm( test_array,1 )       
"""


"""
fd_basis = skfda.FDataBasis(
    basis=basis,
    coefficients= ,
)

fd_basis.plot()
plt.show()        

"""

