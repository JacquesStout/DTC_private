#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:35 2023

@author: ali
"""
import os , glob
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


def outlier_removal(values, qsep=3):

    iqr = abs(np.quantile(values,0.25) - np.quantile(values,0.75))
    median = np.quantile(values,0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values


if 'santorini' in socket.gethostname().split('.')[0]:
    root = '/Users/jas/Downloads/Busa_analysis/AD_decode_bundles/'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
else:
    root = '/Users/ali/Desktop/Nov23/ad_decode_bundle_analysis/AD_decode_bundles/'
    master = '/Users/ali/Desktop/Nov23/ad_decode_bundle_analysis/AD_DECODE_data_stripped.csv'


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
sds = np.zeros([num_groups,12])
means = np.zeros([num_groups,12])

figures_path = '/Users/jas/jacques/AD_Decode/BuSA_analysis'

for bundle_num,bundle in enumerate(bundles):
    
    fig_bundle_path = os.path.join(figures_path,f'bundle_{bundle_num}_boxsquaremodel.png')
    this_bundle_subjs = [i for i in all_subj_bundles if bundle in i]
    
    #bundle_df = pd.DataFrame()
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
        if 'bundle_df' not in locals():
            bundle_df = temp
        else:
            bundle_df = pd.concat([bundle_df,temp])
    
    bundle_df = bundle_df.dropna()
    
    column_names = []
    for i in range(10,40):
        column_names.append("point_"+str(i)+"_fa")
    bundle_df['averageFA'] = np.mean(bundle_df[column_names],1)

    bundle_df_reduced = bundle_df[["averageFA" , "age"]]
#    bundle_df_reduced = bundle_df_reduced.iloc[ 0:2000]
    
   # bundle_df_reduced.boxplot(column='averageFA',by='age')
    #sns.boxplot(x="age", y="averageFA", data=bundle_df_reduced)
    sns.lmplot(x="age", y="averageFA", data=bundle_df_reduced, x_estimator=np.mean,  order=2)
    plt.title(f'Boxplot for bundle {bundle_num+1}',y=0.9)

    plt.savefig(fig_bundle_path)

    list_df = []
    for i in np.arange(num_groups):
        list_df.append(bundle_df_reduced[(bundle_df_reduced['age']>tr_list[i]) & (bundle_df_reduced['age']<=tr_list[i+1])]['averageFA'])
    #group1 = bundle_df_reduced[bundle_df_reduced['age']<=tr1]['averageFA']
    #group2 = bundle_df_reduced[(bundle_df_reduced['age']>tr1) & (bundle_df_reduced['age']<=tr2)]['averageFA']
    #group3 = bundle_df_reduced[bundle_df_reduced['age']>tr2]['averageFA']

    for i,group in enumerate(list_df):
        group = outlier_removal(group,qsep=1.5)
        means[i,bundle_num] = np.mean(group)
        sds[i,bundle_num] = np.std(group)

    del(bundle_df)

    plt.close()

    #bundle_df['agecat'] = bundle_df['age'] > np.median(bundle_df['age'])
    #bundle_df['agecat'] = np.multiply(bundle_df['agecat'], 1) 
    #model = LinearRegression()


    #bundle_df['sex'] 

#datameans.plot.scatter(x=index)

x_labels = [f'Age {int(tr_list[i])}-{int(tr_list[i+1])}' for i in np.arange(num_groups)]

datameans = pd.DataFrame(index= np.arange(12),columns=x_labels)
datameans.iloc[:,:] = means.transpose()

bundle_labels = [f'Bundle {str(num+1)}' for num in np.arange(np.shape(means)[1])]

# Create 12 different plots with different colors
for i in range(datameans.shape[0]):
    fig_plot_path = os.path.join(figures_path,f'bundle_{i}_AgeGrouping_MeanFA.png')
    plt.figure()
    plt.plot(x_labels, means[:, i], marker='o', color=plt.cm.viridis(i / means.shape[0]), label=bundle_labels[i])
    plt.title(f'Change in values for {bundle_labels[i]}')
    plt.xlabel('X-axis Groups')
    plt.ylabel('Change in Values')
    plt.legend()
    plt.savefig(fig_plot_path)
    plt.close()


x_labels = [f'Age {int(tr_list[i])}-{int(tr_list[i+1])}' for i in np.arange(num_groups)]

datameans = pd.DataFrame(index= np.arange(12),columns=x_labels)
datameans.iloc[:,:] = sds.transpose()

bundle_labels = [f'Bundle {str(num+1)}' for num in np.arange(np.shape(sds)[1])]

# Create 12 different plots with different colors
for i in range(datameans.shape[0]):
    fig_plot_path = os.path.join(figures_path,f'bundle_{i}_AgeGrouping_SdFA.png')
    plt.figure()
    plt.plot(x_labels, sds[:, i], marker='o', color=plt.cm.viridis(i / sds.shape[0]), label=bundle_labels[i])
    plt.title(f'Change in values for {bundle_labels[i]}')
    plt.xlabel('X-axis Groups')
    plt.ylabel('Change in Values')
    plt.legend()
    plt.savefig(fig_plot_path)
    plt.close()


'''

    model = "age ~ " 
    for i in range(10,40):
        model=model + " + point_"+str(i)+"_fa "
    
    
    md = smf.mixedlm(model, bundle_df, groups=bundle_df["Subject"])
    mdf = md.fit()

    var_resid = mdf.scale
    var_random_effect = float(mdf.cov_re.iloc[0])
    var_fixed_effect = mdf.predict(bundle_df).var()
    
    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var
    temp_R2 = [ marginal_r2 , conditional_r2]
    if 'R2s' not in locals():   
        R2s = temp_R2
    else: 
        R2s = np.vstack([R2s,temp_R2])
'''
