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
from DTC.file_manager.file_tools import mkcdir, check_files


def outlier_removal(values, qsep=3):

    iqr = abs(np.quantile(values,0.25) - np.quantile(values,0.75))
    median = np.quantile(values,0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values



remote=False

if len(sys.argv)<2:
    #project = 'V0_9_10template_100_6_interhe_majority'
    project = 'V_1_0_10template_100_6_interhe_majority'
else:
    project = sys.argv[1]

loc = 'munin'

if 'santorini' in socket.gethostname().split('.')[0]:
    root = f'/Users/jas/jacques/AD_Decode/BuSA_analysis/{project}'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
else:
    root = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/'
    master = '/Users/ali/Desktop/Dec23/BuSA/AD_DECODE_data_stripped.csv'


if loc=='kos':
    root = f'/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'
elif loc=='munin':
    root = f'/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/{project}'


if remote:
    from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
    from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths
    from DTC.file_manager.computer_nav import glob_remote, load_df_remote

    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
    root = f'/mnt/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'
    inpath, _, _, sftp = get_mainpaths(remote,project = 'AD_Decode', username=username,password=passwd)
else:
    sftp = None

master_df = pd.read_csv(master)

length_cut = 40
if length_cut is None:
    length_str = ''
else:
    length_str = f'_Lengthcut{length_cut}'

figures_path = os.path.join(root,f'Figures{length_str}')
excel_path = os.path.join(root,f'Excels{length_str}')
stats_path = os.path.join(root,'stats')

mkcdir([stats_path,figures_path, excel_path])

if remote:
    all_subj_bundles = glob_remote(stats_path, sftp)
else:
    all_subj_bundles = os.listdir(stats_path)

ref_subj = 'S02224' 
bundles = [i for i in all_subj_bundles if ref_subj in i]
bundles = [ i[6:] for i in bundles]
bundles = sorted(bundles)

bundles_new = []
for bundle in bundles:
    if 'comparison' not in bundle:
        bundles_new.append(bundle)
bundles = bundles_new

num_groups = 5
num_bundles = 6

tr_list = [0] + [np.quantile(master_df['age'],(i+1)*(1/num_groups)) for i in np.arange(num_groups)] + [1+ np.max(master_df['age'])]

#sds = np.zeros([num_groups,num_bundles*2])
#means = np.zeros([num_groups,num_bundles*2])

test=False

if test:
    bundles = [f'_left_bundle_0.xlsx']


columns_list = [f'Side_{bundle.split("_")[1]}_ID_{bundle.split("_")[3].split(".")[0]}' for bundle in bundles]
datameans = pd.DataFrame(columns=(['Age Group'] + columns_list))
datasds = pd.DataFrame(columns=(['Age Group'] + columns_list))

verbose =True

ref = 'mrtrixfa'

figures_box_path = os.path.join(figures_path,f'boxsquares_age_{ref}')
figures_agegrouping_path = os.path.join(figures_path,f'agegrouping_mean_{ref}')

mkcdir([figures_box_path,figures_agegrouping_path])

#meanbundle.split('_')[1]s = {}
#sds = {}

testmode = False

for bundle in bundles:

    if 'left' in bundle:
        side = 'left'
    if 'right' in bundle:
        side = 'right'
    bundle_num = bundle.split('bundle_')[1].split('.')[0]
    
    fig_bundle_path = os.path.join(figures_box_path,f'bundle_{side}_{bundle_num}_boxsquaremodel.png')
    this_bundle_subjs = [i for i in all_subj_bundles if bundle in i]
    this_bundle_subjs = sorted(this_bundle_subjs)
    if testmode:
        this_bundle_subjs = this_bundle_subjs[:3]

    #bundle_df = pd.DataFrame()
    for subj in this_bundle_subjs:

        if remote:
            load_df_remote(os.path.join(stats_path, subj), sftp)
        else:
            temp = pd.read_excel(os.path.join(stats_path, subj))

        #temp = pd.DataFrame()
        temp['Subject']=subj[2:6]
        index = master_df["MRI_Exam"] == int(subj[2:6])
        try:
            temp['age'] = master_df[index]['age'].iloc[0]
            #temp['sex'] = master_df[index]['sex']
        except:
            print(f'Subject {subj[:6]} is missing from excel database')
            continue
        #temp = temp.to_numpy()
        if 'bundle_df' not in locals():
            bundle_df = temp
        else:
            bundle_df = pd.concat([bundle_df,temp])
    
    bundle_df = bundle_df.dropna()

    if length_cut is not None:
        bundle_df = bundle_df[bundle_df['Length'] >= int(length_cut)]

    column_names = []
    for i in range(1,50):
        column_names.append("point_"+str(i)+f"_{ref}")
    bundle_df[f'average{ref}'] = np.mean(bundle_df[column_names],1)

    bundle_df_reduced = bundle_df[[f"average{ref}" , "age"]]
#    bundle_df_reduced = bundle_df_reduced.iloc[ 0:2000]
    
   # bundle_df_reduced.boxplot(column='averageFA',by='age')
    #sns.boxplot(x="age", y="averageFA", data=bundle_df_reduced)
    sns.lmplot(x="age", y=f"average{ref}", data=bundle_df_reduced, x_estimator=np.mean,  order=2)
    plt.title(f'Boxplot for bundle {int(bundle_num)+1}',y=0.9)

    plt.savefig(fig_bundle_path)

    list_df = []
    for i in np.arange(num_groups):
        list_df.append(bundle_df_reduced[(bundle_df_reduced['age']>tr_list[i]) & (bundle_df_reduced['age']<=tr_list[i+1])][f'average{ref}'])
    #group1 = bundle_df_reduced[bundle_df_reduced['age']<=tr1]['averageFA']
    #group2 = bundle_df_reduced[(bundle_df_reduced['age']>tr1) & (bundle_df_reduced['age']<=tr2)]['averageFA']
    #group3 = bundle_df_reduced[bundle_df_reduced['age']>tr2]['averageFA']

    for i,group in enumerate(list_df):
        group = outlier_removal(group,qsep=1.5)
        #means[int(i),int(bundle_num)] = np.mean(group)
        #sds[int(i),int(bundle_num)] = np.std(group)
        datameans.loc[i, 'Age Group'] = f'Age Group {i}'
        datameans.loc[i, f'Side_{side}_ID_{bundle_num}'] = np.mean(group)
        datasds.loc[i, 'Age Group'] = f'Age Group {i}'
        datasds.loc[i, f'Side_{side}_ID_{bundle_num}'] = np.std(group)

    del(bundle_df)

    plt.close()

    if verbose:
        print(f'Finished bundle {bundle_num} side {side}')

    #bundle_df['agecat'] = bundle_df['age'] > np.median(bundle_df['age'])
    #bundle_df['agecat'] = np.multiply(bundle_df['agecat'], 1) 
    #model = LinearRegression()


    #bundle_df['sex'] 

#datameans.plot.scatter(x=index)

x_labels = [f'Age {int(tr_list[i])}-{int(tr_list[i+1])}' for i in np.arange(num_groups)]

#datameans = pd.DataFrame(index= np.arange(12),columns=x_labels)
#datameans.iloc[:,:] = means.transpose()

bundle_labels = [f'Bundle {str(num+1)}' for num in np.arange(np.shape(datameans)[1])]

# Create 12 different plots with different colors
for i,bundle_name in enumerate(columns_list):
    fig_plot_path = os.path.join(figures_agegrouping_path,f'{bundle_name}_AgeGrouping_Mean{ref}.png')
    plt.figure()
    plt.plot(x_labels, datameans.loc[:,bundle_name], marker='o', color=plt.cm.viridis(i / datameans.shape[0]), label=bundle_name)
    plt.title(f'Change in values for {bundle_name}')
    plt.xlabel('X-axis Groups')
    plt.ylabel('Change in Values')
    plt.legend()
    plt.savefig(fig_plot_path)
    plt.close()


# Create 12 different plots with different colors
for i,bundle_name in enumerate(columns_list):
    fig_plot_path = os.path.join(figures_agegrouping_path,f'{bundle_name}_AgeGrouping_Sd{ref}.png')
    plt.figure()
    plt.plot(x_labels, datasds.loc[:,bundle_name], marker='o', color=plt.cm.viridis(i / datameans.shape[0]), label=bundle_name)
    plt.title(f'Change in values for {bundle_name}')
    plt.xlabel('X-axis Groups')
    plt.ylabel('Change in Values')
    plt.legend()
    plt.savefig(fig_plot_path)
    plt.close()
