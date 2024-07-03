#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
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
from DTC.file_manager.file_tools import mkcdir, check_files


def length_hist(length_list, color, filepath = None, title = None):

    fig_hist, ax = plt.subplots(1)
    ax.hist(length_list, color=color)
    ax.set_xlabel('Length')
    ax.set_ylabel('Count')
    if title is not None:
        ax.set_title(title)
    if filepath is not None:
        plt.savefig(filepath)
    plt.close()


def outlier_removal(values, qsep=3):
    iqr = abs(np.quantile(values,0.25) - np.quantile(values,0.75))
    median = np.quantile(values,0.5)
    new_values = values[(values > median - qsep * iqr) & (values < median + qsep * iqr)]
    return new_values


if 'santorini' in socket.gethostname().split('.')[0]:
    root = '/Users/jas/Downloads/Busa_analysis/AD_decode_bundles/'
    master = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
    figures_path = '/Users/jas/jacques/AD_Decode/BuSA_analysis/Figures'
    excel_path = '/Users/jas/jacques/AD_Decode/BuSA_analysis/Excels'
else:
    root = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/stats/'
    master = '/Users/ali/Desktop/Dec23/BuSA/AD_DECODE_data_stripped.csv'
    excel_path = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/Excels'
    figures_path = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/Figures'


if len(sys.argv)<2:
    project = 'V0_9_10template_100_6_interhe_majority'
else:
    project = sys.argv[1]

loc = 'munin'

if loc=='kos':
    root = f'/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_bundle_splitter/{project}'
elif loc=='munin':
    root = f'/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/{project}'

figures_path = os.path.join(root,'Figures')
excel_path = os.path.join(root,'Excels')
stats_path = os.path.join(root,'stats')

mkcdir(excel_path)

master_df = pd.read_csv(master)

all_subj_bundles = os.listdir(stats_path)

ref_subj = 'S02224'
bundles = [i for i in all_subj_bundles if ref_subj in i]
bundles = [ i[6:] for i in bundles]

bundles_new = []
for bundle in bundles:
    if 'comparison' not in bundle:
        bundles_new.append(bundle)
bundles = bundles_new

num_groups = 6

tr_list = [0] + [np.quantile(master_df['age'],(i+1)*(1/num_groups)) for i in np.arange(num_groups)] + [1+ np.max(master_df['age'])]
#tr1 = np.quantile(master_df['age'], 0.3)
#tr2 = np.quantile(master_df['age'], 0.66)
#tr3 = np.quantile(master_df['age'], 1)


#bundle = bundles[0] #
#sds = np.zeros([num_groups,12])
#means = np.zeros([num_groups,12])
#pickle_path = '/Users/ali/Desktop/Dec23/BuSA/variances/'

allvars = {}
allmeans = {}

verbose=True
save_fig = True
test = True

savedatagrid_figs = True

#if test:
#    bundles = [f'_left_bundle_0.xlsx',f'_right_bundle_0.xlsx']

basis = skfda.representation.basis.MonomialBasis(n_basis=10)

total_num_bundles = np.size(bundles)/2

figures_hist_path = os.path.join(figures_path,'histograms_lengths')
mkcdir(figures_hist_path)
verbose=True
for bundle in bundles:

    if 'left' in bundle:
        side = 'left'
    if 'right' in bundle:
        side = 'right'
    bundle_num = bundle.split('bundle_')[1].split('.')[0]

    fig_bundle_path = os.path.join(figures_path,f'bundle_{side}_{bundle_num}_boxsquaremodel.png')
    this_bundle_subjs = [i for i in all_subj_bundles if bundle in i]
    this_bundle_subjs = sorted(this_bundle_subjs)
    if test:
        this_bundle_subjs = [this_bundle_subjs[0]]

    for subj in this_bundle_subjs:
        temp =  pd.read_excel(os.path.join(stats_path, subj))
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
        histfig_path = os.path.join(figures_hist_path,f'{subj[2:6]}_side_{side}_bundle_{bundle_num}.png')
        length_hist(bundle_df.Length.to_list(),(.44, .75, .8),filepath=histfig_path, title = f'{subj[2:6]}_side_{side}_bundle_{bundle_num}')
        if verbose:
            print(f'File saved to {histfig_path}')
