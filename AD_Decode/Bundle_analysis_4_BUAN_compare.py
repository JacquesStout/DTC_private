
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

buan_figures_path = os.path.join(figures_path, f'BUAN_comparisons')

mkcdir(buan_figures_path)

master_df = pd.read_csv(master)

num_bundles = 6

files = [file for file in os.listdir(stats_path) if 'comparison.xlsx' in file and not '$' in file]

# Initialize an empty DataFrame to store the combined data

master_df = pd.read_csv(master)


# Loop through each file and load the DataFrame
for file in files:
    file_path = os.path.join(stats_path, file)

    # Load the DataFrame from the Excel file
    df = pd.read_excel(file_path)
    subj = df['Subject'][0]
    index = master_df["MRI_Exam"] == int(subj[2:6])
    try:
        df['age'] = master_df[index]['age'].iloc[0]
    except:
        print(f'Subject {subj} not found, continue')
        continue
    # Append the DataFrame to the combined DataFrame
    if not 'combined_df' in locals():
        combined_df = df
    else:
        #combined_df = pd.merge(combined_df, df, how='inner', on='Subject')
        combined_df = pd.concat([combined_df, df.iloc[0].to_frame().T], ignore_index=True)

import seaborn as sns
final_df = combined_df

for bun_num in np.arange(num_bundles):

    fig_BUAN_path =os.path.join(buan_figures_path,f'buan_compare_bundle_{bun_num}.png')

    x = pd.to_numeric(final_df.age, errors='coerce').values
    y = pd.to_numeric(final_df[f'BUAN_{bun_num}'], errors='coerce').values

    coefficients = np.polyfit(x, y, 2)
    quadratic_function = np.poly1d(coefficients)

    # Generate values for x-axis for smooth curve
    x_smooth = np.linspace(min(x), max(x), 100)

    # Plot original data using Seaborn scatter plot
    sns.scatterplot(x=x, y=y, label='Original Data')

    # Plot quadratic estimation using Seaborn line plot
    sns.lineplot(x=x_smooth, y=quadratic_function(x_smooth), color='red', label='Quadratic Estimation')
    plt.title(f'BUAN comparison depending on age for bundle {bun_num}')
    plt.savefig(fig_BUAN_path)
    plt.close()