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
from sklearn import datasets, linear_model


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
    root = '/Users/ali/Desktop/Dec23/BuSA/AD_decode_bundles/'
    master = '/Users/ali/Desktop/Dec23/BuSA/AD_DECODE_data_stripped.csv'

mkcdir(excel_path)

excel_data = pd.read_excel(os.path.join(excel_path,'distances_var_FAbundle.xlsx'))

y = excel_data['age']

bundle_number = 6
column_names = [f'bundle_{i}_var' for i in np.arange(bundle_number)]
x = excel_data[column_names]

regr = linear_model.LinearRegression()
regr.fit(x, y)

X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

"""
#plt.scatter(x, y,  color='black')
plt.plot(x, regr.predict(x), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
"""