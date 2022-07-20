#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:59:01 2022

@author: ali
"""

import numpy as np
from DTC.stat_tools.vertex_func import vertex

data_path='/Users/ali/Desktop/Jul/apoe/python_vertex_screen/' 
x=np.load(data_path +"connectivity.npy")
x.shape
xx=x.transpose()
xx.shape
x=xx


####age

y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Age_Months"]








########################### diet
y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Diet"]
y[y=="HFD"]=-1;
y[y=="Control"]=1;
type(y)
y


########################### geno
y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Genotype"]
print(np.unique(y))
y[y=="APOE22"]=1;
y[y=="APOE22HN"]=1;
y[y=="APOE33"]=2;
y[y=="APOE33HN"]=2;
y[y=="APOE44"]=3;
y[y=="APOE44HN"]=3;

y

########################### sex
y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Sex"]
y[y=="male"]=-1;
y[y=="female"]=1;
type(y)
y

########################### geno,diet 
#########################
####################
############3
y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Genotype"]
print(np.unique(y))
y[y=="APOE22"]=1;
y[y=="APOE22HN"]=1;
y[y=="APOE33"]=2;
y[y=="APOE33HN"]=2;
y[y=="APOE44"]=3;
y[y=="APOE44HN"]=3;

y1=y



y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Diet"]
y[y=="HFD"]=-1;
y[y=="Control"]=1;
type(y)
y2=y


y=np.column_stack((y1,y2))





########################### geno,diet , age, sex
#########################
####################
############3
y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Genotype"]
print(np.unique(y))
y[y=="APOE22"]=1;
y[y=="APOE22HN"]=1;
y[y=="APOE33"]=2;
y[y=="APOE33HN"]=2;
y[y=="APOE44"]=3;
y[y=="APOE44HN"]=3;

y1=y



y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Diet"]
y[y=="HFD"]=-1;
y[y=="Control"]=1;
type(y)
y2=y


y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Age_Months"]
y3=y


y=pyreadr.read_r(data_path + 'response.rda')
y=y["response"]
print(y)
y=y["Sex"]
y[y=="male"]=-1;
y[y=="female"]=1;
type(y)
y4=y


#y=np.column_stack((y1,y2, y3, y4))
#y=np.column_stack((y1,y2, y4))
y=np.column_stack((y1,y2))
#y=np.column_stack((y1,y4))



#


#V,c,d, plt = vertex(x,y,z=y4, return_plot=True, verbose=True)
V,c,d, plt = vertex(x,y, return_plot=True, verbose=True)


import pandas
atlas=pandas.read_csv('/Users/ali/Desktop/Jul/apoe/mouse_anatomy.csv')
#noreadcsf=[148,152,161,314,318,327]
# remove white matter too:
noreadcsf=[ 148, 152, 161, 314, 318, 327, 120, 121, 122, 134, 102, 118, 119, 123, 124, 125, 126,
127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
145, 146, 147, 150, 268, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295,
296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
313, 316]


noreadcsf=[x - 1 for x in noreadcsf]



atlas=atlas.drop(atlas.index[noreadcsf])

report=atlas.iloc[V]
report['score']=c
report=report.sort_values('score', ascending=False)

print(report)


