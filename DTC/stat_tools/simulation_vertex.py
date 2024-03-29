#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:09:00 2022

@author: ali
"""

import dcor
import math

import numpy as np
from timeit import timeit
import matplotlib.pyplot as plt
from numpy import random
from DTC.stat_tools.vertex_func import vertex

m=100
y=np.random.random_integers(1, high=3, size=m)

dict_pvals=dict( [ (1, 0.3), (2, 0.4), (3, 0.5) ]  )
p_vals=np.vectorize(dict_pvals.get)(y)


blk_dim=100

m, n = 100, 2*blk_dim

#Matrix = np.array([[[0 for x in range(n)] for y in range(n)] for z in range(m) ])

Matrix=np.zeros((m,n,n))
Matrix[99]



for i in range(Matrix.shape[0]):
    #print(Matrix[i].shape)
    Matrix[i]= np.block([  
    [p_vals[i]* np.ones((blk_dim,blk_dim))+0.1*np.random.normal(0, 1, size=(blk_dim, blk_dim)) , 0.2*np.ones((blk_dim,blk_dim))],
    [0.2*np.ones((blk_dim,blk_dim)), 0.3*np.ones((blk_dim,blk_dim))] 
    ])
    
    
x= Matrix
V,c,d, plt = vertex(x,y,z=y**2, return_plot=True, verbose=True)


