#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eleftherios and Serge

Wenlin make some changes to track on the whole brain
Wenlin add for loop to run all the animals 2018-20-25
"""

from time import time
import numpy as np
import os
import multiprocessing as mp
import pickle
#from DTC.tract_manager import create_tracts, tract_connectome_analysis, diff_handlers.diff_preprocessing
#from DTC.diff_handlers.bvec_handler import extractbvec_fromheader
from DTC.file_manager.BIAC_tools import send_mail
from DTC.file_manager.Daemonprocess import MyPool
from DTC.file_manager.argument_tools import parse_arguments
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from DTC.file_manager.BIAC_tools import isempty

import sys, getopt, glob

subjects = ["N58214", "N58215", "N58216", "N58217", "N58218", "N58219", "N58221", "N58222", "N58223", "N58224",
                "N58225", "N58226", "N58228",
                "N58229", "N58230", "N58231", "N58232", "N58633", "N58635", "N58636", "N58649", "N58650",
                "N58651", "N58653", "N58654",
                'N58408', 'N58714', 'N58740', 'N58477', 'N58734', 'N58309', 'N58792', 'N58302',
                'N58784', 'N58706', 'N58361', 'N58355', 'N58712', 'N58790', 'N58606', 'N58350', 'N58608',
                'N58779', 'N58500', 'N58604', 'N58749', 'N58510', 'N58394', 'N58346', 'N58344', 'N58788', 'N58305',
                'N58514', 'N58794', 'N58733', 'N58655', 'N58735', 'N58310', 'N58400', 'N58708', 'N58780', 'N58512',
                'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58745', 'N58406', 'N58359', 'N58742', 'N58396',
                'N58613', 'N58732', 'N58516', 'N58402']

samos=True
if samos:
    main_folder = "/mnt/paros_MRI/jacques/APOE/"

    diff_preprocessed = os.path.join(main_folder, 'DWI_allsubj_RAS')
    trkpath = os.path.join(main_folder,'TRK_allsubj_RAS')
    figspath = os.path.join(main_folder,"Connectomes_allsubj_RAS")

subjects_all = glob.glob(os.path.join(diff_preprocessed,'*coreg*.nii.gz'))
subjects = []
for subject in subjects_all:
    subject_name = os.path.basename(subject)
    subjects.append(subject_name[:6])
print(subjects)

removed_list = ['N58398', 'N58634', 'N58610', 'N58613', 'N58732', 'N58999','N58219']
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)
subjects = subjects
subject_processes, function_processes = parse_arguments(sys.argv, subjects)

outpathpickle = figspath

atlas_legends = "/mnt/paros_MRI/jacques/atlases/CHASSSYMM3AtlasLegends.xlsx"

stepsize = 2
# accepted values are "small" for one in ten streamlines, "all or "large" for all streamlines,
# "none" or None variable for neither and "both" for both of them

ratio = 1
if ratio == 1:
    saved_streamlines = "_all_"
else:
    saved_streamlines = "_ratio_" + str(ratio)




#mask types => ['FA', 'T1', 'subjspace']
masktype = "RAS"
stepsize = 2
overwrite = False
get_params = False
forcestart = False
picklesave = True
verbose = True
get_params = None
doprune = True
bvec_orient = [1,2,-3]
vol_b0 = [0,1,2]
classifier = "binary"
symmetric = False
inclusive = False
denoise='none'
savefa= True
make_connectomes = True
#classifier types => ["FA", "binary"]
classifiertype = "binary"
brainmask = "subjspace"
brainmask = "RAS"
labeltype='lrordered'
ratio = 1
labelslist = []



if classifiertype == "FA":
    classifiertype = "_fa"
else:
    classifiertype = "_binary"

trkroi = ["wholebrain"]
if len(trkroi)==1:
    roistring = "_" + trkroi[0] #+ "_"
elif len(trkroi)>1:
    roistring="_"
    for roi in trkroi:
        roistring = roistring + roi[0:4]
    roistring = roistring #+ "_"
#str_identifier = '_stepsize_' + str(stepsize) + saved_streamlines+ roistring
str_identifier = '_stepsize_' + str(stepsize) + classifiertype + roistring + saved_streamlines #to be reimplemented if full calc, disabled for now
str_identifier = roistring + saved_streamlines + 'stepsize_' + str(stepsize)
bvec_orient=[1,2,-3]
bvec_orient=[-2,1,3]

tall = time()
tract_results = []


if verbose:
    txt=("Process running with % d max processes available on % d subjects with % d subjects in parallel each using % d processes"
      % (mp.cpu_count(), np.size(subjects), subject_processes, function_processes))
    print(txt)
    send_mail(txt,subject="Main process start msg ")

duration1=time()
overwrite = False
get_params = False
forcestart = False
if forcestart:
    print("WARNING: FORCESTART EMPLOYED. THIS WILL COPY OVER PREVIOUS DATA")
picklesave = True

donelist = []
notdonelist = []
for subject in subjects:
    picklepath_connect = figspath + subject + str_identifier + '_connectomes.p'
    excel_path = figspath + subject + str_identifier + "_connectomes.xlsx"
    if os.path.exists(picklepath_connect) and os.path.exists(excel_path):
        print("The writing of pickle and excel of " + str(subject) + " is already done")
        donelist.append(subject)
    else:
        notdonelist.append(subject)

#str_identifier='_wholebrain_small_stepsize_2'
createmask = True

dwi_results = []
vol_b0 = [0,1,2,3]

labeltype = 'lrordered'
make_connectomes=True
overwrite=False

if subject_processes>1:
    if function_processes>1:
        pool = MyPool(subject_processes)
    else:
        pool = mp.Pool(subject_processes)
    tract_results = pool.starmap_async(create_tracts, [(diff_preprocessed, trkpath, subject, figspath, stepsize, function_processes,
                                                        str_identifier, ratio, brainmask, classifiertype, labelslist, bvec_orient, doprune,
                                                        overwrite, get_params, denoise, verbose) for subject in subjects]).get()
    print(f'overwrite is {overwrite}')
    if make_connectomes:
        tract_results = pool.starmap_async(tract_connectome_analysis, [(diff_preprocessed, trkpath, str_identifier, figspath,
                                                                   subject, atlas_legends, bvec_orient,
                                                                    inclusive,function_processes, forcestart,
                                                                    picklesave, labeltype, symmetric, verbose) for subject in subjects]).get()
    pool.close()
else:
    for subject in subjects:
        tract_results.append(create_tracts(diff_preprocessed, trkpath, subject, figspath, stepsize, function_processes, str_identifier,
                                              ratio, brainmask, classifiertype, labelslist, bvec_orient, doprune, overwrite, get_params, denoise,
                                           verbose))
        print(f'overwrite is {overwrite}')
        #print(f'{diff_preprocessed}, {trkpath}, {str_identifier} \n {figspath}, {subject}, {atlas_legends} \n {bvec_orient}, {brainmask}, {inclusive}, \n {function_processes}, {forcestart}, {picklesave} \n {labeltype}, {verbose}')   
        if make_connectomes:
            tract_results.append(tract_connectome_analysis(diff_preprocessed, trkpath, str_identifier, figspath, subject,
                                                     atlas_legends, bvec_orient, brainmask, inclusive, function_processes,
                                                     forcestart, picklesave, labeltype, verbose))


subject=l[0]
