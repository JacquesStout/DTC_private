#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eleftherios and Serge

Wenlin make some changes to track on the whole brain
Wenlin add for loop to run all the animals 2018-20-25
"""


from DTC.tract_manager.DTC_manager import create_tracts, tract_connectome_analysis, create_tracts_test
from DTC.file_manager.Daemonprocess import MyPool
import multiprocessing as mp
import os
from DTC.file_manager.file_tools import mkcdir, getfromfile
from time import time
from DTC.file_manager.argument_tools import parse_arguments
from DTC.file_manager.computer_nav import get_mainpaths, get_atlas, glob_remote
from DTC.file_manager.computer_nav import get_mainpaths, get_atlas, glob_remote
import sys

remote=False
project='APOE'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, outpath, atlas_folder, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
atlas_legends = get_atlas(atlas_folder, 'CHASSSYMM3')

diff_preprocessed = os.path.join(inpath, "DWI_allsubj_RAS")

if not remote:
    mkcdir([outpath, diff_preprocessed])
else:
    mkcdir([outpath, diff_preprocessed], sftp)

trkpath = os.path.join(inpath,'TRK_allsubj_RAS')

"""
subjects = ['N58214', 'N58215', 'N58216', 'N58217', 'N58218', 'N58219', 'N58221', 'N58222', 'N58223', 'N58224',
            'N58225', 'N58226', 'N58228', 'N58229', 'N58230', 'N58231', 'N58232', 'N58633', 'N58634', 'N58635',
            'N58636', 'N58650', 'N58649', 'N58651', 'N58653', 'N58654']
"""


subjects_all = glob_remote(os.path.join(diff_preprocessed,'*subjspace*coreg*.nii.gz'),sftp)
subjects = []
for subject in subjects_all:
    subject_name = os.path.basename(subject)
    subjects.append(subject_name[:6])

subjects = ['N57442', 'N57504', 'N58400', 'N58611', 'N58613', 'N58859', 'N60101', 'N60056', 'N60064', 'N60092', 'N60088', 'N60093', 'N60097', 'N60095', 'N60068', 'N60072', 'N60058', 'N60103', 'N60060', 'N60190', 'N60225', 'N60198', 'N58612', 'N60062', 'N60070', 'N60221', 'N60223', 'N58610', 'N60229', 'N60188', 'N60192', 'N60194', 'N60219', 'N60231']
subjects = sorted(subjects)

#removed_list = ['N58398', 'N58634', 'N58610', 'N58613', 'N58732', 'N58999','N58219', 'N58394','N58708','N58712','N58747']
removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects)

#subjects = subjects[0::4]
subjects = subjects[firstsubj:lastsubj]

print(subjects)

stepsize = 1
# accepted values are "small" for one in ten streamlines, "all or "large" for all streamlines,
# "none" or None variable for neither and "both" for both of them

ratio = 1
if ratio == 1:
    saved_streamlines = "_all"
    trk_folder_name = ""
else:
    saved_streamlines = "_ratio_" + str(ratio)
    trk_folder_name = "_" + str(ratio)


#mask types => ['FA', 'T1', 'subjspace']
masktype = "RAS"
stepsize = 2
overwrite = False
get_params = False
forcestart = False
picklesave = True
verbose = True
get_params = None
doprune = False
bvec_orient = [1,2,-3]
vol_b0 = [0,1,2]
classifier = "binary"
symmetric = False
inclusive = False
denoise='none'
savefa= True

make_tracts = True
make_connectomes = False

reference_weighting = None
volume_weighting = True
#classifier types => ["FA", "binary"]
classifiertype = "binary"
brainmask = "subjspace"
brainmask = "dwi"
labeltype='lrordered'
ratio = 1
labelslist = []
seedmasktype = 'white'

labelslist = []
dwi_results = []
donelist = []
notdonelist = []

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
str_identifier = roistring + saved_streamlines + '_stepsize_' + str(stepsize)
str_identifier = roistring + saved_streamlines + '_stepsize_' + str(stepsize).replace(".","_")

bvec_orient=[1,2,-3]
bvec_orient=[-2,1,3]

tall = time()
tract_results = []

duration1=time()
overwrite = False
get_params = False
forcestart = False
if forcestart:
    print("WARNING: FORCESTART EMPLOYED. THIS WILL COPY OVER PREVIOUS DATA")
picklesave = True

#str_identifier='_wholebrain_small_stepsize_2'
createmask = True

dwi_results = []
vol_b0 = [0,1,2,3]

labeltype = 'lrordered'

overwrite=False

if classifiertype == "FA":
    classifiertype = "_fa"
else:
    classifiertype = "_binary"

classifier = 'act'

if inclusive:
    inclusive_str = '_inclusive'
else:
    inclusive_str = '_non_inclusive'

if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'

figspath = os.path.join(outpath,"Figures_RAS"+inclusive_str+symmetric_str+saved_streamlines)

if not remote:
    mkcdir([figspath, trkpath])
else:
    mkcdir([figspath, trkpath], sftp)

if make_connectomes:
    for subject in subjects:
        picklepath_connect = figspath + subject + str_identifier + '_connectomes.p'
        excel_path = figspath + subject + str_identifier + "_connectomes.xlsx"
        if os.path.exists(picklepath_connect) and os.path.exists(excel_path):
            print("The writing of pickle and excel of " + str(subject) + " is already done")
            donelist.append(subject)
        else:
            notdonelist.append(subject)

dwi_results = []
tract_results = []

print(f'Overwrite is {overwrite}')

test=True

trkpath = '/Users/jas/jacques/APOE_trks_test'

if test:
    subjects = ['N58784']
    trkpath = '/Users/jas/jacques/APOE_trks_test'

if subject_processes>1:
    if function_processes>1:
        pool = MyPool(subject_processes)
    else:
        pool = mp.Pool(subject_processes)
    if make_tracts:
        tract_results = pool.starmap_async(create_tracts, [(diff_preprocessed, trkpath, subject, figspath, stepsize, function_processes, str_identifier,
                          ratio, brainmask, classifier, labelslist, bvec_orient, doprune, overwrite, get_params, denoise,
                          verbose) for subject
                                                       in subjects]).get()
    if make_connectomes:
        tract_results = pool.starmap_async(tract_connectome_analysis, [(diff_preprocessed, trkpath, str_identifier, figspath,
                                                                       subject, atlas_legends, bvec_orient, brainmask, inclusive,
                                                                       function_processes, overwrite, picklesave, labeltype, symmetric, reference_weighting, volume_weighting, verbose)
                                                                     for subject in subjects]).get()
    pool.close()
else:
    for subject in subjects:
        if make_tracts:
            if test:
                create_tracts_test(diff_preprocessed, trkpath, subject, figspath, stepsize, function_processes, atlas_legends, str_identifier,
                              ratio, brainmask, classifier, labelslist, bvec_orient, doprune, overwrite, get_params, denoise, seedmasktype, labeltype,
                              verbose, sftp)
            else:
                create_tracts(diff_preprocessed, trkpath, subject, figspath, stepsize, function_processes,
                              str_identifier,
                              ratio, brainmask, classifier, labelslist, bvec_orient, doprune, overwrite, get_params,
                              denoise,
                              verbose, sftp)
        #get_diffusionattributes(diff_preprocessed, diff_preprocessed, subject, str_identifier, vol_b0, ratio, bvec_orient,
        #                        masktype, overwrite, verbose)
        if make_connectomes:
            tract_results.append(tract_connectome_analysis(diff_preprocessed, trkpath, str_identifier, figspath, subject,
                                                           atlas_legends, bvec_orient,  brainmask, inclusive,
                                                           function_processes, overwrite, picklesave, labeltype, symmetric, reference_weighting, volume_weighting, verbose, sftp))
    print(tract_results)
