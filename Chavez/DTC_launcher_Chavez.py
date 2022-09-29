
import numpy as np
import glob
from DTC.diff_handlers.bvec_handler import orient_to_str
from DTC.tract_manager.DTC_manager import create_tracts, tract_connectome_analysis, get_diffusionattributes
from DTC.file_manager.Daemonprocess import MyPool
import multiprocessing as mp
import os
from DTC.file_manager.file_tools import mkcdir, getfromfile
from time import time
from DTC.file_manager.argument_tools import parse_arguments
import sys
import socket
import random
from DTC.file_manager.computer_nav import get_mainpaths, get_atlas

remote=True
project='Chavez'

computer_name = socket.gethostname()

if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

inpath, outpath, atlas_folder, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
atlas_legends = get_atlas(atlas_folder, 'CHASSSYMM3')

diff_preprocessed = os.path.join(inpath, "DWI")

if not remote:
    mkcdir([outpath, diff_preprocessed])
else:
    mkcdir([outpath, diff_preprocessed], sftp)

subjects = ['apoe_3_6_CtrlA_male', 'MHI_335_CtrA_male', 'apoe_4_4_CtrlB_female', 'MHI_326_C_male', 'MHI_334_CtrC_female', 'MHI_335_CtrB_female', 'MHI_326_D_female', 'apoe_4_2_A_male', 'apoe_3_4_CtrA_female', 'apoe_4_7_B_male', 'MHI_326_CtrB_male', 'MHI_334_D_female', 'apoe_4_7_A_female', 'apoe_4_2_B_female', 'apoe_3_6_B_male', 'MHI_334_CtrA_male', 'apoe_4_5_CtrA_female', 'apoe_4_4_CtrlD_male', 'apoe_4_4_B_female', 'MHI_326_CtrA_female', 'apoe_3_4_A_female']

subjects.reverse()
removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

subjects.sort()

print(subjects)
subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv,subjects)
subjects = subjects[firstsubj:lastsubj]
#subject_processes=1
#function_processes=10
#mask types => ['FA', 'T1', 'subjspace']
masktype = "subjspace"
stepsize = 2
overwrite = False
get_params = False
forcestart = False
picklesave = True
verbose = True
get_params = None
doprune = True
bvec_orient = [1,2,3]
vol_b0 = [0,1,2]
classifier = "binary"
symmetric = False
inclusive = False
denoise = "coreg"
savefa = True

#reference_weighting = 'fa'
reference_weighting = None
volume_weighting = False
make_tracts = True
make_connectomes = False

classifiertype = "binary"
brainmask = "subjspace"
labeltype='lrordered'
ratio = 1

if ratio == 1:
    saved_streamlines = "_all"
    trk_folder_name = ""
else:
    saved_streamlines = "_ratio_" + str(ratio)
    trk_folder_name = "_" + str(ratio)

#trkpath = os.path.join(inpath, "TRK_MPCA_fixed")
#trkpath = os.path.join(inpath, "TRK_MPCA_100")
#trkpath = os.path.join(inpath, "TRK_MPCA_fixed"+trk_folder_name)
trkpath = os.path.join(inpath, "TRK"+trk_folder_name)

trkroi = ["wholebrain"]
if len(trkroi)==1:
    roistring = "_" + trkroi[0] #+ "_"
elif len(trkroi)>1:
    roistring="_"
    for roi in trkroi:
        roistring = roistring + roi[0:4]
    roistring = roistring #+ "_"
str_identifier = '_stepsize_' + str(stepsize).replace(".","_") + saved_streamlines + roistring

duration1=time()

if forcestart:
    print("WARNING: FORCESTART EMPLOYED. THIS WILL COPY OVER PREVIOUS DATA")

labelslist = []
dwi_results = []
donelist = []
notdonelist = []

if classifiertype == "FA":
    classifiertype = "_fa"
else:
    classifiertype = "_binary"

if inclusive:
    inclusive_str = '_inclusive'
else:
    inclusive_str = '_non_inclusive'

if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'

figspath = os.path.join(outpath,"Figures"+inclusive_str+symmetric_str+saved_streamlines)

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
            tract_results.append(
            create_tracts(diff_preprocessed, trkpath, subject, figspath, stepsize, function_processes, str_identifier,
                          ratio, brainmask, classifier, labelslist, bvec_orient, doprune, overwrite, get_params, denoise,
                          verbose, sftp))
        #get_diffusionattributes(diff_preprocessed, diff_preprocessed, subject, str_identifier, vol_b0, ratio, bvec_orient,
        #                        masktype, overwrite, verbose)
        if make_connectomes:
            tract_results.append(tract_connectome_analysis(diff_preprocessed, trkpath, str_identifier, figspath, subject,
                                                           atlas_legends, bvec_orient,  brainmask, inclusive,
                                                           function_processes, overwrite, picklesave, labeltype, symmetric, reference_weighting, volume_weighting, verbose, sftp))
    print(tract_results)



"""
subjfolder = glob.glob(os.path.join(datapath, "*" + identifier + "*"))[0]
subjbxh = glob.glob(os.path.join(subjfolder, "*.bxh"))
for bxhfile in subjbxh:
    bxhtype = checkbxh(bxhfile, False)
    if bxhtype == "dwi":
        dwipath = bxhfile.replace(".bxh", ".nii.gz")
        break
"""
"""
subjects_all = glob.glob(os.path.join(diff_preprocessed,'*subjspace_dwi*.nii.gz'))
subjects = []
for subject in subjects_all:
    subject_name = os.path.basename(subject)
    subjects.append(subject_name[:6])
"""
