import numpy as np
import multiprocessing as mp
#from file_manager.Daemonprocess import MyPool
import glob
import os, sys
from DTC.diff_handlers.bvec_handler import writebfiles, extractbvals, extractbvals_research, rewrite_subject_bvalues, fix_bvals_bvecs
from time import time
import shutil
from DTC.diff_handlers.diff_preprocessing import launch_preprocessing
from DTC.file_manager.file_tools import mkcdir, largerfile
import shutil
from DTC.file_manager.argument_tools import parse_arguments
from DTC.diff_handlers.bvec_handler import orient_to_str


gunniespath = "/Users/jas/bass/gitfolder/gunnies/"
diffpath = "/Volumes/dusom_civm-atlas/19.abb.14/research/"
outpath = "/Volumes/Data/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/"

SAMBA_inputs_folder = "/Volumes/Data/Badea/Lab/19abb14/"
shortcuts_all_folder = "/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/"
shortcuts_all_folder = None

subjects_folders = glob.glob(os.path.join(diffpath,'diffusion*/'))
subjects = []
for subject_folder in subjects_folders:
    subjects.append(subject_folder.split('diffusion')[1][:6])

#removed_list = ['N58794','N58514','N58305','N58613','N58346','N58344','N58788']
removed_list = []


subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects)

removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

subjects = subjects[firstsubj:lastsubj]

print(subjects)

proc_subjn=""
denoise="None"
recenter=0
proc_name ="diffusion_prep_"+proc_subjn
cleanup = True
masking = "median"
makebtables = False
gettranspose=False
copybtables = True
verbose=True
transpose=None
overwrite=False
ref="coreg"
#btables=["extract","copy","None"]
btables="extract"


#Neither copy nor extract are fully functioning right now, for now the bvec extractor from extractdiffdirs works
#go back to this if ANY issue with bvals/bvecs
#extract is as the name implies here to extract the bvals/bvecs from the files around subject data
#copy takes one known good file for bval and bvec and copies it over to all subjects

subjects_toremove=[]
if btables=="extract":
    for subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        writeformat="tab"
        writeformat="dsi"
        try:
            subjectpath = glob.glob(os.path.join(os.path.join(diffpath, "diffusion*"+subject+"*")))[0]
            subject_outpath = os.path.join(outpath, 'diffusion_prep_' + proc_subjn + subject)
            mkcdir(subject_outpath)
            fbvals, fbvecs = extractbvals(subjectpath, subject, outpath=subject_outpath, writeformat=writeformat, overwrite=overwrite) #extractbvals_research
            #fbvals, fbvecs = rewrite_subject_bvalues(diffpath, subject, outpath=outpath, writeformat=writeformat, overwrite=overwrite)
        except IndexError:
            print(f'skipping subject {subject}')
            subjects_toremove.append(subject)
elif btables=="copy":
    for subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        outpathsubj = os.path.join(outpath,proc_name+subject)
        outpathbval= os.path.join(outpathsubj, proc_subjn + subject+"_bvals.txt")
        outpathbvec= os.path.join(outpathsubj, proc_subjn + subject+"_bvecs.txt")
        outpathrelative = os.path.join(outpath, "relative_orientation.txt")
        mkcdir(outpathsubj)
        newoutpathrelative= os.path.join(outpathsubj, "relative_orientation.txt")
        shutil.copy(outpathrelative, newoutpathrelative)
        if not os.path.exists(outpathbval) or not os.path.exists(outpathbvec) or overwrite:
            mkcdir(outpathsubj)
            writeformat="tab"
            writeformat="dsi"
            overwrite=True
            #bvals = glob.glob(os.path.join(outpath, "*N58214*bvals*.txt"))
            #bvecs = glob.glob(os.path.join(outpath, "*N58214*bvec*.txt"))
            bvals = glob.glob(os.path.join(outpath, "*N60062*bvals*.txt"))
            bvecs = glob.glob(os.path.join(outpath, "*N60062*bvec*.txt"))
            if np.size(bvals)>0 and np.size(bvecs)>0:
                shutil.copy(bvals[0], outpathbval)
                shutil.copy(bvecs[0], outpathbvec)

for rem in subjects_toremove:
    if rem in subjects:
        subjects.remove(rem)

# accepted values are "small" for one in ten streamlines, "all or "large" for all streamlines,
# "none" or None variable for neither and "both" for both of them
nominal_bval=4000
verbose=True
results=[]

if subject_processes>1:
    if function_processes>1:
        pool = MyPool(subject_processes)
    else:
        pool = mp.Pool(subject_processes)
    results = pool.starmap_async(launch_preprocessing, [launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose, overwrite, denoise, recenter,
                             recenter, verbose) for subject in subjects]).get()
else:
    for subject in subjects:
        max_size=0
        subjectpath = glob.glob(os.path.join(os.path.join(diffpath, "diffusion*"+subject+"*")))[0]
        subject_outpath = os.path.join(outpath, 'diffusion_prep_' + proc_subjn + subject)
        max_file=largerfile(subjectpath)
        if os.path.exists(os.path.join(subject_outpath, f'{subject}_subjspace_fa.nii.gz')) and not overwrite:
            print(f'already did subject {subject}')
        elif os.path.exists(os.path.join('/Volumes/Badea/Lab/APOE_symlink_pool/', f'{subject}_subjspace_coreg.nii.gz')) and not overwrite:
            print(f'Could not find subject {subject} in main diffusion folder but result was found in SAMBA prep folder')
        #elif os.path.exists(os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_20APOE01_chass_symmetric3_allAPOE-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_NoNameYet_n32_i5/reg_images/',f'{subject}_rd_to_MDT.nii.gz')) and not overwrite:
        #    print(f'Could not find subject {subject} in main diff folder OR samba init but was in results of SAMBA')
        else:
            print(max_file)
            launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose,
                                 overwrite, denoise, recenter, verbose)
