import numpy as np
import multiprocessing as mp
import glob
import os, sys
from DTC.diff_handlers.bvec_handler import extractbvals
from DTC.diff_handlers.diff_preprocessing import launch_preprocessing
from DTC.file_manager.file_tools import mkcdir, largerfile, getfromfile
import shutil
from DTC.file_manager.argument_tools import parse_arguments
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote, copy_loctoremote, checkfile_exists_remote, load_nifti_remote, make_temppath
from DTC.nifti_handlers.nifti_handler import average_4dslices


gunniespath = "/Users/jas/bass/gitfolder/gunnies/"

project = 'Vitek'
remote = True
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
if remote:
    _, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)

diffpath = "/mnt/paros_MRI/Vitek_UNC/"
outpath = "/Volumes/Data/Badea/Lab/mouse/Vitek_series/diffusion_prep_locale/"

SAMBA_inputs_folder = "/Volumes/Data/Badea/Lab/19abb14/"
SAMBA_inputs_folder = None
shortcuts_all_folder = "/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/"
shortcuts_all_folder = None

subjects_folders = glob_remote(diffpath, sftp)
subjects_folders = [folder for folder in subjects_folders if 'Vitek_' in os.path.basename(folder)]
subjects = []
for subject_folder in subjects_folders:
    subjects.append(subject_folder.split('UNC/')[1][6:12])
#subjects = ['N58309']
#removed_list = ['N58794','N58514','N58305','N58613','N58346','N58344','N58788']

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects)

removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

subjects = subjects[firstsubj:lastsubj]
subjects.sort()
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
btables="copy"


#Neither copy nor extract are fully functioning right now, for now the bvec extractor from extractdiffdirs works
#go back to this if ANY issue with bvals/bvecs
#extract is as the name implies here to extract the bvals/bvecs from the files around subject data
#copy takes one known good file for bval and bvec and copies it over to all subjects

if btables=="extract":
    for subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        writeformat="tab"
        writeformat="dsi"
        subjectpath = glob.glob(os.path.join(os.path.join(diffpath, "diffusion*"+subject+"*")))[0]
        subject_outpath = os.path.join(outpath, 'diffusion_prep_' + proc_subjn + subject)
        mkcdir(subject_outpath)
        fbvals, fbvecs = extractbvals(subjectpath, subject, outpath=subject_outpath, writeformat=writeformat, overwrite=overwrite) #extractbvals_research
        #fbvals, fbvecs = rewrite_subject_bvalues(diffpath, subject, outpath=outpath, writeformat=writeformat, overwrite=overwrite)
elif btables=="copy":
    for subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        outpathsubj = os.path.join(outpath,proc_name+subject)
        outpathbval= os.path.join(outpathsubj, proc_subjn + subject+"_bvals.txt")
        outpathbvec= os.path.join(outpathsubj, proc_subjn + subject+"_bvecs.txt")
        outpathrelative = os.path.join(outpath, "relative_orientation.txt")
        newoutpathrelative= os.path.join(outpathsubj, "relative_orientation.txt")
        mkcdir(outpathsubj)
        shutil.copy(outpathrelative, newoutpathrelative)
        if not os.path.exists(outpathbval) or not os.path.exists(outpathbvec) or overwrite:
            mkcdir(outpathsubj)
            writeformat="tab"
            writeformat="dsi"
            overwrite=True
            bvals = glob.glob(os.path.join(outpath, "bvals*.txt"))
            bvecs = glob.glob(os.path.join(outpath, "bvecs*.txt"))
            if np.size(bvals)>0 and np.size(bvecs)>0:
                shutil.copy(bvals[0], outpathbval)
                shutil.copy(bvecs[0], outpathbvec)

# accepted values are "small" for one in ten streamlines, "all or "large" for all streamlines,
# "none" or None variable for neither and "both" for both of them
nominal_bval=3000
verbose=True
results=[]

toremove = []

if subject_processes>1:
    if function_processes>1:
        pool = MyPool(subject_processes)
    else:
        pool = mp.Pool(subject_processes)

    results = pool.starmap_async(launch_preprocessing, [launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose, overwrite, denoise, recenter,
                              verbose) for subject in subjects]).get()
else:
    for subject in subjects:
        max_size=0
        print(os.path.join(os.path.join(outpath, "diffusion*"+subject+"*")))
        subjectpath = glob_remote(os.path.join(diffpath, "Vitek_"+subject+"*"), sftp)[0]
        max_file=glob.glob(os.path.join(subjectpath,'*_raw_mean.nii.gz'))
        print(max_file)
        #command = gunniespath + "mouse_diffusion_preprocessing.bash"+ f" {subject} {max_file} {outpath}"
        if os.path.exists(os.path.join(shortcuts_all_folder,f'{proc_subjn + subject}_fa.nii.gz')) and os.path.exists(os.path.join(SAMBA_inputs_folder, f'{proc_subjn + subject}_fa.nii.gz')):
            print(f'already did subject {proc_subjn + subject}')
        else:
            #print('notyet')
            launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose, overwrite, denoise,
                             recenter, verbose)
