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


gunniespath = ""

project = 'Vitek'

outpath = "/mnt/munin6/Badea/Lab/mouse/Vitek_series_altvals/diffusion_prep_locale/"

SAMBA_inputs_folder = "/mnt/munin6/Badea/Lab/mouse/Vitek_prep_altvals/"
shortcuts_all_folder = "/mnt/munin6/Badea/Lab/mouse/Vitek_prep_allfiles_altvals/"
mkcdir([SAMBA_inputs_folder, shortcuts_all_folder])
subjects_folders = glob.glob(os.path.join(outpath,'*/'))
#print(subjects_folders)
#subjects = ['01_7_8', '02_7_17', '03_7_9', '04_7_16', '05_7_25', '06_8_8', '08_7_30', '09_7_23', '10_7_31', '11_8_13', '12_8_14', '13_8_6', '14_8_15', '15_8_16', '16_8_21', '17_8_22', '18_8_25']

#for subject_folder in subjects_folders:
#    subjects.append(subject_folder.split('diffusion_prep_locale/')[1][15:22])


subjects = ['01_7_8', '02_7_17', '03_7_9', '04_7_16', '05_7_25', '06_8_8', '08_7_30', '09_7_23', '10_7_31', '11_8_13', '12_8_14', '13_8_6', '14_8_15', '15_8_16', '16_8_21', '17_8_22', '18_8_25']

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects)


removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)


subjects = subjects[firstsubj:lastsubj]
subjects.sort()
print(subjects)



proc_subjn="V"
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
        subject_outpath = glob.glob(os.path.join(os.path.join(outpath, "diffusion*"+subject+"*")))[0]
        print(subject_outpath)
        print(os.path.join(subject_outpath,'*_raw_mean.nii.gz'))
        max_file=glob.glob(os.path.join(subject_outpath,'*_raw*.nii.gz'))[0]
        print(max_file)
        #command = gunniespath + "mouse_diffusion_preprocessing.bash"+ f" {subject} {max_file} {outpath}"
        if os.path.exists(os.path.join(shortcuts_all_folder,f'{proc_subjn + subject}_fa.nii.gz')) and os.path.exists(os.path.join(SAMBA_inputs_folder, f'{proc_subjn + subject}_fa.nii.gz')):
            print(f'already did subject {proc_subjn + subject}')
        else:
            #print('notyet')
            launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose, overwrite, denoise,
                             recenter, verbose)

