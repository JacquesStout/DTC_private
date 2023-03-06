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
remote = False
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
if remote:
    _, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
else:
    sftp=None

#diffpath = "/mnt/paros_MRI/Vitek_UNC/"
diffpath = '/Users/jas/jacques/Vitek_UNC_mean/'
outpath = "/Volumes/Data/Badea/Lab/mouse/Vitek_series_altvals/diffusion_prep_locale/"

SAMBA_inputs_folder = "/Volumes/Data/Badea/Lab/19abb14/"
SAMBA_inputs_folder = None
shortcuts_all_folder = "/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/"
shortcuts_all_folder = None

subjects_folders = glob_remote(os.path.join(diffpath,'Vitek*/'), sftp)
subjects_folders = [folder for folder in subjects_folders if 'Vitek_' in os.path.basename(folder[:-1])]
subjects = []
for subject_folder in subjects_folders:
    subjects.append(subject_folder.split('UNC_mean/')[1][6:12])
#subjects = ['N58309']
#removed_list = ['N58794','N58514','N58305','N58613','N58346','N58344','N58788']

#subjects = ['02_7_17']
subjects = ['01_7_8', '02_7_17', '03_7_9', '04_7_16', '05_7_25', '06_8_8', '08_7_30', '09_7_23', '10_7_31', '11_8_13', '12_8_14', '13_8_6', '14_8_15', '15_8_16', '16_8_21', '17_8_22', '18_8_25']
#subjects = ['02_7_17']
subjects = ['01_7_8', '02_7_17', '03_7_9', '04_7_16', '05_7_25', '06_8_8', '08_7_30', '09_7_23', '10_7_31']
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
masking = "None"
masking = "median"
#masking = 'premade'
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
    for old_subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        subject = old_subject.replace('-','_')
        writeformat="tab"
        writeformat="dsi"
        subjectpath = glob.glob(os.path.join(os.path.join(diffpath, "diffusion*"+subject+"*")))[0]
        subject_outpath = os.path.join(outpath, 'diffusion_prep_' + proc_subjn + subject)
        mkcdir(subject_outpath)
        fbvals, fbvecs = extractbvals(subjectpath, subject, outpath=subject_outpath, writeformat=writeformat, overwrite=overwrite) #extractbvals_research
        #fbvals, fbvecs = rewrite_subject_bvalues(diffpath, subject, outpath=outpath, writeformat=writeformat, overwrite=overwrite)
elif btables=="copy":
    for old_subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        subject = old_subject.replace('-','_')
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
overwrite=False

if subject_processes>1:
    if function_processes>1:
        pool = MyPool(subject_processes)
    else:
        pool = mp.Pool(subject_processes)
    results = pool.starmap_async(launch_preprocessing, [launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose, overwrite, denoise, recenter,
                             recenter, verbose) for subject in subjects]).get()
else:
    for old_subject in subjects:
        max_size=0
        subject = old_subject.replace('-','_')
        subjectpath = glob_remote(os.path.join(diffpath, "Vitek_"+old_subject+"*"), sftp)[0]
        subject_outpath = os.path.join(outpath, 'diffusion_prep_' + proc_subjn + subject)
        max_file=largerfile(subjectpath, sftp)
        toremove = []
        if sftp is not None:
            temp_path = make_temppath(max_file,to_fix=True)
            if not os.path.exists(temp_path):
                sftp.get(max_file, temp_path)
            if temp_path!=max_file:
                toremove.append(temp_path)
            new_temp_path = os.path.join(subject_outpath,os.path.basename(temp_path).replace('.nii.gz',
                                                                                             '_raw_mean.nii.gz'))
            if not os.path.exists(new_temp_path):
                average_4dslices(temp_path, new_temp_path, split=2)
            max_file = new_temp_path
        if os.path.exists(os.path.join(subject_outpath, f'{subject}_subjspace_fa.nii.gz')) and not overwrite:
            print(f'already did subject {subject}, created {subject}_subjspace_fa.nii.gz')
        elif os.path.exists(os.path.join("/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/", f'{subject}_subjspace_coreg.nii.gz')) and not overwrite:
            print(f'Could not find subject {subject} in main diffusion folder but result was found in SAMBA prep folder')
        elif os.path.exists(os.path.join("/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/",f'{proc_subjn + subject}_fa.nii.gz')) and os.path.exists(os.path.join(SAMBA_inputs_folder, f'{proc_subjn + subject}_fa.nii.gz')):
            print(f'already did subject {proc_subjn + subject} shortcuts')
        #elif os.path.exists(os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_20APOE01_chass_symmetric3_allAPOE-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_NoNameYet_n32_i5/reg_images/',f'{subject}_rd_to_MDT.nii.gz')) and not overwrite:
        #    print(f'Could not find subject {subject} in main diff folder OR samba init but was in results of SAMBA')
        else:
            print(max_file)
            launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose,
                                 overwrite, denoise, recenter, verbose)

        for remove in toremove:
            os.remove(remove)
