import numpy as np
import multiprocessing as mp
import glob
import os, sys
from DTC.diff_handlers.bvec_handler import extractbvals
from DTC.diff_handlers.diff_preprocessing import launch_preprocessing, launch_preprocessing_onlydwi
from DTC.file_manager.file_tools import mkcdir, largerfile
import shutil
from DTC.file_manager.argument_tools import parse_arguments


gunniespath = "/Users/jas/bass/gitfolder/gunnies/"
diffpath = "/Volumes/dusom_civm-atlas/18.abb.11/research/"
outpath = "/Volumes/Data/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/"

SAMBA_inputs_folder = "/Volumes/Data/Badea/Lab/19abb14/"
SAMBA_inputs_folder = None
shortcuts_all_folder = "/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/"
shortcuts_all_folder = None

subjects_folders = glob.glob(os.path.join(diffpath,'diffusion*/'))
subjects = []
for subject_folder in subjects_folders:
    subjects.append(subject_folder.split('diffusion')[1][:6])
#subjects = ['N58309']
#removed_list = ['N58794','N58514','N58305','N58613','N58346','N58344','N58788']
#subjects = ['N60127','N60129','N60131','N60133','N60137','N60139','N60157','N60159','N60161','N60163','N60167','N60169']
#subjects = ['N58634', 'N58635', 'N58636', 'N58650', 'N58649', 'N58651', 'N58653', 'N58654', 'N58217', 'N58613', 'N58221', 'N58219', 'N58222', 'N58223', 'N58229', 'N58230', 'N58232', 'N58231', 'N58216', 'N58218', 'N58224', 'N58225', 'N58215', 'N58214', 'N58633', 'N58228', 'N58226']

subjects = ['N58408', 'N58610', 'N58398', 'N58714', 'N58740', 'N58477', 'N58734', 'N58309', 'N58302', 'N58612', 'N58706', 'N58889', 'N58361', 'N58355', 'N59066', 'N58712', 'N58606', 'N58350', 'N58608', 'N58733', 'N58655', 'N58735', 'N58310', 'N58400', 'N58708', 'N58780', 'N58512', 'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58745', 'N58406', 'N58359', 'N58742', 'N58396', 'N58732', 'N58516', 'N58402']

subjects = ['N57437', 'N57446','N57447','N57449','N57451','N57496','N57498','N57500','N57502','N57504','N57513','N57515','N57518','N57520','N57522','N57546','N57548','N57550','N57552','N57554','N57559','N57580','N57582','N57584','N57587','N57590','N57692','N57694','N57700','N57702','N57709','N58214','N58215','N58216','N58217' ,'N58218','N58219','N58221','N58222','N58223' ,'N58224','N58225','N58226','N58228','N58229','N58230','N58231','N58232','N58610','N58612','N58633','N58634','N58635','N58636','N58649','N58650','N58651','N58653','N58654','N58889','N59066','N59109']

subjects = ['N58889']

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects)

removed_list = ['N58610'] #['N58613']
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

subjects.sort()
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
overwrite=True
ref="coreg"
#btables=["extract","copy","None"]
btables="extract"


#Neither copy nor extract are fully functioning right now, for now the bvec extractor from extractdiffdirs works
#go back to this if ANY issue with bvals/bvecs
#extract is as the name implies here to extract the bvals/bvecs from the files around subject data
#copy takes one known good file for bval and bvec and copies it over to all subjects
subjects_toremove = []

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
            print(f'already did subject {subject}, created {subject}_subjspace_fa.nii.gz')
        elif os.path.exists(os.path.join("/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/", f'{subject}_subjspace_coreg.nii.gz')) and not overwrite:
            print(f'Could not find subject {subject} in main diffusion folder but result was found in SAMBA prep folder')
        elif not overwrite and os.path.exists(os.path.join("/Volumes/Data/Badea/Lab/mouse/APOE_symlink_pool_allfiles/",f'{proc_subjn + subject}_fa.nii.gz')) and os.path.exists(os.path.join(SAMBA_inputs_folder, f'{proc_subjn + subject}_fa.nii.gz')):
            print(f'already did subject {proc_subjn + subject} shortcuts')
        #elif os.path.exists(os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_20APOE01_chass_symmetric3_allAPOE-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_NoNameYet_n32_i5/reg_images/',f'{subject}_rd_to_MDT.nii.gz')) and not overwrite:
        #    print(f'Could not find subject {subject} in main diff folder OR samba init but was in results of SAMBA')
        else:
            print(max_file)
            launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose,
                                 overwrite, denoise, recenter, verbose)
            #launch_preprocessing_onlydwi(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
            #                     shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose,
            #                     overwrite, denoise, recenter, verbose)
