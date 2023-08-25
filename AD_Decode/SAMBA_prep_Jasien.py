import numpy as np
#from file_manager.Daemonprocess import MyPool
import multiprocessing as mp
import glob
import os
import sys
from DTC.diff_handlers.bvec_handler import extractbvals, rewrite_subject_bvalues, fix_bvals_bvecs, extractbvals_fromheader
from DTC.diff_handlers.diff_preprocessing import launch_preprocessing
from DTC.file_manager.file_tools import mkcdir, largerfile
from DTC.nifti_handlers.transform_handler import get_transpose
import shutil
from DTC.file_manager.argument_tools import parse_arguments

munin=False
if munin:
    gunniespath = "~/wuconnectomes/gunnies"
    mainpath = "/mnt/munin6/Jasien/ADSB.01/"
    outpath = "/mnt/munin6/Badea/Lab/human/Jasien/diffusion_prep_locale/"
    SAMBA_inputs_folder = "/mnt/munin6/Badea/Lab/mouse/ADDeccode_symlink_pool/"
    shortcuts_all_folder = "/mnt/munin6/Badea/Lab/human/ADDeccode_symlink_pool_allfiles/"
else:
    gunniespath = "/Users/alex/bass/gitfolder/wuconnectomes/gunnies/"
    mainpath="/Volumes/Data/Jasien/ADSB.01/"
    outpath = "/Volumes/Data/Badea/Lab/human/Jasien/diffusion_prep_locale/"
    #bonusshortcutfolder = "/Volumes/Data/Badea/Lab/mouse/ADDeccode_symlink_pool/"
    SAMBA_inputs_folder = None
    shortcuts_all_folder = None
gunniespath = "/Users/jas/bass/gitfolder/gunnies/"

diffpath = os.path.join(mainpath, "Data","Anat")

mkcdir(outpath)

subjects = ['04086', '04129']
subjects = ['04300', '01257', '01277', '04472', '01402']

removed_list = []

for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)
#subjects = ['02842']

#subjects = ["03010", "03033", "03045"]

#subjects = ["02871", "02877", "02898", "02926", "02938", "02939", "02954", "02967", "02987", "02987", "03010", "03017", "03028", "03033", "03034", "03045", "03048"]
#02745 was not fully done, discount
#02771 has 21 images in 4D space even though there should be 23?
#"02812", 02871 is a strange subject, to investigate
#02842, 03028 has apparently a 92 stack ? to investigate

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv,subjects)

subjects = subjects[firstsubj:lastsubj]
print(subjects)

proc_subjn="J"
proc_name ="diffusion_prep_"+proc_subjn
denoise = "mpca"
masking = "bet"
overwrite=False
cleanup = True
atlas = None
gettranspose=False
verbose=True
nominal_bval=2000
if gettranspose:
    transpose = get_transpose(atlas)
ref = "md"
recenter=0

transpose=None

#btables=["extract","copy","None"]
btables="extract"
#Neither copy nor extract are fully functioning right now, for now the bvec extractor from extractdiffdirs works
#go back to this if ANY issue with bvals/bvecs
#extract is as the name implies here to extract the bvals/bvecs from the files around subject data
#copy takes one known good file for bval and bvec and copies it over to all subjects
if btables=="extract":
    for subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        """
        outpathsubj = outpath + "_" + subject
        writeformat="tab"
        writeformat="dsi"
        fbvals, fbvecs = extractbvals(diffpath, subject, outpath=outpath, writeformat=writeformat, overwrite=True)
        """

        writeformat="tab"
        writeformat="dsi"
        subjectpath = glob.glob(os.path.join(os.path.join(diffpath, "*"+subject+"*")))[0]
        bxh_file=largerfile(os.path.join(subjectpath,'*.nii.gz')).replace('.nii.gz','.bxh')
        subject_outpath = os.path.join(outpath, 'diffusion_prep_' + proc_subjn + subject)
        mkcdir(subject_outpath)
        #fbvals, fbvecs = extractbvals(subjectpath, subject, outpath=subject_outpath, writeformat=writeformat, overwrite=overwrite)
        overwrite=True
        if not os.path.exists(os.path.join(subject_outpath,proc_subjn + subject+"_bvals.txt")) or overwrite:
            fbvals, fbvecs, _, _, _, _ = extractbvals_fromheader(bxh_file,
                                                                fileoutpath=os.path.join(subject_outpath,proc_subjn + subject),
                                                                writeformat=writeformat,
                                                                save="all")
            fix_bvals_bvecs(fbvals, fbvecs, writeformat = 'dsi', writeover=True)

        #fbvals, fbvecs = rewrite_subject_bvalues(diffpath, subject, outpath=outpath, writeformat=writeformat, overwrite=overwrite)
elif btables=="copy":
    for subject in subjects:
        #outpathsubj = "/Volumes/dusom_dibs_ad_decode/all_staff/APOE_temp/diffusion_prep_58214/"
        outpathsubj = os.path.join(outpath,proc_name+subject)
        outpathbval= os.path.join(outpathsubj, proc_subjn + subject+"_bvals.txt")
        outpathbvec= os.path.join(outpathsubj, proc_subjn + subject+"_bvecs.txt")
        if not os.path.exists(outpathbval) or not os.path.exists(outpathbvec) or overwrite:
            mkcdir(outpathsubj)
            writeformat="tab"
            writeformat="dsi"
            bvals = glob.glob(os.path.join(outpath, "*bvals*.txt"))
            bvecs = glob.glob(os.path.join(outpath, "*bvec*.txt"))
            if np.size(bvals)>0 and np.size(bvecs)>0:
                shutil.copy(bvals[0], outpathbval)
                shutil.copy(bvecs[0], outpathbvec)
#quickfix was here

# accepted values are "small" for one in ten streamlines, "all or "large" for all streamlines,
# "none" or None variable for neither and "both" for both of them
overwrite = False
results=[]
if subject_processes>1:
    if function_processes>1:
        pool = MyPool(subject_processes)
    else:
        pool = mp.Pool(subject_processes)

    results = pool.starmap_async(launch_preprocessing, [(proc_subjn+subject,
                                                         largerfile(glob.glob(os.path.join(os.path.join(diffpath, "*" + subject + "*")))[0]),
                                                         outpath, cleanup, nominal_bval, SAMBA_inputs_folder, shortcuts_all_folder,
                                                         gunniespath, function_processes, masking, ref, transpose,
                                                         overwrite, denoise, recenter, verbose)
                                                        for subject in subjects]).get()
else:
    for subject in subjects:
        max_size=0
        overwrite=True
        subjectpath = glob.glob(os.path.join(os.path.join(diffpath, "*" + subject + "*")))[0]
        max_file=largerfile(subjectpath)
        #command = gunniespath + "mouse_diffusion_preprocessing.bash"+ f" {subject} {max_file} {outpath}"
        #max_file="/Volumes/Data/Badea/ADdecode.01/Data/Anat/20210522_02842/bia6_02842_003.nii.gz"
        #launch_preprocessing(subject, max_file, outpath, nominal_bval=1000, shortcutpath=shortcutpath, SAMBA_inputfolder = SAMBA_inputfolder, gunniespath="/Users/alex/bass/gitfolder/gunnies/")
        #max_file = '/Volumes/Data/Badea/ADdecode.01/Data/Anat/20210522_02842/bia6_02842_003.nii.gz'
        subject_f = proc_subjn + subject
        if os.path.exists(os.path.join(outpath, proc_name + subject, f'{subject_f}_subjspace_fa.nii.gz')) and not overwrite:
            print(f'already did subject {subject_f}')
        elif os.path.exists(os.path.join('/Volumes/Badea/Lab/mouse/ADDeccode_symlink_pool/', f'{subject_f}_subjspace_coreg.nii.gz')) and not overwrite:
            print(f'Could not find subject {subject_f} in main diffusion folder but result was found in SAMBA prep folder')
        #elif os.path.exists(os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images/',f'{subject_f}_rd_to_MDT.nii.gz')):
        #    print(f'Could not find subject {subject_f} in main diff folder OR samba init but was in results of SAMBA')
        else:
            launch_preprocessing(proc_subjn+subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder, shortcuts_all_folder,
             gunniespath, function_processes, masking, ref, transpose, overwrite, denoise, recenter, verbose)
        #results.append(launch_preprocessing(subject, max_file, outpath))
