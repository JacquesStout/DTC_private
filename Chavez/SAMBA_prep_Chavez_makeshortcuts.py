import numpy as np
import multiprocessing as mp
#from DTC.file_manager.Daemonprocess import MyPool
import glob
import os, sys
from DTC.diff_handlers.bvec_handler import writebfiles, extractbvals, extractbvals_research, rewrite_subject_bvalues, fix_bvals_bvecs
from time import time
import shutil
from DTC.file_manager.file_tools import mkcdir, largerfile
import shutil
from DTC.file_manager.argument_tools import parse_arguments
from DTC.diff_handlers.diff_preprocessing import launch_preprocessing


gunniespath = "~/gunnies/"
mainpath="/mnt/munin6/Badea/Lab/"
diffpath = "/mnt/munin6/Badea/Lab/Chavez_init/"
outpath = "/mnt/munin6/Badea/Lab/mouse/Chavez_series/diffusion_prep_locale/"

SAMBA_inputs_folder = "/mnt/munin6/Badea/Lab/Chavez_prep/"
shortcuts_all_folder = "/mnt/munin6/Badea/Lab/mouse/Chavez_symlink_pool_allfiles/"

if SAMBA_inputs_folder is not None:
    mkcdir(SAMBA_inputs_folder)
if shortcuts_all_folder is not None:
    mkcdir(shortcuts_all_folder)
mkcdir(outpath)
subjects = ['apoe_3_6_CtrlA_male', 'MHI_335_CtrA_male', 'apoe_4_4_CtrlB_female', 'MHI_326_C_male', 'MHI_334_CtrC_female', 'MHI_335_CtrB_female', 'MHI_326_D_female', 'apoe_4_2_A_male', 'apoe_3_4_CtrA_female', 'apoe_4_7_B_male', 'MHI_326_CtrB_male', 'MHI_334_D_female', 'apoe_4_7_A_female', 'apoe_4_2_B_female', 'apoe_3_6_B_male', 'MHI_334_CtrA_male', 'apoe_4_5_CtrA_female', 'apoe_4_4_CtrlD_male', 'apoe_4_4_B_female', 'MHI_326_CtrA_female', 'apoe_3_4_A_female']

"""
subjects_folders = glob.glob(os.path.join(diffpath,'diffusion*/'))
subjects = []
for subject_folder in subjects_folders:
    subjects.append(subject_folder.split('diffusion')[1][:6])
"""
removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

print(subjects)

# subjects = ['N58610', 'N58612', 'N58613']

subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv, subjects)


# subject 'N58610' retired, weird? to investigate
proc_subjn = ""
denoise = "None"
#denoise = "mpca"
recenter = 0
proc_name = "diffusion_prep_" + proc_subjn
cleanup = True
masking = "median_5"
makebtables = False
gettranspose = False
ref = "coreg"
copybtables = True
verbose = True
transpose = None
overwrite = False

proc_name = "diffusion_prep_"

max_processors = 20
if mp.cpu_count() < max_processors:
    max_processors = mp.cpu_count()

# accepted values are "small" for one in ten streamlines, "all or "large" for all streamlines,
# "none" or None variable for neither and "both" for both of them
nominal_bval = 2401
verbose = True
function_processes = np.int(max_processors / subject_processes)
results = []
if subject_processes > 1:
    if function_processes > 1:
        pool = MyPool(subject_processes)
    else:
        pool = mp.Pool(subject_processes)
    results = pool.starmap_async(launch_preprocessing, [
        launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                             shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose, overwrite,
                             denoise, recenter,
                             recenter, verbose) for subject in subjects]).get()
else:
    for subject in subjects:
        max_size = 0
        subjectpath = glob.glob(os.path.join(os.path.join(diffpath,subject + "*")))[0]
        subject_outpath = os.path.join(outpath, 'diffusion_prep_' + proc_subjn + subject)
        max_file = largerfile(subjectpath,identifier=".nii")
        if os.path.exists(os.path.join(shortcuts_all_folder,f'{proc_subjn + subject}_fa.nii.gz')) and os.path.exists(os.path.join(SAMBA_inputs_folder, f'{proc_subjn + subject}_fa.nii.gz')):
            print(f'already did subject {proc_subjn + subject}')
        else:
            launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose,
                                 overwrite, denoise, recenter, verbose)
        # results.append(launch_preprocessing(subject, max_file, outpath))


