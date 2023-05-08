import numpy as np
import multiprocessing as mp
import glob
import os, sys
from DTC.diff_handlers.bvec_handler import extractbvals
from DTC.diff_handlers.diff_preprocessing import launch_preprocessing
from DTC.file_manager.file_tools import mkcdir, largerfile
import shutil
from DTC.file_manager.argument_tools import parse_arguments

#gunniespath = "/mnt/clustertmp/common/rja20_dev/gunnies/"
#gunniespath = "/Users/alex/bass/gitfolder/wuconnectomes/gunnies/"
#diffpath = "/Volumes/dusom_civm-atlas/20.abb.15/research/"
#outpath = "/Volumes/Data/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/"

gunniespath = ""
outpath = "/mnt/munin6/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/"
#bonusshortcutfolder = "/Volumes/Data/Badea/Lab/jacques/APOE_series/19abb14/"
#bonusshortcutfolder = "/mnt/munin6/Badea/Lab/APOE_symlink_pool/"
#bonusshortcutfolder = None
SAMBA_inputs_folder = "/mnt/munin6/Badea/Lab/19abb14/"
shortcuts_all_folder = "/mnt/munin6/Badea/Lab/mouse/APOE_symlink_pool_allfiles/"

mkcdir([SAMBA_inputs_folder, shortcuts_all_folder])


subjects = ["N58214","N58215","N58216","N58217","N58218","N58219","N58221","N58222","N58223","N58224","N58225","N58226","N58228",
            "N58229","N58230","N58231","N58232","N58633","N58634","N58635","N58636","N58649","N58650","N58651","N58653","N58654",
            'N58408', 'N58398', 'N58714', 'N58740', 'N58477', 'N58734', 'N58309', 'N58792', 'N58302',
            'N58784', 'N58706', 'N58361', 'N58355', 'N58712', 'N58790', 'N58606', 'N58350', 'N58608',
            'N58779', 'N58500', 'N58604', 'N58749', 'N58510', 'N58394', 'N58346', 'N58344', 'N58788', 'N58305',
            'N58514', 'N58794', 'N58733', 'N58655', 'N58735', 'N58310', 'N58400', 'N58708', 'N58780', 'N58512',
            'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58745', 'N58406', 'N58359', 'N58742', 'N58396',
            'N58613', 'N58732', 'N58516', 'N58813', 'N58402']
#58610, N58612 removed
subjects = ['N58408', 'N58398', 'N58935', 'N58714', 'N58740', 'N58477', 'N59003', 'N58734', 'N58309', 'N58792', 'N58819', 'N58302', 'N58909', 'N58784', 'N58919', 'N58706', 'N58889', 'N58361', 'N58355', 'N58712', 'N58790', 'N59010', 'N58859', 'N58917', 'N58606', 'N58815', 'N58997', 'N58350', 'N58999', 'N58881', 'N58608', 'N58853', 'N58779', 'N58995', 'N58500', 'N58604', 'N58749', 'N58877', 'N58883', 'N58510', 'N58885', 'N58906', 'N58394', 'N58821', 'N58855', 'N58346', 'N58861', 'N58344', 'N58857', 'N58788', 'N58305', 'N58514', 'N58851', 'N58794', 'N58733', 'N58655', 'N58887', 'N58735', 'N58310', 'N58879', 'N58400', 'N58708', 'N58780', 'N58512', 'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58829', 'N58913', 'N58745', 'N58831', 'N58406', 'N58359', 'N58742', 'N58396', 'N58941', 'N58516', 'N58813', 'N58402']
subjects = ['N60167', 'N60133', 'N60200', 'N60131', 'N60139', 'N60163', 'N60159', 'N60157', 'N60127', 'N60161',
            'N60169', 'N60137', 'N60129']
removed_list = []

subjects_fpath = glob.glob("/mnt/munin6/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/diffusion_prep*")
subjects = []
for subject in subjects_fpath:
    subject_fname = os.path.basename(subject)
    subjects.append(subject_fname.split('diffusion_prep_')[1])

subjects = ['N58302', 'N58303', 'N58305', 'N58309', 'N58310', 'N58344', 'N58346', 'N58350', 'N58355', 'N58359', 'N58361', 'N58394', 'N58396', 'N58398', 'N58400', 'N58402', 'N58404', 'N58406', 'N58408', 'N58477', 'N58500', 'N58510', 'N58512', 'N58514', 'N58516', 'N58604', 'N58606', 'N58608', 'N58610', 'N58611', 'N58612', 'N58655', 'N58706', 'N58708', 'N58712', 'N58714', 'N58732', 'N58733', 'N58734', 'N58735', 'N58740', 'N58742', 'N58745', 'N58747', 'N58749', 'N58751', 'N58779', 'N58780', 'N58784', 'N58788', 'N58790', 'N58792', 'N58794', 'N58813', 'N58815', 'N58819', 'N58821', 'N58829', 'N58831', 'N58851', 'N58853', 'N58855', 'N58857', 'N58859', 'N58861', 'N58877', 'N58879', 'N58881', 'N58883', 'N58885', 'N58887', 'N58889', 'N58906', 'N58909', 'N58913', 'N58915', 'N58917', 'N58919', 'N58935', 'N58941', 'N58946', 'N58948', 'N58952', 'N58954', 'N58995', 'N58997', 'N58999', 'N59003', 'N59005', 'N59010', 'N59022', 'N59026', 'N59033', 'N59035', 'N59039', 'N59041', 'N59065', 'N59066', 'N59072', 'N59076', 'N59078', 'N59080', 'N59097', 'N59099', 'N59109', 'N59116', 'N59118', 'N59120', 'N59136', 'N59140', 'N59141', 'N60056', 'N60058', 'N60060', 'N60062', 'N60064', 'N60068', 'N60070', 'N60072', 'N60088', 'N60092', 'N60093', 'N60095', 'N60097', 'N60101', 'N60103']
#subjects = ['N60103']
#subjects = ['N58302','N58303', 'N58305', 'N58309']

subjects = ['N58610','N60103','N60062','N60101','N60056','N60088','N60064','N60070','N60093','N60097','N60095','N60068','N60072','N60058','N60092','N60060']
subjects = ['N58302', 'N58303', 'N58305', 'N58309', 'N58310', 'N58344', 'N58346', 'N58350', 'N58355', 'N58359', 'N58361', 'N58394', 'N58396', 'N58398', 'N58400', 'N58402', 'N58404', 'N58406', 'N58408', 'N58477', 'N58500', 'N58510', 'N58512', 'N58514', 'N58516', 'N58604', 'N58606', 'N58608', 'N58610', 'N58611', 'N58612', 'N58613', 'N58653', 'N58654', 'N58655', 'N58706', 'N58708', 'N58712', 'N58714', 'N58732', 'N58733', 'N58734', 'N58735', 'N58740', 'N58742', 'N58745', 'N58747', 'N58749', 'N58751', 'N58779', 'N58780', 'N58784', 'N58788', 'N58790', 'N58792', 'N58794', 'N58813', 'N58815', 'N58819', 'N58821', 'N58829', 'N58831', 'N58851', 'N58853', 'N58855', 'N58857', 'N58859', 'N58861', 'N58877', 'N58879', 'N58881', 'N58883', 'N58885', 'N58887', 'N58889', 'N58906', 'N58909', 'N58913', 'N58915', 'N58917', 'N58919', 'N58935', 'N58941', 'N58946', 'N58948', 'N58952', 'N58954', 'N58995', 'N58997', 'N58999', 'N59003', 'N59005', 'N59010', 'N59022', 'N59026', 'N59033', 'N59035', 'N59039', 'N59041', 'N59065', 'N59066', 'N59072', 'N59076', 'N59078', 'N59080', 'N59097', 'N59099', 'N59109', 'N59116', 'N59118', 'N59120', 'N59136', 'N59140', 'N59141', 'N60056', 'N60058', 'N60060', 'N60062', 'N60064', 'N60068', 'N60070', 'N60072', 'N60088', 'N60092', 'N60093', 'N60095', 'N60097', 'N60101', 'N60103']
subjects = ['N58302', 'N58303', 'N58305', 'N58309', 'N58310', 'N58344', 'N58346', 'N58350', 'N58355', 'N58359', 'N58361', 'N58394', 'N58396', 'N58398', 'N58400', 'N58402', 'N58404', 'N58406', 'N58408', 'N58477', 'N58500', 'N58510', 'N58512', 'N58514', 'N58516', 'N58604', 'N58606', 'N58608', 'N58610', 'N58611', 'N58612', 'N58655', 'N58706', 'N58708', 'N58712', 'N58714', 'N58732', 'N58733', 'N58734', 'N58735', 'N58740', 'N58742', 'N58745', 'N58747', 'N58749', 'N58751', 'N58779', 'N58780', 'N58784', 'N58788', 'N58790', 'N58792', 'N58794', 'N58813', 'N58815', 'N58819', 'N58821', 'N58829', 'N58831', 'N58851', 'N58853', 'N58855', 'N58857', 'N58859', 'N58861', 'N58877', 'N58879', 'N58881', 'N58883', 'N58885', 'N58887', 'N58889', 'N58906', 'N58909', 'N58913', 'N58915', 'N58917', 'N58919', 'N58935', 'N58941', 'N58946', 'N58948', 'N58952', 'N58954', 'N58995', 'N58997', 'N58999', 'N59003', 'N59005', 'N59010', 'N59022', 'N59026', 'N59033', 'N59035', 'N59039', 'N59041', 'N59065', 'N59066', 'N59072', 'N59076', 'N59078', 'N59080', 'N59097', 'N59099', 'N59109', 'N59116', 'N59118', 'N59120', 'N59136', 'N59140', 'N59141', 'N60056', 'N60058', 'N60060', 'N60062', 'N60064', 'N60068', 'N60070', 'N60072', 'N60088', 'N60092', 'N60093', 'N60095', 'N60097', 'N60101', 'N60103', 'N60188', 'N60190', 'N60192', 'N60194', 'N60219', 'N60221', 'N60223', 'N60225', 'N60229', 'N60231']
#subjects = ['N60188', 'N60190', 'N60192', 'N60194', 'N60198', 'N60219', 'N60221', 'N60223', 'N60225', 'N60229', 'N60231']
#subjects = ['N58610', 'N58612']
subjects = ['N60167', 'N60133', 'N60200', 'N60131', 'N60139', 'N60163', 'N60159', 'N60157', 'N60127', 'N60161',
            'N60169', 'N60137', 'N60129']

subjects = ['N57437', 'N57442','N57446','N57447','N57449','N57451','N57496','N57498','N57500','N57502','N57504','N57513','N57515','N57518',
'N57520','N57522','N57546','N57548','N57550','N57552','N57554','N57559','N57580','N57582','N57584','N57587','N57590','N57692',
'N57694','N57700','N57702','N57709','N58214','N58215','N58216','N58217' ,'N58218','N58219','N58221','N58222','N58223' ,'N58224',
'N58225','N58226','N58228','N58229','N58230','N58231','N58232','N58610','N58612','N58633','N58634','N58635','N58636','N58649',
'N58650','N58651','N58653','N58654','N58889','N59066','N59109']
subjects = ['N57442']
#subjects = ['N58225', 'N58232', 'N58215', 'N58216', 'N58633', 'N58636', 'N58653', 'N58224', 'N58214', 'N58651', 'N58228', 'N58650', 'N58221', 'N58219', 'N58649', 'N58226', 'N58229', 'N58218', 'N58230', 'N58223', 'N58222', 'N58634', 'N58231', 'N58217', 'N58654', 'N58635']
removed_list = ['N58610', 'N58612']
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)
"""
subjects = ['N58952', 'N58995', 'N58997', 'N58999', 'N59003', 'N59010', 'N59022', 'N59026', 'N59033', 'N59035',
            'N59039', 'N59041', 'N59065', 'N59066', 'N59072', 'N59076', 'N59078', 'N59080', 'N59097', 'N59099',
            'N59109', 'N59116', 'N59118', 'N59120']

removed_list = []
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)
"""

atlas = "/mnt/munin6/Badea/Lab/atlases/chass_symmetric3/chass_symmetric3_DWI.nii.gz"

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

max_processors = 1
if mp.cpu_count() < max_processors:
    max_processors = mp.cpu_count()
subject_processes = np.size(subjects)
if max_processors < subject_processes:
    subject_processes = max_processors
# accepted values are "small" for one in ten streamlines, "all or "large" for all streamlines,
# "none" or None variable for neither and "both" for both of them
nominal_bval=4000
verbose=True
function_processes = np.int(max_processors/subject_processes)
results=[]
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
        subjectpath = glob.glob(os.path.join(os.path.join(outpath, "diffusion*"+subject+"*")))[0]
        max_file=largerfile(subjectpath)
        max_file= os.path.join(subjectpath, "nii4D_"+subject+".nii.gz")
        print(max_file)
        #command = gunniespath + "mouse_diffusion_preprocessing.bash"+ f" {subject} {max_file} {outpath}"
        if os.path.exists(os.path.join(shortcuts_all_folder,f'{proc_subjn + subject}_fa.nii.gz')) and os.path.exists(os.path.join(SAMBA_inputs_folder, f'{proc_subjn + subject}_fa.nii.gz')):
            print(f'already did subject {proc_subjn + subject}')
        else:
            print(f'notyet for subject {subject}')
            shortcuts_subject_fa = os.path.join(shortcuts_all_folder,f'{proc_subjn + subject}_fa.nii.gz')
            SAMBA_subject_fa = os.path.join(shortcuts_all_folder,f'{proc_subjn + subject}_fa.nii.gz')
            print(os.path.exists(shortcuts_subject_fa),os.path.exists(SAMBA_subject_fa),shortcuts_subject_fa,SAMBA_subject_fa)
            
            launch_preprocessing(proc_subjn + subject, max_file, outpath, cleanup, nominal_bval, SAMBA_inputs_folder,
                                 shortcuts_all_folder, gunniespath, function_processes, masking, ref, transpose, overwrite, denoise,
                             recenter, verbose)
            
            
