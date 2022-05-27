
import os, glob
from file_tools import mkcdir, largerfile, getfromfile
from computer_nav import get_mainpaths

"""
subjects = ['N57437','N57440','N57442','N57446','N57447','N57449','N57451','N57496','N57498','N57500','N57502','N57504',
            'N57513','N57515','N57518','N57520','N57522','N57546','N57548','N57550','N57552','N57554','N57559','N57580',
            'N57582','N57584','N57587','N57590','N57692','N57694','N57700','N57702','N57709']
removed_list = ['N57440']
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)
input_folder = '/Volumes/dusom_civm-atlas/19.abb.14/research/'
outpath = '/mnt/paros_MRI/jacques/APOE/DWI_allsubj/'

username, passwd = getfromfile('/Users/jas/samos_connect.rtf')
_, _, _, sftp = get_mainpaths(True,project = 'APOE', username=username,password=passwd)


for subject in subjects:
    subjectpath = glob.glob(os.path.join(os.path.join(input_folder, "diffusion*" + subject + "*")))[0]
    max_file = largerfile(subjectpath)
    subj_outpath = os.path.join(outpath, os.path.basename(max_file))
    print(max_file, subj_outpath)
    sftp.put(max_file,subj_outpath)

"""

subjects = ["N58214", "N58215",
     "N58216", "N58217", "N58218", "N58219", "N58221", "N58222", "N58223", "N58224",
                "N58225", "N58226", "N58228",
                "N58229", "N58230", "N58231", "N58232", "N58633", "N58634", "N58635", "N58636", "N58649", "N58650",
                "N58651", "N58653", "N58654",
                'N58408', 'N58714', 'N58740', 'N58477', 'N58734', 'N58309', 'N58792', 'N58302',
                'N58784', 'N58706', 'N58361', 'N58355', 'N58712', 'N58790', 'N58606', 'N58350', 'N58608',
                'N58779', 'N58500', 'N58604', 'N58749', 'N58510', 'N58394', 'N58346', 'N58344', 'N58788', 'N58305',
                'N58514', 'N58794', 'N58733', 'N58655', 'N58735', 'N58310', 'N58400', 'N58708', 'N58780', 'N58512',
                'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58745', 'N58406', 'N58359', 'N58742', 'N58396',
                'N58613', 'N58732', 'N58516', 'N58402']

subjects = ['N58396','N58794','N58813','N58815','N58819','N58821','N58829','N58831','N58851','N58853','N58855','N58857',
            'N58859','N58861','N58877','N58879','N58881','N58883','N58885','N58887','N58889','N58906','N58909','N58913',
            'N58917','N58919','N58935','N58941','N58952','N58995','N58997','N58999','N59003','N59010','N59022','N59026',
            'N59033','N59035','N59039','N59041','N59065','N59066','N59072','N59076','N59078','N59080','N59097','N59099',
            'N59109','N59116','N59118','N59120']
subjects = ['N58398']
input_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_20APOE01_chass_symmetric3_allAPOE-work/preprocess/'
outpath = '/mnt/paros_MRI/jacques/APOE/DWI_allsubj/'

username, passwd = getfromfile('/Users/jas/samos_connect.rtf')
_, _, _, sftp = get_mainpaths(True,project = 'APOE', username=username,password=passwd)

for subject in subjects:
    subjectpath = os.path.join(input_folder, f'{subject}_dwi_masked.nii.gz')
    subj_outpath = os.path.join(outpath, f'{subject}_dwi_SAMBA_recentered.nii.gz')
    print(subjectpath, subj_outpath)
    if not os.path.exists(subjectpath):
        print(f'could not find {subjectpath}')
        raise Exception
    sftp.put(subjectpath,subj_outpath)