import os
from DTC.tract_manager.tract_handler import reducetractnumber
from DTC.file_manager.file_tools import mkcdir
import random
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile

#path = '/Users/jas/jacques/APOE_subj_to_MDT'
#trk_folder = os.path.join(path,'TRK')
#new_trk_folder = os.path.join(path,'TRK')
#trk_folder = os.path.join(path,'TRK_MPCA_fixed')
#new_trk_folder = os.path.join(path,'TRK_MPCA_fixed_100')
#trk_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/AMD/TRK_rigidaff'
#new_trk_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/AMD/TRK_rigidaff_100'

trk_folder = '/mnt/paros_DB/Projects/AD_Decode/Analysis/TRK_MDT'
new_trk_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT_10'

remote = True

remote=True
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

inpath, _, _, sftp = get_mainpaths(remote,project = 'AD_Decode', username=username,password=passwd)

mkcdir(new_trk_folder)

orig_ratio =1
ratio = 10
stepsize = 2
method= 'decimate'
"""
filelist = os.listdir(trk_folder)
filelist = sorted(filelist)
random.shuffle(filelist)
"""
filelist = glob_remote(os.path.join(trk_folder,'*trk'),sftp)
#filelist.reverse()

test=False
overwrite=False
orig_identifier = get_str_identifier(stepsize, orig_ratio, '', type='mrtrix')
new_identifier = get_str_identifier(stepsize, ratio, '', type='mrtrix')

"""
for filepath in filelist:
    filename, f_ext = os.path.splitext(filepath)
    if f_ext == '.trk' and f'ratio_{str(ratio)}' not in filename:
        newfilename = filename.replace(orig_identifier,new_identifier)
        newfilepath = os.path.join(new_trk_folder, newfilename +f_ext)
        if not os.path.exists(newfilepath):
            print(f'Downsampling from {os.path.join(trk_folder,filepath)} to {newfilepath} with ratio {str(ratio)}')
            reducetractnumber(os.path.join(trk_folder,filepath), newfilepath, getdata=False, ratio=ratio,
                              return_affine=False, verbose=False, sftp=sftp)
            print(f'succesfully created {newfilepath}')
        else:
            print(f'already created {newfilepath}')
"""

try:
    BD = os.environ['BIGGUS_DISKUS']
# os.environ['GIT_PAGER']
except KeyError:
    print('BD not found locally')
    BD = '/mnt/munin2/Badea/Lab/mouse'
    # BD ='/Volumes/Data/Badea/Lab/mouse'
else:
    print("BD is found locally.")

job_descrp = 'downsampler'

GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'

job_descrp = "BuSA"
sbatch_folder_path = os.path.join(BD, job_descrp + '_sbatch/')
mkcdir(sbatch_folder_path)

for filepath in filelist:
    filename, f_ext = os.path.splitext(filepath)
    subj = filename.split('_')[0]
    if f_ext == '.trk' and f'ratio_{str(ratio)}' not in filename:
        newfilename = filename.replace(orig_identifier,new_identifier)
        newfilepath = os.path.join(new_trk_folder, newfilename +f_ext)

        python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/05_stats_splitbundles.py {filepath} {newfilepath}"
        job_name = job_descrp + "_" + subj
        command = os.path.join(GD,
                               "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"