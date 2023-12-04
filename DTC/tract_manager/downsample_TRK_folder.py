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
new_trk_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT_ratio_10'

remote = True

remote=True
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

inpath, _, _, sftp = get_mainpaths(remote,project = 'AD_Decode', username=username,password=passwd)

print(new_trk_folder)
mkcdir([new_trk_folder], sftp)

orig_ratio =1
ratio = 10
stepsize = 2
method= 'decimate'
streamline_type = 'mrtrix'
"""
filelist = os.listdir(trk_folder)
filelist = sorted(filelist)
random.shuffle(filelist)
"""

filelist = glob_remote(os.path.join(trk_folder,'*trk'),sftp)

test=False
overwrite=False
orig_identifier = get_str_identifier(stepsize, orig_ratio, '', type=streamline_type)
new_identifier = get_str_identifier(stepsize, ratio, '', type=streamline_type)

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

filelist = [filelist[0]]
print(filelist)
test = True

verbose = True

for filepath in filelist:
    _, f_ext = os.path.splitext(filepath)
    filename = os.path.basename(filepath)
    subj = filename.split('_')[0]

    print(f'filename is {filename}, f_ext is {f_ext}')
    if f_ext == '.trk' and f'ratio_{str(ratio)}' not in filename:
        newfilename = filename.replace(orig_identifier,new_identifier)

        newfilepath = os.path.join(new_trk_folder, newfilename)

        python_command = f"python /home/jas297/linux/DTC_private/DTC/tract_manager/downsample_TRK_file.py {filepath} {newfilepath} {ratio} {method} {verbose}"
        job_name = job_descrp + "_" + subj
        command = os.path.join(GD,
                               "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        if test:
            print(python_command)
        else:
            os.system(command)
