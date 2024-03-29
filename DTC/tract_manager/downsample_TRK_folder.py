import os
from DTC.tract_manager.tract_handler import reducetractnumber
from DTC.file_manager.file_tools import mkcdir
import random
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from DTC.tract_manager.tract_handler import reducetractnumber

import subprocess

#trk_folder = '/mnt/paros_DB/Projects/AD_Decode/Analysis/TRK_MDT'
#new_trk_folder = '/mnt/paros_WORK/jacques/AD_Decode/TRK_MDT_ratio_10'

#trk_folder = '/mnt/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_MDT'
#new_trk_folder = '/mnt/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_MDT_ratio_10'

#trk_folder = '/mnt/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_MDT_ratio_100'
#new_trk_folder = '/mnt/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_MDT_ratio_1000'

trk_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/ADDECODE/TRK_MDT_fixed'
new_trk_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/ADDECODE/TRK_MDT_fixed_ratio_10'

trk_folder = '/Volumes/Data/Badea/Lab/human/AD_Decode_trk_transfer/TRK_MDT_fixed_ratio_10'
new_trk_folder = '/Volumes/Data/Badea/Lab/human/AD_Decode_trk_transfer/TRK_MDT_fixed_ratio_100'

#remote = True

remote = False

if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

inpath, _, _, sftp = get_mainpaths(remote,project = 'AD_Decode', username=username,password=passwd)

mkcdir([new_trk_folder], sftp)

orig_ratio =10

ratio = 10
stepsize = 2
method= 'decimate'
streamline_type = 'mrtrix'

#methods can be 'decimate', 'top_len', and 'bot_len', where decimate just takes every 'ratio' streamline
#top_len selects the 'ratio' longest streamlines, 'bot_len' selects the 'ratio' shortest streamlines
"""
filelist = os.listdir(trk_folder)
filelist = sorted(filelist)
random.shuffle(filelist)
"""

filelist = glob_remote(os.path.join(trk_folder,'*trk'),sftp)
filelist = sorted(filelist)
test=False
overwrite=False
orig_identifier = get_str_identifier(stepsize, orig_ratio, '', type=streamline_type)
new_identifier = get_str_identifier(stepsize, ratio, '', type=streamline_type)

qsub = False

if qsub:
    job_descrp = 'downsampler'

    try:
        BD = os.environ['BIGGUS_DISKUS']
    # os.environ['GIT_PAGER']
    except KeyError:
        print('BD not found locally')
        BD = '/mnt/munin2/Badea/Lab/mouse'
        # BD ='/Volumes/Data/Badea/Lab/mouse'
    else:
        print("BD is found locally.")
    sbatch_folder_path = os.path.join(BD, job_descrp + '_sbatch/')
    mkcdir(sbatch_folder_path)


GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'

job_descrp = "BuSA"

#filelist = [filelist[0]]
print(filelist)

verbose = True
qsub = False

for filepath in filelist:
    _, f_ext = os.path.splitext(filepath)
    filename = os.path.basename(filepath)
    subj = filename.split('_')[0]

    print(f'filename is {filename}, f_ext is {f_ext}')
    if f_ext == '.trk' and f'ratio_{str(ratio)}' not in filename:
        newfilename = filename.replace(orig_identifier,new_identifier)

        newfilepath = os.path.join(new_trk_folder, newfilename)
        arguments = [filepath, newfilepath, ratio, method, verbose]

        if qsub:
            arguments = [str(arg) for arg in arguments]
            arguments_joined = arguments.join(' ')  # f'{filepath} {newfilepath} {ratio} {method} {verbose}'
            python_command = f"python3 /home/jas297/linux/DTC_private/DTC/tract_manager/downsample_TRK_file.py {arguments}"

            job_name = job_descrp + "_" + subj
            command = os.path.join(GD,
                               "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
            if test:
                print(python_command)
            else:
                os.system(command)
        else:
            #subprocess.run(python_command,shell=True,check=True)
            reducetractnumber(arguments[0],arguments[1],getdata = False, ratio=arguments[2],method = arguments[3],verbose = arguments[4])
