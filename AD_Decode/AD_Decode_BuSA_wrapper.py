import os, subprocess, time
from DTC.file_manager.file_tools import mkcdir, getfromfile
from DTC.file_manager.computer_nav import read_parameters_from_ini
from DTC.file_manager.BIAC_tools import send_mail
import numpy as np
import socket

try:
    BD = os.environ['BIGGUS_DISKUS']
# os.environ['GIT_PAGER']
except KeyError:
    print('BD not found locally')
    BD = '/mnt/munin2/Badea/Lab/mouse'
    # BD ='/Volumes/Data/Badea/Lab/mouse'
else:
    print("BD is found locally.")
# create sbatch folder

def get_running_jobs():
    try:
        output = subprocess.check_output(["qstat"]).decode("utf-8")
        lines = output.split('\n')[2:]  # Skip the header lines
        running_jobs = [line.split()[2] for line in lines if line.strip()]  # Extract job names
        return running_jobs
    except subprocess.CalledProcessError:
        return []

def pause_jobs(verbose=False, email_sending = False):
    while True:
        running_jobs = get_running_jobs()

        if not running_jobs or all(job == 'interact' for job in running_jobs):
            print("All jobs are 'interact'. Stopping the loop.")
            break

        #print("Currently running jobs:", running_jobs)

        time.sleep(5)

    txt = ("Finished with the current jobs")
    if verbose:
        print(txt)
    if email_sending:
        send_mail(txt, subject="Done with qstat jobs")



job_descrp = "BuSA"
sbatch_folder_path = BD + "/ADRC_preprocessing_pipeline/" + job_descrp + '_sbatch/'

mkcdir([sbatch_folder_path])

GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'

if 'santorini' in socket.gethostname().split('.')[0]:
    project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/BuSA_headfiles'
if 'blade' in socket.gethostname().split('.')[0]:
    project_headfile_folder = '/mnt/munin2/Badea/Lab/jacques/BuSA_headfiles'

#project_run_identifier = '202311_10template_test01'
project_run_identifier = '202311_10template_1000'
project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')

parts = ['1','2','3','4','5']

subjects = []
project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')
params = read_parameters_from_ini(project_summary_file)

template_subjects = params['template_subjects']
added_subjects = params['added_subjects']
num_bundles = int(params['num_bundles'])

full_subjects_list = template_subjects + added_subjects

sides = ['left', 'right']
bundle_ids = np.arange(num_bundles)

overwrite = False

if '1' in parts:
    for subj in template_subjects:
        python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/01_split_streams_left_right.py {project_summary_file} {subj}"
        job_name = job_descrp + "_job01_" + subj
        command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        os.system(command)

pause_jobs()

if '2' in parts:
    python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/02_bundle_combinedsubj.py {project_summary_file} {subj}"
    job_name = job_descrp + "_job02_"
    command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    os.system(command)

pause_jobs()

if '3' in parts:
    python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/03_savesplitbundles.py {project_summary_file} {subj}"
    job_name = job_descrp + "_job03_"
    command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    os.system(command)

pause_jobs()

if '4' in parts:
    python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/04_addmoresubjects.py {project_summary_file} {subj}"
    job_name = job_descrp + "_job04_"
    command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    os.system(command)

pause_jobs()

if '5' in parts:
    for subj in full_subjects_list:
        for side in sides:
            for bundle_id in bundle_ids:
                python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/05_stats_splitbundles.py {project_summary_file} {subj} {side} {bundle_id}"
                job_name = job_descrp + "_" + subj
                command = os.path.join(GD ,"submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
                os.system(command)
