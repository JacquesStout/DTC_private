import os, subprocess, time
from DTC.file_manager.file_tools import mkcdir, getfromfile
from DTC.file_manager.computer_nav import read_parameters_from_ini
from DTC.file_manager.BIAC_tools import send_mail
import numpy as np
import socket, glob, datetime

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


def check_for_errors(start_time,folder_path):
    
    # Get a list of files starting with 'sl' in the specified folder
    files = glob.glob(os.path.join(folder_path, 'slurm*'))

    for file_path in files:
        # Check if the file was modified after the current time
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        if modified_time > start_time:
            # Read the file content and check for the word 'Error'
            with open(file_path, 'r') as file:
                content = file.read()
                if 'Error' in content:
                    return False,file_path  # Return False if 'Error' is found in any file

    return True,'allfine'  # Return True if no 'Error' is found in any file


start_time = datetime.datetime.now()

job_descrp = "BuSA"
sbatch_folder_path = os.path.join(BD, job_descrp + '_sbatch/')
mkcdir(sbatch_folder_path)

GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'

if 'santorini' in socket.gethostname().split('.')[0]:
    project_headfile_folder = '/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles'
if 'blade' in socket.gethostname().split('.')[0]:
    project_headfile_folder = '/mnt/munin2/Badea/Lab/jacques/BuSA_headfiles'
import sys
#project_run_identifier = '202311_10template_test01'

if len(sys.argv)<2:
    project_run_identifier = '202311_10template_1000_72_interhe'
else:
    project_run_identifier = sys.argv[1]

if len(sys.argv)<3:
    parts = ['1','2','3','4','5']
else:
    parts = [str(sys.argv[2])]

print(project_run_identifier, parts)

project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')

#parts = ['1','2','3','4','5']
#parts = ['2']
#parts = ['3','4']
subjects = []
project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')
params = read_parameters_from_ini(project_summary_file)

template_subjects = params['template_subjects']
added_subjects = params['added_subjects']
num_bundles = int(params['num_bundles'])

full_subjects_list = template_subjects + added_subjects

sides = ['left', 'right']
bundle_ids = np.arange(num_bundles)

bundle_id_looping = False

overwrite = False

qsub = True

testmode = False

if '1' in parts:
    for subj in template_subjects:
        python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/01_split_streams_left_right.py {project_summary_file} {subj}"
        job_name = job_descrp + "_job01_" + subj
        command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        if testmode:
            print(python_command)
        else:
            os.system(command)
    pause_jobs()
    error_status,error_filepath = check_for_errors(start_time,sbatch_folder_path)
    if error_status is False:
        txt = f'Error found on qstat runs, details found at {error_filepath}'
        raise Exception(txt)

if '2' in parts:
    python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/02_bundle_combinedsubj.py {project_summary_file}"
    job_name = job_descrp + "_job02"
    command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    if testmode:
        print(python_command)
    else:
        os.system(command)
    pause_jobs()
    error_status,error_filepath = check_for_errors(start_time,sbatch_folder_path)
    if error_status is False:
        txt = f'Error found on qstat runs, details found at {error_filepath}'
        raise Exception(txt)


if '3' in parts:
    python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/03_savesplitbundles.py {project_summary_file}"
    job_name = job_descrp + "_job03"
    command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
    if testmode:
        print(python_command)
    else:
        os.system(command)
    pause_jobs()
    error_status,error_filepath = check_for_errors(start_time,sbatch_folder_path)
    if error_status is False:
        txt = f'Error found on qstat runs, details found at {error_filepath}'
        raise Exception(txt)



if '4' in parts:
    for subj in full_subjects_list:
        python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/04_addmoresubjects.py {project_summary_file} {subj}"
        job_name = job_descrp + "_job04"+subj
        command = os.path.join(GD, "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        if qsub:
            if testmode:
                print(python_command)
            else:
                os.system(command)
        else:
            os.system(python_command)
    pause_jobs()
    error_status,error_filepath = check_for_errors(start_time,sbatch_folder_path)
    if error_status is False:
        txt = f'Error found on qstat runs, details found at {error_filepath}'
        raise Exception(txt)



if '5' in parts:
    for subj in full_subjects_list:
        if bundle_id_looping:
            for side in sides:
                for bundle_id in bundle_ids:
                    python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/05_stats_splitbundles.py {project_summary_file} {subj} {side} {bundle_id}"
                    job_name = job_descrp + "_" + subj
                    command = os.path.join(GD ,"submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
                    if testmode:
                        print(python_command)
                    else:
                        os.system(command)
        else:
            python_command = f"python /home/jas297/linux/DTC_private/AD_Decode/05_stats_splitbundles.py {project_summary_file} {subj}"
            job_name = job_descrp + "_" + subj
            command = os.path.join(GD ,"submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
            if testmode:
                print(python_command)
            else:
                os.system(command)
            

    pause_jobs()
    error_status,error_filepath = check_for_errors(start_time,sbatch_folder_path)
    if error_status is False:
        txt = f'Error found on qstat runs, details found at {error_filepath}'
        raise Exception(txt)

