import os, subprocess, time
from DTC.file_manager.file_tools import mkcdir, getfromfile
from DTC.file_manager.computer_nav import read_parameters_from_ini
from DTC.file_manager.BIAC_tools import send_mail
import numpy as np
import socket, glob, datetime, argparse
from DTC.wrapper_tools import parse_list_arg
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths
from DTC.tract_manager.tract_handler import gettrkpath, filter_streamlines, ratio_to_str
import re

# create sbatch folder

def get_running_jobs():
    try:
        output = subprocess.check_output(["qstat"]).decode("utf-8")
        lines = output.split('\n')[2:]  # Skip the header lines
        running_jobs = [line.split()[2] for line in lines if line.strip()]  # Extract job names
        return running_jobs
    except subprocess.CalledProcessError:
        return []


def pause_jobs(verbose=False, email_sending=False):
    while True:
        running_jobs = get_running_jobs()

        if not running_jobs or all(job == 'interact' for job in running_jobs):
            print("All jobs are 'interact'. Stopping the loop.")
            break

        # print("Currently running jobs:", running_jobs)

        time.sleep(5)

    txt = ("Finished with the current jobs")
    if verbose:
        print(txt)
    if email_sending:
        send_mail(txt, subject="Done with qstat jobs")


def check_for_errors(start_time, folder_path):
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
                    return False, file_path  # Return False if 'Error' is found in any file

    return True, 'allfine'  # Return True if no 'Error' is found in any file


def find_matching_files(folder_path, pattern):
    matching_files = []
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the filename matches the specified pattern
        if re.match(pattern, filename):
            matching_files.append(filename)
    return matching_files

#print('0')

if 'santorini' in socket.gethostname().split('.')[0]:
    #project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/BuSA_headfiles'
    project_headfile_folder = '/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles'
    BD = '/Volumes/Data/Badea/Lab/mouse'
    code_folder = '/Users/jas/bass/gitfolder/DTC_private/'
if 'blade' in socket.gethostname().split('.')[0]:
    project_headfile_folder = '/mnt/munin2/Badea/Lab/jacques/BuSA_headfiles'
    code_folder = '~/DTC_private/'
import sys

if 'BD' not in locals():
    try:
        BD = os.environ['BIGGUS_DISKUS']
    # os.environ['GIT_PAGER']
    except KeyError:
        print('BD not found locally')
        BD = '/mnt/munin2/Badea/Lab/mouse'
        # BD ='/Volumes/Data/Badea/Lab/mouse'
    else:
        print("BD is found locally.")

start_time = datetime.datetime.now()

#print('1')

job_descrp = "BuSAreg"
sbatch_folder_path = os.path.join(BD, job_descrp + '_sbatch/')
mkcdir(sbatch_folder_path)

GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'

# project_run_identifier = '202311_10template_test01'

#print('2')

#python3 AD_Decode_BuSA_iterative_wrapper.py --split 4 --proj V0_9_10template_100_6_interhe_majority --id 4 --split 6 --parts 8

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--parts', type=parse_list_arg, help='A list of integers for the code sequences to run (1,2,3)')
parser.add_argument('--split', type=int, help='An integer for splitting')
parser.add_argument('--proj', type=str, help='The project path or name')
parser.add_argument('--id', type=parse_list_arg, help='An integer or a list of integers')


bundle_id_looping = False

overwrite = False

qsub = True

testmode = False


args = parser.parse_args()
bundle_id_orig = args.id

#print(bundle_id_orig)
bundle_split = args.split
project_run_identifier = args.proj
parts = args.parts

if parts is None:
    parts = ['1','2','3']

subjects = []
if os.path.exists(project_run_identifier):
    project_summary_file = project_run_identifier
    project_name = os.path.basename(project_run_identifier).split('.ini')[0]
else:
    project_summary_file = os.path.join(project_headfile_folder, project_run_identifier + '.ini')
    project_name = project_run_identifier
print(project_summary_file)
if not os.path.exists(project_summary_file):
    raise Exception('Could not find project file')
params = read_parameters_from_ini(project_summary_file)

template_subjects = params['template_subjects']
added_subjects = params['added_subjects']
num_bundles = int(params['num_bundles'])

print('4')
print(parts)
full_subjects_list = template_subjects + added_subjects
full_subjects_list = ['S02227','S02386','S02410','S02421','S02666','S02877']
sides = ['left', 'right']

if bundle_id_orig is not None:
    bundle_id_orig = bundle_id_orig[0].replace('all','*')

    if '*' in bundle_id_orig:

        ratio = params['ratio']

        remote_input = bool(params['remote_input'])
        remote_output = bool(params['remote_output'])
        ratiostr = ratio_to_str(ratio, spec_all=False)

        if remote_input or remote_output:
            username, passwd = getfromfile(os.path.join(os.environ['HOME'], 'remote_connect.rtf'))
        else:
            username = None
            passwd = None

        project = params['project']
        outpath, _, _, sftp_out = get_mainpaths(remote_output, project=project, username=username, password=passwd)
        outpath_all = os.path.join(outpath, 'TRK_bundle_splitter')
        proj_path = os.path.join(outpath_all, project_name)

        trk_proj_path = os.path.join(proj_path, 'trk_roi' + ratiostr)

        added_subjects = params['added_subjects']
        bundle_id_orig_new = bundle_id_orig.replace("*","\d+")
        #bundle_ids = glob.glob(os.path.join(trk_proj_path,f'{added_subjects[1]}_bundle_left_{bundle_id_orig_new}.trk'))
        print(bundle_id_orig_new)
        print(trk_proj_path)
        print(os.path.join(trk_proj_path, f'{added_subjects[1]}_bundle_left_{bundle_id_orig_new}.trk'))
        bundle_ids = find_matching_files(trk_proj_path, f'{added_subjects[1]}_bundle_left_{bundle_id_orig_new}.trk')
        print(bundle_ids)
        bundle_ids = [bundle_name.split('left_')[1].split('.trk')[0] for bundle_name in bundle_ids]
        print(bundle_ids)
        #bundle_ids = np.arange(num_bundles)
    else:
        bundle_ids = [bundle_id_orig]
else:
    bundle_ids = [None]

code_specific_folder = os.path.join(code_folder,'AD_Decode','BuSA_split_regions_into_bundles')

#print(bundle_ids)


for bundle_id_orig in bundle_ids:

    if bundle_id_orig is not None:
        bundle_id_qsub = ','.join(bundle_id_orig)
        bundle_id_qsub_id = f'--id {bundle_id_orig}'
    else:
        bundle_id_qsub = ''
        bundle_id_qsub_id = ''


    if '1' in parts:
        python_command = f"python3 {os.path.join(code_specific_folder,'01_regBun_makecentroids.py')} --proj {project_summary_file} --split {bundle_split} {bundle_id_qsub_id}"
        job_name = job_descrp + "_job01"
        command = os.path.join(GD,
                               "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
        if testmode:
            print(python_command)
        else:
            if qsub:
                os.system(command)
            else:
                os.system(python_command)
        if qsub:
            pause_jobs()
            error_status, error_filepath = check_for_errors(start_time, sbatch_folder_path)
            if error_status is False:
                txt = f'Error found on qstat runs, details found at {error_filepath}'
                raise Exception(txt)

    if '2' in parts:
        for subj in full_subjects_list:
            python_command = f"python3 {os.path.join(code_specific_folder,'02_regBun_addsubjects.py')} --proj {project_summary_file}  --split {bundle_split} {bundle_id_qsub_id} --subj {subj}"

            job_name = job_descrp + "_job02_" + subj
            command = os.path.join(GD,
                                   "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
            if testmode:
                if qsub:
                    print(python_command)
                else:
                    print(python_command.split('--subj')[0])
                    break

            else:
                if qsub:
                    os.system(command)
                else:
                    os.system(python_command)
        if qsub:
            pause_jobs()
            error_status, error_filepath = check_for_errors(start_time, sbatch_folder_path)
            if error_status is False:
                txt = f'Error found on qstat runs, details found at {error_filepath}'
                raise Exception(txt)

    if '3' in parts:
        for subj in full_subjects_list:
            python_command = f"python3 {os.path.join(code_specific_folder,'03_regBun_stats.py')} --proj {project_summary_file}  --split {bundle_split} {bundle_id_qsub_id} --subj {subj}"
            job_name = job_descrp + "_job03_" + subj
            command = os.path.join(GD,
                                   "submit_sge_cluster_job.bash") + " " + sbatch_folder_path + " " + job_name + " 0 0 '" + python_command + "'"
            if testmode:
                if qsub:
                    print(python_command)
                else:
                    print(python_command.split('--subj')[0])
                    break
            else:
                if qsub:
                    os.system(command)
                else:
                    os.system(python_command)
        if qsub:
            pause_jobs()
            error_status, error_filepath = check_for_errors(start_time, sbatch_folder_path)
            if error_status is False:
                txt = f'Error found on qstat runs, details found at {error_filepath}'
                raise Exception(txt)
