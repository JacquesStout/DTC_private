import os, subprocess, time
from DTC.file_manager.file_tools import mkcdir, getfromfile
from DTC.file_manager.computer_nav import read_parameters_from_ini
from DTC.file_manager.BIAC_tools import send_mail
import numpy as np
import socket, glob, datetime


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

        # print("Currently running jobs:", running_jobs)

        time.sleep(5)

    txt = ("Finished with the current jobs")
    if verbose:
        print(txt)
    if email_sending:
        send_mail(txt, subject="Done with qstat jobs")


def limit_jobs(limit = 10, verbose=False):
    firstloop = 0
    while True:
        running_jobs = get_running_jobs()

        non_interact_jobs = [job for job in running_jobs if job != 'interact']

        if not firstloop and len(non_interact_jobs)>=limit:
            print(f'Limiting Jobs at {limit} active jobs')
            firstloop=1

        if len(non_interact_jobs)<limit:
            break

        time.sleep(5)

    if firstloop:
        txt = ("Job done")
        if verbose:
            print(txt)


def check_for_errors(start_time ,folder_path):

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
                    return False ,file_path  # Return False if 'Error' is found in any file
    return True ,'allfine'