#! /bin/bash

## 14 March 2022 (Monday), BJ Anderson, Badea Lab, Duke University
# This script pulls data off of our Bruker scanner ("nemo") and copies it to a target directory on a target machine.
# This command uses user name and scan date to decide which data to transfer.
# Future versions will add more flexibility in specifying which data to pull.

# You will most likely be prompted for 2 passwords: the first is the password for the Bruker user
# The second is for the user on the target machine.

## USAGE:
# Sample command:
# pull_data_from_Bruker_scanner.bash nmrsu rja20 feynman ~/raw_pcasl_data/bruker/ 20220125
# First option:
# B_USER=nmrsu; # The username used when scanning on the Bruker
# Second option:
# T_USER=rja20; # The username used to access the target computer
# Third option:
# T_machine=feynman; # Name of the target computer--will usually be 'kefalonia'
# Fourth option:
# T_location=/Users/${T_USER}/raw_pcasl_data/bruker/ # NOTE: This should be the parent directory in which you want to put each directory transferred over

# I usually transfer data for all the data acquired in a single day, using our YYYYMMDD format:
day = 20220125;  # For example, today, January 25, 2022 is 20220125

from DTC.file_manager.file_tools import mkcdir
import os
import socket
from DTC.file_manager.computer_nav import get_sftp, scp_multiple, glob_remote

"""
B_USER =$1;
T_USER =$2;
T_machine =$3;
T_location =$4;
day =$5;
"""
B_USER = 'nmrsu'
T_USER ='jas';
T_machine= 'feynman'
T_location= '/Users/${T_USER}/bruker/raw_pcasl_data'
day ='';

# Archive option
if T_machine == "archive":
    T_machine = 'samos';
    print(f"You have selected the 'archive' option; raw Bruker data will be copied to the project folder ${T_location} if it exists in the database.")
    T_location = f'/mnt/paros_DB/Projects/${T_location}/Data/raw_bruker_data/'
    print(f"New target location: ${T_location}.")
    print(f"Please be sure that ${T_USER} exists on samos; if not, please consider running as user 'alex' instead.")
    mkcdir(os.path.join(T_location,'raw_bruker_data'))
    #archive_check_cmd = "cd ${T_location/raw_bruker_data/}; if [[ ! -d ${T_location} ]];then mkdir -m 775 ${T_location};fi;exit";
    #print("Enter the password for ${T_USER} on ${T_machine} if prompted...")
    #ssh - t ${T_USER} @ samos ${archive_check_cmd};

# Error checking:
computer_name = socket.gethostname()
remote=False

if computer_name != T_machine:
    remote=True

sftp = get_sftp(remote, username=T_USER, password=None)

"""
errors = 1;
error_message = "${error_message}^Unable to connect to target machine, \"${T_machine}\"; please check to make sure you have entered the name correctly and the machine is on.";
fi

if ((${errors}));then
error_message = "${error_message}^Quitting now...";
error_message =$(echo ${error_message} | tr '^' "\n");

print("${error_message}" & & exit
1;

fi
"""
# Run your commands:

# remote_commands="cd /opt/PV6.0.1/data/${B_USER}/; echo \"Enter the password for ${T_USER} on ${T_machine} if prompted...\"; scp -r ${day}_* ${T_USER}@${T_machine}:${T_location};"
remote_commands = "cd /opt/PV6.0.1/data/${B_USER}/; echo \"Enter the password for ${T_USER} on ${T_machine} if prompted...\"; rsync -blurtDv ${day}_* ${T_USER}@${T_machine}:${T_location};"
files_all = glob_remote(f'/opt/PV6.0.1/data/${B_USER}/{day}_*', sftp)
scp_multiple(files_all, T_location)
#print("Enter the password for ${B_USER} on nemo if prompted..."
#ssh - t ${B_USER} @ nemo ${remote_commands};
# echo "screen -d -m  ${remote_commands}" | ssh -t ${B_USER}@nemo;
# Enter the appropriate password when prompted, of course.