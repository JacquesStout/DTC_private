import os
import subprocess

ndir = '/Users/jas/jacques/CS_Project/CS_Data_all/Bruker_niftis'
raw_dir = '/Users/jas/jacques/CS_Project/CS_Data_all/Bruker_data/'

ndir = '/Volumes/dusom_mousebrains/All_Staff/jacques/Bruker_niftis'
raw_dir = '/Volumes/dusom_mousebrains/All_Staff/jacques/Bruker_data/'

if not os.path.isdir(ndir):
    os.mkdir(ndir, mode=0o775)

ncb_path = '/Volumes/Data/Badea/Lab/jacques/Downloads/nanconvert-macos/'
os.environ['PATH'] = ncb_path + ':' + os.environ['PATH']

for data_dir in os.listdir(raw_dir):
    cmd = f'/Users/jas/bass/gitfolder/nanconvert/Scripts/nanbruker -z -v -l -o {ndir} {os.path.join(raw_dir, data_dir)}/'

    if not os.path.isdir(os.path.join(ndir, data_dir)):
        print(f'No results detected for {data_dir}; attempting to generate Niftis with nanconvert:')
        print('Command:')
        print(cmd)
        subprocess.call(cmd, shell=True)
    else:
        print(f'Result folder for {data_dir} already detected; skipping.')