import os
from tract_manager.tract_handler import get_tract_params
from file_manager.computer_nav import get_mainpaths
from file_manager.file_tools import getfromfile, mkcdir
import numpy as np
import pandas as pd

remote=True
project='AD_Decode'
if remote:
    username, passwd = getfromfile('/Users/jas/remote_connect.rtf')
else:
    username = None
    passwd = None

subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361', 'S02363',
        'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451', 'S02469', 'S02473',
        'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654', 'S02666', 'S02670', 'S02686',
        'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771', 'S02781', 'S02802',
        'S02804', 'S02813', 'S02812', 'S02817', 'S02840', 'S02842', 'S02871', 'S02877', 'S02898', 'S02926', 'S02938',
        'S02939', 'S02954', 'S02967', 'S02987', 'S03010', 'S03017', 'S03028', 'S03033', 'S03034', 'S03045', 'S03048',
        'S03069', 'S03225', 'S03265', 'S03293', 'S03308', 'S03321', 'S03343', 'S03350', 'S03378', 'S03391', 'S03394']
#subjects = ['S03321']
removed_list = ["S02745","S02230","S02490","S02523"]

for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

inpath = '/mnt/paros_MRI/jacques/AD_Decode/Analysis/TRK_MPCA_dsistudio/'
_, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
stats_folder = '/Users/jas/jacques/Statistics_ADDecode/'
mkcdir(stats_folder)
verbose = True
csv_stats = os.path.join(stats_folder,
                                  'AD_Decode_dsistudiotrk_basestatistics.csv')
column_names = ['Number of Tracts', 'Minimum Length',
                        'Maximum Length', 'Average Length', 'Std Length']
csv_columns = {}
for i,column_name in enumerate(column_names):
    csv_columns.update({i:column_name})

params_array = np.zeros([np.size(subjects), np.size(column_names)])

for i,subject in enumerate(subjects):
    #trk_file = os.path.join(inpath, f'{subject}_stepsize_2_all_wholebrain_pruned.trk')
    trk_file = os.path.join(inpath, f'{subject}_config_9.trk')
    subject, numtracts, minlength, maxlength, meanlength, stdlength, _, _, _ = \
        get_tract_params(trk_file, subject, verbose=verbose, sftp=sftp)
    params_array[i,:] = [numtracts, minlength, maxlength, meanlength, stdlength]

params_arrayDF = pd.DataFrame()
params_arrayDF['Subjects'] = subjects
dataframes = [params_arrayDF, pd.DataFrame(params_array)]
params_arrayDF = pd.concat(dataframes, axis=1, join='inner')
#params_arrayDF = pd.DataFrame(params_array)
params_arrayDF = params_arrayDF.rename(index=str, columns=csv_columns)
params_arrayDF.to_csv(csv_stats, header=['Subjects']+column_names)
