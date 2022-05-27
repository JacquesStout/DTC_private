import nibabel as nib
import time
import socket, os, getpass, paramiko


filepath = '/Users/jas/bass/gitfolder/wuconnectomes/testfile/S02802_stepsize_2_ratio_100_wholebrain_pruned.trk'
#filepath = '/mnt/paros_MRI/jacques/AD_Decode/Analysis/TRK_MPCA_MDT_100/S02802_stepsize_2_ratio_100_wholebrain_pruned.trk'
filepath = '/Users/jas/bass/gitfolder/wuconnectomes/testfile/S02802_stepsize_2_all_wholebrain_pruned.trk'
time1 = time.time()
nib.streamlines.load(filepath)
print(f'Time taken to load {filepath} was {time.time()-time1} seconds')
