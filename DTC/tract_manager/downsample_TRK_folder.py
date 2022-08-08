import os
from DTC.tract_manager.tract_handler import reducetractnumber
from DTC.file_manager.file_tools import mkcdir
import random

path = '/Users/jas/jacques/APOE_subj_to_MDT'
trk_folder = os.path.join(path,'TRK')
new_trk_folder = os.path.join(path,'TRK')
#trk_folder = os.path.join(path,'TRK_MPCA_fixed')
#new_trk_folder = os.path.join(path,'TRK_MPCA_fixed_100')

mkcdir(new_trk_folder)
ratio = 100
filelist = os.listdir(trk_folder)
filelist = sorted(filelist)
random.shuffle(filelist)
#filelist.reverse()

for filepath in filelist:
    filename, f_ext = os.path.splitext(filepath)
    if f_ext == '.trk' and f'ratio_{str(ratio)}' not in filename:
        newfilename = filename.replace('_all','_ratio_'+str(ratio))
        newfilepath = os.path.join(new_trk_folder, newfilename +f_ext)
        if not os.path.exists(newfilepath):
            print(f'Downsampling from {os.path.join(trk_folder,filepath)} to {newfilepath} with ratio {str(ratio)}')
            reducetractnumber(os.path.join(trk_folder,filepath), newfilepath, getdata=False, ratio=100, return_affine= False, verbose=False)
            print(f'succesfully created {newfilepath}')
        else:
            print(f'already created {newfilepath}')
