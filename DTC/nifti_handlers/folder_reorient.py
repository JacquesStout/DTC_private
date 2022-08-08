from DTC.nifti_handlers.transform_handler import img_transform_exec
import numpy as np
import glob, os
from DTC.file_manager.file_tools import mkcdir

folder = '/Users/jas/jacques/Daniel_test/chass_symmetric3/'
output_folder = '/Users/jas/jacques/Daniel_test/chass_symmetric3_RAS/'
mkcdir(output_folder)
files = glob.glob(os.path.join(folder,'*.nii.gz'))
orientation_in = 'ARS'
orientation_out ='RAS'

verbose=True
rename = False

if np.size(files)>1:
    for file in files:
        img_transform_exec(file,orientation_in,orientation_out, output_path = output_folder, rename=rename, recenter=True, verbose=verbose)

print('done')