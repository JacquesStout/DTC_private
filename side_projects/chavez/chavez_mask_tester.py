from mask_handler import applymask_samespace, median_mask_make
import os
import numpy as np
from file_tools import mkcdir

#raw_dwi = '/Volumes/Data/Badea/Lab/mouse/Chavez_series/diffusion_prep_locale/diffusion_prep_C_20220124_007/C_20220124_007_raw_dwi.nii.gz'
raw_b0 = '/Volumes/Data/Badea/Lab/mouse/Chavez_series/diffusion_prep_locale/diffusion_prep_C_20220124_007/C_20220124_007_b0_dwi.nii.gz'

tmp = '/Volumes/Data/Badea/Lab/mouse/Chavez_series/diffusion_prep_locale/diffusion_prep_C_20220124_007/C_20220124_007_tmp.nii.gz'

outpath_dir = '/Users/jas/jacques/Chavez_test_temp_b0/'
mkcdir(outpath_dir)

for median_radius in np.arange(7,3,-1):
    for numpass in np.arange(7,3,-1):
        outpath = os.path.join(outpath_dir, f'007_mask_rad_{median_radius}_numpass_{numpass}.nii.gz')
        print(outpath)
        if not os.path.exists(outpath):
            print(f'creating file {outpath}')
            median_mask_make(raw_b0, tmp, outpathmask=outpath,
                                             median_radius=median_radius, numpass=numpass)
            print(f'Done')
        else:
            print(f'already made file {outpath}')
