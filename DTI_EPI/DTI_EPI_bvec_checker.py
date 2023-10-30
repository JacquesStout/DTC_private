import numpy as np
import itertools, os
from dipy.io.gradients import read_bvals_bvecs
from DTC.diff_handlers.bvec_handler import orient_to_str, reorient_bvecs, writebvec
import gzip
import shutil

orig_nii_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/13_1_DTI_EPI_seg_30dir_sat_DTI_EPI_seg_30dir_2d_125_125_500_denoised.nii.gz'

bashes = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/dsi_studio_tests/bashes/'
bvec_path_orig = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/13_1_DTI_EPI_seg_30dir_sat_DTI_EPI_seg_30dir_2d_125_125_500.bvec'
bval_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/13_1_DTI_EPI_seg_30dir_sat_DTI_EPI_seg_30dir_2d_125_125_500.bval'

mask_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/mask_orig.nii.gz'

roi_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/dsi_studio_tests/corpus_callosum_orig.nii.gz'

recon_type = 'dti'

bvec_orient1 = (np.array(list(itertools.permutations([1, 2, 3]))))
bvec_orient2 = [elm*[-1, 1, 1] for elm in bvec_orient1]
bvec_orient3 = [elm*[1, -1, 1] for elm in bvec_orient1]
bvec_orient4 = [elm*[1, 1, -1] for elm in bvec_orient1]
bvec_orient5 = [elm*[-1, -1, 1] for elm in bvec_orient1]
bvec_orient6 = [elm*[1, -1, -1] for elm in bvec_orient1]
bvec_orient7 = [[-1,-1,-1]]

bvec_orient_list = np.concatenate((bvec_orient4, bvec_orient1, bvec_orient2, bvec_orient3,bvec_orient5,bvec_orient6,bvec_orient7))

dsi_path = '/Applications/dsi_studio.app/Contents/MacOS//dsi_studio'

i=0
for orient in bvec_orient_list:

    orient_str = orient_to_str(orient)
    bvec_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/dsi_studio_tests/bvecs_all/13{orient_str}.bvec'

    if not os.path.exists(bvec_path):
        read_bvals_bvecs(bval_path,bvec_path_orig)
        bvals,bvecs = read_bvals_bvecs(bval_path,bvec_path_orig)
        bvecs_new = reorient_bvecs(bvecs, orient)
        writebvec(bvecs_new, bvec_path, writeformat = "dsi")

    src_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/dsi_studio_tests/13{orient_str}.src.gz'

    if not os.path.exists(src_path):
        command = f'{dsi_path} --action=src --source={orig_nii_path}  --output={src_path} --bval={bval_path} --bvec={bvec_path}'
        os.system(command)

    fib_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/dsi_studio_tests/13{orient_str}.src.gz.{recon_type}.fib.gz'
    if not os.path.exists(fib_path):
        command = f'{dsi_path} --action=rec --source={src_path}  --output={fib_path} --method=1 --mask={mask_path} --check_btable=0'
        os.system(command)

    trk_path = f'/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone/20231018_221003_8/dsi_studio_tests/13{orient_str}.trk'
    if not os.path.exists(trk_path):
        command = f'{dsi_path} --action=trk --source={fib_path} --roi={roi_path} --fiber_count=50000 --output={trk_path}'
        os.system(command)
        trk_zipped = trk_path.replace('.trk','.trk.gz')
        with gzip.open(trk_zipped, 'rb') as f_in:
            with open(trk_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(trk_zipped)