from DTC.diff_handlers.bvec_handler import writebval, writebvec, extractbvals_vectortxt
import numpy as np

bvals = [0,0] + list((np.zeros(60)+3000).astype(int))
outpath_bval = '/Users/jas/jacques/Vitek_temp_check/bval.txt'
outpath_bvec = '/Users/jas/jacques/Vitek_temp_check/bvec.txt'
writebval(bvals, outpath_bval, writeformat = "dsi", overwrite=False)
directions_path = '/Users/jas/jacques/Vitek_temp_check/60directions.txt'
bvecs = extractbvals_vectortxt(directions_path)
bvecs = list(bvecs)
bvecs = np.array([[0,0,0],[0,0,0]] + bvecs)
writebvec(bvecs, outpath_bvec, subject=None, writeformat = "dsi", overwrite=False)