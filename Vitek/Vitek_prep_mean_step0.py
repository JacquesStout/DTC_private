from DTC.file_manager.file_tools import mkcdir, largerfile, getfromfile
import os, glob
import nibabel as nib
import numpy as np

diffpath = '/Users/jas/jacques/Vitek_UNC/'
outpath_temp = '/Users/jas/jacques/Vitek_UNC_mean/'
mkcdir(outpath_temp)
subjects = ['11_8_13', '12_8_14','13_8_6','14_8_15','15_8_16', '16_8_21', '17_8_22', '18_8_25']
subjects = ['01_7_8', '02_7_17', '03_7_9', '04_7_16', '05_7_25', '06_8_8', '08_7_30', '09_7_23', '10_7_31', '11_8_13',
            '12_8_14','13_8_6','14_8_15','15_8_16', '16_8_21', '17_8_22', '18_8_25']
for subj in subjects:
    subjectpath = glob.glob(os.path.join(os.path.join(diffpath, "Vitek*" + subj + "*")))[0]
    file_path = largerfile(subjectpath, None)
    newsubjpath = subjectpath.replace(diffpath, outpath_temp)
    mkcdir(newsubjpath)
    newfile_path = os.path.join(newsubjpath, os.path.basename(file_path))
    file_nii = nib.load(file_path)
    data = file_nii.get_fdata()
    data_new = (data[:,:,:,:int(np.shape(data)[3]/2)]+ data[:,:,:,int(np.shape(data)[3]/2):])/2
    affine = file_nii._affine
    header = file_nii._header
    new_nii = nib.Nifti1Image(data_new, affine, header)
    nib.save(new_nii, newfile_path)