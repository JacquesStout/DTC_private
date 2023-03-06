
from dipy.data import default_sphere, get_fnames
from dipy.io.image import load_nifti, load_nifti_data
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti


label_fname = get_fnames('stanford_labels')
labels_img = nib.load(label_fname)
print(label_fname)
seed_mask = np.asanyarray(labels_img.dataobj) == 2
seed_mask = seed_mask.astype(np.int32)
seedmask_path = '/Users/jas/jacques/APOE_trks_test/seed_test.nii.gz'
stanford_path = '/Users/jas/jacques/APOE_trks_test/standford_aparc_reduced_test.nii.gz'
save_nifti(seedmask_path, seed_mask, labels_img.affine)
save_nifti(stanford_path, labels_img.dataobj, labels_img.affine)