import nibabel as nib
import pandas as pd
import math
import numpy as np

atlas = pd.read_excel('/Volumes/Data/Badea/Lab/atlases/chass_symmetric3/CHASSSYMM3AtlasLegends.xlsx')

labels_wm = list(atlas[atlas['Subdivisions_7']=='7_whitematter']['index2'])
labels_wm = [int(integral) for integral in labels_wm]

labels_gm = atlas[atlas['Subdivisions_7']!='7_whitematter']
labels_gm = list(labels_gm[labels_gm['Subdivisions_7']!='8_CSF']['index2'])
labels_gm = [x for x in labels_gm if math.isnan(x) == False]
labels_gm = [int(integral) for integral in labels_gm]

labels = '/Users/jas/jacques/Daniel_test/chass_symmetric3_subjspace/chass_symmetric3_labels.nii.gz'

img = nib.load(labels)
data = img.get_fdata()
mask_gm = np.zeros(np.shape(data))
mask_wm = np.zeros(np.shape(data))
affine = img.affine
hdr = img.header

x = len(data)
y = len(data[0])
z = len(data[0][0])

for i in range(x):
    for j in range(y):
        for k in range(z):
            if data[i][j][k] in labels_gm:
                mask_gm[i][j][k] = 1
            else:
                mask_gm[i][j][k] = 0

            if data[i][j][k] in labels_wm:
                mask_wm[i][j][k] = 1
            else:
                mask_wm[i][j][k] = 0



outimg = nib.Nifti1Image(mask_gm, affine, hdr)
nib.save(outimg, '/Users/jas/jacques/Daniel_test/gm_mask.nii.gz')

outimg = nib.Nifti1Image(mask_wm, affine, hdr)
nib.save(outimg, '/Users/jas/jacques/Daniel_test/wm_mask_test.nii.gz')
