import os

"""
diffusion_locale = ''

in=${BIGGUS_DISKUS}/diffusion_prep_21593/21593_dwi.nii.gz;
ref=${BIGGUS_DISKUS}/whitson_symlink_pool/isotropic_reference.nii.gz;
ref1=~/tmp_ref_whitson.nii.gz;

ResampleImage 3 $in $ref 256x256x136 1;
ResampleImageBySpacing 3 $ref $ref1 1 1 1 0;
CopyImageHeaderInformation $ref1 $ref $ref 1 1 1;
"""

input = '/Volumes/Data/Badea/Lab/human/AMD/diffusion_prep_locale/diffusion_prep_H21593/H21593_dwi.nii.gz'
ref = '/Volumes/Data/Badea/Lab/mouse/whitson_symlink_pool/isotropic_reference.nii.gz'
ref1 = '~/tmp_ref_whitson.nii.gz'

cmd = f'ResampleImage 3 {input} {ref} 256x256x136 1'
os.system(cmd)
cmd = f'ResampleImageBySpacing 3 {ref} {ref1} 1 1 1 0'
os.system(cmd)
cmd = f'CopyImageHeaderInformation {ref1} {ref} {ref} 1 1 1'
os.system(cmd)

#os.remove(ref1)