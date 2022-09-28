import time, glob, os
from DTC.file_manager.file_tools import mkcdir

start=time.time()

# paths to input and output directories
overwrite=True
chass_folder = '/Users/jas/jacques/Daniel_test/chass_symmetric3/'

"""
chass_reorient ='/Users/jas/jacques/Daniel_test/chass_symmetric3_subjspace/'


# desired orientations
oldchass_orientation="ARS"
newchass_orientation="LPS"


mkcdir(chass_reorient)
# create output directory
files = glob.glob(os.path.join(chass_folder, '*nii.gz'))

for file in files:
    # image_func_exect()
    newfile = os.path.join(chass_reorient,os.path.basename(file))
    

    if not os.path.exists(newfile) or overwrite:
        img_transform_exec(file, 'ARS', 'LPS', output_path=newfile, recenter_test=True)

        oldnifti = nib.load(newfile)
        newnifti = copy.deepcopy(oldnifti)
        old_affine = copy.deepcopy(oldnifti.affine)
        new_affine = newnifti.affine
        new_affine[1,:] = -old_affine[2,:]
        new_affine[2,:] = -old_affine[1,:]
        new_affine[0,3] = new_affine[0,3] - 15
        new_affine[1,3] = new_affine[1,3] - 1
        new_affine[2,3] = new_affine[2,3] + 15


        nib.save(newnifti,newfile)
        print(f'Saved {newfile}')
"""

overwrite=True
chass_folder = '/Users/jas/jacques/Daniel_test/chass_symmetric3/'
chass_reorient ='/Users/jas/jacques/Daniel_test/chass_symmetric3_MDT/'
MDT_target = '/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_test_i7/median_images/MDT_T1.nii.gz'
label_name = 'chass_symmetric3'

mkcdir(chass_reorient)
# create output directory

final_template_run = '/Volumes/Data/Badea/Lab/mouse/VBM_18abbRAREset_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_test_i7'

MDT_to_atlas_affine = os.path.join(final_template_run, "stats_by_region", "labels", "transforms",
                                   f"MDT_*_to_{label_name}_affine.mat")
atlas_to_MDT = os.path.join(final_template_run, "stats_by_region", "labels", "transforms",
                            f"{label_name}_to_MDT_warp.nii.gz")

files = glob.glob(os.path.join(chass_folder, '*nii.gz'))

for file in files:
    newfile = os.path.join(chass_reorient, os.path.basename(file))
    if not os.path.exists(newfile) or overwrite:
        cmd = f"antsApplyTransforms -v 1 -d 3 -i {file} -o {newfile} -r {MDT_target} -n MultiLabel -t [{MDT_to_atlas_affine},1] {atlas_to_MDT}"
        os.system(cmd)
