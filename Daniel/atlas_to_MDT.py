
import os, glob
from DTC.file_manager.file_tools import mkcdir, check_files

folder = '/Users/jas/jacques/Daniel_test/chass_symmetric3'
output_folder = '/Users/jas/jacques/Daniel_test/chass_symmetric3_MDT'

mkcdir(output_folder)

filepaths = glob.glob(os.path.join(folder,'*.nii.gz'))

reference = '/Volumes/Data/Badea/Lab/mouse/VBM_18APOERAREset02_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_i7/median_images/MDT_T1.nii.gz'
work_dir = '/Volumes/Data/Badea/Lab/mouse/VBM_18APOERAREset02_invivoAPOE1-work/'

mainpath = '/Volumes/Data/Badea/Lab/atlas'
atlas_labels = os.path.join(mainpath, "atlases", "chass_symmetric3", "chass_symmetric3_labels.nii.gz")

label_name = os.path.basename(atlas_labels)
label_name = label_name.split("_labels")[0]

template_type_prefix = os.path.basename(os.path.dirname(glob.glob(os.path.join(work_dir, "T1", "SyN*/"))[0]))
template_runs = glob.glob((os.path.join(work_dir, "T1", template_type_prefix, "*/")))

mymax = -1
if mymax == -1:
    for template_run in template_runs:
        if "rabies" in template_run and template_run[-4:-2] == "_i":
            if int(template_run[-2]) > mymax:
                mymax = int(template_run[-2])
                final_template_run = template_run

atlas_to_MDT = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"{label_name}_to_MDT_warp.nii.gz")
MDT_to_atlas_affine = os.path.join(final_template_run, "stats_by_region", "labels", "transforms",
                                   f"MDT_*_affine.mat")

[MDT_to_atlas_affine, atlas_to_MDT], exists = check_files([MDT_to_atlas_affine, atlas_to_MDT])

for filepath in filepaths:
    newfile_path = os.path.join(output_folder,os.path.basename(filepath))
    cmd = f"antsApplyTransforms -v 1 -d 3 -i {filepath} -o {newfile_path} -r {reference} -n MultiLabel -t [{MDT_to_atlas_affine},1] {atlas_to_MDT}"
    os.system(cmd)
