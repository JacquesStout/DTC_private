from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import create_label_mask, create_labels_frommasks
import nibabel as nib
import numpy as np
import os
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import create_backport_labels, create_MDT_labels, \
    port_to_MDT
from DTC.file_manager.computer_nav import get_mainpaths, checkfile_exists_remote, getremotehome
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import get_info_SAMBA_headfile


mainpath = getremotehome('Lab')

SAMBA_mainpath = os.path.join(mainpath, "mouse")

whitematter_atlas = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/atlas_f_whitematter_LPI_to_RPI_decemberver.nii.gz'
mask_white_outpath = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_whitemattermask.nii.gz'
overwrite = False

if not os.path.exists(mask_white_outpath) or overwrite:
    label_list = list(np.unique(nib.load(whitematter_atlas).get_fdata()))
    label_list.remove(0)
    create_label_mask(whitematter_atlas, label_list, mask_white_outpath, conserve_val = False, exclude = False,verbose=False)
else:
    print(f'{mask_white_outpath} already exists')


greymatter_atlas = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_labels.nii.gz'
mask_grey_outpath = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_greymattermask.nii.gz'
overwrite=False
if not os.path.exists(greymatter_atlas) or overwrite:
    label_list = list(np.unique(nib.load(greymatter_atlas).get_fdata()))
    label_list.remove(0)
    create_label_mask(greymatter_atlas, label_list, greymatter_atlas, conserve_val = False, exclude = False,verbose=False)
else:
    print(f'{greymatter_atlas} already exists')


subject=''
SAMBA_mainpath = os.path.join(mainpath, "mouse")
SAMBA_projectname = "VBM_21ADDecode03_IITmean_RPI_fullrun"
SAMBA_headfile_dir = os.path.join(mainpath, "samba_startup_cache")

SAMBA_headfile = os.path.join(SAMBA_headfile_dir, "jas297_SAMBA_ADDecode.headfile")
verbose=True
_, _, myiteration = get_info_SAMBA_headfile(SAMBA_headfile)

atlas_labels = os.path.join('/Volumes/Data/Badea/Lab/', "atlases","IITmean_RPI","IITmean_RPI_labels.nii.gz")

grey_name = 'IITmean_greymattermask'
white_name = 'IITmean_whitemattermask'

mask_grey_outpath = os.path.join('/Volumes/Data/Badea/Lab/', "atlases","IITmean_RPI",f"{grey_name}.nii.gz")
mask_white_outpath = os.path.join('/Volumes/Data/Badea/Lab/', "atlases","IITmean_RPI",f"{white_name}.nii.gz")


MDT_addon = '_MDT_labels'

grey_MDT_path = os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT/',f"{grey_name}{MDT_addon}.nii.gz")
white_MDT_path = os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT/',f"{white_name}{MDT_addon}.nii.gz")

if not os.path.exists(grey_MDT_path) or overwrite:
    port_to_MDT(mask_grey_outpath, SAMBA_mainpath, SAMBA_projectname, atlas_labels, myiteration=myiteration,
                          overwrite=overwrite, verbose=verbose)
else:
    print(f'Already wrote {grey_MDT_path}')
if not os.path.exists(white_MDT_path) or overwrite:
    port_to_MDT(mask_white_outpath, SAMBA_mainpath, SAMBA_projectname, atlas_labels, myiteration=myiteration,
                      overwrite=overwrite, verbose=verbose)
else:
    print(f'Already wrote {white_MDT_path}')

#create_MDT_labels(subject, SAMBA_mainpath, SAMBA_projectname, atlas_labels, myiteration=myiteration,
#                      overwrite=overwrite, verbose=verbose)


grey_white_label_path = os.path.join('/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT/',f"gwm_labels_MDT.nii.gz")

if not os.path.exists(grey_white_label_path) or overwrite:
    create_labels_frommasks([grey_MDT_path,white_MDT_path],[1,2],grey_white_label_path,verbose=False)
else:
    print(f'Already wrote {grey_white_label_path}')