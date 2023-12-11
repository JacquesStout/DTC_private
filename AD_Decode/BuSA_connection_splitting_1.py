
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import create_label_mask
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
import os
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile

ROI_legends = os.path.join('/Volumes/Data/Badea/Lab/', './atlases/IITmean_RPI/IITmean_RPI_index.xlsx')
converter_lr, converter_comb, index_to_struct_lr, index_to_struct_comb = atlas_converter(ROI_legends)

label_list = []
mask_outpath_list = []

folder_outpath = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT/atlas_MDT_masks'

mkcdir(folder_outpath)

atlas_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT/IITmean_RPI_MDT_labels.nii.gz'

for key in converter_lr.keys():
    if key == 0:
        continue
    label_list.append(key)
    try:
        region_name = (index_to_struct_lr[converter_lr[key]])
    except:
        print('hi')
    if region_name.startswith('left') or region_name.startswith('right'):
        #Removing the left/right at beginning of corresponding variable
        region_name = ('_').join((index_to_struct_lr[converter_lr[key]]).split('-')[1:])
    region_path = os.path.join(folder_outpath, f'{region_name}_MDT.nii.gz')
    mask_outpath_list.append(region_path)

create_label_mask(atlas_path, label_list, mask_outpath_list, conserve_val = False, exclude = False)
