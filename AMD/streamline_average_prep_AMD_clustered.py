import numpy as np
from dipy.io.streamline import load_trk
import warnings
from dipy.tracking.streamline import transform_streamlines
import os, glob
from DTC.nifti_handlers.nifti_handler import getlabeltypemask
from DTC.file_manager.file_tools import mkcdir, getfromfile, check_files
from DTC.tract_manager.tract_handler import ratio_to_str, gettrkpath, gettrkpath_testsftp
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
import socket
from DTC.diff_handlers.connectome_handlers.excel_management import M_grouping_excel_save
import sys
from DTC.file_manager.argument_tools import parse_arguments_function
from DTC.diff_handlers.connectome_handlers.connectome_handler import connectivity_matrix_custom, \
    connectivity_matrix_func
import random
from time import time
from dipy.tracking.streamline import transform_streamlines
from DTC.file_manager.computer_nav import get_mainpaths, get_atlas, load_trk_remote, checkfile_exists_remote

project = 'AMD'

subject = sys.argv[1]
mrtrix = sys.argv[2]

if mrtrix:
    print('getting the mrtrix folder')
else:
    print('getting the none mrtrix folder')

subject = 'H21593'

remote = True
remote = False

# inpath, outpath, atlas_folder, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
#inpath = '/Volumes/dusom_mousebrains/All_Staff/Data/AMD/'
#outpath = '/Users/jas/jacques/Whiston_article/Analysis/AMD/'
#outpath = '/Volumes/dusom_mousebrains/All_Staff/Analysis/AMD/'
inpath = '/mnt/munin2/Badea/Lab/human/AMD_project_23'
outpath = '/mnt/munin2/Badea/Lab/human/AMD_project_23/Analysis'
#atlas_folder = '/Volumes/Data/Badea/Lab/atlases'
atlas_folder = '/mnt/munin2/Badea/Lab/atlases'
sftp = None

if project == 'AMD' or project == 'AD_Decode':
    atlas_legends = get_atlas(atlas_folder, 'IIT')

# Setting identification parameters for ratio, labeling type, etc
ratio = 1
ratio_str = ratio_to_str(ratio)
print(ratio_str)
if ratio_str == '_all':
    folder_ratio_str = ''
    ratio_str = ''
else:
    folder_ratio_str = ratio_str.replace('_ratio', '')

inclusive = False
symmetric = True
fixed = False
overwrite = False

if inclusive:
    inclusive_str = '_inclusive'
else:
    inclusive_str = '_non_inclusive'

if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'

if fixed:
    fixed_str = '_fixed'
else:
    fixed_str = ''

labeltype = 'lrordered'
verbose = True
picklesave = True

function_processes = parse_arguments_function(sys.argv)
print(f'there are {function_processes} function processes')

# if project=='AD_Decode':
#    outpath = os.path.join(outpath,'Analysis')
#    inpath = os.path.join(inpath, 'Analysis')

# TRK_folder = os.path.join(inpath, f'TRK_MPCA_MDT{fixed_str}{folder_ratio_str}')
space_type = 'MDT'

if mrtrix:
    trk_type = '_trix'
else:
    trk_type = ''

#TRK_folder = os.path.join(inpath, f'TRK_{space_type}{fixed_str}{folder_ratio_str}')
TRK_folder = os.path.join(inpath, f'TRK{trk_type}_{space_type}_farun')
#TRK_folder = os.path.join('/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd', 'TRK_trix_MDT')
# TRK_folder = os.path.join(inpath, f'TRK_MDT_real_testtemp{fixed_str}{folder_ratio_str}')

label_folder = os.path.join(inpath, 'DWI')
# trkpaths = glob.glob(os.path.join(TRK_folder, '*trk'))
excel_folder = os.path.join(outpath, f'Excels_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{trk_type}')

mkcdir(excel_folder, sftp)

if not remote and not os.path.exists(TRK_folder):
    raise Exception(f'cannot find TRK folder at {TRK_folder}')

# Initializing dictionaries to be filled
stream_point = {}
stream = {}
groupstreamlines = {}
groupLines = {}
groupPoints = {}
group_qb = {}
group_clusters = {}
groups_subjects = {}

str_identifier = f'*{space_type}{ratio_str}'

_, _, index_to_struct, _ = atlas_converter(atlas_legends, sftp=sftp)
labelmask, labelaffine, labeloutpath, index_to_struct = getlabeltypemask(label_folder, 'MDT', atlas_legends,
                                                                         labeltype=labeltype, verbose=verbose,
                                                                         sftp=sftp)

print(f'Beginning streamline_prep run from {TRK_folder} for folder {excel_folder}')


trkpath, exists = gettrkpath(TRK_folder, subject, str_identifier, pruned=False, verbose=verbose, sftp=sftp)

if not exists:
    txt = f'Could not find subject {subject} at {TRK_folder} with {str_identifier}'
    warnings.warn(txt)

M_xlsxpath = os.path.join(excel_folder, subject + "_connectomes.xlsx")
grouping_xlsxpath = os.path.join(excel_folder, subject + "_grouping.xlsx")

_, exists = check_files([M_xlsxpath, grouping_xlsxpath], sftp=sftp)
if np.all(exists) and not overwrite:
    if verbose:
        print(f'Found written file for subject {subject} at {M_xlsxpath} and {grouping_xlsxpath}')
else:
    t1 = time()

    """
    ##### Alignment checker
    from DTC.tract_manager.tract_handler import reducetractnumber
    tempfilepath = trkpath.replace('.trk','_ratio_100.trk')
    if not os.path.exists(tempfilepath):
        reducetractnumber(trkpath, tempfilepath, getdata=False, ratio=100,
                          return_affine=False, verbose=False)
    trkdatatemp = load_trk_remote(tempfilepath, 'same', sftp)
    header = trkdatatemp.space_attributes
    streamlines_world = transform_streamlines(trkdatatemp.streamlines, np.linalg.inv(labelaffine))
    from dipy.viz import window, actor
    from DTC.visualization_tools.tract_visualize import show_bundles, setup_view_legacy
    import nibabel as nib
    lut_cmap = actor.colormap_lookup_table(
        scale_range=(0.05, 0.3))
    scene = setup_view_legacy(nib.streamlines.ArraySequence(trkdatatemp.streamlines), colors=lut_cmap,
                       ref=labeloutpath, world_coords=True,
                       objectvals=[None], colorbar=True, record=None, scene=None, interactive=True)

    #The alignment is correct if the original streamlines are aligned with the references or labels when world_coords=True,
    #or when the transformed streamliens are aligned with the references or labels when world_coords = False
    """

    trkdata = load_trk_remote(trkpath, 'same', sftp)

    if verbose:
        print(f"Time taken for loading the trk file {trkpath} set was {str((- t1 + time()) / 60)} minutes")
    t2 = time()
    header = trkdata.space_attributes

    streamlines_world = transform_streamlines(trkdata.streamlines, np.linalg.inv(labelaffine))

    if function_processes == 1:

        M, _, _, _, grouping = connectivity_matrix_custom(streamlines_world, np.eye(4), labelmask,
                                                          inclusive=inclusive, symmetric=symmetric,
                                                          return_mapping=True,
                                                          mapping_as_streamlines=False, reference_weighting=None,
                                                          volume_weighting=False)
    else:
        M, _, _, _, grouping = connectivity_matrix_func(streamlines_world, np.eye(4), labelmask,
                                                        inclusive=inclusive,
                                                        symmetric=symmetric, return_mapping=True,
                                                        mapping_as_streamlines=False, reference_weighting=None,
                                                        volume_weighting=False,
                                                        function_processes=function_processes, verbose=False)

    M_grouping_excel_save(M, grouping, M_xlsxpath, grouping_xlsxpath, index_to_struct, verbose=False, sftp=sftp)

    del (trkdata)
    if verbose:
        print(f"Time taken for creating this connectome was set at {str((- t2 + time()) / 60)} minutes")
    if checkfile_exists_remote(grouping_xlsxpath, sftp):
        if verbose:
            print(f'Saved grouping for subject {subject} at {grouping_xlsxpath}')
    # grouping = extract_grouping(grouping_xlsxpath, index_to_struct, np.shape(M), verbose=verbose)
    else:
        raise Exception(f'saving of the excel at {grouping_xlsxpath} did not work')
