
import numpy as np
from dipy.segment.clustering import QuickBundles
from dipy.io.streamline import load_trk, save_trk
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.featurespeed import ResampleFeature
from dipy.io.image import load_nifti
import warnings

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import transform_streamlines
import os, glob, shutil, sys, socket, errno
import pickle
from DTC.nifti_handlers.nifti_handler import getlabeltypemask
from DTC.file_manager.file_tools import mkcdir, check_files
from DTC.tract_manager.tract_handler import ratio_to_str, gettrkpath
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from DTC.tract_manager.tract_save import save_trk_header
from DTC.diff_handlers.connectome_handlers.excel_management import M_grouping_excel_save, extract_grouping
from DTC.file_manager.argument_tools import parse_arguments_function
from DTC.diff_handlers.connectome_handlers.connectome_handler import connectivity_matrix_func
from dipy.tracking.utils import length
from dipy.viz import window, actor
from time import sleep
from dipy.segment.clustering import ClusterCentroid
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import connectivity_matrix
from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from DTC.tract_manager.tract_save import convert_trk_to_tck,convert_tck_to_trk


def get_grouping(grouping_xlsx):
    print('not done yet')


def get_diff_ref(label_folder, subject, ref):
    diff_path = os.path.join(label_folder,f'{subject}_{ref}_to_MDT.nii.gz')
    if os.path.exists(diff_path):
        return diff_path
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), diff_path)

"""
'1 Cerebellum-Cortex_Right---Cerebellum-Cortex_Left 9 1 with weight of 3053.5005\n'
    '2 inferiortemporal_Left---Cerebellum-Cortex_Left 24 1 with weight of 463.1322\n'
    '3 inferiortemporal_Right---inferiorparietal_Right 58 57 with weight of 435.9886\n'
    '4 middletemporal_Right---inferiorparietal_Right 64 57 with weight of 434.9106\n'
    '5 fusiform_Left---Cerebellum-Cortex_Left 22 1 with weight of 402.0991\n'
"""
#target_tuple = (9, 1)
#target_tuple = (64, 57)
#target_tuple = (24, 1)
#target_tuple = (58, 57)
#target_tuple = (64, 57)
#target_tuple = (22, 1)
#target_tuple = (30, 50) #The connectomes to check up on and create groupings clusters for

#target_tuple = (39,32)

#set parameter
num_points1 = 50
distance1 = 1
#group cluster parameter
num_points2 = 50
distance2 = 2

ratio = 1
project = 'AD_Decode'
skip_subjects = True
write_streamlines = True
allow_preprun = False
verbose=True
picklesave=True
overwrite=False
inclusive = False
symmetric = True
write_stats = True
write_txt = True
constrain_groups = True


#target_tuples = [(9, 1), (79, 9), (43, 39), (44, 1), (44, 9), (77, 43), (78,64), (51,9), (74,64), (64,9)]
#target_tuples = [(9, 1), (79, 9), (43, 39)] #superiorfrontal left to precentral left
target_tuples = [(40,25),(44,40),(40,6)]
target_tuples = [(74, 2),(74,40)]
#target_tuples = [(74, 40),(74, 2),(74, 44),(74, 25),(74, 50),(40,6)]
target_tuples = [(78,74),(74,40),(44,40),(40,2),(74,2)]
#target_tuples = [(77, 43)] #superior frontal right to superior frontal left
target_tuples = [(74,2)]
target_tuples = [(40,2),(74,10),(44,40),(78,74),(74,40)]
target_tuples = [(74,10)]
#target_tuples = np.arange(85,127)

extraction = 'edge' #either 'edge' or 'node', defaults to 'edge' (two region connection), node extracts only the single region
csv = 'gm' #'wm' or 'gm' so far

labeltype = 'lrordered'
#reference_img refers to statistical values that we want to compare to the streamlines, say fa, rd, etc
references = ['fa', 'md', 'rd', 'ad', 'b0']
references = ['fa', 'md']
references = ['fa', 'md', 'ln', 'rd', 'ad']

if inclusive:
    inclusive_str = '_inclusive'
else:
    inclusive_str = '_non_inclusive'

computer_name = socket.gethostname()

samos = False
if 'samos' in computer_name:
    mainpath = '/mnt/paros_MRI/jacques/'
    ROI_legends = "/mnt/paros_MRI/jacques/atlases/IITmean_RPI/IITmean_RPI_index.xlsx"
elif 'santorini' in computer_name or 'hydra' in computer_name:
    #mainpath = '/Volumes/Data/Badea/Lab/human/'
    mainpath = '/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/'
    ROI_legends = "/Volumes/Data/Badea/ADdecode.01/Analysis/atlases/IITmean_RPI/IITmean_RPI_index.xlsx"
    #ref_MDT_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images/'
    ref_MDT_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/'
elif 'blade' in computer_name:
    mainpath = '/mnt/munin6/Badea/Lab/human/'
    ROI_legends = "/mnt/munin6/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx"
    #ref_MDT_folder = '/mnt/munin6/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images/'
    ref_MDT_folder = '/mnt/munin6/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/'
else:
    raise Exception('No other computer name yet')

ref_img_path = os.path.join(ref_MDT_folder,'MDT_fa.nii.gz')
# Setting identification parameters for ratio, labeling type, etc
ratio_str = ratio_to_str(ratio,spec_all=False)
print(ratio_str)
if ratio_str == '_all':
    folder_ratio_str = ''
else:
    folder_ratio_str = ratio_str.replace('_ratio', '')

streamline_type = 'mrtrix'
stepsize = 2
if streamline_type == 'mrtrix':
    prune = False
    trkroi = [""]
else:
    prune = True
    trkroi = ["wholebrain"]

str_identifier = get_str_identifier(stepsize, ratio, trkroi, type=streamline_type)
str_identifier = '_MDT'+ str_identifier

labeltype = 'lrordered'

function_processes = parse_arguments_function(sys.argv)
print(f'there are {function_processes} function processes')

if project=='AD_Decode':
    mainpath=os.path.join(mainpath,project,'Analysis')
else:
    mainpath = os.path.join(mainpath, project)

kos=False

mainpath = '/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/'
#TRK_folder = os.path.join(mainpath, f'TRK_MPCA_MDT_fixed{folder_ratio_str}')
TRK_folder = os.path.join(mainpath, 'TRK_MDT'+ratio_str)
TRK_folder = '/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/TRK_MDT_act/'
TCK_folder = '/Volumes/Shared Folder/newJetStor/paros/paros_WORK/jacques/AD_Decode/TCK_MDT_act/'

if not kos:
    mainpath = '/Volumes/Data/Badea/Lab/human/AD_Decode/'
    TCK_folder = '/Volumes/Data/Badea/Lab/human/AD_Decode_trk_transfer/TCK_MDT'
    TRK_folder = '/Volumes/Data/Badea/Lab/human/AD_Decode_trk_transfer/TRK_MDT'

label_folder = os.path.join(mainpath, 'DWI')
if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'


trkpaths = glob.glob(os.path.join(TRK_folder, '*trk'))
streamline_tupled_folders = os.path.join(mainpath, f'Streamlines_tupled_MDT_act{inclusive_str}{symmetric_str}{folder_ratio_str}')
streamline_tupled_folders = os.path.join(mainpath, f'Streamlines_tupled_MDT_mrtrix_act{inclusive_str}{symmetric_str}{folder_ratio_str}')
streamline_tupled_folders = os.path.join(mainpath, f'Streamlines_tupled_MDT_mrtrix_act{inclusive_str}{symmetric_str}{folder_ratio_str}')


#excel_folder = os.path.join(mainpath, f'Excels_MDT{inclusive_str}{symmetric_str}{folder_ratio_str}')
excel_folder = os.path.join('/Volumes/Data/Badea/Lab/mouse/mrtrix_ad_decode/perm_files/')

mkcdir([streamline_tupled_folders,TCK_folder])

trk_files = glob.glob(os.path.join(TRK_folder,'*trk'))
subjects = [os.path.basename(trk_file).split('_')[0] for trk_file in trk_files]
subjects = sorted(subjects)

#removed_list = ['S02230', 'S02654', 'S02490', 'S02523', 'S02745']
removed_list = ['S02390','S02765']
for remove in removed_list:
    if remove in subjects:
        subjects.remove(remove)

if not os.path.exists(TRK_folder):
    raise Exception(f'cannot find TRK folder at {TRK_folder}')

#Initializing dictionaries to be filled
stream_point = {}
stream = {}
groupstreamlines={}
groupstreamlines_orig={}
groupLines = {}
groupPoints = {}
group_qb = {}
group_clusters = {}
groups_subjects = {}


if project == 'APOE':
    raise Exception('not implemented')


feature1 = ResampleFeature(nb_points=num_points1)
metric1 = AveragePointwiseEuclideanMetric(feature=feature1)

feature2 = ResampleFeature(nb_points=num_points2)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)

overwrite=False

mrtrix_connectomes = True

for subject in subjects:

    #print(f'Starting the run for {index_to_struct[target_tuple[0]]} to {index_to_struct[target_tuple[1]]}')
    print(f'Starting the run for subject {subject}')

    trkdata = None

    for target_tuple in target_tuples:

        trkpath, exists = gettrkpath(TRK_folder, subject, str_identifier, pruned=False, verbose=True)
        _, _, index_to_struct, _ = atlas_converter(ROI_legends)

        if extraction == 'edge':
            streamline_tuple_folder_name = f'{index_to_struct[target_tuple[0]]}_to_{index_to_struct[target_tuple[1]]}_MDT{ratio_str}'
        if extraction == 'node':
            streamline_tuple_folder_name = f'{index_to_struct[target_tuple]}_MDT{ratio_str}'

        streamline_tuple_folder_path = os.path.join(streamline_tupled_folders,streamline_tuple_folder_name)
        mkcdir(streamline_tuple_folder_path)
        streamline_tupled_path = os.path.join(streamline_tupled_folders, streamline_tuple_folder_name, f'{subject}_streamlines.trk')

        if os.path.exists(streamline_tupled_path) and not overwrite:
            if extraction == 'edge':
                print(f'Already did {subject} for tuple {index_to_struct[target_tuple[0]]} to {index_to_struct[target_tuple[1]]}')
            if extraction == 'node':
                print(f'Already did {subject} for tuple {index_to_struct[target_tuple]}')
            continue

        if not mrtrix_connectomes:
            if trkdata is None:
                trkdata = load_trk(trkpath, 'same',bbox_valid_check=False)
            header = trkdata.space_attributes
            M_xlsxpath = os.path.join(excel_folder, subject + str_identifier + "_connectomes.xlsx")
            grouping_xlsxpath = os.path.join(excel_folder, subject + "_grouping.xlsx")

            if os.path.exists(grouping_xlsxpath):
                grouping = extract_grouping(grouping_xlsxpath, index_to_struct, None, verbose=verbose)
            else:
                txt = f'Run streamline_average_prep first for this subject {subject}'
                raise Exception(txt)

            target_streamlines_list = grouping[target_tuple[0], target_tuple[1]]

            if np.size(target_streamlines_list) == 0:
                txt = f'Did not have any streamlines for {index_to_struct[target_tuple[0]]} to {index_to_struct[target_tuple[1]]} for subject {subject}'
                warnings.warn(txt)
                continue
            target_streamlines = trkdata.streamlines[np.array(target_streamlines_list)]
            save_trk_header(filepath=streamline_tupled_path, streamlines=target_streamlines, header=header,
                            affine=np.eye(4), verbose=verbose)
        else:
            tck_path = os.path.join(TCK_folder,os.path.basename(trkpath).replace('.trk','.tck'))

            if not os.path.exists(trkpath):
                print(f'Could not find file for subject {subject}')
                continue

            if csv == 'gm':
                assignments_parcels_csv2 = os.path.join(excel_folder, subject+f'_assignments_con_plain_act.csv')
            elif csv == 'wm':
                assignments_parcels_csv2 = os.path.join(excel_folder, subject+f'_assignments_wm_con_plain_act.csv')

            streamline_tupled_tck_path = streamline_tupled_path.replace('.trk','.tck')
            if not os.path.exists(tck_path):
                convert_trk_to_tck(trkpath, tck_path)

            if extraction == 'edge':
                cmd = f'connectome2tck {tck_path} {assignments_parcels_csv2} {streamline_tupled_tck_path} -nodes ' \
                    f'{target_tuple[0]},{target_tuple[1]} -exclusive -files single'.replace("Shared ", "Shared\\ ")
                os.system(cmd)
            if extraction == 'node':
                cmd = f'connectome2tck {tck_path} {assignments_parcels_csv2} ' \
                    f'{os.path.join(streamline_tupled_folders,f"{subject}_node")} ' \
                    f'-files per_node'.replace("Shared ", "Shared\\ ")
                if not os.path.exists(streamline_tupled_tck_path):
                    os.system(cmd)
                node_files = glob.glob(os.path.join(streamline_tupled_folders,f'{subject}_node*'))
                for node_file in node_files:
                    node = int(node_file.split('_node')[1].split('.tck')[0]) + 84
                    streamline_tuple_folder_name = f'{index_to_struct[node]}_MDT{ratio_str}'
                    streamline_tuple_folder_path = os.path.join(streamline_tupled_folders, streamline_tuple_folder_name)
                    mkcdir(streamline_tuple_folder_path)
                    #shutil.move(node_file,streamline_tupled_path)
                    shutil.copy(node_file,streamline_tupled_path)

            convert_tck_to_trk(streamline_tupled_tck_path,streamline_tupled_path,ref_img_path)
            os.remove(streamline_tupled_tck_path)
