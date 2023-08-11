import numpy as np
from dipy.segment.clustering import QuickBundles
from dipy.io.streamline import load_trk, save_trk
from dipy.segment.metric import ResampleFeature, AveragePointwiseEuclideanMetric,mdf
from dipy.io.image import load_nifti
import warnings

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import transform_streamlines
import os, glob
import pickle
from DTC.nifti_handlers.nifti_handler import getlabeltypemask, get_diff_ref
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from DTC.tract_manager.tract_handler import ratio_to_str, gettrkpath, gettrkpath_testsftp
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
import socket
from DTC.tract_manager.tract_save import save_trk_header
from DTC.diff_handlers.connectome_handlers.excel_management import M_grouping_excel_save, extract_grouping
import sys
from DTC.file_manager.argument_tools import parse_arguments_function
from DTC.diff_handlers.connectome_handlers.connectome_handler import connectivity_matrix_func, _to_voxel_coordinates_warning, retweak_points, retweak_point_set
from dipy.tracking.utils import length
import getpass
import random
from DTC.file_manager.computer_nav import get_mainpaths, get_atlas, make_temppath, checkfile_exists_remote, load_trk_remote, \
    remote_pickle, pickledump_remote, load_nifti_remote, remove_remote, glob_remote
import pandas as pd
from dipy.viz import window, actor
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view_legacy
import nibabel as nib


def get_grouping(grouping_xlsx):
    print('not done yet')

#set parameter
num_points1 = 50
distance1 = 1
#group cluster parameter
num_points2 = 50
distance2 = 2

ratio = 1
#projects = ['AD_Decode', 'AMD', 'APOE']
project = 'APOE'

remote = True
remote = False

# inpath, outpath, atlas_folder, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
inpath = '/Volumes/dusom_abadea_nas1/munin_js/APOE_series'
outpath = '/Volumes/dusom_mousebrains/All_Staff/Analysis/APOE'
atlas_folder = '/Volumes/Data/Badea/Lab/atlases'

sftp=None

if project == 'AMD' or project == 'AD_Decode':
    atlas_legends = get_atlas(atlas_folder, 'IIT')
if project == 'APOE':
    atlas_legends = get_atlas(atlas_folder, 'CHASSSYMM3')

#diff_preprocessed = os.path.join(inpath, "DWI")
ref_MDT_folder = os.path.join(inpath, "reg_trix_MDT")
#ref_MDT_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/AMD/DWI_trix_MDT/'
#ref_MDT_folder = os.path.join('/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd', 'DWI_trix_MDT')

skip_subjects = True
write_streamlines = True
allow_preprun = False
verbose=True
picklesave=True
inclusive = False
symmetric = True
write_stats = True
write_txt = True
constrain_groups = True
fixed = False
overwrite=False

labeltype = 'lrordered'
#reference_img refers to statistical values that we want to compare to the streamlines, say fa, rd, etc

#references = ['fa', 'md', 'ln', 'rd', 'ad']
references = ['fa', 'md', 'rd', 'md', 'ad', 'ln']
references = ['ln']

space_type = 'MDT'
other_param = ''

# Setting identification parameters for ratio, labeling type, etc

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

ratio_str = ratio_to_str(ratio)
print(ratio_str)
if ratio_str == '_all':
    folder_ratio_str = ''
else:
    folder_ratio_str = ratio_str.replace('_ratio', '')

str_identifier = f'_stepsize_2{ratio_str}_wholebrain_pruned'
labeltype = 'lrordered'

function_processes = parse_arguments_function(sys.argv)
print(f'there are {function_processes} function processes')

mkcdir(outpath)
#TRK_folder = os.path.join(inpath, f'TRK_MPCA_MDT{fixed_str}{folder_ratio_str}')
#TRK_folder = os.path.join(inpath, f'TRK_rigidaff{fixed_str}{folder_ratio_str}')
#TRK_folder= os.path.join(inpath, f'TRK_trix_farun')
TRK_folder = os.path.join(inpath, f'TRK_{space_type}{fixed_str}{folder_ratio_str}')

label_folder = os.path.join(inpath, 'DWI')
if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'

"""
pickle_folder = os.path.join(outpath, f'Pickle_MDT{inclusive_str}{symmetric_str}{folder_ratio_str}')
centroid_folder = os.path.join(outpath, f'Centroids_MDT{inclusive_str}{symmetric_str}{folder_ratio_str}')
stats_folder = os.path.join(outpath, f'Statistics_MDT{inclusive_str}{symmetric_str}{folder_ratio_str}')
connectome_folder = os.path.join(outpath, f'Excels_MDT{inclusive_str}{symmetric_str}{folder_ratio_str}')
"""


space_type = 'MDT'
target_tuples = [(121, 123), (287, 45), (28, 9), (62, 1)]

#target_tuples = [(62, 28)]
#target_tuples = [(62, 1)]
#target_tuples = [(36,70),(36,28),(70,62),(36,62),(70,28)]
#target_tuples = [(36,62),(70,28)]


pickle_folder = os.path.join(outpath, f'Pickle_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')
#centroid_folder = os.path.join(outpath, f'Centroids_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}')
centroid_folder = os.path.join(outpath, f'Centroids_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')
stats_folder = os.path.join(outpath, f'Statistics_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}')
stats_folder = os.path.join(outpath, f'Statistics_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}')
stats_folder = os.path.join(outpath, f'Statistics_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')
connectome_folder = os.path.join(outpath, f'Excels_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}')
connectome_folder = os.path.join(outpath, f'Excels_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')

figures_folder = os.path.join(outpath, f'Figures_subj_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')

mkcdir([pickle_folder, centroid_folder, stats_folder, connectome_folder, figures_folder],sftp)
if not remote and not os.path.exists(TRK_folder):
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

if project == 'AMD':
    groups_subjects['testing'] = ['H28182']
    groups_subjects['testing2'] = ['H29410']
    groups_subjects['Initial AMD'] = ['H27640', 'H27778', 'H29020', 'H26637', 'H27680', 'H26765', 'H27017', 'H26880', 'H28308', 'H28433', 'H28338', 'H26660', 'H28809', 'H27610', 'H26745', 'H27111', 'H26974', 'H27391', 'H28748', 'H29025', 'H29013', 'H27381', 'H26958', 'H28662', 'H26578', 'H28698', 'H27495', 'H28861', 'H28115', 'H28437', 'H26850', 'H28532', 'H28377', 'H28463', 'H26890', 'H28373', 'H28857', 'H27164', 'H27982']
    groups_subjects['Paired 2-YR AMD'] = ['H22825', 'H21850', 'H29225', 'H29304', 'H29060', 'H23210', 'H21836', 'H29618', 'H22644', 'H22574', 'H22369', 'H29627', 'H29056', 'H22536', 'H23143', 'H22320', 'H22898', 'H22864', 'H29264', 'H22683']
    groups_subjects['Initial Control'] = ['H26949', 'H27852', 'H28029', 'H26966', 'H27126', 'H28068', 'H29161', 'H28955', 'H26862', 'H28262', 'H28856', 'H27842', 'H27246', 'H27869', 'H27999', 'H29127', 'H28325', 'H26841', 'H29044', 'H27719', 'H27100', 'H29254', 'H27682', 'H29002', 'H29089', 'H29242', 'H27488', 'H27841', 'H28820', 'H27163', 'H28869', 'H28208', 'H27686']
    groups_subjects['Paired 2-YR Control'] = ['H29403', 'H22102', 'H29502', 'H22276', 'H29878', 'H29410', 'H22331', 'H22368', 'H21729', 'H29556', 'H21956', 'H22140', 'H23309', 'H22101', 'H23157', 'H21593', 'H21990', 'H22228', 'H23028', 'H21915']
    groups_subjects['Paired Initial Control'] = ['H27852', 'H28029', 'H26966', 'H27126', 'H29161', 'H28955', 'H26862', 'H27842', 'H27999', 'H28325', 'H26841', 'H27719', 'H27100', 'H27682', 'H29002', 'H27488', 'H27841', 'H28820', 'H28208', 'H27686']
    if mrtrix: #removing H27999
        groups_subjects['Paired Initial Control'] = ['H27852', 'H28029', 'H26966', 'H27126', 'H29161', 'H28955',
                                                     'H26862', 'H27842', 'H28325', 'H26841', 'H27719',
                                                     'H27100', 'H27682', 'H29002', 'H27488', 'H27841', 'H28820',
                                                     'H28208', 'H27686']

    groups_subjects['Paired Initial AMD'] = ['H29020', 'H26637', 'H26765', 'H28308', 'H28433', 'H26660', 'H28182', 'H27111', 'H27391', 'H28748', 'H28662', 'H26578', 'H28698', 'H27495', 'H28861', 'H28115', 'H28377', 'H26890', 'H28373', 'H27164']

    #groups to go through
    #groups_all = ['Paired 2-YR AMD','Initial AMD','Initial Control','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']
    groups = ['Paired 2-YR AMD','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']
    #groups = ['Paired 2-YR Control']
    #groups = ['Paired 2-YR AMD', 'Paired 2-YR Control', 'Paired Initial Control']
    groups = ['Paired Initial Control']
    #groups = ['Paired Initial AMD']
    #groups = ['testing']
    #groups = ['testing']
    #groups = ['Paired Initial Control','Paired Initial AMD']
    #groups = ['Paired 2-YR AMD']

    if groups == ['Paired 2-YR AMD','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']:
        allgroups = 'allsubj'
    else:
        allgroups = '-'.join(groups).replace(' ','_')
    #groups = ['testing', 'testing2']
    #groups = ['Paired 2-YR AMD']
    #groups = ['Paired 2-YR AMD']

    #groups = ['Paired 2-YR AMD','Paired 2-YR Control']
    #groups = ['testing']
    str_identifier = '_MDT' + folder_ratio_str
    str_identifier = f'*'

    #for group in groups:
    #    random.shuffle(groups_subjects[group])

    #target_tuples = [(9, 1), (24, 1), (76, 42), (76, 64), (77, 9), (43, 9)]
    #target_tuples = [(9, 1),(24, 1), (76, 42), (76, 64), (77, 9), (43, 9)]
    #target_tuples = [(24, 1)]
    #target_tuples = [(76, 42)]
    #target_tuples = [(76, 64), (77, 9), (43, 9)]
    #target_tuples = [(76, 42)]
    #target_tuples = [(76, 64), (77, 9), (43, 9)]

    #Ctrl vs AMD: 1st visit
    #target_tuples = [(62, 28), (56, 45), (77, 43), (58, 45), (79, 45), (56, 50)]
    #target_tuples = [(62, 28),(58,45),(28,9), (62, 1)]

    #Ctrl vs AMD: 2nd visit
    #target_tuples = [(28, 9), (62, 1), (28, 1), (62, 9), (22, 9), (56, 1)]
    #Ctrl vs AMD: visit change (2-1)
    #target_tuples = [(77, 43), (76, 43), (61, 29), (63, 27), (73, 43), (53, 43)

    #Ctrl vs AMD: 1st visit => VBA analysis
    #target_tuples = [(27,29), (61,63)]
    #Ctrl vs AMD: 2nd visit
    #target_tuples =[(30, 16), (24, 16)]
    #Ctrl vs AMD: visit change (2-1)
    #target_tuples = [(28, 31), (28, 22),(22, 31)]

    #TN-PCA
    #target_tuples = [(62, 28), (56, 45), (77, 43), (58, 45), (79, 45), (56, 50), (28, 9), (62, 1), (28, 1), (62, 9), (22, 9), (56, 1),(77, 43), (76, 43), (61, 29), (63, 27), (73, 43), (53, 43)]
    #VBA
    #target_tuples = [(27,29), (61,63),(30, 16), (24, 16),(28, 31), (28, 22),(22, 31)]
    #TN-PCA / VBA combination
    #target_tuples = [(62, 28), (58, 45), (28, 9), (62, 1), (77, 43), (61, 29)]
    #target_tuples = [(62, 1), (77, 43), (61, 29)]

    removed_list = []

elif project == 'APOE':
    groups_subjects['all'] = ['N57442', 'N57496', 'N57500', 'N57580', 'N57709', 'N58219', 'N58221', 'N58222', 'N58223', 'N58224', 'N58226', 'N58228', 'N58229', 'N58230', 'N58231', 'N58232', 'N58302', 'N58303', 'N58305', 'N58309', 'N58310', 'N58344', 'N58346', 'N58350', 'N58355', 'N58359', 'N58361', 'N58394', 'N58396', 'N58398', 'N58400', 'N58402', 'N58404', 'N58406', 'N58408', 'N58477', 'N58500', 'N58510', 'N58512', 'N58514', 'N58516', 'N58604', 'N58606', 'N58608', 'N58610', 'N58611', 'N58612', 'N58613', 'N58633', 'N58634', 'N58635', 'N58636', 'N58649', 'N58650', 'N58651', 'N58653', 'N58654', 'N58655', 'N58706', 'N58708', 'N58712', 'N58714', 'N58732', 'N58733', 'N58734', 'N58735', 'N58740', 'N58742', 'N58745', 'N58747', 'N58749', 'N58751', 'N58779', 'N58780', 'N58784', 'N58788', 'N58790', 'N58792', 'N58794', 'N58813', 'N58815', 'N58819', 'N58821', 'N58829', 'N58831', 'N58851', 'N58853', 'N58855', 'N58857', 'N58859', 'N58861', 'N58877', 'N58879', 'N58881', 'N58883', 'N58885', 'N58887', 'N58889', 'N58906', 'N58909', 'N58913', 'N58915', 'N58917', 'N58919', 'N58935', 'N58941', 'N58946', 'N58948', 'N58954', 'N58995', 'N58997', 'N58999', 'N59003', 'N59005', 'N59010', 'N59022', 'N59026', 'N59033', 'N59035', 'N59039', 'N59065', 'N59066', 'N59076', 'N59078', 'N59080', 'N59097', 'N59099', 'N59109', 'N59116', 'N59118', 'N59120', 'N59136', 'N59141', 'N60056', 'N60060', 'N60062', 'N60064', 'N60068', 'N60070', 'N60072', 'N60088', 'N60092', 'N60093', 'N60095', 'N60101', 'N60103', 'N60127', 'N60129', 'N60133', 'N60137', 'N60139', 'N60157', 'N60159', 'N60163', 'N60167', 'N60188', 'N60190', 'N60192', 'N60194', 'N60200', 'N60225', 'N60229', 'N60231']
    subjects = []
    #str_identifier = f'*{space_type}{ratio_str}'
    str_identifier = '*smallerTracks2mill'
    groups = ['all']
    removed_list = []
    for group in groups:
        subjects = subjects + groups_subjects[group]

    target_tuples = [(121, 123), (287, 279), (121, 8), (230, 51), (121, 141), (230, 145), (121, 91), (57, 217), (217, 29), (287, 51), (287, 287), (230, 299), (230, 29), (89, 66), (230, 279), (51, 215), (287, 322), (217, 287), (89, 245), (287, 266), (217, 266), (121, 146), (121, 70), (217, 22), (217, 299), (287, 10), (287, 98), (287, 106), (89, 221), (287, 2)]
    target_tuples = [(121, 8), (230, 51), (121, 141), (230, 145), (121, 91), (57, 217), (217, 29), (287, 51), (287, 287), (230, 299), (230, 29), (89, 66), (230, 279), (51, 215), (287, 322), (217, 287), (89, 245), (287, 266), (217, 266), (121, 146), (121, 70), (217, 22), (217, 299), (287, 10), (287, 98), (287, 106), (89, 221), (287, 2)]
    target_tuples = [(121, 8)]
    if groups == ['all']:
        allgroups = 'allsubj'
    else:
        allgroups = '-'.join(groups).replace(' ','_')
else:
    txt = f'{project} not implemented'
    raise Exception(txt)

for group in groups:
    for remove in removed_list:
        if remove in groups_subjects[group]:
            groups_subjects[group].remove(remove)

if constrain_groups:
    group_sizes = []
    for group in groups:
        #group_sizes[group] = np.size(groups_subjects[group])
        group_sizes.append(np.size(groups_subjects[group]))
    group_min = np.min(group_sizes)
    for group in groups:
        groups_subjects[group] = groups_subjects[group][:group_min]
    print(group_sizes)


feature1 = ResampleFeature(nb_points=num_points1)
metric1 = AveragePointwiseEuclideanMetric(feature=feature1)

feature2 = ResampleFeature(nb_points=num_points2)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)

save_subj_grouping = True
print('hi')

for group in groups:
    subjects = groups_subjects[group]
    for subject in subjects:
        grouping_xlsxpath = os.path.join(connectome_folder, subject + str_identifier + "_grouping.xlsx")
        if not checkfile_exists_remote(grouping_xlsxpath, sftp) and not allow_preprun:
            print(subject)

atlas_legends = '/Users/jas/jacques/atlases/CHASSSYMM3AtlasLegends.xlsx'
labelmask, labelaffine, labeloutpath, index_to_struct = getlabeltypemask(label_folder, 'MDT',
                                                                         atlas_legends,
                                                                         labeltype=labeltype,
                                                                         verbose=verbose, sftp=sftp)

for target_tuple in target_tuples:

    for group in groups:
        groupstreamlines[group] = []
        groupstreamlines_orig[group] = []
        for ref in references:
            groupLines[group, ref] = {}
            groupPoints[group, ref] = {}

    _, _, index_to_struct, _ = atlas_converter(atlas_legends)
    print(f'Starting the run for {index_to_struct[target_tuple[0]]} to {index_to_struct[target_tuple[1]]}')
    stats_path = os.path.join(stats_folder, f'{index_to_struct[target_tuple[0]]}_to_{index_to_struct[target_tuple[1]]}_{allgroups}.xlsx')
    #This is the streamline counter for the stats path. Important!
    sl_all = 1

    if sftp is None:
        stats_path_temp = stats_path
    else:
        stats_path_temp = make_temppath(stats_path)

    if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
        import xlsxwriter

        workbook = xlsxwriter.Workbook(stats_path_temp)
        worksheet = workbook.add_worksheet()
        l = 1
        for ref in references:
            """
            worksheet.write(0,l, ref + ' mean')
            worksheet.write(0,l+1, ref + ' min')
            worksheet.write(0,l+2, ref + ' max')
            worksheet.write(0,l+3, ref + ' std')
            l=l+4
            """
            worksheet.write(0, l, ref)
            l += 1

    for group in groups:
        print(f'Going through group {group}')

        group_str = group.replace(' ', '_')
        group_connection_str = group_str + '_' + space_type + ratio_str + '_' + index_to_struct[target_tuple[0]] + '_to_' + index_to_struct[target_tuple[1]]
        centroid_file_path = os.path.join(centroid_folder, group_connection_str + '_centroid.py')
        streamline_file_path = os.path.join(centroid_folder, group_connection_str + '_streamlines.trk')
        excel_SID_path = os.path.join(centroid_folder, group_connection_str + '_streamlineID_subj.xlsx')

        SID = 0
        subjects_ID_list = []
        streams_ID_list = []

        grouping_files = {}
        exists=True

        for ref in references:
            grouping_files[ref,'lines']=(os.path.join(centroid_folder, group_connection_str + '_' + ref + '_lines.py'))
            grouping_files[ref, 'points'] = (os.path.join(centroid_folder, group_connection_str + '_' + ref + '_points.py'))
            list_files, exists = check_files(grouping_files, sftp)

        #(not checkfile_exists_remote(stats_path, sftp) and write_stats)
        if not checkfile_exists_remote(centroid_file_path, sftp) or not np.all(exists) or \
                (not checkfile_exists_remote(streamline_file_path, sftp) and write_streamlines) \
                or not os.path.exists(excel_SID_path) or overwrite:
            subjects = groups_subjects[group]
            subj = 1
            for subject in subjects:

                trkpath, exists = gettrkpath(TRK_folder, subject, str_identifier, pruned=False, verbose=verbose,
                                             sftp=sftp)

                if not exists:
                    txt = f'Could not find subject {subject} at {TRK_folder} with {str_identifier}'
                    warnings.warn(txt)
                    continue
                #streamlines, header, _ = unload_trk(trkpath)
                #if np.shape(groupLines[group, ref])[0] != np.shape(groupstreamlines[group])[0]:
                #    raise Exception('happened from there')

                picklepath_connectome = os.path.join(pickle_folder, subject + '_connectomes.p')
                picklepath_grouping = os.path.join(pickle_folder, subject +  '_grouping.p')

                M_xlsxpath = os.path.join(connectome_folder, subject + "_connectomes.xlsx")
                grouping_xlsxpath = os.path.join(connectome_folder, subject + "_grouping.xlsx")
                #if os.path.exists(picklepath_grouping) and not overwrite:
                #    with open(picklepath_grouping, 'rb') as f:
                #        grouping = pickle.load(f)
                #if checkfile_exists_remote(picklepath_connectome, sftp):
                #    M = remote_pickle(picklepath_connectome, sftp=sftp)
                if checkfile_exists_remote(grouping_xlsxpath, sftp):
                    mygrouping_xlsxpath = glob_remote(grouping_xlsxpath)
                    if np.size(mygrouping_xlsxpath)==1:
                        grouping = extract_grouping(mygrouping_xlsxpath[0], index_to_struct, None, verbose=verbose, sftp=sftp)
                    else:
                        raise Exception
                else:
                    if allow_preprun:

                        trkdata = load_trk_remote(trkpath, 'same', sftp=sftp)
                        header = trkdata.space_attributes

                        streamlines_world = transform_streamlines(trkdata.streamlines, np.linalg.inv(labelaffine))

                        #M, grouping = connectivity_matrix_func(trkdata.streamlines, function_processes, labelmask,
                        #                                       symmetric=True, mapping_as_streamlines=False,
                        #                                       affine_streams=trkdata.space_attributes[0],
                        #                                       inclusive=inclusive)
                        M, grouping = connectivity_matrix_func(streamlines_world, np.eye(4), labelmask, inclusive=inclusive,
                                                 symmetric=symmetric, return_mapping=True, mapping_as_streamlines=False,
                                                 reference_weighting=None,
                                                 volume_weighting=False, verbose=False)
                        M_grouping_excel_save(M, grouping, M_xlsxpath, grouping_xlsxpath, index_to_struct,
                                              verbose=False)
                    else:
                        print(f'skipping subject {subject} for now as grouping file {grouping_xlsxpath} is not calculated. Best rerun it afterwards ^^')

                        raise Exception('Actually just stop it altogether and note the problem here')

                trkdata = load_trk_remote(trkpath, 'same', sftp=sftp)
                header = trkdata.space_attributes

                target_streamlines_list = grouping[target_tuple[0], target_tuple[1]]
                if np.size(target_streamlines_list) == 0:
                    txt = f'Did not have any streamlines for {index_to_struct[target_tuple[0]]} to {index_to_struct[target_tuple[1]]} for subject {subject}'
                    warnings.warn(txt)
                    continue
                target_streamlines = trkdata.streamlines[np.array(target_streamlines_list)]
                target_streamlines_set = set_number_of_points(target_streamlines, nb_points=num_points2)

                if save_subj_grouping:
                    recordsubj_path = os.path.join(figures_folder, f'{subject}_{group_connection_str}.png')
                    trksubj_path = os.path.join(figures_folder, f'{subject}_{group_connection_str}.trk')

                    lut_cmap = actor.colormap_lookup_table(scale_range=(0.05, 0.3))
                    scene = setup_view_legacy(target_streamlines_set, colors=lut_cmap,
                                              ref=labeloutpath, world_coords=True,
                                              objectvals=[None], colorbar=True, record=recordsubj_path, scene=None,
                                              interactive=False)

                    sg = lambda: (s for i, s in enumerate(target_streamlines_set))
                    save_trk_header(filepath=trksubj_path, streamlines=sg, header=header,
                                    affine=np.eye(4), verbose=verbose, sftp=sftp)
                    print(f'Saved Tracts of subject to {trksubj_path}')


                subjects_ID_list += [subject] * np.shape(target_streamlines_set)[0]
                streams_ID_list += list(np.arange(SID, SID + np.shape(target_streamlines_set)[0]))
                SID += np.shape(target_streamlines_set)[0]
                #del(target_streamlines, trkdata)
                #target_qb = QuickBundles(threshold=distance1, metric=metric1)

                if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
                    for i in np.arange(np.shape(target_streamlines_set)[0]):
                        worksheet.write(sl_all+i, 0, subject)
                l = 1

                for ref in references:
                    if ref != 'ln':
                        ref_img_path = get_diff_ref(ref_MDT_folder, subject, ref,sftp=sftp)
                        ref_data, ref_affine, _, _, _ = load_nifti_remote(ref_img_path, sftp=sftp)

                        from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
                        from collections import defaultdict, OrderedDict
                        from itertools import combinations, groupby

                        edges = np.ndarray(shape=(3, 0), dtype=int)
                        lin_T, offset = _mapping_to_voxel(trkdata.space_attributes[0])
                        stream_ref = []
                        stream_point_ref = []
                        from time import time

                        time1 = time()
                        target_streamlines_transformed = transform_streamlines(target_streamlines_set, np.linalg.inv(ref_affine))
                        testmode = False
                        for sl, _ in enumerate(target_streamlines_transformed):
                            # Convert streamline to voxel coordinates
                            #entire = _to_voxel_coordinates(target_streamlines_set[sl], lin_T, offset)
                            cur_row = sl_all + sl

                            voxel_coords = np.round(target_streamlines_transformed[sl]).astype(int)
                            voxel_coords_tweaked = retweak_points(voxel_coords, np.shape(ref_data))
                            ref_values = ref_data[voxel_coords_tweaked[:, 0], voxel_coords_tweaked[:, 1], voxel_coords_tweaked[:, 2]]

                            """
                            cur_row = sl_all + sl
                            entire = _to_voxel_coordinates_warning(target_streamlines_set[sl], lin_T, offset)
                            entire = retweak_points(entire, np.shape(ref_data))
                            i, j, k = entire.T
                            ref_values_orig = ref_data[i, j, k]
                            if np.mean(ref_values_orig) == 0:
                                print('too low a value for old method')
                                testmode=True
                            """

                            stream_point_ref.append(ref_values)
                            stream_ref.append(np.mean(ref_values))

                            if np.mean(ref_values) == 0:
                                print('too low a value for new method')
                                testmode=True

                            if testmode:
                                from DTC.tract_manager.tract_save import save_trk_header

                                small_streamlines_testzone = '/Users/jas/jacques/AMD_testing_zone/single_streamlines'
                                mkcdir(small_streamlines_testzone, sftp)
                                streamline_file_path = os.path.join(small_streamlines_testzone,
                                                                    f'{subject}_streamline_{sl}.trk')
                                # sg = lambda: (s for i, s in enumerate(trkobject[0]))
                                from dipy.tracking import streamline
                                streamlines = streamline.Streamlines([target_streamlines_set[sl]])
                                save_trk_header(filepath=streamline_file_path, streamlines=streamlines, header=header,
                                                affine=np.eye(4), verbose=True, sftp=sftp)
                                testmode=False
                            if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
                                worksheet.write(cur_row, l, np.mean(ref_values))

                        l = l + 1
                    else:
                        stream_ref = list(length(target_streamlines))
                        for sl, ref_val in enumerate(stream_ref):
                            cur_row = sl_all + sl
                            if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
                                worksheet.write(cur_row, l, ref_val)
                        l = l + 1
                    """
                    from dipy.viz import window, actor
                    from tract_visualize import show_bundles, setup_view
                    import nibabel as nib

                    lut_cmap = actor.colormap_lookup_table(
                        scale_range=(0.05, 0.3))

                    scene = setup_view(nib.streamlines.ArraySequence(target_streamlines[33:34]), colors=lut_cmap,
                                       ref=ref_img_path, world_coords=True,
                                       objectvals=[None], colorbar=True, record=None, scene=None, interactive=True)
                    """
                    """
                    if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
                        worksheet.write(subj, l, np.mean(stream_ref))
                        worksheet.write(subj, l+1, np.min(stream_ref))
                        worksheet.write(subj, l+2, np.max(stream_ref))
                        worksheet.write(subj, l+3, np.std(stream_ref))
                        l=l+4
                    """
                    """
                    if not (group, ref) in groupLines.keys():
                        groupLines[group, ref]=(stream_ref)
                    else:
                        groupLines[group, ref].extend(stream_ref)
                        groupPoints[group, ref].extend(stream_point_ref)
                    """
                    try:
                        groupLines[group, ref].update({subject: stream_ref})
                    except KeyError:
                        print('hi')
                    if ref != 'ln':
                        groupPoints[group, ref].update({subject: stream_point_ref})
                sl_all+= np.shape(target_streamlines_set)[0]
                subj += 1
                groupstreamlines[group].extend(target_streamlines_set)

            group_qb[group] = QuickBundles(threshold=distance2, metric=metric2)
            group_clusters[group] = group_qb[group].cluster(groupstreamlines[group])
            if os.path.exists(centroid_file_path) and overwrite:
                remove_remote(centroid_file_path,sftp=sftp)
            if not os.path.exists(centroid_file_path):
                if verbose:
                    print(f'Summarized the clusters for group {group} at {centroid_file_path}')
                pickledump_remote(group_clusters[group], centroid_file_path, sftp=sftp)

            #if np.shape(groupLines[group, ref])[0] != np.shape(groupstreamlines[group])[0]:
            #    raise Exception('happened from there')

            if os.path.exists(streamline_file_path) and overwrite and write_streamlines:
                remove_remote(streamline_file_path,sftp=sftp)
            if not os.path.exists(streamline_file_path) and write_streamlines:
                if verbose:
                    print(f'Summarized the streamlines for group {group} at {streamline_file_path}')
                #pickledump_remote(groupstreamlines[group], streamline_file_path, sftp=sftp)
                sg = lambda: (s for i,s in enumerate(groupstreamlines[group]))
                save_trk_header(filepath= streamline_file_path, streamlines = sg, header = header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp)

            for ref in references:
                if overwrite:
                    if os.path.exists(grouping_files[ref,'lines']):
                        remove_remote(grouping_files[ref,'lines'],sftp=sftp)
                    if os.path.exists(grouping_files[ref,'points']):
                        remove_remote(grouping_files[ref,'points'],sftp=sftp)
                if not os.path.exists(grouping_files[ref,'lines']):
                    if verbose:
                        print(f"Summarized the clusters for group {group} and statistics {ref} at {grouping_files[ref,'lines']}")
                    pickledump_remote(groupLines[group, ref], grouping_files[ref,'lines'], sftp=sftp)

                if not os.path.exists(grouping_files[ref, 'points']):
                    if verbose:
                        print(f"Summarized the clusters for group {group} and statistics {ref} at {grouping_files[ref,'lines']}")
                    pickledump_remote(groupPoints[group, ref], grouping_files[ref,'points'], sftp=sftp)

            pickledump_remote(groupPoints[group, 'ln'], grouping_files['ln', 'points'], sftp=sftp)

            subjects_summary = {'Subject': subjects_ID_list, 'Streamline_ID': streams_ID_list}
            mydf = pd.DataFrame(subjects_summary)
            mydf.to_excel(excel_SID_path)
            print(f'Wrote Subject ID matcher to {excel_SID_path}')

        else:
            print(f'Centroid file was found at {centroid_file_path}, reference files for {references}')
            group_clusters[group] = remote_pickle(centroid_file_path, sftp=sftp)
            for ref in references:
                ref_path_lines = grouping_files[ref, 'lines']
                groupLines[group, ref] = remote_pickle(ref_path_lines, sftp=sftp)
                #ref_path_points = grouping_files[ref, 'points']
                #groupPoints[group, ref] = grouping_files[ref, 'points']


    if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
        workbook.close()
        if sftp is not None:
            sftp.put(stats_path_temp, stats_path)
            os.remove(stats_path_temp)
        # groupstreamlines_orig[group].extend(target_streamlines)