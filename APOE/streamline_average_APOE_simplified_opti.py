import numpy as np
from dipy.segment.clustering import QuickBundles
from dipy.io.streamline import load_trk, save_trk
from dipy.segment.metric import ResampleFeature, AveragePointwiseEuclideanMetric, mdf
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
from DTC.diff_handlers.connectome_handlers.connectome_handler import connectivity_matrix_func, \
    _to_voxel_coordinates_warning, retweak_points, retweak_point_set
from dipy.tracking.utils import length
import getpass
import random
from DTC.file_manager.computer_nav import get_mainpaths, get_atlas, make_temppath, checkfile_exists_remote, \
    load_trk_remote, \
    remote_pickle, pickledump_remote, load_nifti_remote, remove_remote, glob_remote, checkfile_exists_all, \
    checkfile_exists_all_faster
import pandas as pd
from dipy.viz import window, actor
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view_legacy
import nibabel as nib


def get_grouping(grouping_xlsx):
    print('not done yet')

# set parameter
num_points1 = 50
distance1 = 1
# group cluster parameter
num_points2 = 50
distance2 = 2

ratio = 1
# projects = ['AD_Decode', 'AMD', 'APOE']
project = 'APOE'

remote = False

# inpath, outpath, atlas_folder, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
inpath = '/Volumes/dusom_abadea_nas1/munin_js/APOE_series'
outpath = '/Volumes/dusom_mousebrains/All_Staff/Analysis/APOE'
atlas_folder = '/Volumes/Data/Badea/Lab/atlases'

sftp = None

if project == 'AMD' or project == 'AD_Decode':
    atlas_legends = get_atlas(atlas_folder, 'IIT')
if project == 'APOE':
    atlas_legends = get_atlas(atlas_folder, 'CHASSSYMM3')

atlas_legends = '/Users/jas/jacques/atlases/CHASSSYMM3AtlasLegends.xlsx'

# diff_preprocessed = os.path.join(inpath, "DWI")
ref_MDT_folder = os.path.join(inpath, "reg_trix_MDT")
# ref_MDT_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/AMD/DWI_trix_MDT/'
# ref_MDT_folder = os.path.join('/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd', 'DWI_trix_MDT')

skip_subjects = True
write_streamlines = True
allow_preprun = False
verbose = True
picklesave = True
inclusive = False
symmetric = True
write_stats = True
write_txt = True
constrain_groups = True
fixed = False
overwrite = False

labeltype = 'lrordered'
# reference_img refers to statistical values that we want to compare to the streamlines, say fa, rd, etc

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
# TRK_folder = os.path.join(inpath, f'TRK_MPCA_MDT{fixed_str}{folder_ratio_str}')
# TRK_folder = os.path.join(inpath, f'TRK_rigidaff{fixed_str}{folder_ratio_str}')
# TRK_folder= os.path.join(inpath, f'TRK_trix_farun')
TRK_folder = os.path.join(inpath, f'TRK_{space_type}{fixed_str}{folder_ratio_str}')

label_folder = os.path.join(inpath, 'DWI')
if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'

space_type = 'MDT'
target_tuples = [(121, 123), (287, 45), (28, 9), (62, 1)]

pickle_folder = os.path.join(outpath,
                             f'Pickle_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')
centroid_folder = os.path.join(outpath,
                               f'Centroids_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')
stats_folder = os.path.join(outpath, f'Statistics_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}')
stats_folder = os.path.join(outpath, f'Statistics_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}')
stats_folder = os.path.join(outpath,
                            f'Statistics_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')
connectome_folder = os.path.join(outpath, f'Excels_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}')
connectome_folder = os.path.join(outpath,
                                 f'Excels_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')

figures_folder = os.path.join(outpath,
                              f'Figures_subj_{space_type}{inclusive_str}{symmetric_str}{folder_ratio_str}{other_param}')

mkcdir([pickle_folder, centroid_folder, stats_folder, connectome_folder, figures_folder], sftp)
if not remote and not os.path.exists(TRK_folder):
    raise Exception(f'cannot find TRK folder at {TRK_folder}')

# Initializing dictionaries to be filled
stream_point = {}
stream = {}
groupstreamlines = {}
groupstreamlines_orig = {}
groupLines = {}
groupPoints = {}
group_qb = {}
group_clusters = {}
groups_subjects = {}

if project == 'APOE':
    groups_subjects['all'] = ['N57442', 'N57496', 'N57500', 'N57580', 'N57709', 'N58219', 'N58221', 'N58222', 'N58223',
                              'N58224', 'N58226', 'N58228', 'N58229', 'N58230', 'N58231', 'N58232', 'N58302', 'N58303',
                              'N58305', 'N58309', 'N58310', 'N58344', 'N58346', 'N58350', 'N58355', 'N58359', 'N58361',
                              'N58394', 'N58396', 'N58398', 'N58400', 'N58402', 'N58404', 'N58406', 'N58408', 'N58477',
                              'N58500', 'N58510', 'N58512', 'N58514', 'N58516', 'N58604', 'N58606', 'N58608', 'N58610',
                              'N58611', 'N58612', 'N58613', 'N58633', 'N58634', 'N58635', 'N58636', 'N58649', 'N58650',
                              'N58651', 'N58653', 'N58654', 'N58655', 'N58706', 'N58708', 'N58712', 'N58714', 'N58732',
                              'N58733', 'N58734', 'N58735', 'N58740', 'N58742', 'N58745', 'N58747', 'N58749', 'N58751',
                              'N58779', 'N58780', 'N58784', 'N58788', 'N58790', 'N58792', 'N58794', 'N58813', 'N58815',
                              'N58819', 'N58821', 'N58829', 'N58831', 'N58851', 'N58853', 'N58855', 'N58857', 'N58859',
                              'N58861', 'N58877', 'N58879', 'N58881', 'N58883', 'N58885', 'N58887', 'N58889', 'N58906',
                              'N58909', 'N58913', 'N58915', 'N58917', 'N58919', 'N58935', 'N58941', 'N58946', 'N58948',
                              'N58954', 'N58995', 'N58997', 'N58999', 'N59003', 'N59005', 'N59010', 'N59022', 'N59026',
                              'N59033', 'N59035', 'N59039', 'N59065', 'N59066', 'N59076', 'N59078', 'N59080', 'N59097',
                              'N59099', 'N59109', 'N59116', 'N59118', 'N59120', 'N59136', 'N59141', 'N60056', 'N60060',
                              'N60062', 'N60064', 'N60068', 'N60070', 'N60072', 'N60088', 'N60092', 'N60093', 'N60095',
                              'N60101', 'N60103', 'N60127', 'N60129', 'N60133', 'N60137', 'N60139', 'N60157', 'N60159',
                              'N60163', 'N60167', 'N60188', 'N60190', 'N60192', 'N60194', 'N60200', 'N60225', 'N60229',
                              'N60231']
    subjects = []
    # str_identifier = f'*{space_type}{ratio_str}'
    str_identifier = '*smallerTracks2mill'
    groups = ['all']
    removed_list = []
    for group in groups:
        subjects = subjects + groups_subjects[group]

    target_tuples = [(121, 235), (287, 235), (287, 230), (121, 120), (230, 235), (287, 120), (230, 230), (64, 120),
                     (64, 235), (121, 230), (230, 120), (64, 230), (217, 230), (217, 235), (217, 120), (238, 230),
                     (51, 235), (51, 230), (72, 235), (72, 230), (51, 120), (89, 120), (238, 235), (255, 230),
                     (223, 230), (287, 205), (72, 120), (230, 205), (57, 230), (57, 120)]

    if groups == ['all']:
        allgroups = 'allsubj'
    else:
        allgroups = '-'.join(groups).replace(' ', '_')
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
        # group_sizes[group] = np.size(groups_subjects[group])
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

_, _, index_to_struct, _ = atlas_converter(atlas_legends)

stats_path = {}
stats_path_temp = {}

group_connection_str = {}
centroid_file_path = {}
streamline_file_path = {}
excel_SID_path = {}

subjects_ID_list = {}
streams_ID_list = {}

grouping_files = {}

write_stats = False

labelmask, labelaffine, labeloutpath, index_to_struct = getlabeltypemask(label_folder, 'MDT',
                                                                         atlas_legends,
                                                                         labeltype=labeltype,
                                                                         verbose=verbose, sftp=sftp)

for group in groups:

    print(f'Going through group {group}')

    groupstreamlines[group] = []
    groupstreamlines_orig[group] = []
    for ref in references:
        groupLines[group, ref] = {}
        groupPoints[group, ref] = {}

    for target_tuple in target_tuples:

        print(f'Starting the run for {index_to_struct[target_tuple[0]]} to {index_to_struct[target_tuple[1]]}')
        stats_path[target_tuple] = os.path.join(stats_folder, f'{index_to_struct[target_tuple[0]]}_'
        f'to_{index_to_struct[target_tuple[1]]}_{allgroups}.xlsx')
        # This is the streamline counter for the stats path. Important!
        sl_all = 1

        if sftp is None:
            stats_path_temp[target_tuple] = stats_path
        else:
            stats_path_temp[target_tuple] = make_temppath(stats_path)

        group_str = group.replace(' ', '_')
        group_connection_str[group, target_tuple] = group_str + '_' + space_type + ratio_str + '_' + index_to_struct[
            target_tuple[0]] + '_to_' + index_to_struct[target_tuple[1]]
        centroid_file_path[group, target_tuple] = os.path.join(centroid_folder, group_connection_str[
            group, target_tuple] + '_centroid.py')
        streamline_file_path[group, target_tuple] = os.path.join(centroid_folder, group_connection_str[
            group, target_tuple] + '_streamlines.trk')
        excel_SID_path[group, target_tuple] = os.path.join(centroid_folder, group_connection_str[
            group, target_tuple] + '_streamlineID_subj.xlsx')

    """
    if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
        import xlsxwriter

        workbook = xlsxwriter.Workbook(stats_path_temp)
        worksheet = workbook.add_worksheet()
        l = 1
        for ref in references:
            worksheet.write(0, l, ref)
            l += 1
    """

    subjects = groups_subjects[group]

    SID = 0
    subjects_ID_list[target_tuple] = []
    streams_ID_list[target_tuple] = []

    grouping_files[target_tuple] = {}
    exists = True

    for ref in references:
        # grouping_files[ref,'lines']=(os.path.join(centroid_folder, group_connection_str + '_' + ref + '_lines.py'))
        # grouping_files[ref, 'points'] = (os.path.join(centroid_folder, group_connection_str + '_' + ref + '_points.py'))
        list_files, exists = check_files(grouping_files[target_tuple], sftp)

    if not os.path.exists(centroid_file_path[group, target_tuple]) or not np.all(exists) or \
            (not os.path.exists(streamline_file_path[group, target_tuple]) and write_streamlines) \
            or not os.path.exists(excel_SID_path[group, target_tuple]) or overwrite:
        for subject in subjects:

            recordsubj_paths_all = []
            trksubj_paths_all = []

            for target_tuple in target_tuples:
                recordsubj_paths_all.append(
                    os.path.join(figures_folder, f'{subject}_{group_connection_str[group, target_tuple]}.png'))
                trksubj_paths_all.append(
                    os.path.join(figures_folder, f'{subject}_{group_connection_str[group, target_tuple]}.trk'))

            if checkfile_exists_all_faster(recordsubj_paths_all, 0.7) and checkfile_exists_all_faster(trksubj_paths_all,
                                                                                                      0.7):
                continue

            subj = 1

            trkpath, exists = gettrkpath(TRK_folder, subject, str_identifier, pruned=False, verbose=verbose,
                                         sftp=sftp)

            if not exists:
                txt = f'Could not find subject {subject} at {TRK_folder} with {str_identifier}'
                warnings.warn(txt)
                continue

            picklepath_connectome = os.path.join(pickle_folder, subject + '_connectomes.p')
            picklepath_grouping = os.path.join(pickle_folder, subject + '_grouping.p')

            M_xlsxpath = os.path.join(connectome_folder, subject + "_connectomes.xlsx")
            grouping_xlsxpath = os.path.join(connectome_folder, subject + "_grouping.xlsx")

            if checkfile_exists_remote(grouping_xlsxpath, sftp):
                mygrouping_xlsxpath = glob_remote(grouping_xlsxpath)
                if np.size(mygrouping_xlsxpath) == 1:
                    grouping = extract_grouping(mygrouping_xlsxpath[0], index_to_struct, None, verbose=verbose,
                                                sftp=sftp)
                else:
                    raise Exception
            else:
                if allow_preprun:

                    trkdata = load_trk_remote(trkpath, 'same', sftp=sftp)
                    header = trkdata.space_attributes

                    labelmask, labelaffine, labeloutpath, index_to_struct = getlabeltypemask(label_folder, 'MDT',
                                                                                             atlas_legends,
                                                                                             labeltype=labeltype,
                                                                                             verbose=verbose, sftp=sftp)

                    streamlines_world = transform_streamlines(trkdata.streamlines, np.linalg.inv(labelaffine))

                    M, grouping = connectivity_matrix_func(streamlines_world, np.eye(4), labelmask, inclusive=inclusive,
                                                           symmetric=symmetric, return_mapping=True,
                                                           mapping_as_streamlines=False,
                                                           reference_weighting=None,
                                                           volume_weighting=False, verbose=False)
                    M_grouping_excel_save(M, grouping, M_xlsxpath, grouping_xlsxpath, index_to_struct,
                                          verbose=False)
                else:
                    print(
                        f'skipping subject {subject} for now as grouping file {grouping_xlsxpath} is not calculated. Best rerun it afterwards ^^')

                    raise Exception('Actually just stop it altogether and note the problem here')

            trkdata = load_trk_remote(trkpath, 'same', sftp=sftp)
            header = trkdata.space_attributes
            for target_tuple in target_tuples:
                recordsubj_path = os.path.join(figures_folder,
                                               f'{subject}_{group_connection_str[group, target_tuple]}.png')
                trksubj_path = os.path.join(figures_folder,
                                            f'{subject}_{group_connection_str[group, target_tuple]}.trk')

                if os.path.exists(recordsubj_path) and os.path.exists(trksubj_path):
                    continue

                target_streamlines_list = grouping[target_tuple[0], target_tuple[1]]
                if np.size(target_streamlines_list) == 0:
                    txt = f'Did not have any streamlines for {index_to_struct[target_tuple[0]]} to ' \
                        f'{index_to_struct[target_tuple[1]]} for subject {subject}'
                    warnings.warn(txt)
                    continue
                target_streamlines = trkdata.streamlines[np.array(target_streamlines_list)]
                target_streamlines_set = set_number_of_points(target_streamlines, nb_points=num_points2)

                if save_subj_grouping:
                    lut_cmap = actor.colormap_lookup_table(scale_range=(0.05, 0.3))
                    scene = setup_view_legacy(target_streamlines_set, colors=lut_cmap,
                                              ref=labeloutpath, world_coords=True,
                                              objectvals=[None], colorbar=True, record=recordsubj_path, scene=None,
                                              interactive=False)

                    sg = lambda: (s for i, s in enumerate(target_streamlines_set))
                    save_trk_header(filepath=trksubj_path, streamlines=sg, header=header,
                                    affine=np.eye(4), verbose=verbose, sftp=sftp)
                    print(f'Saved Tracts of subject to {trksubj_path}')

                # subjects_ID_list[target_tuple] += [subject] * np.shape(target_streamlines_set)[0]
                # streams_ID_list[target_tuple] += list(np.arange(SID, SID + np.shape(target_streamlines_set)[0]))
                SID += np.shape(target_streamlines_set)[0]

                for ref in references:
                    if ref != 'ln':
                        ref_img_path = get_diff_ref(ref_MDT_folder, subject, ref, sftp=sftp)
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
                        target_streamlines_transformed = transform_streamlines(target_streamlines_set,
                                                                               np.linalg.inv(ref_affine))
                        testmode = False
                        for sl, _ in enumerate(target_streamlines_transformed):
                            # Convert streamline to voxel coordinates
                            # entire = _to_voxel_coordinates(target_streamlines_set[sl], lin_T, offset)
                            cur_row = sl_all + sl

                            voxel_coords = np.round(target_streamlines_transformed[sl]).astype(int)
                            voxel_coords_tweaked = retweak_points(voxel_coords, np.shape(ref_data))
                            ref_values = ref_data[
                                voxel_coords_tweaked[:, 0], voxel_coords_tweaked[:, 1], voxel_coords_tweaked[:, 2]]

                            stream_point_ref.append(ref_values)
                            stream_ref.append(np.mean(ref_values))

                            if np.mean(ref_values) == 0:
                                print('too low a value for new method')
                                testmode = True

                            if testmode:
                                from DTC.tract_manager.tract_save import save_trk_header

                                small_streamlines_testzone = '/Users/jas/jacques/AMD_testing_zone/single_streamlines'
                                mkcdir(small_streamlines_testzone, sftp)
                                streamline_file_path[group, target_tuple] = os.path.join(small_streamlines_testzone,
                                                                                         f'{subject}_streamline_{sl}.trk')
                                from dipy.tracking import streamline

                                streamlines = streamline.Streamlines([target_streamlines_set[sl]])
                                save_trk_header(filepath=streamline_file_path[group, target_tuple],
                                                streamlines=streamlines, header=header,
                                                affine=np.eye(4), verbose=True, sftp=sftp)
                                testmode = False
                            # if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
                            #    worksheet.write(cur_row, l, np.mean(ref_values))

                        # l = l + 1
                    else:
                        stream_ref = list(length(target_streamlines))
                        for sl, ref_val in enumerate(stream_ref):
                            cur_row = sl_all + sl
                            # if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
                            #    worksheet.write(cur_row, l, ref_val)
                        # l = l + 1

                    try:
                        groupLines[group, ref].update({subject: stream_ref})
                    except KeyError:
                        print('hi')
                    if ref != 'ln':
                        groupPoints[group, ref].update({subject: stream_point_ref})
                sl_all += np.shape(target_streamlines_set)[0]
                subj += 1
                groupstreamlines[group].extend(target_streamlines_set)

            group_qb[group] = QuickBundles(threshold=distance2, metric=metric2)
            group_clusters[group] = group_qb[group].cluster(groupstreamlines[group])
            if os.path.exists(centroid_file_path[group, target_tuple]) and overwrite:
                remove_remote(centroid_file_path[group, target_tuple], sftp=sftp)
            if not os.path.exists(centroid_file_path[group, target_tuple]):
                if verbose:
                    print(f'Summarized the clusters for group {group} at {centroid_file_path[group, target_tuple]}')
                pickledump_remote(group_clusters[group], centroid_file_path[group, target_tuple], sftp=sftp)

            # if np.shape(groupLines[group, ref])[0] != np.shape(groupstreamlines[group])[0]:
            #    raise Exception('happened from there')

            if os.path.exists(streamline_file_path[group, target_tuple]) and overwrite and write_streamlines:
                remove_remote(streamline_file_path[group, target_tuple], sftp=sftp)
            if not os.path.exists(streamline_file_path[group, target_tuple]) and write_streamlines:
                if verbose:
                    print(
                        f'Summarized the streamlines for group {group} at {streamline_file_path[group, target_tuple]}')
                # pickledump_remote(groupstreamlines[group], streamline_file_path[group, target_tuple], sftp=sftp)
                sg = lambda: (s for i, s in enumerate(groupstreamlines[group]))
                save_trk_header(filepath=streamline_file_path[group, target_tuple], streamlines=sg, header=header,
                                affine=np.eye(4), verbose=verbose, sftp=sftp)

            for ref in references:
                if overwrite:
                    if os.path.exists(grouping_files[target_tuple][ref, 'lines']):
                        remove_remote(grouping_files[target_tuple][ref, 'lines'], sftp=sftp)
                    if os.path.exists(grouping_files[target_tuple][ref, 'points']):
                        remove_remote(grouping_files[target_tuple][ref, 'points'], sftp=sftp)
                """
                #if not os.path.exists(grouping_files[target_tuple][ref,'lines']):
                #    if verbose:
                #        print(f"Summarized the clusters for group {group} and statistics {ref} at {grouping_files[target_tuple][ref,'lines']}")
                #    pickledump_remote(groupLines[group, ref], grouping_files[target_tuple][ref,'lines'], sftp=sftp)


                if not os.path.exists(grouping_files[target_tuple][ref, 'points']):
                    if verbose:
                        print(f"Summarized the clusters for group {group} and statistics {ref} at {grouping_files[target_tuple][ref,'lines']}")
                    pickledump_remote(groupPoints[group, ref], grouping_files[target_tuple][ref,'points'], sftp=sftp)
                """
            # pickledump_remote(groupPoints[group, 'ln'], grouping_files[target_tuple]['ln', 'points'], sftp=sftp)

            # subjects_summary = {'Subject': subjects_ID_list[target_tuple], 'Streamline_ID': streams_ID_list[target_tuple]}
            # mydf = pd.DataFrame(subjects_summary)
            # mydf.to_excel(excel_SID_path[group, target_tuple])
            print(f'Wrote Subject ID matcher to {excel_SID_path[group, target_tuple]}')

        else:
            print(f'Centroid file was found at '
                  f'{centroid_file_path[group, target_tuple]}, reference files for {references}')
            group_clusters[group] = remote_pickle(centroid_file_path[group, target_tuple], sftp=sftp)
            """
            for ref in references:
                ref_path_lines = grouping_files[target_tuple][ref, 'lines']
                groupLines[group, ref] = remote_pickle(ref_path_lines, sftp=sftp)
            """