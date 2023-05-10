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
from DTC.diff_handlers.connectome_handlers.connectome_handler import connectivity_matrix_func, _to_voxel_coordinates_warning, retweak_points
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
project = 'AMD'

computer_name = socket.gethostname()
remote=False
username = None
passwd = None
if remote:
    if 'santorini' in computer_name:
        username, passwd = getfromfile('/Users/jas/samos_connect.rtf')
    elif 'hydra' in computer_name:
        username, passwd = getfromfile('/Users/alex/jacques/samos_connect.rtf')


#inpath, outpath, atlas_folder, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
inpath = '/Volumes/dusom_mousebrains/All_Staff/Data/AMD/'
outpath = '/Users/jas/jacques/Whiston_article/Analysis/AMD/'
outpath = '/Volumes/dusom_mousebrains/All_Staff/Analysis/AMD/'
atlas_folder = '/Volumes/Data/Badea/Lab/atlases'
sftp=None

if project=='AMD' or project=='AD_Decode':
    atlas_legends = get_atlas(atlas_folder, 'IIT')

diff_preprocessed = os.path.join(inpath, "DWI")
ref_MDT_folder = os.path.join(inpath, "reg_MDT")
#ref_MDT_folder = os.path.join('/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd', 'DWI_trix_MDT')

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
fixed = False

labeltype = 'lrordered'
#reference_img refers to statistical values that we want to compare to the streamlines, say fa, rd, etc

#references = ['fa', 'md', 'ln', 'rd', 'ad']
references = ['fa', 'md', 'rd', 'md', 'ad', 'ln']
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

if project=='AD_Decode':
    outpath = os.path.join(outpath, 'Analysis')
    inpath = os.path.join(inpath, 'Analysis')

TRK_folder = os.path.join(inpath, f'TRK_MPCA_MDT{fixed_str}{folder_ratio_str}')
TRK_folder = os.path.join(inpath, f'TRK_rigidaff{fixed_str}{folder_ratio_str}')

label_folder = os.path.join(inpath, 'DWI')
if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'

other_param = '_mrtrix'
other_param = ''

if other_param=='_mrtrix':
    target_tuples = [(62, 28),(58,45), (28, 9), (62, 1)]
    target_tuples = [(36,70),(36,28),(70,62),(36,62),(70,28)]
    #target_tuples = [(36,62),(70,28)]
    TRK_folder = os.path.join('/Volumes/dusom_mousebrains/All_Staff/Nariman_mrtrix_amd', 'TRK_trix_MDT')
    space_type = 'MDT'
else:
    space_type = 'affinerigid'
    target_tuples = [(62, 28),(58,45),(28,9), (62, 1)]
    target_tuples = [(36,70),(36,28),(70,62),(36,62),(70,28)]


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
    groups_subjects['Paired Initial AMD'] = ['H29020', 'H26637', 'H26765', 'H28308', 'H28433', 'H26660', 'H28182', 'H27111', 'H27391', 'H28748', 'H28662', 'H26578', 'H28698', 'H27495', 'H28861', 'H28115', 'H28377', 'H26890', 'H28373', 'H27164']

    #groups to go through
    #groups_all = ['Paired 2-YR AMD','Initial AMD','Initial Control','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']
    groups = ['Paired 2-YR AMD','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']
    #groups = ['Paired Initial Control','Paired Initial AMD']
    #groups = ['Paired Initial AMD']
    #groups = ['testing']
    #groups = ['testing']
    #groups = ['Paired Initial Control','Paired Initial AMD']

    if groups == ['Paired 2-YR AMD','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']:
        allgroups = 'allsubj'
    else:
        allgroups = '-'.join(groups).replace(' ','_')

    #groups = ['Paired 2-YR AMD','Paired 2-YR Control']

    str_identifier = '_MDT' + folder_ratio_str
    str_identifier = f'*'

    #TN-PCA
    #target_tuples = [(62, 28), (56, 45), (77, 43), (58, 45), (79, 45), (56, 50), (28, 9), (62, 1), (28, 1), (62, 9), (22, 9), (56, 1),(77, 43), (76, 43), (61, 29), (63, 27), (73, 43), (53, 43)]
    #VBA
    #target_tuples = [(27,29), (61,63),(30, 16), (24, 16),(28, 31), (28, 22),(22, 31)]
    #TN-PCA / VBA combination
    #target_tuples = [(62, 28), (58, 45), (28, 9), (62, 1), (77, 43), (61, 29)]
    #target_tuples = [(62, 1), (77, 43), (61, 29)]

    removed_list = []

elif project == 'APOE':
    raise Exception('not implemented')

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

overwrite=False
save_subj_grouping = True
print('hi')

for group in groups:
    subjects = groups_subjects[group]
    for subject in subjects:
        grouping_xlsxpath = os.path.join(connectome_folder, subject + str_identifier + "_grouping.xlsx")
        if not checkfile_exists_remote(grouping_xlsxpath, sftp) and not allow_preprun:
            print(subject)

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

        if not checkfile_exists_remote(centroid_file_path, sftp) or not np.all(exists) or \
                (not checkfile_exists_remote(streamline_file_path, sftp) and write_streamlines) or \
                (not checkfile_exists_remote(stats_path, sftp) and write_stats) or overwrite:
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

                picklepath_connectome = os.path.join(pickle_folder, subject + str_identifier + '_connectomes.p')
                picklepath_grouping = os.path.join(pickle_folder, subject + str_identifier + '_grouping.p')
                M_xlsxpath = os.path.join(connectome_folder, subject + str_identifier + "_connectomes.xlsx")
                grouping_xlsxpath = os.path.join(connectome_folder, subject + str_identifier + "_grouping.xlsx")
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

                        labelmask, labelaffine, labeloutpath, index_to_struct = getlabeltypemask(label_folder, 'MDT', atlas_legends,
                                                     labeltype=labeltype, verbose=verbose, sftp=sftp)

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
                        print(f'skipping subject {subject} for now as grouping file is not calculated. Best rerun it afterwards ^^')
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
                    labelmask, labelaffine, labeloutpath, index_to_struct = getlabeltypemask(label_folder, 'MDT',
                                                                                             atlas_legends,
                                                                                             labeltype=labeltype,
                                                                                             verbose=verbose, sftp=sftp)
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
                        for sl, _ in enumerate(target_streamlines_set):
                            # Convert streamline to voxel coordinates
                            #entire = _to_voxel_coordinates(target_streamlines_set[sl], lin_T, offset)
                            cur_row = sl_all + sl
                            entire = _to_voxel_coordinates_warning(target_streamlines_set[sl], lin_T, offset)
                            entire = retweak_points(entire, np.shape(ref_data))
                            i, j, k = entire.T
                            ref_values = ref_data[i, j, k]
                            stream_point_ref.append(ref_values)
                            stream_ref.append(np.mean(ref_values))
                            if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
                                worksheet.write(cur_row, l, np.mean(ref_values))
                        l = l + 1
                    else:
                        stream_ref = list(length(target_streamlines))
                        for sl, ref_val in enumerate(stream_ref):
                            cur_row = sl_all + sl
                            worksheet.write(cur_row, l, ref_val)
                        l = l + 1

                    try:
                        groupLines[group, ref].update({subject: stream_ref})
                    except KeyError:
                        print('hi')
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

            """
            if not os.path.exists(streamline_file_path_orig) and write_streamlines:
                if verbose:
                    print(f'Summarized the streamlines for group {group} at {streamline_file_path}')
                pickle.dump(groupstreamlines_orig[group], open(streamline_file_path, "wb"))
                save_trk_header(filepath= streamline_file_path, streamlines = groupstreamlines_orig[group], header = header, affine=np.eye(4), verbose=verbose)
            """

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

        else:
            print(f'Centroid file was found at {centroid_file_path}, reference files for {references}')
            group_clusters[group] = remote_pickle(centroid_file_path, sftp=sftp)
            for ref in references:
                ref_path_lines = grouping_files[ref, 'lines']
                groupLines[group, ref] = remote_pickle(ref_path_lines, sftp=sftp)
                #ref_path_points = grouping_files[ref, 'points']
                #groupPoints[group, ref] = grouping_files[ref, 'points']

        subjects_summary = {'Subject': subjects_ID_list, 'Streamline_ID': streams_ID_list}
        mydf = pd.DataFrame(subjects_summary)
        mydf.to_excel(excel_SID_path)
        print(f'Wrote Subject ID matcher to {excel_SID_path}')

    if write_stats and (not checkfile_exists_remote(stats_path, sftp) or overwrite):
        workbook.close()
        if sftp is not None:
            sftp.put(stats_path_temp, stats_path)
            os.remove(stats_path_temp)
