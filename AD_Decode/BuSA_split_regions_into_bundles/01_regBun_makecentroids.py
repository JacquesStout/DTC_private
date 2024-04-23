import numpy as np
import os, fury, sys

from DTC.tract_manager.DTC_manager import get_str_identifier, check_dif_ratio
from dipy.viz import window, actor
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote, pickledump_remote, remote_pickle, copy_remotefiles, write_parameters_to_ini, read_parameters_from_ini
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from dipy.tracking.streamline import transform_streamlines
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import sleep
from dipy.tracking.streamline import set_number_of_points
from DTC.tract_manager.tract_save import save_trk_header
import time
import nibabel as nib
import copy
import socket
from DTC.tract_manager.tract_handler import ratio_to_str, filter_streamlines
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles
import argparse
from DTC.wrapper_tools import parse_list_arg
from DTC.tract_manager.tract_handler import gettrkpath


def check_streamline(streamline):
    for i in range(len(streamline) - 1):
        point1 = streamline[i]
        point2 = streamline[i + 1]

        distance = np.linalg.norm(point2 - point1)

        if distance > 10:

            firstpart = streamline[:i + 1]
            secondpart = streamline[i + 1:]
            if np.linalg.norm(firstpart - firstpart[0]) < 0.5:
                if len(firstpart)==1:
                    secondpart = secondpart[1:]
                    continue
                else:
                    return(secondpart)
            elif np.linalg.norm(secondpart - secondpart[0]) < 0.5:
                return(firstpart)
            else:
                streamline_1 = check_streamline(firstpart)
                streamline_2 = check_streamline(secondpart)
                if len(streamline_1)>len(streamline_2):
                    return(streamline_1)
                else:
                    return(streamline_2)

    return(streamline)


def fix_badwarp_streamlines(streamlines):
    streamlines_pruned = []
    streamlines_onlypruned = []

    for id, streamline in enumerate(streamlines):
        streamline = check_streamline(streamline)
        streamlines_pruned.append(streamline)
    return streamlines_pruned

#python3 01_regBun_makecentroids.py --proj /Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/V_1_0_10template_100_6_interhe_majority.ini --id 0 --split 3
#--proj /Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/V_1_0_10template_100_6_interhe_majority.ini --id 0 --split 3

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--split', type=int, help='An integer for splitting')
parser.add_argument('--proj', type=str, help='The project path or name')
parser.add_argument('--id', type=parse_list_arg, help='The ID for the bundle subtype, can be a single value or a list of values [0 1 4], etc')

args = parser.parse_args()
bundle_split = args.split
project_summary_file = args.proj

bundle_id_orig = args.id

project_headfile_folder = '/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/'

if project_summary_file is None:
    project_headfile_folder = '/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/'
    project_run_identifier = 'V0_9_10template_100_6_interhe_majority'
    project_summary_file = os.path.join(project_headfile_folder, project_run_identifier + '.ini')
else:
    project_run_identifier = os.path.basename(project_summary_file).split('.')[0]

if not os.path.exists(project_summary_file):
    project_summary_file = os.path.join(project_headfile_folder,project_summary_file+'.ini')

if not os.path.exists(project_summary_file):
    txt = f'Could not find configuration file at {project_summary_file}'
    raise Exception(txt)
else:
    params = read_parameters_from_ini(project_summary_file)

#locals().update(params) #This line will add to the code the variables specified above from the config file, namely

project = params['project']
streamline_type = params['streamline_type']
test = params['test']
ratio = params['ratio']
stepsize = params['stepsize']
template_subjects = params['template_subjects']
setpoints = params['setpoints']
points_resample = int(params['points_resample'])
remote_output = bool(params['remote_output'])
remote_input = bool(params['remote_input'])
streamline_lr_inclusion = params['streamline_lr_inclusion']
length_threshold = int(params['length_threshold'])

distance = int(params['distance'])

bundle_points = int(params['bundle_points'])
path_TRK = params['path_trk']

if bundle_id_orig is None:
    bundle_split = int(params['num_bundles'])
    bundle_id_txt = ''
elif bundle_id_orig is not None and bundle_split is None:
    raise Exception('Must specify the number of bundles to subsplit into')
else:
    bundle_id_txt = '_'+str(bundle_id_orig[0])
    if np.size((bundle_id_orig[0]).split('_')) >= 3:
        distance = distance / 3
        removed_list = params['removed_list']
        template_subjects = template_subjects + params['added_subjects']
        for remove in removed_list:
            if remove in template_subjects:
                template_subjects.remove(remove)

"""
if bundle_split is None:
    try:
        bundle_split = int(params['bundle_split'])
    except:
        bundle_split = 6
"""

overwrite=True
verbose = False

if remote_input or remote_output:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

if streamline_type == 'mrtrix':
    prune = False
    trkroi = [""]
else:
    prune = True
    trkroi = ["wholebrain"]

str_identifier = get_str_identifier(stepsize, ratio, trkroi, type=streamline_type)
#str_identifier = '_streamlines'

if 'santorini' in socket.gethostname().split('.')[0]:
    lab_folder = '/Volumes/Data/Badea/Lab'
if 'blade' in socket.gethostname().split('.')[0]:
    lab_folder = '/mnt/munin2/Badea/Lab'

if project == 'AD_Decode':
    SAMBA_MDT = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz')
    MDT_mask_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT')
    ref_MDT_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images')
    anat_path = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz')


_, _, _, sftp_in = get_mainpaths(remote_input,project = project, username=username,password=passwd)
outpath, _, _, sftp_out = get_mainpaths(remote_output,project = project, username=username,password=passwd)

ratiostr = ratio_to_str(ratio,spec_all=False)

outpath_all = os.path.join(outpath, 'TRK_bundle_splitter')
proj_path = os.path.join(outpath_all,project_run_identifier)
figures_proj_path = os.path.join(proj_path, 'Figures')
centroids_proj_path = os.path.join(figures_proj_path,'Centroids')
trk_proj_path = os.path.join(proj_path, 'trk_roi'+ratiostr)

srr = StreamlineLinearRegistration()

streams_dict = {}

streamlines_template = {}

combined_trk_folder = os.path.join(proj_path, 'combined_TRK')
streams_dict_picklepaths = {}

timings = []

streams_dict_side = {}
timings.append(time.perf_counter())

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratiostr)

mkcdir([proj_path,combined_trk_folder,pickle_folder,figures_proj_path,centroids_proj_path],sftp_out)


trktemplate_paths = os.path.join(combined_trk_folder, f'streamlines_template{bundle_id_txt}.trk')
streams_dict_picklepaths = os.path.join(combined_trk_folder, f'streams_dict.py')
streamlines_template = nib.streamlines.array_sequence.ArraySequence()

timings.append(time.perf_counter())

sides = ['left', 'right']

overwrite=True

if bundle_id_orig is None and (not checkfile_exists_remote(trktemplate_paths, sftp_out) \
        or not checkfile_exists_remote(streams_dict_picklepaths, sftp_out) or overwrite):

    num_streamlines_all = 0
    i = 1
    for subject in template_subjects:
        subj_trk, trkexists = gettrkpath(path_TRK, subject, str_identifier, pruned=prune, verbose=False, sftp=sftp_in)

        if not trkexists:
            txt = (f'Could not find subject {subject+str_identifier} at {path_TRK}')
            raise Exception('txt')

        if 'header' not in locals():
            streamlines_temp_data = load_trk_remote(subj_trk, 'same', sftp_in)
            header = streamlines_temp_data.space_attributes
            streamlines_temp = streamlines_temp_data.streamlines
            del streamlines_temp_data
        else:
            streamlines_temp = load_trk_remote(subj_trk, 'same', sftp_in).streamlines


        num_streamlines_subj = len(streamlines_temp)


        streams_dict_side[subject] = np.arange(num_streamlines_all, num_streamlines_all + num_streamlines_subj)

        if verbose:
            timings.append(time.perf_counter())
            print(f'Loaded subject {subject} from {subj_trk}, took {timings[-1] - timings[-2]} seconds')

        num_streamlines_all += num_streamlines_subj
        i += 1


        side_mask_path = os.path.join(MDT_mask_folder, f'IITmean_RPI_MDT_mask_left.nii.gz')
        roi_mask_left = nib.load(side_mask_path)
        streamlines_template.extend(filter_streamlines(streamlines_temp, roi_mask=roi_mask_left, world_coords=True,
                                             include=streamline_lr_inclusion,
                                             threshold=length_threshold))


        side_mask_path = os.path.join(MDT_mask_folder, f'IITmean_RPI_MDT_mask_right.nii.gz')
        roi_mask_right = nib.load(side_mask_path)
        streamlines_right_rev = filter_streamlines(streamlines_temp, roi_mask=roi_mask_right, world_coords=True,
                                                   include=streamline_lr_inclusion,
                                                   threshold=length_threshold)

        affine_flip = np.eye(4)
        affine_flip[0, 0] = -1
        affine_flip[0, 3] = 0

        streamlines_right_rev = transform_streamlines(streamlines_right_rev, affine_flip,
                                                      in_place=False)

        streamlines_template.extend(streamlines_right_rev)

        del streamlines_temp


    streamlines_template = fix_badwarp_streamlines(streamlines_template)

    sg = lambda: (s for i, s in enumerate(streamlines_template))
    save_trk_header(filepath=trktemplate_paths, streamlines=sg, header=header,
                    affine=np.eye(4), verbose=verbose, sftp=sftp_out)
    timings.append(time.perf_counter())
    print(f'Saved streamlines at {trktemplate_paths}, took {timings[-1] - timings[-2]} seconds')
    pickledump_remote(streams_dict_side, streams_dict_picklepaths, sftp_out)
    timings.append(time.perf_counter())
    print(f'Saved dictionary at {streams_dict_picklepaths}, took {timings[-1] - timings[0]} seconds')
elif checkfile_exists_remote(trktemplate_paths, sftp_out) and not overwrite:
    print(f'already wrote {trktemplate_paths} and {streams_dict_picklepaths}')
    streamlines_template_main = load_trk_remote(trktemplate_paths, 'same', sftp_out)
    streamlines_template = streamlines_template_main.streamlines
    header = streamlines_template_main.space_attributes
    streams_dict_side = remote_pickle(streams_dict_picklepaths, sftp=sftp_out)
else:
    trkpaths = {}

    for side in sides:

        if side == 'all':
            side_str = ''
        else:
            side_str = f'_{side}'

        if bundle_id_orig is not None:
            bundle_id_orig_txt = side_str + '_' + bundle_id_orig[0]
        else:
            bundle_id_orig_txt = side_str

        for subject in template_subjects:

            #trkpaths[subject, 'right'] = os.path.join(trk_proj_path, f'{subject}_right_bundle_{bundle_id_orig}.trk')
            #trkpaths[subject, 'left'] = os.path.join(trk_proj_path, f'{subject}_left_bundle_{bundle_id_orig}.trk')
            trkpath_subj = os.path.join(trk_proj_path, f'{subject}_bundle{bundle_id_orig_txt}.trk')

            if 'header' not in locals():
                streamlines_temp_data = load_trk_remote(trkpath_subj, 'same', sftp_out)
                header = streamlines_temp_data.space_attributes
                streamlines_temp = streamlines_temp_data.streamlines
                del streamlines_temp_data
            else:
                streamlines_temp = load_trk_remote(trkpath_subj, 'same', sftp_out).streamlines

            if side == 'left' or side=='all':
                streamlines_template.extend(streamlines_temp)
            elif side == 'right':
                affine_flip = np.eye(4)
                affine_flip[0, 0] = -1
                affine_flip[0, 3] = 0
                streamlines_template.extend(transform_streamlines(streamlines_temp, affine_flip,
                                                               in_place=False))
            else:
                raise Exception('unrecognized side, this should never happen, fix sides to be either left and right, or all')


    sg = lambda: (s for i, s in enumerate(streamlines_template))
    save_trk_header(filepath=trktemplate_paths, streamlines=sg, header=header,
                    affine=np.eye(4), verbose=verbose, sftp=sftp_out)
    timings.append(time.perf_counter())
    print(f'Saved streamlines at {trktemplate_paths}, took {timings[-1] - timings[-2]} seconds')
    #pickledump_remote(streams_dict_side, streams_dict_picklepaths, sftp_out)
    #timings.append(time.perf_counter())
    #print(f'Saved dictionary at {streams_dict_picklepaths}, took {timings[-1] - timings[0]} seconds')

feature2 = ResampleFeature(nb_points=bundle_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
qb = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=bundle_split)

num_streamlines = {}
streams_dict_picklepaths = {}
centroids_perside = {}

"""
side = 'left'

side_str = f'_{side}'

bundle_id_orig_txt = side_str

side_mask_path = os.path.join(MDT_mask_folder, f'IITmean_RPI_MDT_mask_left.nii.gz')
roi_mask_left = nib.load(side_mask_path)
streamlines_all = filter_streamlines(streamlines_template, roi_mask=roi_mask_left, world_coords=True,
                                      include=streamline_lr_inclusion,
                                      threshold=length_threshold)

side_mask_path = os.path.join(MDT_mask_folder, f'IITmean_RPI_MDT_mask_right.nii.gz')
roi_mask_right = nib.load(side_mask_path)
streamlines_right_rev = filter_streamlines(streamlines_template, roi_mask=roi_mask_right, world_coords=True,
                                      include=streamline_lr_inclusion,
                                      threshold=length_threshold)


streamlines_right_rev = transform_streamlines(streamlines_right_rev, affine_flip,
                                                   in_place=False)

streamlines_all.extend(streamlines_right_rev)
"""

affine_flip = np.eye(4)
affine_flip[0, 0] = -1
affine_flip[0, 3] = 0

bundles = qb.cluster(streamlines_template)
num_streamlines = bundles.clusters_sizes()

top_bundles = sorted(range(len(num_streamlines)), key=lambda i: num_streamlines[i], reverse=True)[:]
ordered_bundles = []
centroids_side = {}
centroids_side['left'] = []
centroids_side['right'] = []

bundle_split_used = bundle_split

while min(num_streamlines)<np.mean(num_streamlines)/5:
    bundle_split_used += 1
    qb = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=bundle_split_used)
    bundles = qb.cluster(streamlines_template)
    num_streamlines = bundles.clusters_sizes()

    top_bundles = sorted(range(len(num_streamlines)), key=lambda i: num_streamlines[i], reverse=True)[:bundle_split]

    num_streamlines = sorted(num_streamlines,reverse=True)[:bundle_split]


for bundle in top_bundles[:bundle_split]:
    ordered_bundles.append(bundles.clusters[bundle])
    centroids_side['left'].append(bundles.clusters[bundle].centroid)

for bundle_toflip in centroids_side['left']:
    centroids_side['right'].append(np.array(transform_streamlines(bundle_toflip, affine_flip, in_place=False)))


for side in sides:

    if side == 'all':
        side_str = ''
    else:
        side_str = f'_{side}'

    if bundle_id_orig is not None:
        #bundle_id_orig_txt = side_str + '_'.join(bundle_id_orig) + '_'
        bundle_id_orig_txt = side_str + '_' + bundle_id_orig[0]
    else:
        bundle_id_orig_txt = side_str

    pickled_centroids = os.path.join(pickle_folder, f'bundles_centroids{bundle_id_orig_txt}.py')
    if not checkfile_exists_remote(pickled_centroids, sftp_out) or overwrite:
        # pickledump_remote(bundles.centroids,pickled_centroids,sftp_out)
        pickledump_remote(centroids_side[side], pickled_centroids, sftp_out)
        print(f'Saved centroids at {pickled_centroids}, took {timings[-1] - timings[-2]} seconds')
    else:
        print(f'Centroids at {pickled_centroids} already exist')
    timings.append(time.perf_counter())

    for bundle_id in np.arange(bundle_split):
        full_bundle_id = bundle_id_orig_txt + f'_{bundle_id}'
        sg = lambda: (s for i, s in enumerate(centroids_side[side][bundle_id:bundle_id+1]))
        filepath_bundle = os.path.join(centroids_proj_path, f'centroid_bundle{full_bundle_id}.trk')
        save_trk_header(filepath=filepath_bundle, streamlines=sg, header=header, affine=np.eye(4), verbose=verbose,
                        sftp=sftp_out)

"""
else:

    for side in sides:

        side_str = f'_{side}'

        if bundle_id_orig is not None:
            bundle_id_orig_txt = side_str + '_'.join(bundle_id_orig) + '_'
        else:
            bundle_id_orig_txt = side_str

        side_mask_path = os.path.join(MDT_mask_folder, f'IITmean_RPI_MDT_mask{side_str}.nii.gz')
        roi_mask_side= nib.load(side_mask_path)

        streamlines_side = filter_streamlines(streamlines_template, roi_mask=roi_mask_side, world_coords=True, include=streamline_lr_inclusion,
                           threshold=length_threshold)

        bundles = qb.cluster(streamlines_side)
        num_streamlines = bundles.clusters_sizes()

        top_bundles = sorted(range(len(num_streamlines)), key=lambda i: num_streamlines[i], reverse=True)[:]
        ordered_bundles = []
        centroids = []
        for bundle in top_bundles[:bundle_split]:
            ordered_bundles.append(bundles.clusters[bundle])
            centroids.append(bundles.clusters[bundle].centroid)

        pickled_centroids = os.path.join(pickle_folder, f'bundles_centroids{bundle_id_orig_txt}_split_{bundle_split}.py')
        #pickled_centroids = os.path.join(pickle_folder, f'bundles_centroids_split_{bundle_split}.py')

        if not checkfile_exists_remote(pickled_centroids, sftp_out) or overwrite:
            # pickledump_remote(bundles.centroids,pickled_centroids,sftp_out)
            pickledump_remote(centroids, pickled_centroids, sftp_out)
            print(f'Saved centroids at {pickled_centroids}, took {timings[-1] - timings[-2]} seconds')
        else:
            print(f'Centroids at {pickled_centroids} already exist')
        timings.append(time.perf_counter())

        for new_bundle_id in np.arange(num_bundles):
            full_bundle_id = bundle_id_orig_txt + side_str + f'{new_bundle_id}'
            sg = lambda: (s for i, s in enumerate(centroids[new_bundle_id:new_bundle_id+1]))
            filepath_bundle = os.path.join(centroids_proj_path, f'centroid_bundle{full_bundle_id}.trk')
            save_trk_header(filepath=filepath_bundle, streamlines=sg, header=header, affine=np.eye(4), verbose=verbose,
                            sftp=sftp_out)
        del bundles
"""