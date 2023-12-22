


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
from DTC.tract_manager.tract_handler import ratio_to_str
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles
import argparse
from DTC.wrapper_tools import parse_list_arg

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--id', type=parse_list_arg, help='An integer or a list of integers')
parser.add_argument('--split', type=int, help='An integer for splitting')
parser.add_argument('--proj', type=str, help='The project path or name')

args = parser.parse_args()
bundle_id_orig = args.id
bundle_id_orig_txt = '_'.join(bundle_id_orig)
bundle_split = args.split
project_summary_file = args.proj

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

distance = int(params['distance'])
bundle_points = int(params['bundle_points'])
num_bundles = int(params['num_bundles'])
path_TRK = params['path_trk']

if bundle_split is None:
    try:
        bundle_split = int(params['bundle_split'])
    except:
        bundle_split = 6


overwrite=False
verbose = False

if remote_output:
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

if 'santorini' in socket.gethostname().split('.')[0]:
    lab_folder = '/Volumes/Data/Badea/Lab'
if 'blade' in socket.gethostname().split('.')[0]:
    lab_folder = '/mnt/munin2/Badea/Lab'

if project == 'AD_Decode':
    SAMBA_MDT = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_dwi.nii.gz')
    MDT_mask_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/atlas_to_MDT')
    ref_MDT_folder = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images')
    anat_path = os.path.join(lab_folder,'mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz')


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
mkcdir([combined_trk_folder],sftp_out)
trktemplate_paths = {}
streams_dict_picklepaths = {}

allsides = ['right', 'right_f', 'left', 'left_f', 'combined']

timings = []

streams_dict_side = {}
timings.append(time.perf_counter())

pickle_folder = os.path.join(proj_path, 'pickle_roi'+ratiostr)

trkpaths = {}

affine_flip = np.eye(4)
affine_flip[0, 0] = -1
affine_flip[0, 3] = 0

comb_sides = ['left','right_f']
centroid_sides =['left','right']
dict_revtracker = {'right_f': 'right', 'left_f': 'left', 'right': 'right', 'left': 'left'}
reversesides = ['left_f','right_f']

for side in comb_sides:
    streamlines_template[side] = nib.streamlines.array_sequence.ArraySequence()

streamlines_combined = nib.streamlines.array_sequence.ArraySequence()

for side in comb_sides:

    for subject in template_subjects:
        trkpaths[subject, 'right'] = os.path.join(trk_proj_path, f'{subject}_right_bundle_{bundle_id_orig_txt}.trk')
        trkpaths[subject, 'left'] = os.path.join(trk_proj_path, f'{subject}_left_bundle_{bundle_id_orig_txt}.trk')

        if 'header' not in locals():
            streamlines_temp_data = load_trk_remote(trkpaths[subject, dict_revtracker[side]], 'same', sftp_out)
            header = streamlines_temp_data.space_attributes
            streamlines_temp = streamlines_temp_data.streamlines
            del streamlines_temp_data
        else:
            streamlines_temp = load_trk_remote(trkpaths[subject, dict_revtracker[side]], 'same', sftp_out).streamlines

        if side not in reversesides:
            streamlines_template[side].extend(streamlines_temp)
        else:
            affine_flip = np.eye(4)
            affine_flip[0, 0] = -1
            affine_flip[0, 3] = 0
            streamlines_template[side].extend(transform_streamlines(streamlines_temp, affine_flip,
                                                           in_place=False))
for side in comb_sides:
    streamlines_combined.extend(streamlines_template[side])

feature2 = ResampleFeature(nb_points=bundle_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
qb = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=bundle_split)

num_streamlines = {}
streams_dict_picklepaths = {}
centroids_perside = {}

bundles = qb.cluster(streamlines_combined)
num_streamlines[side] = bundles.clusters_sizes()
print(f'side {side} has {num_streamlines[side]}')

top_bundles = sorted(range(len(num_streamlines[side])), key=lambda i: num_streamlines[side][i], reverse=True)[:]
ordered_bundles = []
for bundle in top_bundles[:bundle_split]:
    ordered_bundles.append(bundles.clusters[bundle])

centroids_perside['combined'] = []
for i in np.arange(bundle_split):
    centroids_perside['combined'].append(ordered_bundles[i].centroid)

centroids_perside['left'] = centroids_perside['combined']

centroids_perside['right'] = []

affine_flip = np.eye(4)
affine_flip[0, 0] = -1
affine_flip[0, 3] = 0
for bundle_toflip in centroids_perside['left']:
    centroids_perside['right'].append(np.array(transform_streamlines(bundle_toflip, affine_flip, in_place=False)))

timings.append(time.perf_counter())
print(f'Organized top {bundle_split} bundles for left and right side, took {timings[-1] - timings[-2]} seconds')

for side in centroid_sides:
    pickled_centroids = os.path.join(pickle_folder, f'bundles_centroids_{side}_{bundle_id_orig_txt}_split_{bundle_split}.py')
    if not checkfile_exists_remote(pickled_centroids, sftp_out) or overwrite:
        # pickledump_remote(bundles.centroids,pickled_centroids,sftp_out)
        pickledump_remote(centroids_perside[side], pickled_centroids, sftp_out)
        print(f'Saved centroids at {pickled_centroids}, took {timings[-1] - timings[-2]} seconds')
    else:
        print(f'Centroids at {pickled_centroids} already exist')
    timings.append(time.perf_counter())

    for new_bundle_id in np.arange(bundle_split):
        full_bundle_id = bundle_id_orig_txt + f'_{new_bundle_id}'
        sg = lambda: (s for i, s in enumerate(centroids_perside[side][new_bundle_id:new_bundle_id+1]))
        filepath_bundle = os.path.join(centroids_proj_path, f'centroid_{side}_bundle_{full_bundle_id}.trk')
        save_trk_header(filepath=filepath_bundle, streamlines=sg, header=header, affine=np.eye(4), verbose=verbose,
                        sftp=sftp_out)
del bundles
