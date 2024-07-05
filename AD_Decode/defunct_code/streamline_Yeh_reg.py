import os, glob
import numpy as np
from DTC.file_manager.computer_nav import load_trk_remote
from DTC.tract_manager.tract_save import save_trk_header
from DTC.file_manager.computer_nav import get_mainpaths, load_nifti_remote, load_trk_remote, loadmat_remote
from time import sleep
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.data import two_cingulum_bundles
from dipy.tracking.streamline import set_number_of_points
from dipy.viz import window, actor
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.streamline import deform_streamlines, transform_streamlines
from scipy.io import loadmat
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles
import pandas as pd


def show_both_bundles(bundles, colors=None, show=True, fname=None):

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
        lines_actor.RotateX(-90)
        lines_actor.RotateZ(90)
        scene.add(lines_actor)
    if show:
        window.show(scene)
    if fname is not None:
        sleep(1)
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


def length_cut_streamlines(streamlines,threshold_min=0,threshold_max =None):
    length_all = list(streamlines._lengths)
    if threshold_max is not None:
        target_streamlines = [s for s,len in zip(streamlines,length_all) if len > threshold_min and len<threshold_max]
    else:
        target_streamlines = [s for s,len in zip(streamlines,length_all) if len > threhsold_min]

    streamlines = ArraySequence(target_streamlines)
    return streamlines

def load_matrix_in_any_format(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    elif ext == '.mat':
        # .mat are actually dictionnary. This function support .mat from
        # antsRegistration that encode a 4x4 transformation matrix.
        transfo_dict = loadmat(filepath)
        lps2ras = np.diag([-1, -1, 1])

        rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
        trans = transfo_dict['AffineTransform_float_3_3'][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))

    return data


mainpath = '/Volumes/Data/Badea/Lab/'
outpath = os.path.join(mainpath, 'human/AD_Decode_trk_transfer')

subj = 'S00775'

subset = True
if subset:
    subset_str = '_subset'
else:
    subset_str = ''

trk_target_path = os.path.join(mainpath,f'human/AD_Decode_trk_transfer/TRK/{subj}_smallerTracks2mill.trk')

target_preprocess_path = f'/Volumes/Data/Badea/Lab/human/' \
    f'AD_Decode_trk_transfer/TRK_aligned/{subj}_smallerTracks2mill_preprocessed.trk'

trk_target_aligned_path = f'/Volumes/Data/Badea/Lab/human/' \
    f'AD_Decode_trk_transfer/TRK_aligned/{subj}_smallerTracks2mill_aligned{subset_str}.trk'

trk_model_folder = '/Users/jas/jacques/Yeh_tracks/'
trk_model_all_path = '/Users/jas/jacques/Yeh_tracks/Yeh_all.trk'
path_trk_tempdir = os.path.join(outpath, 'TRK_transition')

verbose = False
sftp_out = None
overwrite=True
apply_preprocess = True
display = True

if not os.path.exists(trk_target_aligned_path) or overwrite:

    if not os.path.exists(trk_model_all_path):
        trk_files = glob.glob(os.path.join(trk_model_folder,'*.trk'))
        for trk_file in trk_files:
            if not 'model_streamlines' in locals():
                model_streamlines_trk = load_trk_remote(trk_file,'same')
                model_streamlines = model_streamlines_trk.streamlines
                header = model_streamlines_trk.space_attributes
            else:
                model_streamlines.extend(load_trk_remote(trk_file,'same').streamlines)
        sg = lambda: (s for i, s in enumerate(model_streamlines))
        save_trk_header(filepath=trk_model_all_path, streamlines=sg, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp_out)
    else:
        model_streamlines_orig = load_trk_remote(trk_model_all_path,'same').streamlines

    #cb_subj1, cb_subj2 = two_cingulum_bundles()

    if apply_preprocess:

        if not os.path.exists(target_preprocess_path):
            target_trk = load_trk_remote(trk_target_path,'same')
            target_streamlines = target_trk.streamlines
            header = target_trk.space_attributes

            transform_to_init_mat_path = os.path.join(path_trk_tempdir, f'{subj}_subj_to_init.mat')

            transform_to_init = transform_to_init_mat_path
            toinit_reorient_struct = loadmat_remote(transform_to_init, sftp_out)

            var_name = list(toinit_reorient_struct.keys())[0]
            toinit_ants = toinit_reorient_struct[var_name]
            # toinit_mat = convert_ants_vals_to_affine(toinit_ants)
            # toinit_mat = convert_ants_vals_to_affine(toinit_ants)
            toinit_mat = load_matrix_in_any_format(transform_to_init_mat_path)

            toinit_trans_mat = np.eye(4)
            toinit_trans_mat[:, 3] = toinit_mat[:, 3]
            # toinit_trans_mat[2,3] = toinit_trans_mat[2,3]+4
            streamlines_init_1 = transform_streamlines(target_streamlines, np.linalg.inv(toinit_trans_mat))

            # streamlines_init = transform_streamlines(streamlines_postreorient, np.linalg.inv(toinit_mat))
            toinit_rot_mat = np.eye(4)
            toinit_rot_mat[:3, :3] = toinit_mat[:3, :3]
            target_streamlines_init = transform_streamlines(streamlines_init_1, np.linalg.inv(toinit_rot_mat))

            save_trk_header(filepath=target_preprocess_path, streamlines=target_streamlines_init, header=header,
                    affine=np.eye(4), verbose=verbose, sftp=sftp_out)
            target_streamlines_orig = target_streamlines_init
        else:
            target_trk = load_trk_remote(target_preprocess_path, 'same')
            target_streamlines_orig = target_trk.streamlines
            header = target_trk.space_attributes

    else:
        target_trk = load_trk_remote(trk_target_path, 'same').streamlines
        target_streamlines_orig = target_trk.streamlines
        header = target_trk.space_attributes

    fix_streamlines = True

    if fix_streamlines:
        threshold_min = 50
        threshold_max = 250

        target_streamlines = length_cut_streamlines(target_streamlines_orig,threshold_min=threshold_min, threshold_max=threshold_max)
        model_streamlines = length_cut_streamlines(model_streamlines_orig,threshold_min=threshold_min, threshold_max=threshold_max)


    if subset:
        subset_size = np.size(model_streamlines._lengths)
        subset_size = 50000
        target_streamlines_subset = select_random_set_of_streamlines(target_streamlines,subset_size)
        model_streamlines_subset = select_random_set_of_streamlines(model_streamlines,subset_size)
    else:
        target_streamlines_subset = target_streamlines
        model_streamlines_subset = model_streamlines

    target_streamlines_set = set_number_of_points(target_streamlines, 20)
    model_streamlines_set = set_number_of_points(model_streamlines, 20)

    #Bundling of target
    feature2 = ResampleFeature(nb_points=20)
    metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
    qb = QuickBundles(threshold=15, metric=metric2)

    bundles = qb.cluster(target_streamlines_set)
    num_streamlines_bundles = bundles.clusters_sizes()
    top_bundles = sorted(range(len(num_streamlines_bundles)), key=lambda i: num_streamlines_bundles[i], reverse=True)[:]

    i=0
    target_bundles = []
    for bundle in top_bundles:
        if num_streamlines_bundles[bundle]>50:
            target_bundles.append(bundles.clusters[bundle])
        else:
            break

    #Bundling of model (might be unnecessary, find alternative??)
    feature2 = ResampleFeature(nb_points=20)
    metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
    qb = QuickBundles(threshold=15, metric=metric2)

    bundles = qb.cluster(model_streamlines_set)
    num_streamlines_bundles = bundles.clusters_sizes()
    top_bundles = sorted(range(len(num_streamlines_bundles)), key=lambda i: num_streamlines_bundles[i], reverse=True)[:]

    i=0
    model_bundles = []
    for bundle in top_bundles:
        if num_streamlines_bundles[bundle]>50:
            model_bundles.append(bundles.clusters[bundle])
        else:
            break

    srr = StreamlineLinearRegistration()

    model_centroids = [bundle.centroid for bundle in model_bundles]
    target_centroids = [bundle.centroid for bundle in target_bundles]

    srm = srr.optimize(static=model_centroids, moving=target_centroids)

    target_streamlines_aligned = srm.transform(target_streamlines)
    target_streamlines_aligned_set = srm.transform(target_streamlines_set)

    sg_al = lambda: (s for i, s in enumerate(target_streamlines_aligned))
    save_trk_header(filepath=trk_target_aligned_path, streamlines=sg_al, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp_out)

    if display:
        show_both_bundles([model_streamlines, target_streamlines],
                          colors=[window.colors.orange, window.colors.red],
                          show=False,
                          fname='before_registration.png')

        show_both_bundles([model_streamlines, target_streamlines_aligned],
                          colors=[window.colors.orange, window.colors.red],
                          show=False,
                          fname='after_registration.png')


metadf_path = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx'
meta_df = pd.read_excel(metadf_path)

wm_regions = meta_df.loc[meta_df['Region type'] == 'White', 'abbreviation_IIT'].tolist()

for wm_region in wm_regions:
    trk_file = os.path.join(trk_model_folder,f'{wm_region}.trk')

trk_files = glob.glob(os.path.join(trk_model_folder, '*.trk'))
for trk_file in trk_files:

    if not 'model_streamlines' in locals():
        model_streamlines_trk = load_trk_remote(trk_file, 'same')
        model_streamlines = model_streamlines_trk.streamlines
        header = model_streamlines_trk.space_attributes
    else:
        model_streamlines.extend(load_trk_remote(trk_file, 'same').streamlines)
sg = lambda: (s for i, s in enumerate(model_streamlines))