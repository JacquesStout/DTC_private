from os.path import join as pjoin

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import dipy.core.optimize as opt
from dipy.io.streamline import load_trk
import dipy.tracking.life as life
from dipy.viz import window, actor, colormap as cmap
import os
    # We'll need to know where the corpus callosum is from these variables:
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data, load_nifti
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
import nibabel as nib
from DTC.tract_manager.tract_handler import get_trk_params
import sys


def convert_tck_to_trk(input_file, output_file, ref):
    header = {}

    nii = nib.load(ref)
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    tck = nib.streamlines.load(input_file)
    nib.streamlines.save(tck.tractogram, output_file, header=header)

inpath = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/'
diff_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded.nii.gz'

data, affine, hardi_img = load_nifti(diff_path, return_img=True)
bval_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/15_dirs_checked.bval'
bvec_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/15_dirs_checked.bvec'
bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs)

fiber_model = life.FiberModel(gtab)
inv_affine = np.linalg.inv(hardi_img.affine)

fig_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/figures_Life'
ref_mask_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/Bruker_diffusion_test_15_bounded_mask.nii.gz'
cs_vals = [1,4,6,8,10,12]
#cs_vals = [1,4,6,8,10]
#cs_vals = []
#cs_vals.append(sys.argv[1])
num_tracks = '2mill'

get_tract_stats = True
get_life_figs = False

for cs_val in cs_vals:
    trk_path_folder = os.path.join(inpath, f'{cs_val}_perm_RAS')
    trk_path = os.path.join(trk_path_folder,f'{cs_val}_smallerTracks{num_tracks}.trk')
    if not os.path.exists(trk_path):
        tck_path = os.path.join(trk_path_folder,f'{cs_val}_smallerTracks{num_tracks}.tck')
        if os.path.exists(tck_path):
            convert_tck_to_trk(tck_path,trk_path,ref_mask_path)
        else:
            txt = f'Could not find either {tck_path} or {trk_path}'
            raise Exception(txt)

    trk_sft = load_trk(trk_path, 'same', bbox_valid_check=False)
    streamlines = trk_sft.streamlines

    if get_tract_stats:
        numtracts, minlength, maxlength, meanlength, stdlength = get_trk_params(streamlines)
        txt_path = os.path.join(trk_path_folder,f'CS_{cs_val}_{num_tracks}.txt')
        txt = f'Number of streamlines: {numtracts}\nMinimum length: {minlength}\nMaximum length: {maxlength}\nMean length: {meanlength}\nStd of length: {stdlength}'
        print(txt)
        with open(txt_path, 'w') as f:
            f.write(txt)

    if get_life_figs:
        fiber_fit = fiber_model.fit(data, streamlines, affine=np.eye(4))

        fig, ax = plt.subplots(1)
        ax.hist(fiber_fit.beta, bins=100, histtype='step')
        ax.set_xlabel('Fiber weights')
        ax.set_ylabel('# fibers')
        fig.savefig(os.path.join(fig_path,f'CS_{cs_val}_{num_tracks}_beta_histogram.png'))

        model_predict = fiber_fit.predict()
        model_error = model_predict - fiber_fit.data
        model_rmse = np.sqrt(np.mean(model_error[:, 10:] ** 2, -1))

        beta_baseline = np.zeros(fiber_fit.beta.shape[0])
        pred_weighted = np.reshape(opt.spdot(fiber_fit.life_matrix, beta_baseline),
                                   (fiber_fit.vox_coords.shape[0],
                                    np.sum(~gtab.b0s_mask)))
        mean_pred = np.empty((fiber_fit.vox_coords.shape[0], gtab.bvals.shape[0]))
        S0 = fiber_fit.b0_signal

        mean_pred[..., gtab.b0s_mask] = S0[:, None]
        mean_pred[..., ~gtab.b0s_mask] =\
            (pred_weighted + fiber_fit.mean_signal[:, None]) * S0[:, None]
        mean_error = mean_pred - fiber_fit.data
        mean_rmse = np.sqrt(np.mean(mean_error ** 2, -1))

        fig, ax = plt.subplots(1)
        ax.hist(mean_rmse - model_rmse, bins=100, histtype='step')
        ax.text(0.2, 0.9, 'Median RMSE, mean model: %.2f' % np.median(mean_rmse),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.text(0.2, 0.8, 'Median RMSE, LiFE: %.2f' % np.median(model_rmse),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel('RMS Error')
        ax.set_ylabel('# voxels')
        fig.savefig(os.path.join(fig_path,f'CS_{cs_val}_{num_tracks}_error_histograms.png'))

