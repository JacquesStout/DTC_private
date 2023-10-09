import os
from DTC.file_manager.file_tools import mkcdir
import nibabel as nib
from dipy.viz import regtools
import numpy as np
import matplotlib.pyplot as plt

def double_view_compare(L, R, slice_index=None, slice_type=1, ltitle='Left',
                   rtitle='Right', fname=None, **fig_kwargs):

    #Taken from regtools.overlay_slices, removed the overlya aspect

    # Normalize the intensities to [0,255]
    sh = L.shape
    L = np.asarray(L, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    L = 255 * (L - L.min()) / (L.max() - L.min())
    R = 255 * (R - R.min()) / (R.max() - R.min())
    # Create the color image to draw the overlapped slices into, and extract
    # the slices (note the transpositions)
    if slice_type == 0:
        if slice_index is None:
            slice_index = sh[0] // 2
        colorImage = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
        ll = np.asarray(L[slice_index, :, :]).astype(np.uint8).T
        rr = np.asarray(R[slice_index, :, :]).astype(np.uint8).T
    elif slice_type == 1:
        if slice_index is None:
            slice_index = sh[1] // 2
        colorImage = np.zeros(shape=(sh[2], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, slice_index, :]).astype(np.uint8).T
        rr = np.asarray(R[:, slice_index, :]).astype(np.uint8).T
    elif slice_type == 2:
        if slice_index is None:
            slice_index = sh[2] // 2
        colorImage = np.zeros(shape=(sh[1], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, :, slice_index]).astype(np.uint8).T
        rr = np.asarray(R[:, :, slice_index]).astype(np.uint8).T
    else:
        print("Slice type must be 0, 1 or 2.")
        return

    fig = regtools._tile_plot([ll, rr],
                     [ltitle, rtitle],
                     cmap=plt.cm.gray, origin='lower')

    # Save the figure to disk, if requested
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', **fig_kwargs)

CS_orig = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/Bruker_diffusion_test_15_bounded.nii.gz'
CS_folder = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/'

outpath_diffs = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/difference_tests'
outpath_figs = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/Figures'
mkcdir([outpath_figs,outpath_diffs])

CS_vals = ['4','5.9998','8','10.0009','11.9985']

orients = [0,1,2]

for CS_val in CS_vals:

    CS_val_simp = int(np.round(float(CS_val)))

    if float(CS_val) <7:
        TV_val = 0.0025
    else:
        TV_val = 0.005

    L1_val = 0.0025

    CS_path = os.path.join(CS_folder,
                        f'Bruker_diffusion_test_15_0.045_af_{CS_val}x__TV_and_L1_wavelet_{TV_val}_{L1_val}_bart_recon.nii.gz')

    orig_data = nib.load(CS_orig).get_fdata()
    CS_data = nib.load(CS_path).get_fdata()
    for orient in orients:
        #slice = 50
        slice = int(np.round((np.shape(orig_data)[orient])/2))

        orients_str = ['sagittal', 'coronal', 'axial']

        fig_path_1 = os.path.join(outpath_figs, f'original_vs_CS_{CS_val}x_TV_{TV_val}_L1_{L1_val}_orient_{orients_str[orient]}_slice_{slice}.png')
        fig_path_2 = os.path.join(outpath_figs, f'original_vs_diff_{CS_val}x_TV_{TV_val}_L1_{L1_val}_orient_{orients_str[orient]}_slice_{slice}.png')


        double_view_compare(orig_data[:,:,:,0], CS_data[:,:,:,0], slice, orient,
                                "Original", "Compressed",fig_path_1)
        double_view_compare(orig_data[:,:,:,0], np.abs(orig_data[:,:,:,0]-CS_data[:,:,:,0]), slice, orient,
                                "Original", "Difference",fig_path_2)

"""
regtools.overlay_slices(orig_data[:,:,:,0], CS_data[:,:,:,0], slice, orient,
                        "Original", "Compressed",fig_path_1)
regtools.overlay_slices(orig_data[:,:,:,0], np.abs(orig_data[:,:,:,0]-CS_data[:,:,:,0]), slice, orient,
                        "Original", "Difference",fig_path_2)
"""
"""
diffpath = os.path.join(outpath_diffs,
                        f'Bruker_diffusion_test_15_0.045_af_{CS_val}x__TV_and_L1_wavelet_{TV_val}_{L1_val}_bart_recon_diff.nii.gz')

if not os.path.exists(diffpath):
"""