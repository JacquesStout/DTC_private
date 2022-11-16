import numpy as np
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
from DTC.diff_handlers.connectome_handlers.connectome_handler import _to_voxel_coordinates_warning, retweak_points
from dipy.tracking.streamline import Streamlines
import nibabel as nib

def roi_target_lesserror(streamlines, affine, target_mask, include=True, verbose=False):
    target_mask = np.array(target_mask, dtype=bool, copy=True)
    lin_T, offset = _mapping_to_voxel(affine)
    #yield
    # End of initialization

    for sl in streamlines:
        ind = _to_voxel_coordinates_warning(sl, lin_T, offset)
        i, j, k = ind.T
        try:
            state = target_mask[i, j, k]
        except IndexError:
            entire = retweak_points(ind, target_mask.shape, verbose=verbose)
            i, j, k = entire.T
            state = target_mask[i, j, k]
            #raise ValueError("streamlines points are outside of target_mask")

        #include if even a single point of streamline is in mask
        if include is True or include=='all':
            if state.any() == True:
                yield sl
        #do not include at all if even a single point of streamline is not in mask
        elif include is False or include=='only_mask':
            if not np.invert(state).any():
                yield sl

        #only include the streamline points in mask (will look strange if the streamline loops back)
        elif include =='clip':
            yield sl[state]


def filter_streamlines(roi_mask, streamlines, include= 'all', label_list = None, world_coords = False, interactive=False):
    #interactive: Enables/disables interactive visualization

    if isinstance(roi_mask, str):
        roi_mask = nib.load(roi_mask)
    if isinstance(roi_mask, nib.Nifti1Image):
        affine = roi_mask.affine
        if label_list is not None:
            roi_mask_new = np.zeros(shape=roi_mask.shape, dtype=np.bool_)
            for i in np.arange(roi_mask.shape[0]):
                for j in np.arange(roi_mask.shape[1]):
                    for k in np.arange(roi_mask.shape[2]):
                        if roi_mask[i][j][k] in label_list:
                            roi_mask_new[i][j][k] = True
                        else:
                            roi_mask_new[i][j][k] = False
            roi_mask = roi_mask_new
        else:
            roi_mask = roi_mask.get_fdata()
            roi_mask = roi_mask.astype(np.bool_)


    if not isinstance(streamlines, nib.streamlines.ArraySequence):
        try:
            streamlines = streamlines.streamlines
        except:
            raise Exception('Unrecognizable streamline variable')
        if not isinstance(streamlines, nib.streamlines.ArraySequence):
            raise Exception('Unrecognizable streamline variable')

    roi_streamlines=streamlines
    #roi_streamlines = utils.target(streamlines, affine, roi_mask)
    roi_streamlines = roi_target_lesserror(streamlines, affine, roi_mask, include= include)
    roi_streamlines = Streamlines(roi_streamlines)

    from dipy.viz import window, actor, colormap as cmap

    # Make display objects
    if interactive:

        if world_coords:
            roi_streamlines = transform_streamlines(roi_streamlines, np.linalg.inv(affine))

        color = cmap.line_colors(roi_streamlines)
        roi_streamlines_actor = actor.line(roi_streamlines, color)
        ROI_actor = actor.contour_from_roi(roi_mask, color=(1., 1., 0.), opacity=0.1)

        r = window.Scene()

        r.add(roi_streamlines_actor)
        r.add(ROI_actor)

        # Save figures
        # window.record(r, n_frames=1, out_path='corpuscallosum_axial.png',
        #   size=(800, 800))
        print('ROI mask created - look at python window!')

        window.show(r)
        r.set_camera(position=[-1, 0, 0], focal_point=[0, 0, 0], view_up=[0, 0, 1])
        # window.record(r, n_frames=1, out_path='corpuscallosum_sagittal.png',
        #             size=(800, 800))

    return roi_streamlines