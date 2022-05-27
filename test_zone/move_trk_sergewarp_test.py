
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
import nibabel as nib
import numpy as np
from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration,
                                 transform_origins)
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import deform_streamlines, transform_streamlines
from nifti_handlers.nifti_handler import extract_nii_info
from dipy.viz import window, actor
import os
from tract_manager.tract_save import save_trk_header
from file_manager.file_tools import mkcdir
from visualization_tools.visualization_tools.tract_visualize import show_bundles, setup_view
from dipy.io.utils import get_reference_info, is_header_compatible
from dipy.tracking.streamline import transform_streamlines
from nibabel.streamlines.array_sequence import ArraySequence
from scipy.ndimage import map_coordinates
from scilpy.tracking.tools import smooth_line_gaussian, smooth_line_spline
from dipy.io.stateful_tractogram import StatefulTractogram, Space


def cut_invalid_streamlines(sft):
    """ Cut streamlines so their longest segment are within the bounding box.
    This function keeps the data_per_point and data_per_streamline.
    Parameters
    ----------
    sft: StatefulTractogram
        The sft to remove invalid points from.
    Returns
    -------
    new_sft : StatefulTractogram
        New object with the invalid points removed from each streamline.
    cutting_counter : int
        Number of streamlines that were cut.
    """
    if not len(sft):
        return sft, 0

    # Keep track of the streamlines' original space/origin
    space = sft.space
    origin = sft.origin

    sft.to_vox()
    sft.to_corner()

    copy_sft = copy.deepcopy(sft)
    epsilon = 0.001
    indices_to_remove, _ = copy_sft.remove_invalid_streamlines()

    new_streamlines = []
    new_data_per_point = {}
    new_data_per_streamline = {}
    for key in sft.data_per_point.keys():
        new_data_per_point[key] = []
    for key in sft.data_per_streamline.keys():
        new_data_per_streamline[key] = []

    cutting_counter = 0
    for ind in range(len(sft.streamlines)):
        # No reason to try to cut if all points are within the volume
        if ind in indices_to_remove:
            best_pos = [0, 0]
            cur_pos = [0, 0]
            for pos, point in enumerate(sft.streamlines[ind]):
                if (point < epsilon).any() or \
                        (point >= sft.dimensions - epsilon).any():
                    cur_pos = [pos+1, pos+1]
                if cur_pos[1] - cur_pos[0] > best_pos[1] - best_pos[0]:
                    best_pos = cur_pos
                cur_pos[1] += 1

            if not best_pos == [0, 0]:
                new_streamlines.append(
                    sft.streamlines[ind][best_pos[0]:best_pos[1]-1])
                cutting_counter += 1
                for key in sft.data_per_streamline.keys():
                    new_data_per_streamline[key].append(
                        sft.data_per_streamline[key][ind])
                for key in sft.data_per_point.keys():
                    new_data_per_point[key].append(
                        sft.data_per_point[key][ind][best_pos[0]:best_pos[1]-1])
            else:
                logging.warning('Streamlines entirely out of the volume.')
        else:
            new_streamlines.append(sft.streamlines[ind])
            for key in sft.data_per_streamline.keys():
                new_data_per_streamline[key].append(
                    sft.data_per_streamline[key][ind])
            for key in sft.data_per_point.keys():
                new_data_per_point[key].append(sft.data_per_point[key][ind])
    new_sft = StatefulTractogram.from_sft(new_streamlines, sft,
                                          data_per_streamline=new_data_per_streamline,
                                          data_per_point=new_data_per_point)

    # Move the streamlines back to the original space/origin
    sft.to_space(space)
    sft.to_origin(origin)

    new_sft.to_space(space)
    new_sft.to_origin(origin)

    return new_sft, cutting_counter


def upsample_tractogram(
    sft, nb, point_wise_std=None,
    streamline_wise_std=None, gaussian=None, spline=None, seed=None
):
    """
    Generate new streamlines by either adding gaussian noise around
    streamlines' points, or by translating copies of existing streamlines
    by a random amount.
    Parameters
    ----------
    sft : StatefulTractogram
        The tractogram to upsample
    nb : int
        The target number of streamlines in the tractogram.
    point_wise_std : float
        The standard deviation of the gaussian to use to generate point-wise
        noise on the streamlines.
    streamline_wise_std : float
        The standard deviation of the gaussian to use to generate
        streamline-wise noise on the streamlines.
    gaussian: float
        The sigma used for smoothing streamlines.
    spline: (float, int)
        Pair of sigma and number of control points used to model each
        streamline as a spline and smooth it.
    seed: int
        Seed for RNG.
    Returns
    -------
    new_sft : StatefulTractogram
        The upsampled tractogram.
    """
    assert bool(point_wise_std) ^ bool(streamline_wise_std), \
        'Can only add either point-wise or streamline-wise noise' + \
        ', not both nor none.'

    rng = np.random.RandomState(seed)

    # Get the number of streamlines to add
    nb_new = nb - len(sft.streamlines)

    # Get the streamlines that will serve as a base for new ones
    indices = rng.choice(
        len(sft.streamlines), nb_new)
    new_streamlines = sft.streamlines.copy()

    # For all selected streamlines, add noise and smooth
    for s in sft.streamlines[indices]:
        if point_wise_std:
            noise = rng.normal(scale=point_wise_std, size=s.shape)
        elif streamline_wise_std:
            noise = rng.normal(
                scale=streamline_wise_std, size=s.shape[-1])
        new_s = s + noise
        if gaussian:
            new_s = smooth_line_gaussian(new_s, gaussian)
        elif spline:
            new_s = smooth_line_spline(new_s, spline[0],
                                       spline[1])

        new_streamlines.append(new_s)

    new_sft = StatefulTractogram.from_sft(new_streamlines, sft)
    return new_sft

def transform_warp_sft(sft, linear_transfo, target, inverse=False,
                       reverse_op=False, deformation_data=None,
                       remove_invalid=True, cut_invalid=False):
    """ Transform tractogram using a affine Subsequently apply a warp from
    antsRegistration (optional).
    Remove/Cut invalid streamlines to preserve sft validity.
    Parameters
    ----------
    sft: StatefulTractogram
        Stateful tractogram object containing the streamlines to transform.
    linear_transfo: numpy.ndarray
        Linear transformation matrix to apply to the tractogram.
    target: Nifti filepath, image object, header
        Final reference for the tractogram after registration.
    inverse: boolean
        Apply the inverse linear transformation.
    reverse_op: boolean
        Apply both transformation in the reverse order
    deformation_data: np.ndarray
        4D array containing a 3D displacement vector in each voxel.
    remove_invalid: boolean
        Remove the streamlines landing out of the bounding box.
    cut_invalid: boolean
        Cut invalid streamlines rather than removing them. Keep the longest
        segment only.
    Return
    ----------
    new_sft : StatefulTractogram
    """

    # Keep track of the streamlines' original space/origin
    space = sft.space
    origin = sft.origin
    dtype = sft.streamlines._data.dtype

    sft.to_rasmm()
    sft.to_center()

    if len(sft.streamlines) == 0:
        return StatefulTractogram(sft.streamlines, target,
                                  Space.RASMM)

    if inverse:
        linear_transfo = np.linalg.inv(linear_transfo)

    if not reverse_op:
        streamlines = transform_streamlines(sft.streamlines,
                                            linear_transfo)
    else:
        streamlines = sft.streamlines

    if deformation_data is not None:
        if not reverse_op:
            affine, _, _, _ = get_reference_info(target)
        else:
            affine = sft.affine

        # Because of duplication, an iteration over chunks of points is
        # necessary for a big dataset (especially if not compressed)
        streamlines = ArraySequence(streamlines)
        nb_points = len(streamlines._data)
        cur_position = 0
        chunk_size = 1000000
        nb_iteration = int(np.ceil(nb_points/chunk_size))
        inv_affine = np.linalg.inv(affine)

        while nb_iteration > 0:
            max_position = min(cur_position + chunk_size, nb_points)
            points = streamlines._data[cur_position:max_position]

            # To access the deformation information, we need to go in VOX space
            # No need for corner shift since we are doing interpolation
            cur_points_vox = np.array(transform_streamlines(points,
                                                            inv_affine)).T

            x_def = map_coordinates(deformation_data[..., 0],
                                    cur_points_vox.tolist(), order=1)
            y_def = map_coordinates(deformation_data[..., 1],
                                    cur_points_vox.tolist(), order=1)
            z_def = map_coordinates(deformation_data[..., 2],
                                    cur_points_vox.tolist(), order=1)

            # ITK is in LPS and nibabel is in RAS, a flip is necessary for ANTs
            #final_points = np.array([-1*x_def, -1*y_def, z_def])
            #No modif required
            final_points = np.array([x_def, y_def, z_def])
            final_points += np.array(points).T

            streamlines._data[cur_position:max_position] = final_points.T
            cur_position = max_position
            nb_iteration -= 1

    if reverse_op:
        streamlines = transform_streamlines(streamlines,
                                            linear_transfo)

    streamlines._data = streamlines._data.astype(dtype)
    new_sft = StatefulTractogram(streamlines, target, Space.RASMM,
                                 data_per_point=sft.data_per_point,
                                 data_per_streamline=sft.data_per_streamline)
    if cut_invalid:
        new_sft, _ = cut_invalid_streamlines(new_sft)
    elif remove_invalid:
        new_sft.remove_invalid_streamlines()

    # Move the streamlines back to the original space/origin
    sft.to_space(space)
    sft.to_origin(origin)

    new_sft.to_space(space)
    new_sft.to_origin(origin)

    return new_sft

def show_template_bundles(bundles, image, show=True, fname=None):

    scene = window.Scene()

    if isinstance(image, nib.nifti1.Nifti1Image):
        image_data = np.asarray(image.dataobj)
        affine = image._affine
    elif isinstance(image, str):
        image_data, affine = nib.load(image)
    else:
        affine = np.eye(4)
    template_actor = actor.slicer(image_data, affine)
    scene.add(template_actor)

    lines_actor = actor.streamtube(bundles, window.colors.orange,
                                   linewidth=0.3)
    scene.add(lines_actor)

    if show:
        window.show(scene)
    if fname is not None:
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


mainpath = '/Users/jas/jacques/AD_Decode_warp_test/'

trk_prewarp_fixed = os.path.join(mainpath,'TRK_save_moving/S02227_stepsize_2_ratio_100_wholebrain_preprocess_postrigid_affine_fixed.trk')
trk_postwarp = os.path.join(mainpath,'TRK_save_moving/S02227_stepsize_2_ratio_100_postwarp_quicktest.trk')
trk_postwarp_svtsave = os.path.join(mainpath,'TRK_save_moving/S02227_stepsize_2_ratio_100_postwarp_quicktest_savetracto.trk')
figspath = os.path.join(mainpath,'Figures/')
runno_to_MDT = os.path.join(mainpath,'Transforms/S02227_to_MDT_warp.nii.gz')
MDT_to_runno = os.path.join(mainpath,'Transforms/MDT_to_S02227_warp.nii.gz')
warping = 'dipy'
mkcdir(figspath)

overwrite = False
view_streamlines = False
save_warp = False


if not os.path.exists(trk_prewarp_fixed):
    from tract_manager.tract_handler import trk_fixer
    trk_preprocess_prewarp = os.path.join(mainpath,'TRK_save_moving/S02227_stepsize_2_ratio_100_wholebrain_preprocess_postrigid_affine.trk')
    trk_fixer(trk_preprocess_prewarp, trk_prewarp_fixed, verbose=True)

streamlines_prewarp_obj = load_trk(trk_prewarp_fixed,'same')
streamlines_prewarp = streamlines_prewarp_obj.streamlines
header = streamlines_prewarp_obj.space_attributes

moving_path = os.path.join(mainpath,'NII_save_moving/S02227_fa_postrigid_affine.nii.gz')
static_path = os.path.join(mainpath,'DWI_MDT/S02227_fa_postwarp.nii.gz')
moving = nib.load(moving_path)
static = nib.load(static_path)

lut_cmap = actor.colormap_lookup_table(
    scale_range=(0.01, 0.5))
scene = None

metric = CCMetric(3)
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
static_data = np.asarray(static.dataobj)
moving_data = np.asarray(moving.dataobj)

affine_map = transform_origins(static_data, static._affine, moving_data, moving._affine)

warped_image_dipy_path = os.path.join(mainpath,'DWI_MDT/S02227_fa_postwarp_dipywarp.nii.gz')

warping = 'precalc'
#warping = 'precalc'

overwrite= True
if warping == 'dipy':
    if not os.path.exists(warped_image_dipy_path) or overwrite:
        mapping = sdr.optimize(static_data, moving_data, static._affine, moving._affine)
        warped_moving = mapping.transform(moving_data)
        warp = mapping.get_forward_field()
        #warp = = mapping.get_backward_field()
        warped_image_test = nib.Nifti1Image(warp, static._affine)
        nib.save(warped_image_test, warped_image_dipy_path)
        print(f'Saved the warped image to {warped_image_dipy_path}')
    else:
        warp, warp_affine, vox_size, header_warp, ref_info = extract_nii_info(warped_image_dipy_path)
elif warping == 'precalc':
    warp, warp_affine, vox_size, header_warp, ref_info = extract_nii_info(MDT_to_runno)
    warp = warp[:, :, :, 0, :]


vox_size = moving.header.get_zooms()[0]
target_isocenter = np.diag(np.array([vox_size, vox_size, vox_size, 1]))

target_isocenter = moving._affine


test_streamlines_1 = transform_streamlines(streamlines_prewarp, np.linalg.inv(warp_affine),
                      in_place=False)

streamlines_1 = StatefulTractogram(test_streamlines_1, static, Space.RASMM,
                                 data_per_point=streamlines_prewarp_obj.data_per_point,
                                 data_per_streamline=streamlines_prewarp_obj.data_per_streamline)


view_streamlines=False
if view_streamlines:
    scene = setup_view(test_streamlines_1[:100], colors=lut_cmap, ref=warp[:,:,:,0], world_coords=False, objectvals=[None],
                       colorbar=False, record=None, scene=scene)

affine = np.eye(4)
linear_transfo = affine

streamlines_1 = StatefulTractogram(test_streamlines_1, static, Space.RASMM,
                                 data_per_point=streamlines_prewarp_obj.data_per_point,
                                 data_per_streamline=streamlines_prewarp_obj.data_per_streamline)

mni_trksl = transform_warp_sft(streamlines_1, linear_transfo, static, inverse=False,
                       reverse_op=False, deformation_data=None,
                       remove_invalid=False, cut_invalid=False)

view_streamlines=False
if view_streamlines:
    scene = setup_view(mni_trksl.streamlines[:100], colors=lut_cmap, ref=warp[:,:,:,0], world_coords=False, objectvals=[None],
                       colorbar=False, record=None, scene=scene)

mni_streamlines = transform_streamlines(mni_trksl.streamlines, warp_affine,
                      in_place=False)
overwrite=True
if (not os.path.exists(trk_postwarp) or overwrite):
    save_trk_header(filepath=trk_postwarp, streamlines=mni_streamlines, header=header,
                    affine=np.eye(4), verbose=True)


scene = setup_view(mni_streamlines, colors=lut_cmap, ref=warp[:,:,:,0], world_coords=False,
                   objectvals=[None],
                   colorbar=True, record=None, scene=scene)

#mni_streamlines_revert = transform_streamlines(mni_streamlines, np.linalg.inv(moving._affine),
#                      in_place=False)

#show_template_bundles(mni_streamlines, moving, show=True,
#                      fname=os.path.join(figspath, 'streamlines_prewarp.png'))


if (not os.path.exists(trk_postwarp) or overwrite):
    save_trk_header(filepath=trk_postwarp, streamlines=mni_streamlines, header=header,
                    affine=np.eye(4), verbose=True)



from fury import actor, window

show_template_bundles(mni_streamlines, static, show=True,
                      fname=os.path.join(figspath,'streamlines_DSN_MNI.png'))

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

sft = StatefulTractogram(mni_streamlines, static, Space.RASMM)
save_tractogram(sft, trk_postwarp_svtsave, bbox_valid_check=False)