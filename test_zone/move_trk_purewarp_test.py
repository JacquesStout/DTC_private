from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
import numpy as np
import nibabel as nib
from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration,
                                 transform_origins)
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import deform_streamlines, transform_streamlines
from nifti_handlers.nifti_handler import extract_nii_info

from dipy.viz import window, actor
import os
from tract_manager.tract_save import save_trk_header
from file_manager.file_tools import mkcdir

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

from visualization_tools.visualization_tools.tract_visualize import show_bundles, setup_view

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
    #streamlines_prewarp = load_trk(trk_preprocess_prewarp,'same').streamlines

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

if view_streamlines:

    #show_template_bundles(streamlines_prewarp, moving, show=True,
    #                      fname=os.path.join(figspath,'streamlines_prewarp.png'))
    scene = setup_view(streamlines_prewarp[:], colors=lut_cmap, ref=moving_path, world_coords=True, objectvals=[None],
                       colorbar=True, record=None, scene=scene)

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
        warped_image_test = nib.Nifti1Image(warped_moving, static._affine)
        nib.save(warped_image_test, warped_image_dipy_path)
        print(f'Saved the warped image to {warped_image_dipy_path}')
    else:
        warp, warp_affine, vox_size, header_warp, ref_info = extract_nii_info(warped_image_dipy_path)
elif warping == 'precalc':
    warp, warp_affine, vox_size, header_warp, ref_info = extract_nii_info(MDT_to_runno)
    warp = warp[:, :, :, 0, :]

visualize = False
if visualize:
    from dipy.viz import regtools

    regtools.overlay_slices(static_data, warped_moving, None, 0, 'Static', 'Moving',
                            None)
    regtools.overlay_slices(static_data, warped_moving, None, 1, 'Static', 'Moving',
                            None)
    regtools.overlay_slices(static_data, warped_moving, None, 2, 'Static', 'Moving',
                            None)

vox_size = moving.header.get_zooms()[0]
target_isocenter = np.diag(np.array([vox_size, vox_size, vox_size, 1]))

# Take the off-origin affine capturing the extent contrast between the mean b0
# image and the template
origin_affine = affine_map.affine.copy()
origin_affine[0,0] = -origin_affine[0,0]
origin_affine[1,1] = -origin_affine[1,1]
"""
origin_affine[0][3] = -origin_affine[0][3]
origin_affine[1][3] = -origin_affine[1][3]
origin_affine[2][3] = origin_affine[2][3]/vox_size

#origin_affine[0][3] = -origin_affine[0][3]
#origin_affine[1][3] = -origin_affine[1][3]
#origin_affine[2][3] = origin_affine[2][3] / vox_size
#origin_affine[1][3] = origin_affine[1][3] / vox_size ** 2

# Apply the deformation and correct for the extents

origin_affine[1][3] = origin_affine[1][3]/vox_size**2
"""
#target_isocenter = moving._affine

#test_streamlines_1 = transform_streamlines(streamlines_prewarp, (moving._affine),
#                      in_place=False)

target_isocenter = moving._affine
#target_isocenter[2,3] = -target_isocenter[2,3]
test_streamlines_1 = transform_streamlines(streamlines_prewarp, np.linalg.inv(warp_affine),
                      in_place=False)

#scene = setup_view(test_streamlines_1, colors=lut_cmap, ref=warp[:,:,:,0], world_coords=False,
#                   objectvals=[None],
#                   colorbar=True, record=None, scene=scene)

#show_template_bundles(test_streamlines_1, static, show=True,
#                      fname=os.path.join(figspath,'streamlines_DSN_MNI.png'))
"""
mni_streamlines = deform_streamlines(
    test_streamlines_1, deform_field=warp,
    stream_to_current_grid=np.eye(4),
    current_grid_to_world=np.eye(4), stream_to_ref_grid=np.eye(4),
    ref_grid_to_world=np.eye(4))

mni_streamlines_revert = nib.streamlines.ArraySequence(transform_streamlines(mni_streamlines, (warp_affine),
                      in_place=False))

scene = setup_view(mni_streamlines_revert, colors=lut_cmap, ref=warp[:,:,:,0], world_coords=False,
                   objectvals=[None],
                   colorbar=True, record=None, scene=scene)

show_template_bundles(mni_streamlines_revert, static, show=True,
                      fname=os.path.join(figspath,'streamlines_DSN_MNI.png'))

if (not os.path.exists(trk_postwarp) or overwrite):
    save_trk_header(filepath=trk_postwarp, streamlines=mni_streamlines_revert, header=header,
                    affine=np.eye(4), verbose=True)


show_template_bundles(mni_streamlines, static, show=True,
                      fname=os.path.join(figspath,'streamlines_DSN_MNI.png'))
"""
"""
mni_streamlines = deform_streamlines(
    test_streamlines_1, deform_field=warp,
    stream_to_current_grid=target_isocenter,
    current_grid_to_world=origin_affine, stream_to_ref_grid=target_isocenter,
    ref_grid_to_world=np.eye(4))
"""
mni_streamlines = deform_streamlines(
    test_streamlines_1, deform_field=warp,
    stream_to_current_grid=target_isocenter,
    current_grid_to_world=origin_affine, stream_to_ref_grid=np.eye(4),
    ref_grid_to_world=np.eye(4))

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