import numpy as np
import nibabel as nib
import os.path as op
import os
from file_tools import mkcdir

dipy_testzone = '/Users/jas/jacques/dipy_testzone'

if not op.exists(op.join(dipy_testzone,'lr-superiorfrontal.trk')):
    from streamline_tools import *
    vox_size = hardi_img.header.get_zooms()[0]
else:
    from dipy.core.gradients import gradient_table
    from dipy.data import get_fnames
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti_data, load_nifti, save_nifti

    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
    vox_size = hardi_img.header.get_zooms()[0]
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

from dipy.data.fetcher import (fetch_mni_template, read_mni_template)
from dipy.align.reslice import reslice

fetch_mni_template()
img_t2_mni = read_mni_template("a", contrast="T2")

new_zooms = (2., 2., 2.)
data2, affine2 = reslice(np.asarray(img_t2_mni.dataobj), img_t2_mni.affine,
                         img_t2_mni.header.get_zooms(), new_zooms)
img_t2_mni = nib.Nifti1Image(data2, affine=affine2)

b0_idx_stanford = np.where(gtab.b0s_mask)[0]
b0_data_stanford = data[..., b0_idx_stanford]


from dipy.segment.mask import median_otsu

b0_masked_stanford, _ = median_otsu(b0_data_stanford,
                vol_idx=list(range(b0_data_stanford.shape[-1])),
                median_radius=4, numpass=4)

mean_b0_masked_stanford = np.mean(b0_masked_stanford, axis=3,
                                  dtype=data.dtype)

from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration,
                                 transform_origins)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D,
                                   AffineTransform3D)

static = np.asarray(img_t2_mni.dataobj)
static_affine = img_t2_mni.affine
moving = mean_b0_masked_stanford
moving_affine = hardi_img.affine

affine_map = transform_origins(static, static_affine, moving, moving_affine)

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

level_iters = [10, 10, 5]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affine_reg = AffineRegistration(metric=metric, level_iters=level_iters,
                                sigmas=sigmas, factors=factors)
transform = TranslationTransform3D()

params0 = None
translation = affine_reg.optimize(static, moving, transform, params0,
                                  static_affine, moving_affine)
transformed = translation.transform(moving)
transform = RigidTransform3D()

rigid_map = affine_reg.optimize(static, moving, transform, params0,
                                static_affine, moving_affine,
                                starting_affine=translation.affine)
transformed = rigid_map.transform(moving)
transform = AffineTransform3D()

affine_reg.level_iters = [1000, 1000, 100]
highres_map = affine_reg.optimize(static, moving, transform, params0,
                                  static_affine, moving_affine,
                                  starting_affine=rigid_map.affine)
transformed = highres_map.transform(moving)

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

metric = CCMetric(3)
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                       highres_map.affine)
warped_moving = mapping.transform(moving)

from dipy.viz import regtools

regtools.overlay_slices(static, warped_moving, None, 0, 'Static', 'Moving',
                        op.join(dipy_testzone,'transformed_sagittal.png'))
regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Moving',
                        op.join(dipy_testzone,'transformed_coronal.png'))
regtools.overlay_slices(static, warped_moving, None, 2, 'Static', 'Moving',
                        op.join(dipy_testzone,'transformed_axial.png'))

from dipy.io.streamline import load_tractogram

sft = load_tractogram(op.join(dipy_testzone,'lr-superiorfrontal.trk', 'same'))


from dipy.tracking.streamline import deform_streamlines

# Create an isocentered affine
target_isocenter = np.diag(np.array([-vox_size, vox_size, vox_size, 1]))

# Take the off-origin affine capturing the extent contrast between the mean b0
# image and the template
origin_affine = affine_map.affine.copy()

origin_affine[0][3] = -origin_affine[0][3]
origin_affine[1][3] = -origin_affine[1][3]

origin_affine[2][3] = origin_affine[2][3]/vox_size


origin_affine[1][3] = origin_affine[1][3]/vox_size**2

# Apply the deformation and correct for the extents
mni_streamlines = deform_streamlines(
    sft.streamlines, deform_field=mapping.get_forward_field(),
    stream_to_current_grid=target_isocenter,
    current_grid_to_world=origin_affine, stream_to_ref_grid=target_isocenter,
    ref_grid_to_world=np.eye(4))

from dipy.viz import has_fury


def show_template_bundles(bundles, show=True, fname=None):

    scene = window.Scene()
    template_actor = actor.slicer(static)
    scene.add(template_actor)

    lines_actor = actor.streamtube(bundles, window.colors.orange,
                                   linewidth=0.3)
    scene.add(lines_actor)

    if show:
        window.show(scene)
    if fname is not None:
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


if has_fury:

    from fury import actor, window

    show_template_bundles(mni_streamlines, show=False,
                          fname=op.join(dipy_testzone,'streamlines_DSN_MNI.png'))

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

sft = StatefulTractogram(mni_streamlines, img_t2_mni, Space.RASMM)

save_tractogram(sft, op.join(dipy_testzone,'mni-lr-superiorfrontal.trk'), bbox_valid_check=False)