#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eleftherios and Serge

Wenlin make some changes to track on the whole brain
Wenlin add for loop to run all the animals 2018-20-25
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from dipy.io.streamline import load_trk
from os.path import splitext
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
import pickle

from types import ModuleType, FunctionType
from gc import get_referents

import smtplib

import os, re, sys, io, struct, socket, datetime
from email.mime.text import MIMEText
import glob

from dipy.tracking.utils import unique_rows


from time import time
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.local_tracking import LocalTracking
from dipy.direction import peaks
from nibabel.streamlines import detect_format
from dipy.io.utils import (create_tractogram_header,
                           get_reference_info)
from dipy.viz import window, actor

from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box

import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool



from scipy.ndimage.morphology import binary_dilation
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
import matplotlib.pyplot as plt

from dipy.denoise.localpca import mppca
#from denoise_processes import mppca
from dipy.denoise.gibbs import gibbs_removal

from random import randint

from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib
import matplotlib.pyplot as plt
import dipy.tracking.life as life
#import JSdipy.tracking.life as life
from dipy.viz import window, actor, colormap as cmap
import dipy.core.optimize as opt

l = ['N54717', 'N54718', 'N54719', 'N54720', 'N54722', 'N54759', 'N54760', 'N54761', 'N54762', 'N54763', 'N54764',
     'N54765', 'N54766', 'N54770', 'N54771', 'N54772', 'N54798', 'N54801', 'N54802', 'N54803', 'N54804', 'N54805',
     'N54806', 'N54807', 'N54818', 'N54824', 'N54825', 'N54826', 'N54837', 'N54838', 'N54843', 'N54844', 'N54856',
     'N54857', 'N54858', 'N54859', 'N54860', 'N54861', 'N54873', 'N54874', 'N54875', 'N54876', 'N54877', 'N54879',
     'N54880', 'N54891', 'N54892', 'N54893', 'N54897', 'N54898', 'N54899', 'N54900', 'N54915', 'N54916', 'N54917']


def save_trk_heavy_duty(fname, streamlines, affine, vox_size=None, shape=None, header=None):
    """ Saves tractogram files (*.trk)

    Parameters
    ----------
    fname : str
        output trk filename
    streamlines : list of 2D arrays, generator or ArraySequence
        Each 2D array represents a sequence of 3D points (points, 3).
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
    vox_size : array_like (3,), optional
        The sizes of the voxels in the reference image (default: None)
    shape : array, shape (dim,), optional
        The shape of the reference image (default: None)
    header : dict, optional
        Metadata associated to the tractogram file(*.trk). (default: None)
    """
    if vox_size is not None and shape is not None:
        if not isinstance(header, dict):
            header = {}
        header[Field.VOXEL_TO_RASMM] = affine.copy()
        header[Field.VOXEL_SIZES] = vox_size
        header[Field.DIMENSIONS] = shape
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(affine))

    tractogram = nib.streamlines.LazyTractogram(streamlines)
    tractogram.affine_to_rasmm = affine
    trk_file = nib.streamlines.TrkFile(tractogram, header=header)
    nib.streamlines.save(trk_file, fname)


# please set the parameter here

mypath = '/Users/alex/code/Wenlin/data/wenlin_data/'  # wenlin make this change
# mypath = ''

outpath = '/Users/alex/bass/testdata/' + 'btable_sanitycheck/'

# ---------------------------------------------------------
tall = time()
for j in range(1):
    print(j + 1)
    runno = l[j]
    subject=runno
    #    fdwi = mypath + 'N54900_nii4D_RAS.nii.gz'
    #
    #    ffalabels = mypath + 'fa_labels_warp_N54900_RAS.nii.gz'
    #
    #    fbvals = mypath + 'N54900_RAS_ecc_bvals.txt'
    #
    #    fbvecs = mypath + 'N54900_RAS_ecc_bvecs.txt'

    #   wenlin make this change
    fdwi = mypath + '4Dnii/' + runno + '_nii4D_RAS.nii.gz'

    ffalabels = mypath + 'labels/' + 'fa_labels_warp_' + runno + '_RAS.nii.gz'

    fbvals = mypath + '4Dnii/' + runno + '_RAS_ecc_bvals.txt'

    fbvecs = mypath + '4Dnii/' + runno + '_RAS_ecc_bvecs.txt'

    labels, affine_labels = load_nifti(ffalabels)

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

    # Correct flipping issue
    bvecs = np.c_[bvecs[:, 0], bvecs[:, 1], -bvecs[:, 2]]

    gtab = gradient_table(bvals, bvecs)

    data, affine, vox_size = load_nifti(fdwi, return_voxsize=True)

    # Build Brain Mask
    bm = np.where(labels == 0, False, True)
    mask = bm

    sphere = get_sphere('repulsion724')

    from dipy.reconst.dti import TensorModel

    tensor_model = TensorModel(gtab)

    t1 = time()
    tensor_fit = tensor_model.fit(data, mask)
    #    save_nifti('bmfa.nii.gz', tensor_fit.fa, affine)
    #   wenlin make this change-adress name to each animal
    save_nifti(outpath + 'bmfa' + runno + '.nii.gz', tensor_fit.fa, affine)
    fa = tensor_fit.fa
    duration1 = time() - t1
    # wenlin make this change-adress name to each animal
    #    print('DTI duration %.3f' % (duration1,))
    print(runno + ' DTI duration %.3f' % (duration1,))

    # Compute odfs in Brain Mask
    t2 = time()

    csa_model = CsaOdfModel(gtab, 6)

    csa_peaks = peaks_from_model(model=csa_model,
                                 data=data,
                                 sphere=peaks.default_sphere,  # issue with complete sphere
                                 mask=mask,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 parallel=False)

    duration2 = time() - t2
    print(duration2) \
        # wenlin make this change-adress name to each animal
    #    print('CSA duration %.3f' % (duration2,))
    print(runno + ' CSA duration %.3f' % (duration2,))

    t3 = time()

    # from dipy.tracking.local import ThresholdTissueClassifier
    # classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)
    #classifier = BinaryTissueClassifier(bm)  # Wenlin Make this change

    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

    classifier = ThresholdStoppingCriterion(csa_peaks.gfa, 0.25)

    # generates about 2 seeds per voxel
    # seeds = utils.random_seeds_from_mask(fa > .2, seeds_count=2,
    #                                      affine=np.eye(4))

    # generates about 2 million streamlines
    # seeds = utils.seeds_from_mask(fa > .2, density=1,
    #                              affine=np.eye(4))

    seeds = utils.seeds_from_mask(mask, density=1,
                                  affine=np.eye(4))

    streamlines_generator = LocalTracking(csa_peaks, classifier,
                                          seeds, affine=np.eye(4), step_size=2)

    outpathsubject = outpath + subject
    verbose=True
    saved_tracts = "small"
    stringstep="_2"
    trkheader=create_tractogram_header("place.trk", *get_reference_info(fdwi))
    if saved_tracts == "small" or saved_tracts == "both":
        sg_small = lambda: (s for i, s in enumerate(streamlines_generator) if i % 10 == 0)
        outpathtrk = outpathsubject + "_bmCSA_detr_small_wenlin_quicktest" + stringstep + ".trk"
        save_trk_heavy_duty(outpathtrk, streamlines=sg_small,
                            affine=affine, header=trkheader,
                            shape=mask.shape, vox_size=vox_size)
        print("Tract files were saved at " + outpathtrk)
    else:
        outpathtrk = None
    if saved_tracts == "large" or saved_tracts == "both" or saved_tracts == "all":
        sg = lambda: (s for s in streamlines_generator)
        outpathtrk = outpathsubject + "bmCSA_detr_all_wenlin" + stringstep + ".trk"
        save_trk_heavy_duty(outpathtrk, streamlines=sg,
                            affine=affine, header=trkheader,
                            shape=mask.shape, vox_size=vox_size)
        if verbose:
            print("Tract files were saved at "+outpathtrk)
    if saved_tracts == "none" or saved_tracts is None:
        print("Tract files were not saved")

    duration3 = time() - t3
    print(duration3)
    # wenlin make this change-adress name to each animal
    #    print('Tracking duration %.3f' % (duration3,))
    print(runno + ' Tracking duration %.3f' % (duration3,))

duration_all = time() - tall
print('All animals tracking finished, running time is {}'.format(duration_all))