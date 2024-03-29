#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:34:51 2020

@author: alex
Now part of DTC pipeline. Used to save and unload heavy trk files
"""


import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from dipy.io.streamline import load_trk, save_trk
from dipy.io.utils import create_tractogram_header
from dipy.io.stateful_tractogram import StatefulTractogram
import time, os
import numpy as np

def make_tractogram_object(fname, streamlines, affine, vox_size=None, shape=None, header=None):
    """ Saves tractogram object for future use

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
    return tractogram, trk_file

def save_trk_header(filepath, streamlines, header, affine=np.eye(4), fix_streamlines = False, verbose=False, sftp=None):

    myheader = create_tractogram_header(filepath, *header)
    if isinstance(streamlines, nib.streamlines.ArraySequence):
        trk_sl = lambda: (s for s in streamlines)
    else:
        trk_sl = streamlines
    if verbose:
        print(f'Saving streamlines to {filepath}')
        time1 = time.perf_counter()
    save_trk_heavy_duty(filepath, streamlines=trk_sl,
                        affine=affine, header=myheader, fix_streamlines = fix_streamlines, return_tractogram=False, sftp=sftp)
    if verbose:
        time2 = time.perf_counter()
        print(f'Saved in {time2 - time1:0.4f} seconds')

def save_trk_heavy_duty(fname, streamlines, affine, vox_size=None, shape=None, header=None, fix_streamlines = False,
                        return_tractogram=False,sftp=None):
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

    if not isinstance(streamlines,StatefulTractogram):
        #if type(streamlines)==list:
        #    streamlines = nib.streamlines.array_sequence.ArraySequence(streamlines)
        tractogram = nib.streamlines.LazyTractogram(streamlines)
        tractogram.affine_to_rasmm = affine
    else:
        tractogram = streamlines
    if fix_streamlines:
        tractogram.remove_invalid_streamlines()
    trk_file = nib.streamlines.TrkFile(tractogram, header=header)
    if sftp is None:
        nib.streamlines.save(trk_file, fname)
    else:
        temp_path = f'{os.path.join(os.path.expanduser("~"), os.path.basename(fname))}'
        nib.streamlines.save(trk_file, temp_path)
        try:
            sftp.put(temp_path, fname)
            os.remove(temp_path)
        except Exception as e:
            os.remove(temp_path)
            raise Exception(e)
    if return_tractogram:
        return tractogram, trk_file

def unload_trk(tractogram_path, reference='same'):
    """ Similar functionality as the older version of load_trk, as it directly
    extracts the streams and header instead of returning a Tractogram object

    Parameters
    ----------
    tractogram_path: the file path of the tractogram data ( path/tract.trk )
    reference: the file used for the header information. if 'same', use the hdr from tractogram file
    """

    if reference.lower() == "same":
        print("Reference taken directly from file")
    tract_obj = load_trk(tractogram_path, reference)
    streams_control = tract_obj.streamlines
    try:
        hdr_control = tract_obj.space_attribute
    except:
        hdr_control = tract_obj.space_attributes
    return streams_control, hdr_control, tract_obj


def convert_tck_to_trk(input_file, output_file, ref):
    header = {}

    nii = nib.load(ref)
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    tck = nib.streamlines.load(input_file)
    nib.streamlines.save(tck.tractogram, output_file, header=header)


def convert_trk_to_tck(input_file, output_file):
    from dipy.io.streamline import load_tractogram, save_tractogram

    #cc_trk = load_tractogram(input_file, reference_file, bbox_valid_check=False)
    #save_tractogram(cc_trk, output_file, bbox_valid_check=False)
    """
    try:
        tractogram = load_trk(trk_path, 'same')
    except:
        tractogram = load_trk_spe(trk_path, 'same')

    import warnings

    if nib.streamlines.detect_format(trk_path) is not nib.streamlines.TrkFile:
        warnings.warn("Skipping non TRK file: '{}'".format(tractogram))

    if output_filename is None:
        output_filename = trk_path[:-4] + '.tck'

    if os.path.isfile(output_filename) and not overwrite:
        warnings.warn("Skipping existing file: '{}'. Set overwrite to true".format(output_filename))

    """

    trk = nib.streamlines.load(input_file)
    nib.streamlines.save(trk.tractogram, output_file)

    return
