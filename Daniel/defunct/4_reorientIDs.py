import os, glob, shutil, copy
from DTC.file_manager.file_tools import mkcdir, getfromfile
import pandas as pd
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote, copy_loctoremote, checkfile_exists_remote, save_nifti_remote
import numpy as np
import nibabel as nib

def angle_to_matrix(angle, axis, degrees=True):
    if degrees:
        angle = angle * (np.pi/180)
    matrix = np.eye(3)
    if axis == 0:
        matrix[1,1] = np.cos(angle)
        matrix[1,2] = - np.sin(angle)
        matrix[2,1] = np.sin(angle)
        matrix[2,2] = np.cos(angle)
    if axis == 1:
        matrix[0,0] = np.cos(angle)
        matrix[0,2] = np.sin(angle)
        matrix[2,0] = - np.sin(angle)
        matrix[2,2] = np.cos(angle)
    if axis == 2:
        matrix[0,0] = np.cos(angle)
        matrix[0,1] = - np.sin(angle)
        matrix[1,0] = np.sin(angle)
        matrix[1,1] = np.cos(angle)
    return(matrix)

sub22040411_fix = np.eye(4)
sub22040411_fix[0,3] = 3.5
sub22040411_fix[1,3] = 1.0
sub22040411_fix[2,3] = 0.0

sub22050910_fix = np.eye(4)
sub22050910_fix[0,3] = -0.73
sub22050910_fix[1,3] = 1.3
sub22050910_fix[2,3] = -0.06

sub22050913_fix = np.eye(4)
angle = -10
sub22050913_fix[0,3] = 0
sub22050913_fix[1,3] = 0
sub22050913_fix[2,3] = 0
sub22050913_fix[:3,:3] = angle_to_matrix(angle,2)

sub2205094_fix = np.eye(4)
angle = 12
sub2205094_fix[0,3] = 0
sub2205094_fix[1,3] = 3
sub2205094_fix[2,3] = -2
sub2205094_fix[:3,:3] = angle_to_matrix(angle,2)
sub2205094_fix[:3,:3] = np.matmul(angle_to_matrix(-10,0),sub2205094_fix[:3,:3])

sub2206068_fix = np.eye(4)
angle = 10
sub2206068_fix[0,3] = 1.7
sub2206068_fix[1,3] = 1
sub2206068_fix[2,3] = 0
sub2206068_fix[:3,:3] = angle_to_matrix(angle,2)

sub2206064_fix = np.eye(4)
angle = -8
sub2206064_fix[0,3] = 0
sub2206064_fix[1,3] = 2
sub2206064_fix[2,3] = -1
sub2206064_fix[:3,:3] = angle_to_matrix(angle,0)

sub2206063_fix = np.eye(4)
angle = -8
sub2206063_fix[0,3] = -0.4
sub2206063_fix[1,3] = 0.4
sub2206063_fix[2,3] = -2.8
sub2206063_fix[:3,:3] = angle_to_matrix(angle,0)
sub2206063_fix[:3,:3] = np.matmul(angle_to_matrix(5,1),sub2206063_fix[:3,:3])

sub2206068_fix = np.eye(4)
angle = 8
sub2206068_fix[0,3] = 0
sub2206068_fix[1,3] = 0
sub2206068_fix[2,3] = 0
sub2206068_fix[:3,:3] = angle_to_matrix(angle,2)
sub2206068_fix[:3,:3] = np.matmul(angle_to_matrix(-5,1),sub2206068_fix[:3,:3])

sub2206065_fix = np.eye(4)
angle = -8
sub2206065_fix[0,3] = 0
sub2206065_fix[1,3] = 0
sub2206065_fix[2,3] = -2
sub2206065_fix[:3,:3] = angle_to_matrix(angle,0)

sub2206067_fix = np.eye(4)
angle = -8
sub2206067_fix[0,3] = 0.5
sub2206067_fix[1,3] = 3
sub2206067_fix[2,3] = -0.8
sub2206067_fix[:3,:3] = angle_to_matrix(angle,0)
sub2206067_fix[:3,:3] = np.matmul(angle_to_matrix(8,2),sub2206067_fix[:3,:3])
sub2206067_fix[:3,:3] = np.matmul(angle_to_matrix(-4,1),sub2206067_fix[:3,:3])

sub2206062_fix = np.eye(4)
angle = -8
sub2206062_fix[0,3] = 0
sub2206062_fix[1,3] = 0.6
sub2206062_fix[2,3] = -1.5
sub2206062_fix[:3,:3] = angle_to_matrix(angle,0)

sub2204047_fix = np.eye(4)
angle = 9
sub2204047_fix[0,3] = 1.8
sub2204047_fix[1,3] = 2.8
sub2204047_fix[2,3] = 0
sub2204047_fix[:3,:3] = angle_to_matrix(angle,2)
sub2204047_fix[:3,:3] = np.matmul(angle_to_matrix(-4,1),sub2204047_fix[:3,:3])

sub22060611_fix = np.eye(4)
angle = -10
sub22060611_fix[0,3] = 0.5
sub22060611_fix[1,3] = 0
sub22060611_fix[2,3] = -2
sub22060611_fix[:3,:3] = angle_to_matrix(angle,2)
sub22060611_fix[:3,:3] = np.matmul(angle_to_matrix(8,1),sub22060611_fix[:3,:3])
sub22060611_fix[:3,:3] = np.matmul(angle_to_matrix(-10,0),sub22060611_fix[:3,:3])

sub22060612_fix = np.eye(4)
angle = 4
sub22060612_fix[0,3] = 0
sub22060612_fix[1,3] = 1.5
sub22060612_fix[2,3] = -2
sub22060612_fix[:3,:3] = angle_to_matrix(angle,2)
sub22060612_fix[:3,:3] = np.matmul(angle_to_matrix(4,1),sub22060612_fix[:3,:3])
sub22060612_fix[:3,:3] = np.matmul(angle_to_matrix(-4,0),sub22060612_fix[:3,:3])

sub22060613_fix = np.eye(4)
angle = -15
sub22060613_fix[0,3] = -0.9
sub22060613_fix[1,3] = -1.4
sub22060613_fix[2,3] = -2
sub22060613_fix[:3,:3] = angle_to_matrix(angle,2)
sub22060613_fix[:3,:3] = np.matmul(angle_to_matrix(8,1),sub22060613_fix[:3,:3])
sub22060613_fix[:3,:3] = np.matmul(angle_to_matrix(-6,0),sub22060613_fix[:3,:3])

sub22060614_fix = np.eye(4)
angle = -7
sub22060614_fix[0,3] = -0.4
sub22060614_fix[1,3] = 2.4
sub22060614_fix[2,3] = -1
sub22060614_fix[:3,:3] = angle_to_matrix(angle,0)
#sub22060614_fix[:3,:3] = np.matmul(angle_to_matrix(8,1),sub22060614_fix[:3,:3])
#sub22060614_fix[:3,:3] = np.matmul(angle_to_matrix(-6,0),sub22060614_fix[:3,:3])

sub2206069_fix = np.eye(4)
angle = -10
sub2206069_fix[0,3] = 0.9
sub2206069_fix[1,3] = 0.5
sub2206069_fix[2,3] = -0.5
sub2206069_fix[:3,:3] = angle_to_matrix(angle,0)
sub2206069_fix[:3,:3] = np.matmul(angle_to_matrix(-4,1),sub2206069_fix[:3,:3])
sub2206069_fix[:3,:3] = np.matmul(angle_to_matrix(-2,2),sub2206069_fix[:3,:3])

sub2205097_fix = np.eye(4)
angle = 12
sub2205097_fix[0,3] = -0.4
sub2205097_fix[1,3] = 3
sub2205097_fix[2,3] = -1.2
sub2205097_fix[:3,:3] = angle_to_matrix(angle,2)
sub2205097_fix[:3,:3] = np.matmul(angle_to_matrix(-6,0),sub2205097_fix[:3,:3])
sub2205097_fix[:3,:3] = np.matmul(angle_to_matrix(6,1),sub2205097_fix[:3,:3])

sub22050914_fix = np.eye(4)
angle = -7
sub22050914_fix[0,3] = 0
sub22050914_fix[1,3] = 0
sub22050914_fix[2,3] = -1
sub22050914_fix[:3,:3] = angle_to_matrix(angle,2)
sub22050914_fix[:3,:3] = np.matmul(angle_to_matrix(-6,0),sub22050914_fix[:3,:3])
#sub22050914_fix[:3,:3] = np.matmul(angle_to_matrix(-6,0),sub22050914_fix[:3,:3])

animal_fix = {'sub-22050914':sub22050914_fix,'sub-22050913':sub22050913_fix,'sub-22040411':sub22040411_fix, 'sub-22050910':sub22050910_fix}
animal_fix = {'sub-2205097':sub2205097_fix,'sub-22060614':sub22060614_fix,'sub-22060613':sub22060613_fix,'sub-22060611':sub22060611_fix,'sub-22060612':sub22060612_fix,'sub-2204047':sub2204047_fix,'sub-2206062':sub2206062_fix,'sub-2206067':sub2206067_fix,'sub-2206065':sub2206065_fix,'sub-2206063':sub2206063_fix,'sub-2206068':sub2206068_fix,'sub-2206069':sub2206069_fix,'sub-2206064':sub2206064_fix,'sub-2206068':sub2206068_fix,'sub-2205094':sub2205094_fix,'sub-22050914':sub22050914_fix,'sub-22050913':sub22050913_fix,'sub-22050910':sub22050910_fix}
#animal_fix = {'sub-22050914':sub22050914_fix}

remote=False
overwrite=True

BRUKER_init = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS/'

project='Daniel'
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
if remote:
    _, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
    BRUKER_f = '/mnt/paros_WORK/daniel/project/BRUKER_organized_JS_reorient/'
else:
    BRUKER_f = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_reorient/'
    sftp=None

mkcdir(BRUKER_f,sftp)

for key in animal_fix:
    folder = os.path.join(BRUKER_init,key)
    folder_anat = os.path.join(folder,'ses-1','anat')
    folder_func = os.path.join(folder,'ses-1','func')

    rare_prepath = os.path.join(folder_anat, f'{key}_ses-1_T1w.nii.gz')
    func_prepath = os.path.join(folder_func, f'{key}_ses-1_bold.nii.gz')

    folder = os.path.join(BRUKER_f,key)
    folder_anat = os.path.join(folder,'ses-1','anat')
    folder_func = os.path.join(folder,'ses-1','func')
    mkcdir([folder,os.path.join(folder,'ses-1'),folder_anat,folder_func], sftp)
    rare_newpath = os.path.join(folder_anat, f'{key}_ses-1_T1w.nii.gz')
    func_newpath = os.path.join(folder_func, f'{key}_ses-1_bold.nii.gz')

    if not os.path.exists(rare_newpath) or overwrite:
        oldnifti = nib.load(rare_prepath)
        newnifti = copy.deepcopy(oldnifti)
        old_affine = copy.deepcopy(oldnifti.affine)
        new_affine = newnifti.affine
        matrix_fix = animal_fix[key]
        new_affine[:3,3] = new_affine[:3,3] + matrix_fix[:3,3]
        new_affine[:3,:3] = np.matmul(matrix_fix[:3,:3],new_affine[:3,:3])
        save_nifti_remote(newnifti,rare_newpath,sftp=sftp)
        print(f'Saved {rare_newpath}')

    if not os.path.exists(func_newpath) or overwrite:
        oldnifti = nib.load(func_prepath)
        newnifti = copy.deepcopy(oldnifti)
        old_affine = copy.deepcopy(oldnifti.affine)
        new_affine = newnifti.affine
        matrix_fix = animal_fix[key]
        new_affine[:3,3] = new_affine[:3,3] + matrix_fix[:3,3]
        new_affine[:3,:3] = np.matmul(matrix_fix[:3,:3],new_affine[:3,:3])
        save_nifti_remote(newnifti,func_newpath,sftp=sftp)
        print(f'Saved {func_newpath}')