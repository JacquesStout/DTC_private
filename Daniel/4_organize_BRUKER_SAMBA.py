import os, shutil, copy
from DTC.file_manager.file_tools import mkcdir
from DTC.nifti_handlers.transform_handler import img_transform_exec, get_affine_transform, rigid_reg
import nibabel as nib
import numpy as np
from dipy.io.image import save_nifti
import pandas as pd
from DTC.file_manager.computer_nav import save_nifti_remote
from dipy.align.reslice import reslice
import glob, sys
from DTC.file_manager.argument_tools import parse_arguments
import platform
from DTC.file_manager.computer_nav import get_mainpaths, glob_remote, copy_loctoremote, checkfile_exists_remote, load_nifti_remote
from DTC.file_manager.file_tools import mkcdir, getfromfile
from DTC.nifti_handlers.atlas_handlers.mask_handler import applymask_samespace, median_mask_make, mask_fixer
import warnings

subjects = ['sub22040413', 'sub22040411', 'sub2204041', 'sub22040410', 'sub2204042', 'sub2204043', 'sub2204044',
            'sub2204045', 'sub2204046', 'sub2204047', 'sub2204048', 'sub2204049', 'sub2205091', 'sub22050910',
            'sub22050911', 'sub22050912', 'sub22050913', 'sub22050914', 'sub2205094', 'sub2205097', 'sub2205098',
            'sub2206061', 'sub22060610', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub2206062',
            'sub2206063', 'sub2206064', 'sub2206065', 'sub2206066', 'sub2206067', 'sub2206068', 'sub2206069']

cancelled_subj = ['sub2204041', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub2206063']
for cancelled in cancelled_subj:
    if cancelled in subjects:
        subjects.remove(cancelled)

#subjects = ['sub2206061', 'sub22060610', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub2206062']
#subjects = ['sub2206061', 'sub22060610', 'sub22060611']
#subjects = ['sub22060612', 'sub22060613', 'sub22060614', 'sub2206062']
#subjects = ['sub2206063', 'sub2206064', 'sub2206065', 'sub2206066']


def full_split_nii(subj_path, output_folder, sftp=None):
    img = nib.load(subj_path)
    data = img.get_fdata()

    for i in np.arange(np.shape(data)[3]):
        if i<10:
            istr=f'000{i}'
        elif i<100:
            istr=f'00{i}'
        elif i<1000:
            istr=f'0{i}'
        else:
            istr=str(i)
        nii_temp = nib.Nifti1Image(data[...,i], img.affine)
        temp_path = os.path.join(output_folder, f'{subj}_{istr}.nii.gz')
        save_nifti_remote(nii_temp, temp_path, sftp)

subjects.sort()
subject_processes, function_processes, firstsubj, lastsubj = parse_arguments(sys.argv,subjects)
#firstsubj=23
#lastsubj=26
subjects = subjects[firstsubj:lastsubj]
#subjects = ['sub22040411']
print(subjects)

overwrite = False
verbose = True
recenter = False
remote = True
toclean = True
tempcheck=False

project='Daniel'

if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None

if remote:
    _, _, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)
    orig_dir = '/mnt/paros_WORK/daniel/project/BRUKER_organized_JS_combined_v3/'
    new_dir = '/mnt/paros_WORK/daniel/project/BRUKER_organized_JS_SAMBAD_5'
else:
    orig_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_combined/'
    new_dir = '/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/BRUKER_organized_JS_SAMBAD_5'
    sftp = None

copytype = "truecopy"
ext = '.nii.gz'
native_ref = ''

#orig_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS/'
#orig_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_combined/'

"""
if 'hydra' in platform.node():
    orig_dir = '/Users/alex/jacques/BRUKER_organized_JS/'
if 'lefkada' in platform.node():
    orig_dir = '/Users/alex/temp_BRUKER_run/BRUKER_organized_JS'
"""

#new_dir = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_SAMBAD_3/'

#nii_temp_dir = '/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/temp_nii'
nii_temp_dir = '/Users/jas/jacques/Daniel_test/temp_nii'

#MDT_baseimages = '/Volumes/Data/Badea/Lab/mouse/VBM_18APOERAREset02_invivoAPOE1-work/preprocess'
MDT_baseimages = '/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/MDT_base/'

#transforms_folder = '/Users/jas/jacques/Daniel_test/Transforms_2'
transforms_folder = '/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/Transforms'

mkcdir(nii_temp_dir)
mkcdir(new_dir, sftp)

#ref_t1 = '/Volumes/Data/Badea/Lab/mouse/VBM_18APOERAREset02_invivoAPOE1-work/T1/SyN_0p5_3_0p5_T1/JS_rabies_i7/median_images/MDT_T1.nii.gz'
ref_t1 = '/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/MDT_T1.nii.gz'

transf_level = 'torigid'
#transf_level = 'towarp'

#csv_summary_path = '/Users/jas/jacques/Daniel_test/FMRI_mastersheet.xlsx'
csv_summary_path = '/Volumes/Data/Badea/Lab/jacques/APOE_func_proc/FMRI_mastersheet.xlsx'
csv_summary = pd.read_excel(csv_summary_path)

apply_mask = True

temp_check = True
toclean = False

num_images = 600

overwrite=True

for subj in subjects:

    line = csv_summary.loc[csv_summary['D_name'] == int(subj.replace('sub',''))]
    try:
        newid = 'sub'+str(list(line['RARE_name'])[0])
    except:
        print(f'could not find associated animal ID for this folder {subj}, skipping')
        continue

    #print(f'running move for subject {newid}')
    trans = os.path.join(transforms_folder, f"{newid}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(transforms_folder, f"{newid}_rigid.mat")
    affine_orig = os.path.join(transforms_folder, f"{newid}_affine.mat")
    runno_to_MDT = os.path.join(transforms_folder, f'{newid}_to_MDT_warp.nii.gz')
    MDT_to_subject = os.path.join(transforms_folder, f"MDT_to_{newid}_warp.nii.gz")


    subj_anat = os.path.join(orig_dir, subj.replace('b', 'b-'), 'ses-1', 'anat',
                            f'{subj.replace("b", "b-")}_ses-1_T1w{ext}')
    subj_mask = os.path.join(orig_dir, subj.replace('b', 'b-'), 'ses-1', 'anat',
                            f'{subj.replace("b", "b-")}_mask{ext}')
    subj_func = os.path.join(orig_dir, subj.replace('b', 'b-'), 'ses-1', 'func',
                             f'{subj.replace("b", "b-")}_ses-1_bold{ext}')

    folder_ID = os.path.join(new_dir, subj.replace('b', 'b-'))
    folder_ses = os.path.join(folder_ID, 'ses-1')
    folder_anat = os.path.join(folder_ses, 'anat')
    folder_func = os.path.join(folder_ses, 'func')
    mkcdir([folder_ID,folder_ses,folder_anat,folder_func],sftp)
    subj_anat_sambad = os.path.join(folder_anat, f'{subj.replace("b", "b-")}_ses-1_T1w{ext}')
    subj_func_sambad = os.path.join(folder_func, f'{subj.replace("b", "b-")}_ses-1_bold.nii.gz')

    #if not os.path.exists(subj_func_sambad) or not os.path.exists(subj_func_sambad):
    #    print(f'Still missing subject {subj}')
    #    continue
    #else:
    #    continue

    if not checkfile_exists_remote(subj_anat_sambad, sftp) or not checkfile_exists_remote(subj_func_sambad, sftp) or \
            overwrite:
        overwrite=False
        func_reorient = os.path.join(nii_temp_dir, f'{subj}_func_reorient{ext}')
        func_reorient_affined = os.path.join(nii_temp_dir, f'{subj}_func_reorient_affined{ext}')
        target_base = os.path.join(MDT_baseimages, f'{newid}_T1_masked.nii.gz')
        SAMBA_preprocess = os.path.join(nii_temp_dir, f'{subj}_preprocess{ext}')

        print(f'Reorienting the file {subj_anat}')

        if not os.path.exists(func_reorient) or overwrite:
            if sftp is not None:
                subj_func_temp = os.path.join(nii_temp_dir,os.path.basename(subj_func))
                sftp.get(subj_func, subj_func_temp)
            else:
                subj_func_temp = subj_func
            img_transform_exec(subj_func_temp, 'LPI', 'ALS', output_path=func_reorient, verbose=True)
            if sftp is not None:
                os.remove(subj_func_temp)
        if not os.path.exists(SAMBA_preprocess) or overwrite:
            shutil.copy(target_base, SAMBA_preprocess)

        if not os.path.exists(func_reorient_affined) or overwrite:
            _, affine, _, _, _ = load_nifti_remote(subj_anat, sftp)
            affine_recentered = nib.load(SAMBA_preprocess).affine

            transform = get_affine_transform(affine, affine_recentered)

            affine_func = nib.load(func_reorient).affine
            affine_func_new = np.eye(4)
            affine_func_new[:3, :3] = np.dot(affine_func[:3, :3], transform[:3, :3])
            affine_func_new[:3, 3] = [0, 0, 0]
            affine_func_new[1, :] = - affine_func_new[1, :]
            affine_func_new[:3, 3] = affine_func[:3, 3] - transform[:3, 3]

            func_reorient_affined_nii = nib.Nifti1Image(nib.load(func_reorient).get_fdata(), affine_func_new)
            nib.save(func_reorient_affined_nii, func_reorient_affined)
            print(f'Saved {func_reorient_affined}')

        func_reorient_recentered_reslicedtargetaff = os.path.join(nii_temp_dir, f'{subj}_func_reorient_rigid_reslicedtargetaff{ext}')

        if not os.path.exists(func_reorient_recentered_reslicedtargetaff) or overwrite:

            target = nib.load(SAMBA_preprocess)
            target_data = target.get_fdata()
            target_grid2world = target.affine

            moving = nib.load(func_reorient_affined)
            moving_data = moving.get_fdata()
            moving_grid2world = moving.affine

            from dipy.align.imaffine import transform_centers_of_mass

            target = nib.load(SAMBA_preprocess)
            target_data = target.get_fdata()
            target_grid2world = target.affine

            moving = nib.load(func_reorient_affined)
            moving_data = moving.get_fdata()
            moving_grid2world = moving.affine

            moved, trans_affine = rigid_reg(target_data, target_grid2world,
                                            moving_data, moving_grid2world)

            toreslice = True
            resliced = np.zeros(list(np.shape(
                reslice(moved[..., 0], moving_grid2world, [0.1, 0.1, 0.1], [0.3, 0.3, 0.3])[
                    0])) + [np.shape(moving_data)[3]])
            if toreslice:
                resliced, reslice_affine = reslice(moved, trans_affine,
                                                              [0.1, 0.1, 0.1], [0.3, 0.3, 0.3])
                _, targetreslice_affine = reslice(moved[...,0], target_grid2world,
                                                              [0.1, 0.1, 0.1], [0.3, 0.3, 0.3])
                save_nifti(func_reorient_recentered_reslicedtargetaff,resliced,targetreslice_affine)
                print(f'saved {func_reorient_recentered_reslicedtargetaff} with affine \n{targetreslice_affine}')
            else:
                save_nifti(func_reorient_recentered_reslicedtargetaff, moved, target_grid2world)
                print(f'Saved {func_reorient_recentered_reslicedtargetaff}')

        if toclean:
            tempfiles = [func_reorient,func_reorient_affined]
            for tmpfile in tempfiles:
                os.remove(tmpfile)

        if transf_level=='torigid':
            overwrite=True
            if not checkfile_exists_remote(subj_anat_sambad, sftp) or overwrite:
                if sftp is not None:
                    subj_anat_sambad_temp = os.path.join(nii_temp_dir, os.path.basename(subj_anat_sambad))
                else:
                    subj_anat_sambad_temp = subj_anat_sambad
                cmd = f"antsApplyTransforms -v 1 -d 3 -i {SAMBA_preprocess} -o {subj_anat_sambad_temp} -r {SAMBA_preprocess} -n MultiLabel -t [{rigid},0] [{trans},0]"
                os.system(cmd)
                if apply_mask:
                    if checkfile_exists_remote(subj_mask, sftp):
                        mask_nii = load_nifti_remote(subj_mask, sftp, return_nii=True)
                        applymask_samespace(subj_anat_sambad_temp, mask_nii, outpath=subj_anat_sambad_temp)
                    else:
                        warnings.warn('Could not apply mask, continue')
                if sftp is not None:
                    sftp.put(subj_anat_sambad_temp,subj_anat_sambad)
                    os.remove(subj_anat_sambad_temp)

            overwrite=False
            if not os.path.exists(subj_func_sambad) or overwrite:
                split_subject_folder = os.path.join(nii_temp_dir,f'{subj}_split')
                mkcdir(split_subject_folder)
                if np.size(glob.glob(os.path.join(split_subject_folder,'*.nii.gz')))<num_images:
                    full_split_nii(func_reorient_recentered_reslicedtargetaff,split_subject_folder)
                split_subject_files = glob.glob(os.path.join(split_subject_folder,'*.nii.gz'))
                split_subject_files.sort()

                split_subject_folder_transf = os.path.join(nii_temp_dir,f'{subj}_split_transf')
                mkcdir(split_subject_folder_transf)
                if np.size(glob.glob(os.path.join(split_subject_folder_transf, '*.nii.gz')))<num_images:
                    for split_subj_file in split_subject_files:
                        output_transf = os.path.join(split_subject_folder_transf, os.path.basename(split_subj_file))
                        if not os.path.exists(output_transf):
                            cmd = f"antsApplyTransforms -v 1 -d 3 -i {split_subj_file} -o {output_transf} -r {SAMBA_preprocess} -n MultiLabel -t [{rigid},0] [{trans},0]"
                            os.system(cmd)

                time_origin = 0
                time_spacing = float(os.popen(f'fslval {func_reorient_recentered_reslicedtargetaff} pixdim4').read().split(' \n')[0])


                if sftp is not None:
                    subj_func_sambad_temp = os.path.join(nii_temp_dir, os.path.basename(subj_func_sambad))
                else:
                    subj_func_sambad_temp = subj_func_sambad
                cmd = f'ImageMath 4 {subj_func_sambad_temp} TimeSeriesAssemble {time_spacing} {time_origin} {split_subject_folder_transf}/*'
                os.system(cmd)
                if sftp is not None:
                    sftp.put(subj_func_sambad_temp,subj_func_sambad)
                    os.remove(subj_func_sambad_temp)

        if transf_level=='tofullwarp':
            if not os.path.exists(subj_anat_sambad) or overwrite:
                cmd = f"antsApplyTransforms -v 1 -d 3 -i {SAMBA_preprocess} -o {subj_anat_sambad} -r {SAMBA_preprocess} -n MultiLabel -t {runno_to_MDT} [{affine_orig},0] [{rigid},0] [{trans},0]"
                os.system(cmd)
            if not os.path.exists(subj_func_sambad) or overwrite:
                split_subject_folder = os.path.join(nii_temp_dir,f'{subj}_split')
                mkcdir(split_subject_folder)
                if np.size(glob.glob(os.path.join(split_subject_folder,'*.nii.gz')))<num_images:
                    full_split_nii(func_reorient_recentered_reslicedtargetaff,split_subject_folder)
                split_subject_files = glob.glob(os.path.join(split_subject_folder,'*.nii.gz'))
                split_subject_files.sort()

                split_subject_folder_transf = os.path.join(nii_temp_dir,f'{subj}_split_transf')
                mkcdir(split_subject_folder_transf)
                if np.size(glob.glob(os.path.join(split_subject_folder_transf, '*.nii.gz')))<num_images:
                    for split_subj_file in split_subject_files:
                        output_transf = os.path.join(split_subject_folder_transf, os.path.basename(split_subj_file))
                        if not os.path.exists(output_transf):
                            cmd = f"antsApplyTransforms -v 1 -d 3 -i {split_subj_file} -o {output_transf} -r {SAMBA_preprocess} -n MultiLabel -t {runno_to_MDT} [{affine_orig},0] [{rigid},0] [{trans},0]"
                            os.system(cmd)

                time_origin = 0
                time_spacing = float(os.popen(f'fslval {func_reorient_recentered_reslicedtargetaff} pixdim4').read().split(' \n')[0])
                if sftp is not None:
                    subj_func_sambad_temp = os.path.join(nii_temp_dir, os.path.basename(subj_func_sambad))
                else:
                    subj_func_sambad_temp = subj_func_sambad
                cmd = f'ImageMath 4 ${subj_func_sambad} TimeSeriesAssemble ${time_spacing} ${time_origin} ${split_subject_folder_transf}/*'
                os.system(cmd)
                if sftp is not None:
                    sftp.put(subj_func_sambad_temp,subj_func_sambad)
                    os.remove(subj_func_sambad_temp)

        if toclean and os.path.exists(subj_func_sambad):
            tempfiles = [func_reorient_recentered_reslicedtargetaff]
            for tmpfile in tempfiles:
                os.remove(tmpfile)

            tempfolders = [split_subject_folder_transf,split_subject_folder]
            for tmpfolder in tempfolders:
                os.rmdir(tmpfolder)
    else:
        print(f'already wrote {subj_anat_sambad} and {subj_func_sambad}')