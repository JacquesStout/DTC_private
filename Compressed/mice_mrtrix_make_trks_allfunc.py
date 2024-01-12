import os, shutil, re, copy, socket
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti, save_nifti
import numpy as np
from DTC.diff_handlers.bvec_handler import extractbvals_from_method, reorient_bvecs_files,fix_bvals_bvecs
import warnings
import nibabel as nib
from os.path import splitext
from nibabel.tmpdirs import InTemporaryDirectory

def splitpath(path):
    dirpath = os.path.dirname(path)
    name = os.path.basename(path).split('.')[0]
    if '.' in os.path.basename(path):
        ext = '.' + '.'.join(os.path.basename('/dwiMDT_NoNameYet_n32_i5/stats_by_region/labels/transforms/chass_symmetric3_to_MDT_warp.nii.gz').split('.')[1:])
    else:
        ext = ''
    return dirpath, name, ext


def mkcdir(folderpaths, sftp=None):
    #creates new folder only if it doesnt already exists

    if isinstance(folderpaths, str):
        folderpaths = [folderpaths]

    if sftp is None:
        if np.size(folderpaths) == 1:
            if not os.path.exists(folderpaths[0]):
                os.mkdir(folderpaths[0])
        else:
            for folderpath in folderpaths:
                if not os.path.exists(folderpath):
                    os.mkdir(folderpath)
    else:
        if np.size(folderpaths) == 1:
            try:
                sftp.chdir(folderpaths[0])
            except:
                sftp.mkdir(folderpaths[0])
        else:
            for folderpath in folderpaths:
                try:
                    sftp.chdir(folderpath)
                except:
                    sftp.mkdir(folderpath)


def img_transform_exec(img, current_vorder, desired_vorder, output_path=None, write_transform=0, rename = True, recenter=False, recenter_test= False, recenter_eye=False, recenter_flipaffine=False, verbose=False):

    is_RGB = 0;
    is_vector = 0;
    is_tensor = 0;

    if not os.path.exists(img):
        raise('Nifti img file does not exists')
    for char in current_vorder:
        if char.islower():
            warnings.warn("Use uppercase for current order")
            current_vorder=current_vorder.upper()
    for char in desired_vorder:
        if char.islower():
            warnings.warn("Use uppercase for desired order")
            current_vorder=desired_vorder.upper()

    ordervals='RLAPSI'
    if not ordervals.find(current_vorder[0]) and not ordervals.find(current_vorder[1]) and not ordervals.find(current_vorder[2]):
        raise TypeError('Please use only R L A P S or I for current voxel order')

    if not ordervals.find(desired_vorder[0]) and not ordervals.find(desired_vorder[1]) and not ordervals.find(desired_vorder[2]):
        raise TypeError('Please use only R L A P S or I for desired voxel order')

    dirname, filename, ext = splitpath(img)
    if ext=='.nii.gz':
        filename = filename + ext
    else:
        filename = filename + "." + ext

    if rename:
        output_name = filename.replace('.nii', '_' + desired_vorder + '.nii')
    else:
        output_name = filename

    if output_path is None:
        output_path = os.path.join(dirname,output_name)
    if output_path.find('.')==-1:
        mkcdir(output_path)
        output_path = os.path.join(output_path,output_name)

    out_dir, _, _ = splitpath(output_path)
    affine_out = os.path.join(out_dir, current_vorder + '_to_' + desired_vorder + '_affine.pickle')

    overwrite = True;
    if os.path.isfile(output_path) and not overwrite and (not write_transform or (write_transform and os.path.exists(affine_out))):
        warnings.warn('Existing output:%s, not regenerating', output_path);

    orig_string = 'RLAPSI';
    flip_string = 'LRPAIS';
    orig_current_vorder = current_vorder;

    nii = nib.load(img)

    nii_data = nii.get_fdata()
    hdr = nii.header

    dims = nii.shape
    if np.size(dims) > 6:
        raise('Image has > 5 dimensions')
    elif np.size(dims) < 3:
        raise('Image has < 3 dimensions')

    new_data = nii_data
    affine = nii._affine
    vox_size = nii.header.get_zooms()[:3]
    new_vox_size = vox_size

    if desired_vorder!=orig_current_vorder:
        if ((np.size(dims) > 4) and (dims(5) == 3)):
            is_vector = 1;
        elif ((np.size(dims) > 5) and (dims(5) == 6)):
            is_tensor = 1;

        #x_row = [1, 0, 0];
        #y_row = [0, 1, 0];
        #z_row = [0, 0, 1];
        x_row = affine[0,:]
        y_row = affine[1,:]
        z_row = affine[2,:]
        flipping = [1,1,1]
        xpos=desired_vorder.find(current_vorder[0])
        if xpos == -1:
            if verbose:
                print('Flipping first dimension')
            val=0
            new_data = np.flip(new_data, 0)
            orig_ind=orig_string.find(current_vorder[0])
            current_vorder = current_vorder[0:val] + flip_string[orig_ind] + current_vorder[val+1:]
            if is_vector:
                new_data[:,:,:,0,1]=-new_data[:,:,:,0,1]
            x_row = [-1 * val for val in x_row]
            flipping[0] = -1

        ypos=desired_vorder.find(current_vorder[1])
        if ypos == -1:
            if verbose:
                print('Flipping second dimension')
            val=1
            new_data = np.flip(new_data, 1)
            orig_ind=orig_string.find(current_vorder[1])
            current_vorder = current_vorder[0:val] + flip_string[orig_ind] + current_vorder[val+1:]
            if is_vector:
                new_data[:,:,:,0,2]=-new_data[:,:,:,0,2]
            y_row = [-1 * val for val in y_row]
            flipping[1] = -1


        zpos=desired_vorder.find(current_vorder[2])
        if zpos == -1:
            if verbose:
                print('Flipping third dimension')
            val=2
            new_data = np.flip(new_data, 2)
            orig_ind=orig_string.find(current_vorder[2])
            current_vorder = current_vorder[0:val] + flip_string[orig_ind] + current_vorder[val+1:]
            if is_vector:
                new_data[:,:,:,0,2]=-new_data[:,:,:,0,2]
            z_row = [-1 * val for val in z_row]
            flipping[2] = -1


        xpos = current_vorder.find(desired_vorder[0])
        ypos = current_vorder.find(desired_vorder[1])
        zpos = current_vorder.find(desired_vorder[2])


        if verbose:
            print(['Dimension order is:' + str(xpos) + ' ' + str(ypos) + ' ' + str(zpos)] )
        if not os.path.isfile(output_path) or overwrite:
            if np.size(dims) == 5:
                if is_tensor:
                    new_data = new_data.tranpose(xpos, ypos, zpos, 3, 4)
                else:
                    if is_vector:# =>> honestly looking at the original code, this doesnt really make sense to me, so deactivated for now. Will raise warning in case it happens
                        warnings.warn('is vector not properly implemented')
                        #    new[:,:,:,1,:] = new[:,:,:,1].transpose(xpos, ypos, zpos)
                        #new=new(:,:,:,[xpos, ypos, zpos]);
                    new_data.transpose(xpos, ypos, zpos, 3, 4)
            elif np.size(dims) == 4:
                if is_RGB:
                    ('is rgb not properly implemented')
                    #new=new(:,:,:,[xpos, ypos, zpos]);
                new_data = new_data.transpose(xpos, ypos, zpos, 3)
                new_vox_size = [0] * len(vox_size)
                new_vox_size = [vox_size[[xpos, ypos, zpos].index(x)] for x in sorted([xpos, ypos, zpos])]
            elif np.size(dims) == 3:
                new_data = new_data.transpose(xpos, ypos, zpos)
                new_vox_size = [0] * len(vox_size)
                new_vox_size = [vox_size[[xpos, ypos, zpos].index(x)] for x in sorted([xpos, ypos, zpos])]

        if not os.path.isfile(affine_out) and write_transform:
            intermediate_affine_matrix = [x_row , y_row, z_row];
            iam = intermediate_affine_matrix;
            affine_matrix_for_points = [iam[xpos,:], iam[ypos,:], iam[zpos,:]]
            affine_matrix_for_images = np.inv(affine_matrix_for_points)
            am4i = affine_matrix_for_images
            affine_matrix_string = [am4i[1,:] + am4i[2,:] + am4i[3,:] + '0 0 0']
            affine_fixed_string = ['0', '0', '0',];
            try:
                #write_affine_xform_for_ants(affine_out,affine_matrix_string,affine_fixed_string);
                #needs to be implemented if this is a desired functionality
                print("nope, not implemented")
            except:
                print("nope, not implemented")

    origin=affine[0:3,3]
    if desired_vorder != orig_current_vorder:
        trueorigin=origin*[x_row[0],y_row[1],z_row[2]]
        trueorigin[2]=trueorigin[2]*(-1)
    else:
        trueorigin = origin

    newaffine=np.zeros([4,4])

    newaffine[0:3,0:3]=affine[0:3,0:3]
    newaffine[3,:]=[0,0,0,1]
    newaffine[:3,3]=trueorigin

    #affine_transform = np.array([x_row, y_row, z_row, affine[3, :]])
    #affine_transform.transpose(xpos, ypos, zpos)
    if recenter:
        """
        newdims = np.shape(new_data)
        newaffine[0,3] = -(newdims[0] * newaffine[0,0]* 0.5)+0.045
        newaffine[1,3] = -(newdims[1] * newaffine[1,1]* 0.5)+0.045#+1
        newaffine[2,3] = -(newdims[2] * newaffine[2,2]* 0.5)+0.045
        """
        newaffine = recenter_affine(np.shape(new_data),newaffine)
    elif recenter_test:
        newaffine[:3,3] = origin * flipping
    elif recenter_eye:
        newaffine = np.eye(4)
        d = np.einsum('ii->i', newaffine)
        d[0:3] *= nii.header['pixdim'][1:4]
        #newaffine = recenter_affine(np.shape(new_data),newaffine)
        newaffine = recenter_affine_test(np.shape(new_data), newaffine)
        newaffine[:3,3] += 0.1
        #newaffine[0,:]=x_row
        #newaffine[1,:]=y_row
        #newaffine[2,:]=z_row
        #newhdr.srow_x=[newaffine[0,0:3]]
        #newhdr.srow_y=[newaffine[1,0:3]]
        #newhdr.srow_z=[newaffine[2,0:3]]
        #newhdr.pixdim=hdr.pixdim
    elif recenter_flipaffine:
        newaffine = affine_transform

    else:
        ## ADDED THIS ELSE OPTION FOR CONSISTENCY, MIGHT NEED TO SWITCH BACK AFTERWARDS IF OTHER STUFF BREAKS!!!!!
        newaffine = affine

    if new_vox_size!=vox_size:
        newaffine[:3,:3] = new_vox_size*(newaffine[:3,:3]/vox_size)

    new_nii=nib.Nifti1Image(new_data, newaffine, hdr)
    output_path = str(output_path)
    if verbose:
        print(f'Saving nifti file to {output_path}')
    nib.save(new_nii, output_path)
    if verbose:
        print(f'Saved')
    """
    #newnii.hdr.dime.intent_code = nii.hdr.dime.intent_code
    new_nii=nib.Nifti1Image(new, newaffine, newhdr)
    output_pathold=output_path
    output_path = str(output_path)
    output_path=output_path.replace('.nii', '_test.nii')
    nib.save(new_nii, output_path)
    output_path2=output_path.replace('.nii','_2.nii')
    new_nii_oldaffine=nib.Nifti1Image(new, affine, hdr)
    nib.save(new_nii_oldaffine,output_path2)
    #newnii.hdr.dime.intent_code = nii.hdr.dime.intent_code
    nib_test=nib.load(output_path)
    nib_test2=nib.load(output_path)
    nib_test_target=nib.load(output_pathold)
    hdr_test=nib_test._header
    hdr_test2=nib_test2._header
    hdr_target = nib_test_target._header
    print('hi')
    """


def median_mask_make(inpath, outpath=None, outpathmask=None, median_radius=4, numpass=4, binary_dilation_val=None,
                     vol_idx=None, affine=None, verbose=False, overwrite=False):
    if type(inpath) == str:
        data, affine = load_nifti(inpath)
        if outpath is None:
            outpath = inpath.replace(".nii", "_masked.nii")
        elif outpath is None and outpathmask is None:
            outpath = inpath.replace(".nii", "_masked.nii")
            outpathmask = inpath.replace(".nii", "_mask.nii")
        elif outpathmask is None:
            outpathmask = outpath.replace(".nii", "_mask.nii")
    else:
        data = inpath
        if affine is None:
            raise Exception('Needs affine')
        if outpath is None:
            raise Exception('Needs outpath')
    if os.path.exists(outpath) and os.path.exists(outpathmask) and not overwrite:
        print('Already wrote mask')
        return outpath, outpathmask
    data = np.squeeze(data)
    data_masked, mask = median_otsu(data, median_radius=median_radius, numpass=numpass, dilate=binary_dilation_val,
                                    vol_idx=vol_idx)
    save_nifti(outpath, data_masked.astype(np.float32), affine)
    save_nifti(outpathmask, mask.astype(np.float32), affine)
    if verbose:
        print(f'Saved masked file to {outpath}, saved mask to {outpathmask}')
    return outpath, outpathmask


def extractbvals_from_method(source_file,outpath=None,tonorm=True,verbose=False):

    bvals = bvecs = None

    filename = os.path.split(source_file)[1]
    if outpath is None:
        outpath = os.path.join(os.path.split(source_file)[0],'diffusion')
    with open(source_file, 'rb') as source:
        if verbose: print('INFO    : Extracting acquisition parameters')
        bvals = []
        bvecs = []
        i=0
        num_bvec_found = 0
        num_bval_found = 0

        bvec_start = False
        bval_start = False
        for line in source:

            pattern1 = 'PVM_DwDir='
            rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)

            pattern2 = 'PVM_DwEffBval'
            rx2 = re.compile(pattern2, re.IGNORECASE|re.MULTILINE|re.DOTALL)

            if bvec_start:
                if num_bvec_found < num_bvecs:
                    vals = str(line).split("\'")[1].split(' \\')[0].split('\\n')[0].split(' ')
                    for val in vals:
                        bvecs.append(val)
                        num_bvec_found = np.size(bvecs)
                else:
                    bvec_start = False
                if num_bvec_found == num_bvecs:
                    bvecs = np.reshape(bvecs,bvec_shape)
                """
                pattern_end = '$$'
                rxend = re.compile(pattern_end, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for a in rxend.findall(str(line)):
                    bvec_start = False
                """
            if bval_start:
                if num_bval_found < num_bvals:
                    vals = str(line).split("\'")[1].split(' \\')[0].split('\\n')[0].split(' ')
                    for val in vals:
                        bvals.append(val)
                        num_bval_found = np.size(bvals)
                else:
                    bval_start = False

            for a in rx1.findall(str(line)):
                bvec_start = True
                shape = str(line).split('(')[1]
                shape = str(shape).split(')')[0]
                bvec_shape = shape.split(',')
                bvec_shape = [int(i) for i in bvec_shape]
                num_bvecs = bvec_shape[0] * bvec_shape[1]

            for a in rx2.findall(str(line)):
                if 'num_bvals' not in locals():
                    bval_start = True
                    shape = str(line).split('(')[1]
                    shape = str(shape).split(')')[0]
                    num_bvals = int(shape)


    while np.shape(bvecs)[0] < np.size(bvals):
        bvecs = np.insert(bvecs, 0, 0, axis=0)
    if 'bvecs' in locals() and np.size(bvecs) > 0:
        bvecs_file = outpath+ "_bvec.txt"
        File_object = open(bvecs_file, "w")
        for bvec in bvecs:
            File_object.write(str(bvec[0]) + " " + str(bvec[1]) + " " + str(bvec[2]) + "\n")
        File_object.close()
    if 'bvals' in locals() and np.size(bvals)>0:
        bvals_file = outpath+ "_bval.txt"
        bvals = '\n'.join(bvals)
        File_object = open(bvals_file, "w")
        File_object.write(bvals)
        File_object.close()

    return bvecs_file


def read_bvecs(fbvecs):
    bvecs=[]
    if fbvecs is None or not fbvecs:
        bvecs.append(None)
    else:
        if isinstance(fbvecs, str):
            base, ext = splitext(fbvecs)
            if ext in ['.bvals', '.bval', '.bvecs', '.bvec', '.txt', '.eddy_rotated_bvecs', '']:
                with open(fbvecs, 'r') as f:
                    content = f.read()
                # We replace coma and tab delimiter by space
                with InTemporaryDirectory():
                    tmp_fname = "tmp_bvals_bvecs.txt"
                    with open(tmp_fname, 'w') as f:
                        f.write(re.sub(r'(\t|,)', ' ', content))
                    bvecs.append(np.squeeze(np.loadtxt(tmp_fname)))
            elif ext == '.npy':
                bvecs.append(np.squeeze(np.load(fbvecs)))
            else:
                e_s = "File type %s is not recognized" % ext
                raise ValueError(e_s)
        else:
            raise ValueError('String with full path to file is required')
    if np.shape(bvecs)[0]==1:
        bvecs=bvecs[0]
    return bvecs


def reorient_bvecs(bvecs, bvec_orient=[1,2,3]):

    #bvals, bvecs = get_bvals_bvecs(mypath, subject)
    bvec_sign = bvec_orient/np.abs(bvec_orient)
    bvecs = np.c_[bvec_sign[0]*bvecs[:, np.abs(bvec_orient[0])-1], bvec_sign[1]*bvecs[:, np.abs(bvec_orient[1])-1],
                  bvec_sign[2]*bvecs[:, np.abs(bvec_orient[2])-1]]
    return bvecs


def writebvec(bvecs, outpath, subject=None, writeformat="line", overwrite=False):
    if os.path.isdir(outpath) and subject is not None:
        bvec_file = os.path.join(outpath, subject + "_bvecs.txt")
    else:
        bvec_file = outpath
    if np.size(np.shape(bvecs)) == 1:
        bvecs = np.resize(bvecs, [3, int(np.size(bvecs) / 3)]).transpose()
    if np.shape(bvecs)[0] == 3:
        bvecs = bvecs.transpose()
    if os.path.exists(bvec_file):
        if overwrite:
            os.remove(bvec_file)
        else:
            print(f'already wrote {bvec_file}')
            return bvec_file
    if writeformat == "dsi":
        with open(bvec_file, 'w') as File_object:
            for i in [0, 1, 2]:
                for j in np.arange(np.shape(bvecs)[0]):
                    bvec = round(float(bvecs[j, i]), 5)
                    File_object.write(str(bvec) + "\t")
                File_object.write("\n")
            File_object.close()
    elif writeformat == "line":
        with open(bvec_file, 'w') as File_object:
            for i in [0, 1, 2]:
                for j in np.arange(np.shape(bvecs)[0]):
                    if bvecs[j, i] == 0:
                        bvec = 0
                    elif bvecs[j, i] == 1:
                        bvec = 1
                    else:
                        bvec = bvecs[j, i]
                    File_object.write(str(bvec) + " ")
            File_object.close()
        # shutil.copyfile(bvecs[0], bvec_file)
    elif writeformat == "classic":
        File_object = open(bvec_file, "w")
        for bvec in bvecs:
            File_object.write(str(bvec[0]) + " " + str(bvec[1]) + " " + str(bvec[2]) + "\n")
        File_object.close()
    elif writeformat == 'BRUKER':
        File_object = open(bvec_file, "w")
        num_dirs = np.shape(bvecs)[0]
        File_object.write(f'[directions={num_dirs}]\n')
        for i, bvec in enumerate(bvecs):
            if i + 1 != num_dirs:
                File_object.write(f'Vector[{i}]= ({str(bvec[0])}, {str(bvec[1])}, {str(bvec[2])})\n')
            else:
                File_object.write(f'Vector[{i}]= ({str(bvec[0])}, {str(bvec[1])}, {str(bvec[2])})')
    return bvec_file


def reorient_bvecs_files(bvec_file, bvec_file_reoriented, bvec_orig_orient = '', bvec_new_orient = ''):

    orig_string = 'RLAPSI'
    flip_string = 'LRPAIS'

    flip_todo = [0,0,0]

    current_vorder = bvec_orig_orient

    val = 0
    xpos = bvec_new_orient.find(current_vorder[val])
    if xpos!=-1:
        flip_todo[val] = xpos+1
    else:
        flip_todo[val] = -1 * (bvec_new_orient.find(orig_string[flip_string.find(current_vorder[val])])+1)

    val = 1
    ypos = bvec_new_orient.find(current_vorder[val])
    if ypos!=-1:
        flip_todo[val] = ypos+1
    else:
        flip_todo[val] = -1 * (bvec_new_orient.find(orig_string[flip_string.find(current_vorder[val])])+1)

    val = 2
    zpos = bvec_new_orient.find(current_vorder[val])
    if zpos!=-1:
        flip_todo[val] = zpos+1
    else:
        flip_todo[val] = -1 * (bvec_new_orient.find(orig_string[flip_string.find(current_vorder[val])])+1)

    bvecs = read_bvecs(bvec_file)
    reoriented_bvecs = reorient_bvecs(bvecs, bvec_orient = flip_todo)
    writebvec(reoriented_bvecs,bvec_file_reoriented,writeformat='classic')

    return(bvec_file)


def read_bvals(fbvals,fbvecs, sftp = None):

    vals=[]
    if sftp is not None:
        temp_path_bval = f'{os.path.join(os.path.expanduser("~"), os.path.basename(fbvals))}'
        temp_path_bvec = f'{os.path.join(os.path.expanduser("~"), os.path.basename(fbvecs))}'
        sftp.get(fbvals, temp_path_bval)
        sftp.get(fbvecs, temp_path_bvec)
        fbvals = temp_path_bval
        fbvecs = temp_path_bvec

    for this_fname in [fbvals, fbvecs]:
        # If the input was None or empty string, we don't read anything and
        # move on:
        if this_fname is None or not this_fname:
            vals.append(None)
        else:
            if isinstance(this_fname, str):
                base, ext = splitext(this_fname)
                if ext in ['.bvals', '.bval', '.bvecs', '.bvec', '.txt', '.eddy_rotated_bvecs', '']:
                    with open(this_fname, 'r') as f:
                        content = f.read()
                    # We replace coma and tab delimiter by space
                    with InTemporaryDirectory():
                        tmp_fname = "tmp_bvals_bvecs.txt"
                        with open(tmp_fname, 'w') as f:
                            f.write(re.sub(r'(\t|,)', ' ', content))
                        vals.append(np.squeeze(np.loadtxt(tmp_fname)))
                elif ext == '.npy':
                    vals.append(np.squeeze(np.load(this_fname)))
                else:
                    e_s = "File type %s is not recognized" % ext
                    raise ValueError(e_s)
            else:
                raise ValueError('String with full path to file is required')

    if sftp is not None:
        os.remove(fbvals)
        os.remove(fbvecs)
    # Once out of the loop, unpack them:
    bvals, bvecs = vals[0], vals[1]
    return bvals, bvecs


def badpath_fixer(path):
    fixed_path = re.sub('\?|!|\(|\)|', '', path)
    fixed_path = re.sub('-| ', '_', fixed_path)
    return fixed_path


def make_temppath(path, to_fix=False):
    #print(socket.gethostname())
    if 'blade' in socket.gethostname().split('.')[0]:
        temp_folder = '/mnt/munin2/Badea/Lab/jacques/temp'
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
    else:
        temp_folder = os.path.expanduser("~")
    temppath = f'{os.path.join(temp_folder, os.path.basename(path).split(".")[0]+"_temp."+ ".".join(os.path.basename(path).split(".")[1:]))}'
    #print(temppath)
    if to_fix:
        return badpath_fixer(temppath)
    else:
        return temppath


def fix_bvals_bvecs(fbvals, fbvecs, b0_threshold=50, atol=1e-2, outpath=None, identifier = "_fix",
                    writeformat="classic", writeover = False, sftp=None):
    """
    Read b-values and b-vectors from disk

    Parameters
    ----------
    fbvals : str
       Full path to file with b-values. None to not read bvals.
    fbvecs : str
       Full path of file with b-vectors. None to not read bvecs.

    Returns
    -------
    bvals : array, (N,) or None
    bvecs : array, (N, 3) or None

    Notes
    -----
    Files can be either '.bvals'/'.bvecs' or '.txt' or '.npy' (containing
    arrays stored with the appropriate values).
    """

    # Loop over the provided inputs, reading each one in turn and adding them
    # to this list:
    if type(fbvals)==str and type(fbvecs)==str:
        bvals, bvecs = read_bvals(fbvals,fbvecs, sftp)
    else:
        bvals = fbvals
        bvecs = fbvecs
        if outpath is None:
            raise Exception('Must feed some path information for saving the fixed files')
    # If bvecs is None, you can just return now w/o making more checks:
    if bvecs is None:
        return bvals, bvecs

    if bvecs.ndim != 2:
        if np.shape(bvecs)[0]%3 == 0:
            bvecs_new = np.zeros([3, int(np.shape(bvecs)[0] / 3)])
            bvecs_new[0, :] = bvecs[:int(np.shape(bvecs)[0] / 3)]
            bvecs_new[1, :] = bvecs[int(np.shape(bvecs)[0] / 3):int(2 * np.shape(bvecs)[0] / 3)]
            bvecs_new[2, :] = bvecs[int(2 * np.shape(bvecs)[0] / 3):]
            bvecs = bvecs_new
        else:
            raise IOError('bvec file should be saved as a two dimensional array')
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T

    if bvecs.shape[1] == 4:
        if np.max(bvecs[:,0]) > b0_threshold:
            if bvals is None:
                bvals = bvec[0,:]
            bvecs = np.delete(bvecs,0,1)

    # If bvals is None, you don't need to check that they have the same shape:
    if bvals is None:
        return bvals, bvecs

    if len(bvals.shape) > 1:
        raise IOError('bval file should have one row')

    if max(bvals.shape) != max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')

    from dipy.core.geometry import vector_norm

    bvecs_close_to_1 = abs(vector_norm(bvecs) - 1) <= atol

    if bvecs.shape[1] != 3:
        raise ValueError("bvecs should be (N, 3)")
    dwi_mask = bvals > b0_threshold

    correctvals = [i for i, val in enumerate(bvecs_close_to_1) if val and dwi_mask[i]]
    incorrectvals = [i for i, val in enumerate(bvecs_close_to_1) if not val and dwi_mask[i]]
    if np.size(correctvals) == 0:
        warnings.warn('Bvalues are wrong, will try to rewrite to appropriate values')
        vector_norm(bvecs[dwi_mask])[0]
        bvals_new = copy.copy(bvals)
        bvals_new[dwi_mask] = bvals[dwi_mask] * vector_norm(bvecs[dwi_mask])[0] * vector_norm(bvecs[dwi_mask])[0]
        #bvals = bvals_new
        #baseline_bval = bvals[dwi_mask][0]
        for i in incorrectvals:
            if dwi_mask[i]:
                bvecs[i,:] = bvecs[i,:] / np.sqrt(bvals_new[i]/bvals[i])

        bvecs_close_to_1 = abs(vector_norm(bvecs) - 1) <= atol
        bvals = bvals_new

    if not np.all(bvecs_close_to_1[dwi_mask]):
        correctvals = [i for i,val in enumerate(bvecs_close_to_1) if val and dwi_mask[i]]
        incorrectvals = [i for i,val in enumerate(bvecs_close_to_1) if not val and dwi_mask[i]]
        baseline_bval = bvals[correctvals[0]]
        if np.any(bvals[incorrectvals]==baseline_bval):
            warnings.warn('Bvalues are wrong, will try to rewrite to appropriate values')
            bvals_new = bvals
            for i in incorrectvals:
                if dwi_mask[i]:
                    bvals_new[i] = round(bvals[i] * np.power(vector_norm(bvecs[i,:]),2),1)
            bvals = bvals_new
        for i in incorrectvals:
            if dwi_mask[i]:
                bvecs[i,:] = bvecs[i,:] / np.sqrt(bvals[i]/baseline_bval)
        bvecs_close_to_1 = abs(vector_norm(bvecs) - 1) <= atol
        if not np.all(bvecs_close_to_1[dwi_mask]):
            incorrectvals = [i for i, val in enumerate(bvecs_close_to_1) if not val and dwi_mask[i]]
            raise ValueError("The vectors in bvecs should be unit (The tolerance "
                             "can be modified as an input parameter)")


    if outpath is None:
        base, ext = splitext(fbvals)
    else:
        base=os.path.join(outpath,os.path.basename(fbvals).replace(".txt",""))
        ext=".txt"
    if writeover==True:
        txt = f'copying over the bval file {fbvals} and bvec file {fbvecs}'
        warnings.warn(txt)
    else:
        fbvals = base + identifier + ext
    if writeformat == "classic":
        if sftp is None:
            np.savetxt(fbvals, bvals)
        else:
            np.savetxt(make_temppath(fbvals),bvals)
            sftp.put(make_temppath(fbvals),fbvals)
            os.remove(make_temppath(fbvals))
    if writeformat=="dsi":
        with open(fbvals, 'w') as File_object:
            for bval in bvals:
                if bval>10:
                    bval = int(round(bval))
                else:
                    bval=0
                File_object.write(str(bval) + "\t")
    if writeformat =='mrtrix':
        np.savetxt(fbvals, bvals.reshape(1, np.size(bvals)), fmt='%.2f')


    #base, ext = splitext(fbvecs)
    basevecs = base.replace("bval","bvec")
    if not writeover:
        fbvecs = basevecs + identifier + ext
    if writeformat=="classic":
        if sftp is None:
            np.savetxt(fbvecs, bvecs)
        else:
            np.savetxt(make_temppath(fbvecs),bvecs)
            sftp.put(make_temppath(fbvecs),fbvecs)
            os.remove(make_temppath(fbvecs))
            #    with open(fbvecs, 'w') as f:
    #        f.write(str(bvec))
    if writeformat=="dsi":
        with open(fbvecs, 'w') as File_object:
            for i in [0,1,2]:
                for j in np.arange(np.shape(bvecs)[0]):
                    if bvecs[j,i]==0:
                        bvec=0
                    else:
                        bvec=round(bvecs[j,i],3)
                    File_object.write(str(bvec)+"\t")
                File_object.write("\n")
            File_object.close()

    if writeformat=='mrtrix':
        np.savetxt(fbvecs, bvecs.T, fmt='%f')


    return fbvals, fbvecs


verbose=False
overwrite=False
masking = True
cleanup = False
checked_bvecs = True
#
subj = '20220905_14'
orig_recon_nii = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/11_CS_DWI_bart_recon.nii.gz'
method_path = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke/20220905_14/11/method'
in_path = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/'
out_path = os.path.join(in_path,'trks')
temp_path = os.path.join(in_path,'temp')

mkcdir([out_path,temp_path])

orient_start = 'ARI'
orient_end = 'RAS'

basename = os.path.basename(orig_recon_nii)

if orient_start!=orient_end:
    recon_nii = os.path.join(temp_path,basename.replace('.nii.gz',f'_{orient_end}.nii.gz'))
    if not os.path.exists(recon_nii) or overwrite:
        img_transform_exec(orig_recon_nii, orient_start, orient_end, output_path=recon_nii, recenter_test=True)
else:
    recon_nii = os.path.join(temp_path,basename)
    if not os.path.exists(recon_nii) or overwrite:
        shutil.copy(orig_recon_nii,recon_nii)

mask_nii_path = recon_nii.replace('.nii.gz','_mask.nii.gz')
masked_nii_path = recon_nii.replace('.nii.gz','_masked.nii.gz')

recon_mif = recon_nii.replace('.nii.gz','.mif')

bval_orig_path = os.path.join(in_path, f'{subj}_bval.txt')
bvec_orig_path = os.path.join(in_path, f'{subj}_bvec.txt')

bval_path = os.path.join(temp_path,f'{subj}_bval_fix.txt')
bvec_path = os.path.join(temp_path,f'{subj}_bvec_fix.txt')

bval_checked_path = os.path.join(temp_path,f'{subj}_checked.bval')
bvec_checked_path = os.path.join(temp_path,f'{subj}_checked.bvec')

if not os.path.exists(bvec_orig_path) or not os.path.exists(bvec_orig_path) or overwrite:
    extractbvals_from_method(method_path, outpath=os.path.join(in_path, f'{subj}'), tonorm=True, verbose=False)

if orient_start!=orient_end:
    bvecs_reoriented_path = os.path.join(temp_path,f'{subj}_{orient_end}.bvec')
    if not os.path.exists(bvecs_reoriented_path) or overwrite:
        reorient_bvecs_files(bvec_orig_path,bvecs_reoriented_path,orient_start,orient_end)
    bvec_orig_path = bvecs_reoriented_path

if not os.path.exists(bval_path) or not os.path.exists(bvec_path) or overwrite:
    fix_bvals_bvecs(bval_orig_path, bvec_orig_path, outpath=temp_path, b0_threshold=300,writeformat='mrtrix')

if not os.path.exists(recon_mif) or overwrite:
    os.system(
        'mrconvert ' + recon_nii + ' ' + recon_mif + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000


if not os.path.exists(mask_nii_path) or overwrite:

    median_radius = 4
    numpass = 7
    binary_dilation_val = 1

    b0_dwi_mif_temp = os.path.join(temp_path, f'orig_b0_mean_temp.mif')
    if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        command = 'dwiextract ' + recon_mif + ' - -bzero | mrmath - mean ' + b0_dwi_mif_temp + ' -axis 3 -force'
        #if not os.path.exists(b0_dwi_mif_temp) or overwrite:
        print(command)
        os.system(command)


    b0_dwi_nii_temp = os.path.join(temp_path, f'orig_b0_mean_temp.nii.gz')
    #if not os.path.exists(b0_dwi_nii_temp):
    if not os.path.exists(b0_dwi_nii_temp) or overwrite:
        os.system(f'mrconvert {b0_dwi_mif_temp} {b0_dwi_nii_temp} -force')

    median_mask_make(b0_dwi_nii_temp, masked_nii_path, median_radius=median_radius,
                     binary_dilation_val=binary_dilation_val,
                     numpass=numpass, outpathmask=mask_nii_path, verbose=verbose, overwrite=True)

    if cleanup:
        os.remove(b0_dwi_mif_temp)
        os.remove(b0_dwi_nii_temp)


if not os.path.exists(bval_checked_path) or not os.path.exists(bvec_checked_path) and checked_bvecs:
    os.system(
        f'dwigradcheck ' + recon_mif + ' -fslgrad ' + bvec_path + ' ' + bval_path + ' -mask ' + mask_nii_path + ' -number 100000 -export_grad_fsl ' + bvec_checked_path + ' ' + bval_checked_path + ' -force')


fastrun = False

keep_10mil = True

denoise = True


if checked_bvecs:
    coreg_bvecs = bvec_checked_path
    coreg_bvals = bval_checked_path
    bvec_string = ''
else:
    coreg_bvecs = bvec_path
    coreg_bvals = bval_path
    bvec_string = '_orig'


if denoise:
    denoise_str = '_denoised'
else:
    denoise_str = ''

wmfod_norm_mif = os.path.join(out_path, f'{subj}_wmfod_norm.mif')
gmfod_norm_mif = os.path.join(out_path, f'{subj}_gmfod_norm.mif')
csffod_norm_mif = os.path.join(out_path, f'{subj}_csffod_norm.mif')

if denoise:
    output_denoise_nii = recon_nii.replace('.nii.gz','_denoised.nii.gz')
    output_denoise_mif = recon_nii.replace('.nii.gz','_denoised.mif')
    if (not os.path.exists(output_denoise_mif) and not os.path.exists(output_denoise_nii)) or overwrite:
        os.system('dwidenoise ' + recon_nii + ' ' + output_denoise_nii + ' -force')
    recon_nii = output_denoise_nii
    recon_mif = output_denoise_mif
    if not os.path.exists(recon_mif) or overwrite:
        os.system(
            'mrconvert ' + recon_nii + ' ' + recon_mif + ' -fslgrad ' + bvec_checked_path + ' ' + bval_checked_path + ' -bvalue_scaling false -force')  # turn off the scaling otherwise bvals becomes 0 4000 1000 instead of 2000

if fastrun:
    smallerTracks = os.path.join(out_path, f'{subj}_smallerTracks10000{bvec_string}{denoise_str}.tck')
else:
    smallerTracks = os.path.join(out_path, f'{subj}_smallerTracks2mill{bvec_string}{denoise_str}.tck')

tracks_10M_tck = os.path.join(out_path, f'{subj}_smallerTracks10mill{bvec_string}{denoise_str}.tck')

if not os.path.exists(smallerTracks) or (keep_10mil and not os.path.exists(tracks_10M_tck)):

    # Estimating the Basis Functions:
    if not os.path.exists(wmfod_norm_mif) or overwrite:
        wm_txt = os.path.join(out_path, subj + '_wm.txt')
        gm_txt = os.path.join(out_path, subj + '_gm.txt')
        csf_txt = os.path.join(out_path, subj + '_csf.txt')
        voxels_mif = os.path.join(out_path, subj + '_voxels.mif')

        ##Right now we are using the RESAMPLED mif of the 4D miff, to be discussed
        if not os.path.exists(voxels_mif) or not os.path.exists(wm_txt) or not os.path.exists(
                gm_txt) or not os.path.exists(csf_txt) or overwrite:
            command = f'dwi2response dhollander {recon_mif} {wm_txt} {gm_txt} {csf_txt} -voxels {voxels_mif} -mask {mask_nii_path} -scratch {out_path} -fslgrad {coreg_bvecs} {coreg_bvals} -force'
            print(command)
            os.system(command)

        # Applying the basis functions to the diffusion data:
        wmfod_mif = os.path.join(out_path, subj + '_wmfod.mif')
        gmfod_mif = os.path.join(out_path, subj + '_gmfod.mif')
        csffod_mif = os.path.join(out_path, subj + '_csffod.mif')


        if not os.path.exists(wmfod_mif) or overwrite:
            #command = 'dwi2fod msmt_csd ' + subj_mif_path + ' -mask ' + mask_mif_path + ' ' + wm_txt + ' ' + wmfod_mif + ' ' + gm_txt + ' ' + gmfod_mif + ' ' + csf_txt + ' ' + csffod_mif + ' -force'
            #Only doing white matter in mouse brain
            command = f'dwi2fod msmt_csd {recon_mif} -mask {mask_nii_path} {wm_txt} {wmfod_mif} -force'
            print(command)
            os.system(command)

        if not os.path.exists(wmfod_norm_mif) or not os.path.exists(gmfod_norm_mif) or not os.path.exists(
                csffod_norm_mif) or overwrite:
            #command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' ' + gmfod_mif + ' ' + gmfod_norm_mif + ' ' + csffod_mif + ' ' + csffod_norm_mif + ' -mask ' + mask_mif_path + '  -force'
            command = 'mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' -mask ' + mask_nii_path + '  -force'
            print(command)
            os.system(command)

    gmwmSeed_coreg_mif = mask_nii_path

    if fastrun:
        command = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.1 -select 10000 ' + wmfod_norm_mif + ' ' + smallerTracks + ' -force'
        print(command)
        os.system(command)
    else:

        if not os.path.exists(tracks_10M_tck):
            command = 'tckgen -backtrack -seed_image ' + gmwmSeed_coreg_mif + ' -maxlength 250 -cutoff 0.1 -select 10000000 ' + wmfod_norm_mif + ' ' + tracks_10M_tck + ' -force'
            print(command)
            os.system(command)

        if not os.path.exists(smallerTracks):
            command = 'tckedit ' + tracks_10M_tck + ' -number 2000000 ' + smallerTracks + ' -force'
            print(command)
            os.system(command)

if verbose:
    print(f'Created {smallerTracks}')

if cleanup:
    shutil.rmtree(out_path)
