
from DTC.file_manager.computer_nav import splitpath
import os
import numpy as np
import nibabel as nib
import warnings
from file_tools import mkcdir, largerfile

def recenter_affine(shape, affine, return_translation = False):

    origin=np.round([val/2 for val in shape])
    origin=origin[0:3]
    trueorigin=-origin + [1,1,1]

    newaffine=np.zeros([4,4])

    newaffine[0:3,0:3]=affine[0:3,0:3]
    newaffine[3,:]=affine[3,:]

    trueorigin = np.matmul(newaffine[0:3,0:3],trueorigin)
    newaffine[:3,3]=trueorigin

    if return_translation:
        translation = trueorigin - affine[3,0:3]
        translation_mat = np.eye(4)
        translation_mat[:3, 3] = translation
        return newaffine, translation, translation_mat
    else:
        return newaffine


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

    nii_data = nii.get_data()
    hdr = nii.header

    dims = nii.shape
    if np.size(dims) > 6:
        raise('Image has > 5 dimensions')
    elif np.size(dims) < 3:
        raise('Image has < 3 dimensions')

    new_data = nii_data
    affine = nii._affine

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
            elif np.size(dims) == 3:
                new_data = new_data.transpose(xpos, ypos, zpos)

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

    new_nii=nib.Nifti1Image(new_data, newaffine, hdr)
    output_path = str(output_path)
    if verbose:
        print(f'Saving nifti file to {output_path}')
    nib.save(new_nii, output_path)
    if verbose:
        print(f'Saved')


original_orientation = 'ARI'
new_orientation = 'RAS'
file_path = '.nii.gz'
reorientedfile_path = '.nii.gz'
#reorientedfile_path = file_path.replace('.nii',f'{new_orientation}.nii')

img_transform_exec(file_path,'ARI','RAS',reorientedfile_path)