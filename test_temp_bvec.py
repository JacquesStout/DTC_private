import numpy as np

from os.path import splitext

from nibabel.tmpdirs import InTemporaryDirectory
import os, re
from computer_nav import make_temppath




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

def fix_bvals_bvecs(fbvals, fbvecs, b0_threshold=50, atol=1e-2, outpath=None, identifier = "_fix", writeformat="classic",sftp=None):
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
    bvals, bvecs = read_bvals(fbvals,fbvecs, sftp)

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
    if not np.all(bvecs_close_to_1[dwi_mask]):
        correctvals = [i for i,val in enumerate(bvecs_close_to_1) if val and dwi_mask[i]]
        incorrectvals = [i for i,val in enumerate(bvecs_close_to_1) if not val and dwi_mask[i]]
        baseline_bval = bvals[correctvals[0]]
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
    #base, ext = splitext(fbvecs)
    basevecs = base.replace("bvals","bvecs")
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

    return fbvals, fbvecs