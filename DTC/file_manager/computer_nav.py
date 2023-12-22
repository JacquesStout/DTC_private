
import socket, os, getpass, paramiko, glob
#from DTC.file_manager.file_tools import getremotehome
import fnmatch
import numpy as np
import pickle
import nibabel as nib
import shutil, re
import pandas as pd
import configparser
from DTC.tract_manager.tract_save import save_trk_heavy_duty
from dipy.io.utils import create_tractogram_header


def write_parameters_to_ini(file_path, parameters):
    config = configparser.ConfigParser()
    config.read_dict({'parameters': parameters})

    with open(file_path, 'w') as config_file:
        config.write(config_file)


def read_parameters_from_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)

    # Retrieve parameters from the 'parameters' section
    parameters = dict(config.items('parameters'))

    # Convert values to appropriate types
    for key, value in parameters.items():
        if value.lower() == 'true':
            parameters[key] = True
        elif value.lower() == 'false':
            parameters[key] = False
        elif '[' in value:
            parameters[key] = [val.replace('[','').replace(']','').strip() for val in value.split(',')]
        else:
            parameters[key] = str(value)

    return parameters


def getremotehome(computer):
    import re
    homepaths_file = os.path.join(os.path.expanduser('~'), 'homepaths.rtf')
    computer = ''.join(filter(str.isalpha, computer))
    if os.path.exists(homepaths_file):
        with open(homepaths_file, 'rb') as source:
            for line in source:
                username_str = f'{computer} home'
                rx1 = re.compile(username_str, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for a in rx1.findall(str(line)):
                    homepath = str(line).split('=')[1]
                    homepath = homepath.split('\\')[0]
                    homepath = homepath.strip()
    else:
        txt = f'could not find connection parameters at {homepaths_file}'
        print(txt)
        return None
    return homepath


def get_scp(server,username=None,password=None):

    #home = getremotehome(server)
    if username is None or password is None:
        username = input("Username:")
        password = getpass.getpass("Password for " + username + ":")
    # inpath = inpath.split(":")[1]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(server, username=username, password=password)
    sftp = ssh.open_sftp()

    return sftp


def create_sftp_connection(hostname, port, username, password):
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def get_mainpaths(remote=False, project='any',username=None,password=None):
    computer_name = socket.gethostname()
    project_rename = {'Chavez':'21.chavez.02','AD_Decode':'AD_Decode','APOE':'APOE','AMD':'AMD','Daniel':'Daniel', 'Vitek':'Vitek', 'ADRC':'ADRC'}
    sftp = None
    computer_n = computer_name.split('.')[0]

    if not remote:
        home = getremotehome(computer_n)
        inpath = home
        outpath = home
        atlas_folder = os.path.join(home,'atlases')
    else:
        remote_path = getremotehome('remote')
        if '.' in remote_path:
            server = remote_path.split('.')[0]
        elif ':' in remote_path:
            server = remote_path.split(':')[0]
        home = getremotehome(server)
        inpath = home
        """
        if "@" in inpath:
            inpath = inpath.split("@")
            username = inpath[0]
            server = inpath[1].split(".")[0]
            password = getpass.getpass()
        """
        if username is None or password is None:
            username = input("Username:")
            password = getpass.getpass("Password for " + username + ":")
        #inpath = inpath.split(":")[1]
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        ssh.connect(server, username=username, password=password)
        sftp = ssh.open_sftp()

        outpath=inpath
        atlas_folder = os.path.join(inpath,'../atlases/')
    if project == 'Chavez':
        outpath = os.path.join(outpath, project_rename[project], 'Analysis')
        inpath = os.path.join(inpath, project_rename[project], 'Analysis')
    else:
        outpath = os.path.join(outpath, project_rename[project])
        inpath = os.path.join(inpath, project_rename[project])

    return inpath, outpath, atlas_folder, sftp


def splitpath(path):
    dirpath = os.path.dirname(path)
    name = os.path.basename(path).split('.')[0]
    if '.' in os.path.basename(path):
        ext = '.' + '.'.join(os.path.basename('/dwiMDT_NoNameYet_n32_i5/stats_by_region/labels/transforms/chass_symmetric3_to_MDT_warp.nii.gz').split('.')[1:])
    else:
        ext = ''
    return dirpath, name, ext


def get_sftp(remote, username=None, password=None):
    computer_name = socket.gethostname()
    server= getremotehome('remotename')
    if remote and not server in computer_name:
        if username is None:
            username = input("Username:")
        if password is None:
            password = getpass.getpass("Password for " + username + ":")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        ssh.connect(server, username=username, password=password)
        sftp = ssh.open_sftp()
    else:
        sftp=None

    return sftp


def get_atlas(atlas_folder, atlas_type):
    if atlas_type == 'IIT':
        index_path = os.path.join(atlas_folder,'IITmean_RPI','IITmean_RPI_index.xlsx')
    elif atlas_type == 'CHASSSYMM3':
        index_path = os.path.join(atlas_folder,'CHASSSYMM3AtlasLegends.xlsx')
    else:
        raise Exception('unknown atlas')
    return index_path


def badpath_fixer(path):
    fixed_path = re.sub('\?|!|\(|\)|', '', path)
    fixed_path = re.sub('-| ', '_', fixed_path)
    return fixed_path


def make_temppath(path, to_fix=False):
    if 'blade' in socket.gethostname().split('.')[0]:
        temp_folder = '/mnt/munin2/Badea/Lab/jacques/temp'
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
    else:
        temp_folder = os.path.expanduser("~")
    temppath = f'{os.path.join(temp_folder, os.path.basename(path).split(".")[0]+"_temp."+ ".".join(os.path.basename(path).split(".")[1:]))}'
    if to_fix:
        return badpath_fixer(temppath)
    else:
        return temppath


def load_nifti_remote(niipath, sftp=None, return_nii = False):
    from DTC.nifti_handlers.nifti_handler import get_reference_info

    if sftp is None:
        img = nib.load(niipath)
        data = img.get_fdata()
        vox_size = img.header.get_zooms()[:3]
        affine = img.affine
        header = img.header
        ref_info = get_reference_info(niipath)
    else:
        temp_path = f'{os.path.join(os.path.expanduser("~"), os.path.basename(niipath))}'
        sftp.get(niipath, temp_path)
        try:
            img = nib.load(temp_path)
            data = img.get_data()
            vox_size = img.header.get_zooms()[:3]
            affine = img.affine
            header = img.header
            ref_info = get_reference_info(temp_path)
            os.remove(temp_path)
        except Exception as e:
            os.remove(temp_path)
            raise Exception(e)
    if return_nii:
        return img
    else:
        return data, affine, vox_size, header, ref_info


def save_nifti_remote(niiobject,niipath, sftp):

    if sftp is None:
        nib.save(niiobject, niipath)
    else:
        nib.save(niiobject, make_temppath(niipath))
        sftp.put(make_temppath(niipath),niipath)
        os.remove(make_temppath(niipath))
    return


def save_df_remote(df,df_path, sftp):

    if sftp is None:
        df.to_excel(df_path)
    else:
        df.to_excel(make_temppath(df_path))
        sftp.put(make_temppath(df_path),df_path)
        os.remove(make_temppath(df_path))
    return


def load_df_remote(df_path, sftp):

    if sftp is None:
        db = pd.read_excel(df_path)
    else:
        sftp.get(df_path,make_temppath(df_path))
        db = pd.read_excel(make_temppath(df_path))
        os.remove(make_temppath(df_path))
    return db


def remove_remote(path, sftp=None):
    if sftp is None:
        os.remove(path)
    else:
        sftp.remove(path)


def read_bvals_bvecs_remote(fbvals, fbvecs, sftp):
    from dipy.io.gradients import read_bvals_bvecs

    if sftp is not None:
        temp_path_bval = f'{os.path.join(os.path.expanduser("~"), os.path.basename(fbvals))}'
        temp_path_bvec = f'{os.path.join(os.path.expanduser("~"), os.path.basename(fbvecs))}'
        sftp.get(fbvals, temp_path_bval)
        sftp.get(fbvecs, temp_path_bvec)
        try:
            bvals, bvecs = read_bvals_bvecs(temp_path_bval, temp_path_bvec)
            os.remove(temp_path_bval)
            os.remove(temp_path_bvec)
        except Exception as e:
            os.remove(temp_path_bval)
            os.remove(temp_path_bvec)
            raise Exception(e)
    else:
        bvals, bvecs = read_bvals_bvecs(fbvals,fbvecs)
    return bvals, bvecs


def remote_pickle(picklepath, sftp=None,erase_temp=True):
    if sftp is not None:
        picklepath_tconnectome = make_temppath(picklepath)
        sftp.get(picklepath, picklepath_tconnectome)
    else:
        picklepath_tconnectome = picklepath
    with open(picklepath_tconnectome, 'rb') as f:
        M = pickle.load(f)
    if sftp is not None and erase_temp:
        os.remove(picklepath_tconnectome)
    return M


def load_trk_remote(trkpath,reference,sftp=None):
    #from dipy.io.streamline import load_trk
    from DTC.tract_manager.streamline_nocheck import load_trk as load_trk_spe
    if sftp is not None:
        temp_path = f'{os.path.join(os.path.expanduser("~"), os.path.basename(trkpath))}'
        sftp.get(trkpath, temp_path)
        try:
            trkdata = load_trk_spe(temp_path, reference)
            #trkdata = load_trk(temp_path, reference)
            os.remove(temp_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception(e)
    else:
        trkdata = load_trk_spe(trkpath, reference)
    return trkdata


def save_trk_remote(trkpath,streamlines,header, affine=np.eye(4),sftp=None):
    from DTC.tract_manager.tract_save import save_trk_heavy_duty
    if sftp is not None:
        temp_path =make_temppath(trkpath)
        myheader = create_tractogram_header(temp_path, *header)
        lambda_streamlines = lambda: (s for s in streamlines)
        save_trk_heavy_duty(temp_path, streamlines=lambda_streamlines,
                            affine=affine, header=myheader)
        sftp.put(temp_path,trkpath)
        os.remove(temp_path)
    else:
        myheader = create_tractogram_header(trkpath, *header)
        lambda_streamlines = lambda: (s for s in streamlines)
        save_trk_heavy_duty(trkpath, streamlines=lambda_streamlines,
                            affine=affine, header=myheader)
    return


def save_fig_remote(figpath, sftp=None):
    import matplotlib.pyplot as plt
    if sftp is not None:
        temp_path =make_temppath(figpath)
        plt.savefig(temp_path)
        sftp.put(temp_path,figpath)
        os.remove(temp_path)
    else:
        plt.savefig(figpath)


def loadmat_remote(matpath, sftp=None):
    from scipy.io import loadmat
    if sftp is not None:
        temp_path = f'{os.path.join(os.path.expanduser("~"), os.path.basename(matpath))}'
        sftp.get(matpath, temp_path)
        try:
            mymat = loadmat(temp_path)
            os.remove(temp_path)
        except Exception as e:
            os.remove(temp_path)
            raise Exception(e)
    else:
        mymat = loadmat(matpath)
    return mymat

"""
def savemat_remote(matpath, mat_dict, sftp = None):
    from scipy.io import savemat
    
    create_ants_transform
    if sftp is not None:
        savemat(matpath, mat_dict, appendmat=False, long_field_names=True, do_compression=False, )
"""

def true_loadmat(matpath, sftp=None):
    struct = loadmat_remote(matpath, sftp)
    var_name = list(struct.keys())[0]
    mat = struct[var_name]
    return mat


def ants_loadmat(matpath, sftp=None):
    old_mat = true_loadmat(matpath, sftp=None)
    ants_mat =np.eye(4)

    ants_mat[:3,:3] = old_mat[:-3].reshape((3,3))
    ants_mat[:3, 3] = old_mat[-3:].reshape(3)
    return(ants_mat)


def scp_multiple(list,outpath,sftp=None,overwrite=False):
    for filepath in list:
        newfilepath = os.path.join(outpath,os.path.basename(filepath))
        if not overwrite and os.path.exists(newfilepath):
            print(f'{newfilepath} already exists')
        else:
            print(f'Copying {filepath} to {newfilepath}')
            if sftp is not None:
                sftp.get(filepath,newfilepath)
            else:
                import shutil
                shutil.copy(filepath, newfilepath)

def regexify(string):
    newstring = ('^'+string+'$').replace('*','.*')
    return newstring

def glob_remote(path, sftp=None):
    match_files = []
    if sftp is not None:
        if '*' in path:
            pathdir, pathname = os.path.split(path)
            pathname = regexify(pathname)
            allfiles = sftp.listdir(pathdir)
            for file in allfiles:
                if re.search(pathname, file) is not None:
                    match_files.append(os.path.join(pathdir,file))
        elif '.' not in path and 'method' not in path and 'fid' not in path:
            allfiles = sftp.listdir(path)
            for filepath in allfiles:
                match_files.append(os.path.join(path, filepath))
            return match_files
        else:
            dirpath = os.path.dirname(path)
            try:
                sftp.stat(dirpath)
            except:
                return match_files
            allfiles = sftp.listdir(dirpath)
            #if '*' in path:
            #    for filepath in allfiles:
            #            match_files.append(os.path.join(dirpath,filepath))
            #else:
            for filepath in allfiles:
                if fnmatch.fnmatch(os.path.basename(filepath), os.path.basename(path)):
                    match_files.append(os.path.join(dirpath, filepath))
    else:
        if '.' not in path:
            match_files = glob.glob(path)
        else:
            dirpath = os.path.dirname(path)
            if not os.path.exists(dirpath):
                return(match_files)
            else:
                allfiles = glob.glob(os.path.join(dirpath,'*'))
                for filepath in allfiles:
                    if fnmatch.fnmatch(os.path.basename(filepath), os.path.basename(path)):
                        match_files.append(os.path.join(dirpath, filepath))
    return(match_files)


##Copys a file to a sftp new path, if not sftp, simply copy
def copy_loctoremote(file,newfile,sftp=None):
    if sftp is None:
        shutil.copy(file,newfile)
    else:
        sftp.put(file,newfile)


##Copys a sftp file to a new sftp path, if not sftp, simple copy
def copy_remotefiles(file,newfile,sftp=None):
    if sftp is not None:
        temp_path = make_temppath(file)
        sftp.get(file, temp_path)
        sftp.put(temp_path,newfile)
        os.remove(temp_path)
    else:
        shutil.copy(file,newfile)


def pickledump_remote(var,path,sftp=None):
    if sftp is None:
        pickle.dump(var, open(path, "wb"))
    else:
        temp_path = make_temppath(path)
        pickle.dump(var, open(temp_path, "wb"))
        sftp.put(temp_path, path)
        os.remove(temp_path)


def checkfile_exists_remote(path, sftp=None):
    match_files = glob_remote(path,sftp)
    if np.size(match_files)>0:
        return True
    else:
        return False

def checkfile_exists_all(paths, sftp=None, percent = None):
    exists=True
    i=0
    for path in paths:
        if not checkfile_exists_remote(path, sftp):
            exists=False
        else:
            i+=1
    if percent is None:
        return exists
    else:
        exists = (i/np.size(paths))>percent
        return exists

def checkfile_exists_all_faster(paths, percent = None):
    exists=True
    i=0
    for path in paths:
        if not os.path.exists(path):
            exists=False
        else:
            i+=1
    if percent is None:
        return exists
    else:
        exists = (i/np.size(paths))>percent
        return exists


def checkallfiles(paths, sftp=None):
    existing = True
    for path in paths:
        match_files = glob_remote(path, sftp)
        if not match_files:
            existing= False
    return existing
