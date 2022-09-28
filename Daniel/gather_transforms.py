from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile
import os, glob, shutil
from DTC.file_manager.computer_nav import getremotehome, get_mainpaths
from pathlib import Path

mainpath = getremotehome('Lab')

SAMBA_mainpath = os.path.join(mainpath, "mouse")
SAMBA_projectname = "VBM_18APOERAREset02_invivoAPOE1"
SAMBA_work_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-work")

register_target = 'T1'

transforms_folder = '/Users/jas/jacques/Daniel_test/Transforms'
transforms_folder = '/Users/jas/jacques/Daniel_test/Transforms_2'
mkcdir(transforms_folder)

template_type_prefix = os.path.basename(os.path.dirname(glob.glob(os.path.join(SAMBA_work_folder,register_target,"SyN*/"))[0]))
template_runs = glob.glob((os.path.join(SAMBA_work_folder,register_target,template_type_prefix,"*/")))

subjects = ['sub2204041','sub22040410','sub22040411','sub22040413','sub2204042','sub2204043','sub2204044','sub2204045','sub2204046','sub2204047','sub2204048','sub2204049','sub2205091','sub22050910','sub22050911','sub22050912','sub22050913','sub22050914','sub2205094','sub2205097','sub2205098','sub2206061','sub22060610','sub22060611','sub22060612','sub22060613','sub22060614','sub2206062','sub2206063','sub2206064','sub2206065','sub2206066','sub2206067','sub2206068','sub2206069']
#subjects = ['sub2204048']
subjects = ['sub22040413', 'sub22040411', 'sub22040401', 'sub22040410', 'sub22040402', 'sub22040403', 'sub22040404',
            'sub22040405', 'sub22040406', 'sub22040407', 'sub22040408', 'sub22040409', 'sub22050901', 'sub22050910',
            'sub22050911', 'sub22050912', 'sub22050913', 'sub22050914', 'sub22050904', 'sub22050907', 'sub22050908',
            'sub22060601', 'sub22060610', 'sub22060611', 'sub22060612', 'sub22060613', 'sub22060614', 'sub22060602',
            'sub22060603', 'sub22060604', 'sub22060605', 'sub22060606', 'sub22060607', 'sub22060608', 'sub22060609']

overwrite= True
verbose=True
remote=False
copytype = "truecopy"

if remote:
    _, outpath, _, sftp = get_mainpaths(remote, project=project, username=username, password=passwd)

mymax = -1
for template_run in template_runs:
    if "JS_rabies" in template_run and template_run[-4:-2]=="_i":
        if int(template_run[-2])>mymax:
            mymax=int(template_run[-2])
            final_template_run=template_run
if mymax==-1:
    for template_run in template_runs:
        if "dwiMDT_Control_n72" in template_run and template_run[-4:-2]=="_i":
            if int(template_run[-2])>mymax:
                mymax=int(template_run[-2])
                final_template_run=template_run
if mymax == -1:
    raise Exception(f"Could not find template runs in {os.path.join(mainpath, f'{SAMBA_projectname}-work',register_target,template_type_prefix)}")

for subject in subjects:
    trans = os.path.join(SAMBA_work_folder, "preprocess", "base_images", "translation_xforms",
                         f"{subject}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(SAMBA_work_folder, register_target, f"{subject}_rigid.mat")
    affine = os.path.join(SAMBA_work_folder, register_target, f"{subject}_affine.mat")
    runno_to_MDT = os.path.join(final_template_run, "reg_diffeo", f"{subject}_to_MDT_warp.nii.gz")
    burn_dir = os.path.join(SAMBA_mainpath, "burn_after_reading")
    MDT_to_subject = os.path.join(final_template_run,"reg_diffeo",f"MDT_to_{subject}_warp.nii.gz")

    #MDT_to_atlas_affine = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"MDT_*_to_{label_name}_affine.mat")
    #atlas_to_MDT = os.path.join(final_template_run,"stats_by_region","labels","transforms",f"{label_name}_to_MDT_warp.nii.gz")

    #affine_mat_path = os.path.join(burn_dir, f'{subject}_affine.txt')

    #transform_files = [trans, rigid, affine, affine_mat_path, runno_to_MDT]
    transform_files = [trans, rigid, affine, runno_to_MDT, MDT_to_subject]

    for filepath in transform_files:
        if os.path.exists(filepath):
            if Path(filepath).is_symlink():
                filepath=Path(filepath).resolve()
            filename = os.path.basename(filepath)
            filenewpath = os.path.join(transforms_folder, filename)
            if remote:
                try:
                    sftp.chdir(transforms_folder)
                except IOError:
                    sftp.mkdir(transforms_folder)
            else:
                mkcdir(transforms_folder)

            if not os.path.isfile(filenewpath) or overwrite:
                if copytype=="shortcut":
                    if remote:
                        raise Exception("Can't build shortcut to remote path")
                    else:
                        buildlink(filepath, filenewpath)
                        if verbose:
                            print(f'Built link for {filepath} at {filenewpath}')
                elif copytype=="truecopy":
                    if remote:
                        if not overwrite:
                            try:
                                sftp.stat(filenewpath)
                                if verbose:
                                    print(f'file at {filenewpath} exists')
                            except IOError:
                                if verbose:
                                    print(f'copying file {filepath} to {filenewpath}')
                                sftp.put(filepath, filenewpath)
                        else:
                            if verbose:
                                print(f'copying file {filepath} to {filenewpath}')
                            sftp.put(filepath, filenewpath)
                    else:
                        shutil.copy(filepath, filenewpath)
                        if verbose:
                            print(f'copying file {filepath} to {filenewpath}')
        else:
            print(f'Could not find {filepath}')