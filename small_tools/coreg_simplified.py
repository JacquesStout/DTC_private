
import os, shutil, subprocess, glob
from file_tools import mkcdir
from DTC.file_manager.file_tools import largerfile, mkcdir, getext, buildlink
import numpy as np
#from DTC.file_manager.file_tools import mkcdir
import getpass

def coreg_run(nii_path, subj, dti, outpath, ANTSPATH = '~/Ants/install/bin', qsub=False):
        
    job_desc = "co_reg";  # e.g. co_reg
    job_shorthand = "Reg";  # "Reg" for co_reg
    ext = "nii.gz";
    
    sbatch_file = '';
    
    print(f"Processing subj: {subj}")
    
    if not os.path.exists(nii_path):
        #print(f"ABORTING: Input file does not exist: {nii_path}")
        raise FileNotFoundError(f"ABORTING: Input file does not exist: {nii_path}")
    

    #YYY =(PrintHeader nii_path 2 | cut -d 'x' -f4);
    try:
        YYY=str(subprocess.check_output(['PrintHeader', '/Volumes/Data/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/diffusion_prep_N60056/nii4D_N60056_masked.nii.gz', '2'])).split('x')[-1].split('\\n')[0]
    except FileNotFoundError:
        os.environ["PATH"] = f"{ANTSPATH}/:" + os.environ["PATH"]
        YYY=str(subprocess.check_output(['PrintHeader', '/Volumes/Data/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/diffusion_prep_N60056/nii4D_N60056_masked.nii.gz', '2'])).split('x')[-1].split('\\n')[0]

    XXX = int(YYY) - 1;
    
    print(f"Total number of volumes: {YYY}")
    print(f"Number of independently oriented volumes: {XXX}")
    
    if XXX<10:
        zeros = '0'
    elif XXX<100:
        zeros = '00'
    else:
        zeros = '000'

    discard = '1000';
    discard = discard.replace(zeros,'')
    zero = '0';
    zero_pad = zeros.replace(zero,'')
    
    inputs = f"{outpath}/{job_desc}_{subj}_m{zeros}-inputs/";
    work = f"{outpath}/{job_desc}_{subj}_m{zeros}-work/";
    results = f"{outpath}/{job_desc}_{subj}_m{zeros}-results/";
    
    vol_zero = f"{inputs}{subj}_m{zeros}.{ext}";
    
    print(f"Target for coregistration: {vol_zero}")

    mkcdir(inputs)
    mkcdir(work)
    
    sbatch_folder = os.path.join(work, "sbatch/")
    mkcdir(sbatch_folder)
    mkcdir(results)

    prefix = f"{outpath}/{job_desc}_{subj}_m{zeros}-inputs/{subj}_m.nii.gz";

    if not os.path.exists(vol_zero):
        if not os.path.exists(os.path.join(prefix.replace('_m.nii','_m1000.nii'))):
            print("Splitting up nii4D volume...")
            os.system(f'{ANTSPATH}/ImageMath 4 {prefix} TimeSeriesDisassemble {nii_path}')

    allfiles = glob.glob(os.path.join(inputs,'*.nii.gz'))
    for file in allfiles:
        file_link=f'{file.replace(f"_m{discard}","_m")}'
        if not os.path.exists(file_link):
            buildlink(file, file_link)

    work_vol_zero=f"{work}/{job_shorthand}_{subj}_m{zeros}.{ext}";
    reassemble_list=f"{work_vol_zero} ";
    jid_list='';
    
    # for nn in (seq 1 XXX);do

    if not os.path.exists(work_vol_zero):
        buildlink(vol_zero,work_vol_zero)

    print("Dispatching co-registration jobs to the cluster:")
    
    # Note the following line is necessarily complicated, as...
    # the common sense line ('for nn in {01..$XXX}') does not work...
    # https://stackoverflow.com/questions/169511/how-do-i-iterate-over-a-range-of-numbers-defined-by-variables-in-bash
    for nn in np.arange(XXX):
        num_digits=len(str(nn))
        num_zeros=len(zeros) - num_digits
        num_string=''
        for i in np.arange(num_zeros):
            num_string += '0'
        num_string += str(nn) #should be one zero for <10, nothing for 10+

        # num_string=$nn;
        vol_xxx=f"{inputs}{subj}_m{num_string}.{ext}";
        out_prefix=f"{results}xform_{subj}_m{num_string}.{ext}";
        xform_xxx=f"{out_prefix}0GenericAffine.mat";
        vol_xxx_out=f"{work}/{job_shorthand}_{subj}_m{num_string}.{ext}";
        reassemble_list=f"{reassemble_list} {vol_xxx_out} ";

        name=f"{job_desc}_{subj}_m{num_string}";
        sbatch_file=f"{sbatch_folder}/{name}.bash";
        # source_sbatch="${BIGGUS_DISKUS}/sinha_co_reg_nii_path_qsub_master.bash";
        # cp ${source_sbatch} ${sbatch_file};

        if not os.path.exists(xform_xxx) or not os.path.exists(vol_xxx_out):

            sbatch_file = f"{sbatch_folder}/{name}.bash";

            reg_cmd=f"{ANTSPATH}/antsRegistration  --float -d 3 -v  -m Mattes[ {vol_zero},{vol_xxx},1,32,regular,0.3 ] -t Affine[0.05] -c [ 100x100x100,1.e-5,15 ] -s 0x0x0vox -f 4x2x1 -u 1 -z 1 -o {out_prefix}"
            apply_cmd=f"{ANTSPATH}/antsApplyTransforms -d 3 -e 0 -i {vol_xxx} -r {vol_zero} -o {vol_xxx_out} -n Linear -t {xform_xxx}  -v 0 --float"

            if qsub:
                USER = getpass.getuser()
                with open(sbatch_file, 'a') as f1:
                    f1.write( "#!/bin/bash" + os.linesep)
                    f1.write( "#\$ -l h_vmem=8000M,vf=8000M" + os.linesep)
                    f1.write( f"#\$ -M {USER}@duke.edu" + os.linesep)
                    f1.write( "#\$ -m ea" + os.linesep)
                    f1.write( f"#\$ -o {sbatch_folder}"'/slurm-$JOB_ID.out' + os.linesep)
                    f1.write( f"#\$ -e {sbatch_folder}"'/slurm-$JOB_ID.out' + os.linesep)
                    f1.write( f"#\$ -N {name}" + os.linesep)
                    f1.write(f"#\$ -N {name}" + os.linesep)
                    f1.write(reg_cmd + os.linesep)
                    f1.write(apply_cmd + os.linesep)
                    os.system(f'bash {sbatch_file}')

            if not os.path.exists(xform_xxx):
                os.system(reg_cmd)
            if not os.path.exists(vol_xxx_out):
                os.system(apply_cmd)

        """
        echo "${reg_cmd}" >> ${sbatch_file};
        echo "${apply_cmd}" >> ${sbatch_file};
            if ! command -v qsub & > / dev / null | |["$qsub" == "0"]; then
        bash ${sbatch_file};
        else
        """

    reg_nii_path=f"{results}/{job_shorthand}_{subj}_nii_path.{ext}"
    assemble_cmd=f"{ANTSPATH}/ImageMath 4 {reg_nii_path} TimeSeriesAssemble 1 0 {reassemble_list}"
    # if [[ 1 -eq 2 ]];then # Uncomment when we want to short-circuit this to OFF
    if not os.path.exists(reg_nii_path):
        name=f"assemble_nii_path_{job_desc}_{subj}_m{zeros}";
        assemble_cmd = f"{assemble_cmd}"

        if qsub:
            sbatch_file=f"{sbatch_folder}/{name}.bash";
            f1.write("#!/bin/bash" + os.linesep)
            f1.write("#\$ -l h_vmem=32000M,vf=32000M" + os.linesep)
            f1.write(f"#\$ -M {USER}@duke.edu" + os.linesep)
            f1.write("#\$ -m ea" + os.linesep)
            f1.write(f"#\$ -o {sbatch_folder}"'/slurm-$JOB_ID.out' + os.linesep)
            f1.write(f"#\$ -e {sbatch_folder}"'/slurm-$JOB_ID.out' + os.linesep)
            f1.write(f"#\$ -N {name}" + os.linesep)
            f1.write(assemble_cmd + os.linesep)
            os.system(f'bash {sbatch_file}')
        else:
            os.system(assemble_cmd)

    reg_nii_path_f = os.path.join(outpath,f'{job_shorthand}_{subj}_nii_path.{ext}')
    shutil.move(reg_nii_path, reg_nii_path_f)

nii_path = '/Users/jas/jacques/coreg_testzone/positive_rawdata_MVMS11AZD_1_05_1_reverse_EPI_DTI_Blip_TR180_E5_masked.nii.gz'
work_dir = '/Users/jas/jacques/coreg_testzone/'
subj_name = 'V11_8_13'
outpath = '/Users/jas/jacques/coreg_testzone/outpath'
Antspath = '/Users/jas/Ants/install/bin/'
ext = '.nii.gz'
overwrite=False

coreg_nii_old = f'{outpath}/co_reg_{subj_name}_m00-results/Reg_{subj_name}_nii_path{ext}';
coreg_nii = os.path.join(outpath, f'Reg_{subj_name}_nii_path{ext}')

if not os.path.exists(coreg_nii) or overwrite:
    coreg_run(nii_path, subj_name, dti=0, outpath=outpath, ANTSPATH=Antspath)
    #shutil.move(coreg_nii_old, coreg_nii)
else:
    print(f'Path coreg already exits {coreg_nii}')