
import os, shutil, subprocess
from file_tools import mkcdir

def coreg_run(nii_path, subj, dti, outpath, ANTSPATH = '~/Ants/install/bin'):
        
    job_desc = "co_reg";  # e.g. co_reg
    job_shorthand = "Reg";  # "Reg" for co_reg
    ext = "nii.gz";
    
    sbatch_file = '';
    
    print(f"Processing subj: {subj}")
    
    if not os.path.exists(nii_path):
        #print(f"ABORTING: Input file does not exist: {nii_path}")
        raise FileNotFoundError(f"ABORTING: Input file does not exist: {nii_path}")
    

    #YYY =(PrintHeader nii_path 2 | cut -d 'x' -f4);
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
    
    sbatch_folder = os.path.join({work}, "sbatch/")
    mkcdir(sbatch_folder)
    mkcdir(results)

    prefix = "${outpath}/${job_desc}_${subj}_m${zeros}-inputs/${subj}_m.nii.gz";

    if not os.path.exists(vol_zero):
        if not os.path.exists(os.path.join(prefix,'_m/_m1000')):
            print("Splitting up nii4D volume...")
            os.system(f'{ANTSPATH}/ImageMath 4 {prefix} TimeSeriesDisassemble {nii_path}')

    if not os.path.exists(os.path.join(prefix,'_m/_m1000')):
        print("Splitting up nii_path volume...")
        os.system(f'{ANTSPATH}/ImageMath 4 {prefix} TimeSeriesDisassemble {nii_path}')

    for file in $(ls ${inputs});do
    new_file=${inputs} / ${file / _m${discard} / _m};
    if[[! -e ${new_file}]];then
    ln -s $file ${new_file};
    fi
    done
    fi
    
    work_vol_zero="${work}/${job_shorthand}_${subj}_m${zeros}.${ext}";
    reassemble_list="${work_vol_zero} ";
    jid_list='';
    
    # for nn in $(seq 1 $XXX);do
    
    if[[! -e ${work_vol_zero}]];then
    ln -s ${vol_zero} ${work_vol_zero};
    fi
    echo "Dispatching co-registration jobs to the cluster:";
    
    # Note the following line is necessarily complicated, as...
    # the common sense line ('for nn in {01..$XXX}') does not work...
    # https://stackoverflow.com/questions/169511/how-do-i-iterate-over-a-range-of-numbers-defined-by-variables-in-bash
    for nn in $(eval echo "{${zero_pad}1..$XXX}");do
    num_digits="${#nn}"
    num_zeros=$((${  # zeros} - $num_digits))
    num_string=''
    if[[$num_zeros -gt 0]]; then
    for ((i=0; i < $num_zeros; i++)); do
    num_string += '0'
    done
    fi
    num_string += $nn
    
    # if [[ $nn -lt 10 ]];then
    #    #num_string="0${nn}";
    #    num_string="0${nn}";
    # fi
    
    # num_string=$nn;
    vol_xxx="${inputs}${subj}_m${num_string}.${ext}";
    out_prefix="${results}xform_${subj}_m${num_string}.${ext}";
    xform_xxx="${out_prefix}0GenericAffine.mat";
    vol_xxx_out="${work}/${job_shorthand}_${subj}_m${num_string}.${ext}";
    reassemble_list="${reassemble_list} ${vol_xxx_out} ";
    
    name="${job_desc}_${subj}_m${num_string}";
    sbatch_file="${sbatch_folder}/${name}.bash";
    # source_sbatch="${BIGGUS_DISKUS}/sinha_co_reg_nii_path_qsub_master.bash";
    # cp ${source_sbatch} ${sbatch_file};
    if[[! -e ${xform_xxx}]] | |[[! -e ${vol_xxx_out}]];then
    
    echo "#!/bin/bash" > ${sbatch_file};
    echo "#\$ -l h_vmem=8000M,vf=8000M" >> ${sbatch_file};
    echo "#\$ -M ${USER}@duke.edu" >> ${sbatch_file};
    echo "#\$ -m ea" >> ${sbatch_file};
    echo "#\$ -o ${sbatch_folder}"'/slurm-$JOB_ID.out' >> ${sbatch_file};
    echo "#\$ -e ${sbatch_folder}"'/slurm-$JOB_ID.out' >> ${sbatch_file};
    echo "#\$ -N ${name}" >> ${sbatch_file};
    
    reg_cmd="if [[ ! -e ${xform_xxx} ]];then ${ANTSPATH}/antsRegistration  --float -d 3 -v  -m Mattes[ ${vol_zero},${vol_xxx},1,32,regular,0.3 ] -t Affine[0.05] -c [ 100x100x100,1.e-5,15 ] -s 0x0x0vox -f 4x2x1 -u 1 -z 1 -o ${out_prefix};fi";
    apply_cmd="if [[ ! -e ${vol_xxx_out} ]];then ${ANTSPATH}/antsApplyTransforms -d 3 -e 0 -i ${vol_xxx} -r ${vol_zero} -o ${vol_xxx_out} -n Linear -t ${xform_xxx}  -v 0 --float;fi";
    
    echo "${reg_cmd}" >> ${sbatch_file};
    echo "${apply_cmd}" >> ${sbatch_file};
    
    if ! command -v qsub & > / dev / null | |["$qsub" == "0"]; then
    bash ${sbatch_file};
    else
    # cmd="qsub -terse  -b y -V ${sbatch_file}";
    cmd="qsub -terse -V ${sbatch_file}";
    echo $cmd;
    job_id=$($cmd | tail -1);
    echo "JOB ID = ${job_id}; Job Name = ${name}";
    new_sbatch_file=${sbatch_file / ${name} / ${job_id}_${name}};
    mv ${sbatch_file} ${new_sbatch_file};
    jid_list="${jid_list}${job_id},";
    fi
    fi
    done
    
    # Trim trailing comma from job id list:
    jid_list=${jid_list %, };
    
    reg_nii_path="${results}/${job_shorthand}_${subj}_nii_path.${ext}";
    assemble_cmd="${ANTSPATH}/ImageMath 4 ${reg_nii_path} TimeSeriesAssemble 1 0 ${reassemble_list}";
    # if [[ 1 -eq 2 ]];then # Uncomment when we want to short-circuit this to OFF
    if[[! -f ${reg_nii_path}]];then
    name="assemble_nii_path_${job_desc}_${subj}_m${zeros}";
    sbatch_file="${sbatch_folder}/${name}.bash";
    
    echo "#!/bin/bash" > ${sbatch_file};
    echo "#\$ -l h_vmem=32000M,vf=32000M" >> ${sbatch_file};
    echo "#\$ -M ${USER}@duke.edu" >> ${sbatch_file};
    echo "#\$ -m ea" >> ${sbatch_file};
    echo "#\$ -o ${sbatch_folder}"'/slurm-$JOB_ID.out' >> ${sbatch_file};
    echo "#\$ -e ${sbatch_folder}"'/slurm-$JOB_ID.out' >> ${sbatch_file};
    echo "#\$ -N ${name}" >> ${sbatch_file};
    if[["x${jid_list}x" != "xx"]];then
    echo "#\$ -hold_jid ${jid_list}" >> ${sbatch_file};
    fi
    echo "${assemble_cmd}" >> ${sbatch_file};
    
    if ! command -v qsub & > / dev / null | |["$qsub" == "0"]; then
    bash ${sbatch_file};
    else
    ass_cmd="qsub -terse -V ${sbatch_file}";
    echo $ass_cmd;
    job_id=$($ass_cmd | tail -1);
    echo "JOB ID = ${job_id}; Job Name = ${name}";
    new_sbatch_file=${sbatch_file / ${name} / ${job_id}_${name}};
    mv ${sbatch_file} ${new_sbatch_file};
    fi
fi

nifti_init = ''
subj_name = ''
outpath = ''

nii_path=$1;
subj=$2
dti=$3
outpath=$4

coreg_nii_old = f'{outpath}/co_reg_{D_subj}_m00-results/Reg_{D_subj}_nii_path{ext}';
coreg_nii = os.path.join(work_dir, f'Reg_{D_subj}_nii_path{ext}')
if not cleanup:
    coreg_nii = coreg_nii_old
if not os.path.exists(coreg_nii) or overwrite:
    if not os.path.exists(coreg_nii_old) or overwrite:
        temp_cmd = os.path.join(gunniespath,
                                'co_reg_4d_stack_tmpnew.bash') + f' {denoised_nii} {D_subj} 0 {outpath} 0';
        os.system(temp_cmd)
    if cleanup:
        shutil.move(coreg_nii_old, coreg_nii)