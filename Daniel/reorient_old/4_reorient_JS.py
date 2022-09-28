import time, glob, os
from DTC.file_manager.file_tools import mkcdir
from DTC.nifti_handlers.transform_handler import img_transform_exec
import nibabel as nib
import copy

start=time.time()

# paths to input and output directories
overwrite=True
lab_data = '/Users/jas/jacques/Daniel_test/BRUKER_organized_JS/'
lab_data_reoriented='/Users/jas/jacques/Daniel_test/BRUKER_organized_JS_reoriented'



# desired orientations
new_anat_orientation="RPS"
new_func_orientation="RPS"

# create output directory
mkcdir(lab_data_reoriented)

folders = glob.glob(os.path.join(lab_data,'*/'))

# iterate through subjects
for folder in folders:
    subject = os.path.basename(os.path.dirname(folder))
    print("starting processing for subject {subject}")
    mkcdir(os.path.join(lab_data_reoriented,f'{subject}'))
    mkcdir(os.path.join(lab_data_reoriented,f'{subject}','ses_1'))
    anat_folder = os.path.join(lab_data_reoriented,f'{subject}','ses_1','anat')
    func_folder = os.path.join(lab_data_reoriented,f'{subject}','ses_1','func')
    mkcdir(anat_folder)
    mkcdir(func_folder)
    print(f"processing anat for subject {subject}")
    # reorient anat image
    files=glob.glob(os.path.join(folder,'ses_1','anat','*'))
    for file in files:
        # image_func_exect()
        newfile = os.path.join(anat_folder,os.path.basename(file))
        if not os.path.exists(newfile) or overwrite:
            #img_transform_exec(file, 'LPS', new_anat_orientation, output_path=newfile, recenter_test=True)
            oldnifti = nib.load(file)
            newnifti = copy.deepcopy(oldnifti)
            new_affine = newnifti.affine
            new_affine[0,:] = newnifti.affine[0,:] * [-1,-1,-1,-1]
            nib.save(newnifti,newfile)
            print(f'Saved {newfile}')
        #c3d ${files[0]} -orient ${new_anat_orientation} -o ${lab_data_reoriented}/${subject}/ses-1/anat/${subject}_ses-1_T1w.nii.gz

    print(f"processing func for subject {subject}")
    # reorient func images
    counter_1=1
    files = glob.glob(os.path.join(folder, 'ses_1', 'func', '*'))
    for file in files:
        newfile = os.path.join(func_folder,os.path.basename(file))
        if not os.path.exists(newfile) or overwrite:
            #img_transform_exec(file, 'LPI', new_func_orientation, output_path=newfile, recenter_test=True)
            oldnifti = nib.load(file)
            newnifti = copy.deepcopy(oldnifti)
            new_affine = newnifti.affine
            new_affine[0,:] = newnifti.affine[0,:] * [-1,-1,-1,-1]
            #new_affine[2,:] = newnifti.affine[2,:] * [-1,-1,-1,-1]
            nib.save(newnifti,newfile)
            print(f'Saved {newfile}')
        # image_func_exect()
        #c3d ${files[0]} -orient ${new_func_orientation} -o ${lab_data_reoriented}/${subject}/ses-1/anat/${subject}_ses-1_T1w.nii.gz
        """
        disassembly_outputs=${lab_data_reoriented}/${subject}/ses-1/func/disassembly_outputs
        reorient_outputs=${lab_data_reoriented}/${subject}/ses-1/func/reorient_outputs
        
        mkdir ${disassembly_outputs}
        mkdir ${reorient_outputs}
        
        echo "disassembling func ${counter_1} for subject ${subject}"
        ImageMath 4 ${disassembly_outputs}/split.nii.gz TimeSeriesDisassemble ${func}
        
        counter_2=0
        
        echo "reorienting components of func ${counter_1} for subject ${subject}"
        for file in ${disassembly_outputs}/*; do
            printf -v j "%04d" $counter_2
            c3d ${file} -orient ${new_func_orientation} -o ${reorient_outputs}/reorient_${j}.nii.gz
            let "counter_2++"
        done
        
        time_spacing=$(fslval ${func} pixdim4)
        time_origin=0
        
        echo "time spacing = ${time_spacing}"
        echo "time origin = ${time_origin}"
        echo "reassembling func ${counter_1} for subject ${subject}"
        ImageMath 4 ${lab_data_reoriented}/${subject}/ses-1/func/${subject}_ses-1_run-${counter_1}_bold.nii.gz TimeSeriesAssemble ${time_spacing} ${time_origin} ${reorient_outputs}/*
        let "counter_1++"
        
        rm -r ${disassembly_outputs}
        rm -r ${reorient_outputs}
        """

duration = (time.time() - start)
print(duration)