import os, glob
import matlab.engine
from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile, glob_remote

#folders = glob.glob('/Users/jas/jacques/CS_Project/CS_Data_all/Bruker_data/20230419_165630_211001_21_1_DEV_18abb11_DEV_1_1/12*')

data_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_data/'
results_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_results/'

matlab = matlab.engine.start_matlab()

matlab.addpath('/Users/jas/bass/gitfolder/Compressed_sensing_recon')
matlab.addpath('/Users/jas/bass/gitfolder/Compressed_sensing_recon/NIFTI_20140122_v2/')

make_recon = True
run_recon = False

#subjects = ['20230830_142204_230605_20_apoe_18abb11_1_1']
subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'*/'))]

#subjects = ['20221117_163120_Dummy_17November2022_v12_18abb11_DEV_1_1']

for subject in subjects:
    subject_path = os.path.join(data_path, subject)

    scans_paths = glob.glob(os.path.join(subject_path,'*'))
    scans_paths.sort()
    for scan_path in scans_paths:

        if not os.path.basename(scan_path).isdigit():
            continue

        scanno = int(os.path.basename(scan_path))

        methodpath = os.path.join(scan_path, 'method')
        fidpath = os.path.join(scan_path,'fid')

        spat_resol = False
        size_matrix = False
        bval_get = False
        dir_shape = ''
        method_name = ''
        TR = ''
        TE = ''
        CS_val = ''
        if os.path.exists(methodpath):
            with open(methodpath, 'r') as file:
                for line in file:
                    if line.startswith('##$Method='):
                        method_name = line.split(':')[1].split('>')[0].strip()


                    if spat_resol:
                        spat_resol_vals = line.split('\n')[0].split(' ')
                        spat_resol = False
                    if line.startswith('##$PVM_SpatResol='):
                        spat_resol = True

                    if size_matrix:
                        size_matrix_vals = line.strip()
                        size_matrix = False
                    if line.startswith('##$PVM_Matrix'):
                        size_matrix = True

                    if bval_get:
                        bval = line.strip()
                        bval_get = False
                    if line.startswith('##$PVM_DwBvalEach'):
                        bval_get = True

                    if line.startswith('##$PVM_ScanTime='):
                        try:
                            scan_time = int(float(line.split('=')[1].strip()))/3600000
                        except:
                            print('hi')

                    if line.startswith('##$PVM_DwNDiffExp='):
                        dir_shape = line.split('=')[1].strip()
                    if line.startswith('##$PVM_RepetitionTime='):
                        TR = line.split('=')[1].strip()
                    if line.startswith('##$PVM_EchoTime='):
                        TE = line.split('=')[1].strip()
                    if line.startswith('##$CS_Matrix'):
                        CS_val = line.split('=')[1].strip()

        if method_name == 'cs_DtiStandard':
            if not os.path.exists(fidpath):
                continue
            else:
                sizefid_mb = os.path.getsize(fidpath)/(1024*1024)
                if sizefid_mb<20:
                    continue

            result_subj_path = os.path.join(results_path,subject)
            result_scanno_path = os.path.join(results_path,subject,str(scanno))
            mkcdir([result_subj_path, result_scanno_path])
            reconned_nii_path = os.path.join(result_scanno_path,f'{scanno}_CS_DWI_bart_recon.nii.gz')
            if not os.path.exists(reconned_nii_path) and run_recon:
                matlab.Bruker_DWI_CS_recon_JSchanges(subject_path, scanno, result_scanno_path, 0, 0, nargout=0)

            if int(dir_shape)>40 and int(sizefid_mb)>1000:
                print(f'For {scan_path},\nThe method is {method_name}\nThe spat_resol is {spat_resol_vals}\nThe size of the fid is {sizefid_mb}MB\nTR/TE is {TR}/{TE}\nThe number of volumes is {dir_shape}\nThe size of the matrix is {size_matrix_vals}\nThe CS undersampling value is {CS_val}\nThe bvalue is {bval}\nThe scan time is {scan_time} h\nThe scan time per volume is {int(scan_time)/int(dir_shape)} h\n\n\n')

        #print(f'For {scan_path},\n the method is {method_name},\n the spat_resol is {spat_resol_vals}')
