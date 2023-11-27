import os, glob
#
from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile, glob_remote
import subprocess

#folders = glob.glob('/Users/jas/jacques/CS_Project/CS_Data_all/Bruker_data/20230419_165630_211001_21_1_DEV_18abb11_DEV_1_1/12*')

data_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_data/'
results_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_results/'
niftis_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/CS_Data_all/Bruker_niftis/'

run_recon = False

if run_recon:
    import matlab.engine
    matlab = matlab.engine.start_matlab()

    matlab.addpath('/Users/jas/bass/gitfolder/Compressed_sensing_recon')
    matlab.addpath('/Users/jas/bass/gitfolder/Compressed_sensing_recon/NIFTI_20140122_v2/')


allowed_methods = ['cs_DtiStandard','DtiEpi','nmrsuDtiStandardAlex']
allowed_methods = ['cs_DtiStandard','nmrsuDtiStandardAlex']
#allowed_methods = ['cs_DtiStandard']
#allowed_methods = ['cs_DtiStandard','pCASL_FcFLASHv2']
#subjects = ['20230830_142204_230605_20_apoe_18abb11_1_1']
subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'*/'))]
subjects = ['20230621_124300_230508_14_apoe_18abb11_1_1', '20230621_142852_230508_14_CStesting_18abb11_DEV_1_1', '20230621_143413_230508_14_CStesting_2_18abb11_DEV_1_1']
subjects = ['20231011_091614_230925_6_18abb11_1_1','','20231011_155125_230925_11_apoe_18abb11_apoe_1_1']
subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'20231011*/'))]
subjects = ['20221117_163120_Dummy_17November2022_v12_18abb11_DEV_1_1']
subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'20231024*/'))]
subjects = ['20231024_163510_221101_22_apoe_18abb11_1_1']
subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'*/'))]
subjects = ['20231011_155125_230925_11_apoe_18abb11_apoe_1_1']
subjects = ['20231107_101335_210222_17_exvivotestCS_v2_18abb11_DEV_1_1']
subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'20221115*/'))]
#subjects = ['20231004_105847_230918_11_apoe_18abb11_APOE_1_1']
#subjects = ['20231024_175640_210222_17_exvivotestCS_apoe_18abb11_1_1']
#subjects = ['20231101_111601_221128_14_apoe_18abb11_1_1']
#subjects = ['20231024_143134_221128_9_apoe_rev_phase_18abb11_1_1']

#subjects = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path,'202211*/'))]
#subjects = ['20231017_140603_230925_16_apoe_18abb11_1_1']
#subjects = ['20221117_163120_Dummy_17November2022_v12_18abb11_DEV_1_1']


fid_size_lim_mb = 0
dir_min_lim = 1

verbose = False

read_all_methods = True

#info_type => 'Show_niipath' or 'Show_methodinfo'
info_type = 'Show_methodinfo'

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
        dir_shape = '1'
        method_name = ''
        TR = ''
        TE = ''
        CS_val = '1'
        bval = 'NA'
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
                            scan_time_seconds = int(float(line.split('=')[1].strip()))/1000
                            scan_time_hours = scan_time_seconds/3600
                        except:
                            print('hi')

                    if line.startswith('##$PVM_DwNDiffExp='):
                        dir_shape = line.split('=')[1].strip()
                    if line.startswith('##$PVM_RepetitionTime='):
                        TR = line.split('=')[1].strip()
                    if line.startswith('##$PVM_EchoTime='):
                        TE = line.split('=')[1].strip()
                    if line.startswith('##$User_CS_factor'):
                        CS_val = line.split('=')[1].strip()

        if not os.path.exists(fidpath):
            continue
        else:
            sizefid_mb = os.path.getsize(fidpath)/(1024*1024)
            if sizefid_mb<fid_size_lim_mb:
                continue

        if method_name == 'cs_DtiStandard' or method_name == 'nmrsuDtiStandardAlex':
            result_subj_path = os.path.join(results_path,subject)
            result_scanno_path = os.path.join(results_path,subject,str(scanno))
            mkcdir([result_subj_path, result_scanno_path])
            reconned_nii_path = os.path.join(result_scanno_path,f'{scanno}_CS_DWI_bart_recon.nii.gz')
            scanno_nii_path = reconned_nii_path
            if run_recon and not os.path.exists(reconned_nii_path) and int(dir_shape)>=dir_min_lim and int(sizefid_mb)>=fid_size_lim_mb:
                mat_command = (f"{subject_path}, {scanno}, {result_scanno_path}, 0, 0, nargout=0")
                matlab.Bruker_DWI_CS_recon_JSchanges(subject_path, scanno, result_scanno_path, 0, 0, nargout=0)

        else:
            cmd = f'/Users/jas/bass/gitfolder/nanconvert/Scripts/nanbruker -z -v -l -o {niftis_path} {subject_path}/'
            if not os.path.isdir(os.path.join(niftis_path, subject)):
                print(f'No results detected for {subject}; attempting to generate Niftis with nanconvert:')
                print('Command:')
                print(cmd)
                subprocess.call(cmd, shell=True)
            elif verbose:
                print(f'Result folder for {subject} already detected; skipping.')

            try:
                scanno_nii_path = glob.glob(os.path.join(niftis_path,subject,f'{scanno}_*'))[0]
            except IndexError:
                scanno_nii_path = None

        if method_name in allowed_methods:

            if int(dir_shape)>=dir_min_lim and int(sizefid_mb)>=fid_size_lim_mb:
                #scan_time_seconds = scan_time*
                scan_t_sec_vol = (scan_time_seconds)/int(dir_shape)
                if info_type == 'Show_methodinfo':
                    print(f'For {scan_path},\nThe method is {method_name}\nThe spat_resol is {spat_resol_vals}\nThe size of the fid is {sizefid_mb}MB\nTR/TE is {TR}/{TE}\nThe number of volumes is {dir_shape}\nThe size of the matrix is {size_matrix_vals}\nThe CS undersampling value is {CS_val}\nThe bvalue is {bval}\n'
                          f'The scan time is {int(scan_time_seconds//3600)}h{int((scan_time_seconds - 3600*(scan_time_seconds//3600))//60)}min{int((scan_time_seconds - 3600*(scan_time_seconds//3600))%60)}s \n'
                          f'The scan time per volume is {int(scan_t_sec_vol//3600)}h{int((scan_t_sec_vol - 3600*(scan_t_sec_vol//3600))//60)}min{int((scan_t_sec_vol - 3600*(scan_t_sec_vol//3600))%60)}s\n\n\n')
                #if make_excel:
                #    result_subj_path = os.path.join(results_path, subject)
                #    df
                elif info_type == 'Show_niipath':
                    print(f'{scanno_nii_path}')
        elif read_all_methods:
            if info_type == 'Show_methodinfo':
                print(f'For {scan_path},\nThe method is {method_name}')
            else:
                print(f'{scanno_nii_path}')

        #print(f'For {scan_path},\n the method is {method_name},\n the spat_resol is {spat_resol_vals}')
