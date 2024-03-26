import os, glob
import pandas as pd
import numpy as np

xlsx_path = '/Users/jas/Downloads/MasterSheet_ExperimentsfMRI.xlsx'
xlsx_path_r = '/Users/jas/Downloads/MasterSheet_ExperimentsfMRI_JSrevised.xlsx'

allowed_methods = ['cs_DtiStandard','DtiEpi','nmrsuDtiStandardAlex']

#dti_seq = ['cs_DtiStandard','nmrsuDtiStandardAlex']
func_seq = ['T2S_EPI']
diff_seq = ['DtiEpi','reversed_DtiEpi']
CEST_seq = ['mircen_CEST_RARE']
allowed_methods = ['cs_DtiStandard','nmrsuDtiStandardAlex','T2S_EPI','DtiEpi','reversed_DtiEpi']

fid_size_lim_mb = 0
dir_min_lim = 1

verbose = False

read_all_methods = True

#info_type => 'Show_niipath' or 'Show_methodinfo'
info_type = 'Show_methodinfo'

main_df = pd.read_excel(xlsx_path)
main_df['CEST'] = ''
cols_int = ['NameGroup','Weight','Mn2Cl pump','Age_Imaging']
main_df[cols_int] = main_df[cols_int].fillna(0)
main_df[cols_int] = main_df[cols_int].astype(int)

main_df['DOB'] = main_df['DOB'].dt.strftime('%m/%d/%y')
main_df['Perfusion '] = main_df['Perfusion '].dt.strftime('%m/%d/%y')


data_path_folder = '/Volumes/dusom_mousebrains/All_Staff/Projects/Bruker_Repo/Data/Raw_Bruker_Data/18abb11/'

data_paths = [folder_path.split('/')[-2] for folder_path in glob.glob(os.path.join(data_path_folder,'20*/'))]
data_paths = [subj for subj in data_paths if 'Dummy' not in subj and 'phantom' not in subj]

subjects = main_df.BadeaID

"""
approved_list = ['20220510_093518_220307_1_apoeb_18abb11_apoe_1_1','20220518_100450_220307_10_apoe_B_18abb11_apoe_1_1',
                 '20220518_123306_220307_12_apoe_18abb11_apoe_1_1','20220810_165730_220606_3_apoe_18abb11_apoe3_1_3',
                 '20220907_090946_220704_16_2_apoe_channel3_18abb11_apoe_1_1', '20221101_145118_220905_10_18abb11_1_1',
                 '20221101_163819_220905_11_18abb11_apoe_1_2','20230808_084919_220905_21_apoe_18bb11_1_1','20221117_092316_221003_5_apoeb_18abb11_apoe_1_1',
                 '20221117_092316_221003_5_apoeb_18abb11_apoe_1_1','20230802_084535_221003_20_apoe_b_18abb11_1_1','20230503_103756_221128_4_apoe_c_18abb11_1_1',
                 '20230222_125027_230117_7_2_APOE_18abb11_1_1','20240221_170558_230213_4_b_apoe_18abb11_1_1','20230321_105740_230213_12_APOE_18abb11_1_1',
                 '20230321_173357_230213_18_apoe_18abb11_1_1','20240213_114431_230313_2_apoe_b_18abb11_1_1','20230419_085813_230313_15_apoe_18abb11_apoe_1_1',
                 '20230516_121314_230410_3_apoe_18abb11_apoe_b_1_2','20230517_104009_230410_8_apoe2_18abb11_apoe_1_1','20230523_105804_230410_11_apoe_18abb11_apoe_1_1',
                 '20230613_090025_230508_6_apoe_8abb11_apoe_1_1','20230621_124300_230508_14_apoe_18abb11_1_1','20230726_111342_230605_12_apoe_230605_12_18abb11b_1_2',
                 '20220810_094434_220606_10_apoe_18abb11_apoe2_1_2']


renamed_subjs = {'220307_13':'220307_12_apoe_B','220404_1':'220401_1','220606_5':'28_220606_5real','220606_6':
                '220606_5_','221101_1':'221101_01','221101_2':'221101_02','221101_4':'221101_04','230410_7':'230410_7apoe'}
notfound_subjs = ['220606_15','220606_16','220606_17','220606_19','220606_20','220606_21','220606_22','220704_5',
                  '220704_10','220704_11','220704_18','220704_19','220808_18','220808_19',
                  '220808_20','220808_21','220808_22','221003_24','221101_3','230313_9','230410_18','230605_14']

double_subjs = {'230605_15':'20240215_093555_230605_15_apoe_b_18abb11_1_1','230605_16':'20240215_113610_230605_16_apoe_b_18abb11_1_1',
                '230605_17':'20240227_142731_230605_17_APOE_18abb11_1_1','230605_18':'20240227_155923_230605_18_APOE_18abb11_1_1',
                '230605_19':'20240228_091332_230605_19_apoe_b_18abb11_1_1','230605_20':'20240220_090512_230605_20_apoe_exo_18abb11_1_1',
                '230605_21':'20240220_104336_230605_21_apoe_exo_18abb11_1_1','230605_22':'20240220_122028_230605_22_apoe_exo_18abb11_1_1',
                '230605_23':'20240220_135308_230605_23_apoe_exo_18abb11_1_1','230605_24':'20240220_162907_230605_24_apoe_exo_18abb11_1_1',
                '230605_25':'20240221_085207_230605_25_apoe_exo_18abb11_1_1'}
"""
#approved_list = ['20220518_123306_220307_12_apoe_18abb11_apoe_1_1','20220810_165730_220606_3_apoe_18abb11_apoe3_1_3']
approved_list = ['20220518_123306_220307_12_apoe_18abb11_apoe_1_1','20221101_145118_220905_10_18abb11_1_1',
                 '20230808_084919_220905_21_apoe_18bb11_1_1','20240221_170558_230213_4_b_apoe_18abb11_1_1',
                 '20240213_114431_230313_2_apoe_b_18abb11_1_1']
renamed_subjs = {}
double_subjs = {'230605_15':'20240215_093555_230605_15_apoe_b_18abb11_1_1','230605_16':'20240215_113610_230605_16_apoe_b_18abb11_1_1',
               '230605_17':'20240227_142731_230605_17_APOE_18abb11_1_1','230605_18':'20240227_155923_230605_18_APOE_18abb11_1_1',
                '230605_19':'20240228_091332_230605_19_apoe_b_18abb11_1_1','230605_20':'20240220_090512_230605_20_apoe_exo_18abb11_1_1',
                '230605_21':'20240220_104336_230605_21_apoe_exo_18abb11_1_1', '230605_22':'20240220_122028_230605_22_apoe_exo_18abb11_1_1',
                '230605_23':'20240220_135308_230605_23_apoe_exo_18abb11_1_1','230605_24':'20240220_162907_230605_24_apoe_exo_18abb11_1_1',
                '230605_25':'20240221_085207_230605_25_apoe_exo_18abb11_1_1'
                }
notfound_subjs = ['230810_1','220704_10','221003_24','221101_3','230313_9','230810_3','230810_4','240212_1','240212_2',
                  '240212_3','240212_4','240212_5','240212_6','240212_7','240212_8','240212_9','240212_10','240212_11',
                  '240212_12']

for subject in subjects:

    if subject is np.nan:
        continue

    subject_id = subject.replace('-','_')
    found =0

    if list(main_df.loc[main_df['BadeaID'] == subject, 'Bruker_folder'])[0] is not np.nan:

        substr = list(main_df.loc[main_df['BadeaID'] == subject, 'Bruker_folder'])[0]

        if len(substr)<10:
            data_path_indices = [index for index, string in enumerate(data_paths) if substr+'_' in string]
        else:
            data_path_indices = [index for index, string in enumerate(data_paths) if substr in string]

    elif subject_id in renamed_subjs.keys():
        data_path_indices = [index for index, string in enumerate(data_paths) if renamed_subjs[subject_id] in string]
    else:
        data_path_indices = [index for index, string in enumerate(data_paths) if subject_id + '_' in string]

    if subject_id in notfound_subjs:
        continue

    if subject_id in double_subjs.keys():
        subject_path = os.path.join(data_path_folder, double_subjs[subject_id])
    elif len(data_path_indices)<1:
        print(f'Could not find subject {subject_id}')
        if not found:
            main_df.loc[main_df['BadeaID'] == subject, 'fMRI'] = 0
            main_df.loc[main_df['BadeaID'] == subject, 'invivoDWI'] = 0
            main_df.loc[main_df['BadeaID'] == subject, 'CEST'] = 0
            main_df.loc[main_df['BadeaID'] == subject, 'MRI_Scan_date'] = np.nan
            continue

    elif len(data_path_indices)>1:
        data_allfound = [data_paths[data_path_indice] for data_path_indice in data_path_indices]
        found=0
        for data_found in data_allfound:
            if data_found in approved_list:
                found=1
                subject_path = os.path.join(data_path_folder, data_found)
                break
        if found==0:
            print(f'Too many instances for subject {subject_id}, investigate')
            if not found:
                main_df.loc[main_df['BadeaID'] == subject, 'fMRI'] = 0
                main_df.loc[main_df['BadeaID'] == subject, 'invivoDWI'] = 0
                main_df.loc[main_df['BadeaID'] == subject, 'CEST'] = 0
                main_df.loc[main_df['BadeaID'] == subject, 'MRI_Scan_date'] = np.nan
                continue
    else:
        subject_path = os.path.join(data_path_folder, data_paths[data_path_indices[0]])

    diff =0
    CEST = 0
    func =0

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

        allowed_methods = ['cs_DtiStandard', 'nmrsuDtiStandardAlex', 'T2S_EPI', 'DtiEpi', 'reversed_DtiEpi']

        if 'method_name' not in locals():
            print('hi')
        if method_name == diff_seq[0] and sizefid_mb>300:
            diff += 0.5
        if method_name == diff_seq[1] and sizefid_mb>20:
            diff+=3
        if method_name in func_seq:
            func = 1
        if method_name in CEST_seq:
            CEST+=0.5
        #if method_name==CEST_seq[1]:
        #    CEST+=3

    if diff == 3.5:
        diff=1
    elif diff ==0:
        diff=0
    elif diff>3.5:
        diff=1
        print(f'Too many diffusion acquisitions for {subject_id}, investigate')

    if int(CEST)==1:
        CEST=int(CEST)
    elif CEST==0:
        CEST=0
    elif int(CEST)>1:
        CEST=1
        print(f'Too many CEST for subject {subject_id}, investigate')

    try:
        if list(main_df.loc[main_df['BadeaID'] == subject, 'fMRI'])[0] != func:
            print(f'Problem with func detection for subject {subject_id}, investigate')
            main_df.loc[main_df['BadeaID'] == subject, 'fMRI'] = func
    except IndexError:
        print('hi')

    main_df.loc[main_df['BadeaID'] == subject, 'invivoDWI'] = diff

    #else:
    #    main_df.loc[main_df['BadeaID'] == subject, 'fMRI'] = func
    main_df.loc[main_df['BadeaID'] == subject, 'CEST'] = CEST
    main_df.loc[main_df['BadeaID'] == subject, 'MRI_Scan_date']  = os.path.basename(subject_path).split('_')[0]

main_df.to_excel(xlsx_path_r,na_rep='', index=False)