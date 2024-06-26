import os, glob

"""
#folders = glob.glob('/Users/jas/jacques/CS_Project/CS_Data_all/Bruker_data/20230419_165630_211001_21_1_DEV_18abb11_DEV_1_1/12*')

data_path = '/Users/jas/jacques/CS_Project/CS_Data_all/Bruker_data/'
#data_path = '/Volumes/Data/Jasien/ADSB.01/Data/Anat/'
#data_path = '/Users/jas/jacques/AD_Decode/raw_temp/'
subjects = ['20230830_142204_230605_20_apoe_18abb11_1_1']
subjects = ['20230621_142852_230508_14_CStesting_18abb11_DEV_1_1']
subjects = ['20231004_102859_220905_10_b_apoe_18abb11_APOE_1_1']
#subjects = ['20221026_04086']
"""

data_path = '/Volumes/dusom_mousebrains/All_Staff/jacques/CS_Project/CS_data_all/Bruker_data'
subjects = ['20231004_105847_230918_11_apoe_18abb11_APOE_1_1']


for subject in subjects:
    subject_path = os.path.join(data_path, subject)

    scans_paths = glob.glob(os.path.join(subject_path,'*'))
    scans_paths.sort()
    for scan_path in scans_paths:

        if not os.path.basename(scan_path).isdigit():
            continue

        filepath = os.path.join(scan_path, 'method')

        spat_resol = False
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                for line in file:
                    if line.startswith('##$Method='):
                        method_name = line.split(':')[1].split('>')[0].strip()
                    if spat_resol:
                        spat_resol_vals = line.split('\n')[0].split(' ')
                        spat_resol = False
                    if line.startswith('##$PVM_SpatResol='):
                        spat_resol = True

            print(f'For {scan_path},\n the method is {method_name},\n the spat_resol is {spat_resol_vals}')
