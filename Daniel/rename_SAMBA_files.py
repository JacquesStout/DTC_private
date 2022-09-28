import os
import glob
import pandas as pd
import numpy as np

RARE_path = '/Volumes/Data/Badea/Lab/jacques/RARE_folder'

BRUKER_fmri = '/Users/jas/jacques/Daniel_test/FMRI_mastersheet.xlsx'

allsubj = glob.glob(os.path.join(RARE_path,'sub*'))

database = pd.read_excel(BRUKER_fmri)

#APOE2_HN = list(database[database['Genotype'] == 'E2HN']['D_name'])
#APOE2 = list(database[database['Genotype'] == 'E22']['D_name'])
#APOE2_all = APOE2_HN + APOE2
#print(APOE2_all)

for subjpath in allsubj:
    subj= subjpath.split('/')[-1]
    subj = subj.replace('sub','')
    subj = subj.split('_')[0]
    val = str(np.array(database[database['D_name']==int(subj)]['RARE_name'])[0])
    #subj = subj.replace('sub-22','22')
    #print(subj)
    if val != subj:
        subjpath_new = subjpath.replace(subj, val)
        print(subjpath, subjpath_new)
        os.rename(subjpath, subjpath_new)