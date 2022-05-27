import os, glob
import pandas as pd
from file_tools import file_rename


chavez_Ids_path = '/Volumes/Data/Badea/Lab/RaulChavezValdez/Chavez_ID.xlsx'
chavez_path = '/Volumes/Data/Badea/Lab/RaulChavezValdez/'

chavez_Ids = pd.read_excel(chavez_Ids_path)

orig_names = chavez_Ids.Original_Names
new_names = chavez_Ids.IDs

folder_paths = glob.glob(os.path.join(chavez_path,'*/'))

for orig_name, new_name in zip(orig_names,new_names):
    for folder_path in folder_paths:
        if folder_path.split('/')[-2] == new_name:
            file_rename(folder_path, orig_name, new_name, test=False)

