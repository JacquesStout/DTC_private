import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from DTC.diff_handlers.connectome_handlers.excel_management import get_group

connectome_path = '/Volumes/Data/Badea/Lab/mouse/Jasien_mrtrix_pipeline/connectomes/'
connectome_func_path = os.path.join(connectome_path,'functional_conn')
connectome_output = '/Users/jas/jacques/Jasien/connectome_figures'

excel_path = '/Users/jas/jacques/Jasien/Jasien_list.xlsx'

color = 'nipy_spectral'

#subjects = ['J04086', 'J04129', 'J04300', 'J01257', 'J01277', 'J04472', 'J01402','J01501','J01516','J04602','J01541']
#Lesion = ['Lumbar','Lumbar','Lumbar','Lumbar','Lumbar','Sacral','Sacral','LLumbar','Lumbar','Lumbar','Sacral']
#Genotype = ['APOE33','APOE33','APOE33','APOE33','APOE34','APOE33','APOE33','APOE33','APOE34','APOE24','APOE34']
#Ambulatory = ['W','W','A','W','W','A','A','A','W','W','FA']

#subjects = ['J01501','J01516','J04602','J01541']
#Lesion = ['LLumbar','Lumbar','Lumbar','Sacral']
#Genotype = ['APOE33','APOE34','APOE24','APOE34']
#Ambulatory = ['A','W','W','A']

subjects = ['J04086', 'J04129', 'J04300', 'J01257', 'J01277', 'J04472', 'J01402','J01501','J01516','J04602','J01541']
subj_val_base = {}
subj_column = 'RUNNO'
group_columns = ['Lesion','Genotype','Ambulatory']
#group_columns = ['Lesion']

group_vals = pd.read_excel(excel_path)
group_vals = group_vals[[subj_column]+group_columns]

make_group_connectomes = True
group_simplify = True


if group_simplify:

    group_vals = group_vals.replace(
            {'APOE24': 'APOE4', 'APOE34': 'APOE4', 'APOE33': 'APOE3'}, regex=True)

    group_vals = group_vals.replace({'FA': 'A'}, regex=True)
    group_vals = group_vals.replace({'LLumbar': 'Lumbar'}, regex=True)

    group_types = {'APOE4':'Genotype','APOE3':'Genotype','A':'Ambulatory','W':'Ambulatory','Lumbar':'Lesion','Sacral':'Lesion'}


connectome_group = {}
num_group = {}

#[struct,func_ts,func_FC] ###func_ts doesn't work yet, due to it being over 600 time points
con_type = 'func_FC'

make_group_connectomes = False

full_title = True

for i,subject in enumerate(subjects):

    if con_type == 'struct':
        connectome_subject_path = os.path.join(connectome_path,subject,f'{subject}_distances.csv')
        connectome_figure_path = os.path.join(connectome_output,f'{subject}_distances.png')
    elif con_type =='func_ts':
        connectome_subject_path = os.path.join(connectome_func_path, f'time_serts_{subject}.csv')
        connectome_figure_path = os.path.join(connectome_output,f'{subject}_serts.png')
    elif con_type =='func_FC':
        connectome_subject_path = os.path.join(connectome_func_path, f'time_serFC_{subject}.csv')
        connectome_figure_path = os.path.join(connectome_output, f'{subject}_serFC.png')

    if not os.path.exists(connectome_subject_path):
        print(f'Could not find {os.path.basename(connectome_subject_path)}, skipping {subject}')
        continue

    df = pd.read_csv(connectome_subject_path, index_col=0)
    connectivity_matrix = df.to_numpy()

    plt.figure(figsize=(8, 6))

    try:
        plt.imshow(connectivity_matrix, cmap=color, interpolation='nearest')
    except:
        print(f'Color {color} is invalid')
    #plt.colorbar(label='Connectivity Strength')
    if full_title:
        txt = f'Connectivity Subject {subject}'
        for group in group_columns:
            txt+=f', {group_vals[group_vals[subj_column]==subject][group]}'
        plt.title(txt)
    else:
        plt.title(f'Connectivity Subject {subject}')
    #plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45)
    #plt.yticks(np.arange(len(df.index)), df.index)
    plt.tight_layout()

    # Display the figure
    plt.savefig(connectome_figure_path)
    plt.close()

    if make_group_connectomes:
        for group in group_columns:
            group_col = group_vals[group_vals[subj_column]==subject][group].iloc[0]
            if not group_col in connectome_group.keys():
                connectome_group[group_col] = connectivity_matrix
                num_group[group_col] = 1
            else:
                connectome_group[group_col]+=connectivity_matrix
                num_group[group_col] += 1

if make_group_connectomes:
    for key in connectome_group.keys():

        group_type = group_types[key]

        connectome_figure_path = os.path.join(connectome_output,f'{group_type}_{key}_average.png')

        connectome_group_f = connectome_group[key]/num_group[key]

        plt.figure(figsize=(8, 6))

        try:
            plt.imshow(connectome_group_f, cmap=color, interpolation='nearest')
        except:
            print(f'Color {color} is invalid')
        #plt.colorbar(label='Connectivity Strength')

        txt = f'Average Connectivity, {group_type} {key}'
        plt.title(txt)

        #plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45)
        #plt.yticks(np.arange(len(df.index)), df.index)
        plt.tight_layout()

        # Display the figure
        plt.savefig(connectome_figure_path)
        plt.close()

"""
colors = ['autumn', 'bone','copper', 'flag','gray','hot','hsv','jet','pink','prism','spring','summer','winter','magma','inferno','plasma','viridis','nipy_spectral']

for color in colors:
    # Create a figure and plot the connectivity matrix
    plt.figure(figsize=(8, 6))
    try:
        plt.imshow(connectivity_matrix, cmap=color, interpolation='nearest')
    except:
        print(f'Color {color} is invalid')
    plt.colorbar(label='Connectivity Strength')
    plt.title('Connectivity Matrix')
    #plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45)
    #plt.yticks(np.arange(len(df.index)), df.index)
    plt.tight_layout()

    # Display the figure
    plt.savefig(os.path.join(connectome_folder,f'{color}_connectome.png'))
"""