import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

connectome_path = '/Volumes/Data/Badea/Lab/mouse/Jasien_mrtrix_pipeline/connectomes/'
connectome_output = '/Users/jas/jacques/Jasien/connectome_figures'

color = 'nipy_spectral'

subjects = ['J04086', 'J04129', 'J04300', 'J01257', 'J01277', 'J04472', 'J01402']
Lesion = ['Lumbar','Lumbar','Lumbar','Lumbar','Lumbar','Sacral','Sacral']
Genotype = ['APOE33','APOE33','APOE33','APOE33','APOE34','APOE33','APOE33']
Ambulatory = ['W','W','A','W','W','A','A']

full_title = True

for i,subject in enumerate(subjects):
    connectome_subject_path = os.path.join(connectome_path,subject,f'{subject}_distances.csv')
    connectome_figure_path = os.path.join(connectome_output,f'{subject}_distances.png')

    df = pd.read_csv(connectome_subject_path, index_col=0)
    connectivity_matrix = df.to_numpy()

    plt.figure(figsize=(8, 6))
    try:
        plt.imshow(connectivity_matrix, cmap=color, interpolation='nearest')
    except:
        print(f'Color {color} is invalid')
    #plt.colorbar(label='Connectivity Strength')
    if full_title:
        plt.title(f'Connectivity Subject {subject}, {Lesion[i]}, {Ambulatory[i]}, {Genotype[i]}')
    else:
        plt.title(f'Connectivity Subject {subject}')
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