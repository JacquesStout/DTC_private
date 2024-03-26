import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from DTC.diff_handlers.connectome_handlers.excel_management import get_group
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
#from scipy.spatial.distance import mahalanobis, canberra
import scipy.spatial.distance
from scipy.stats import ks_2samp
import statsmodels.api as sm


def spearman_correlation_matrix(matrix1, matrix2):
    # Ensure matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")

    # Initialize an empty matrix for correlation values
    correlation_matrix = np.zeros_like(matrix1, dtype=float)

    # Calculate Spearman correlation for each pair of corresponding elements
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            correlation_matrix[i, j], _ = spearmanr(matrix1[i, j], matrix2[i, j])

    return correlation_matrix

connectome_path = '/Volumes/Data/Jasien/ADSB.01/Analysis/connectomes/tract_conn'
connectome_func_path = '/Volumes/Data/Jasien/ADSB.01/Analysis/connectomes/functional_conn/'
connectome_output = '/Users/jas/jacques/Jasien/connectome_figures'


excel_path = '/Users/jas/jacques/Jasien/Jasien_list.xlsx'

color = 'nipy_spectral'
#color = 'bwr'
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

make_group_connectomes = True
colorbars = True
full_title = True

connectome_group_list = {}
connectome_subj_list = {}

con_type = 'func_FC'

#dist_type = 'KS'
dist_type = 'KS'

for i,subject in enumerate(subjects):


    if con_type == 'distances':
        connectome_subject_path = os.path.join(connectome_path,f'{subject}_distances.csv')
        #connectome_figure_path = os.path.join(connectome_output,f'{subject}_distances.png')
    elif con_type == 'struct':
        connectome_subject_path = os.path.join(connectome_path,f'{subject}_conn_plain.csv')
        #connectome_figure_path = os.path.join(connectome_output,f'{subject}_conn_plain.png')
    elif con_type =='func_ts':
        connectome_subject_path = os.path.join(connectome_func_path, f'time_serts_{subject}.csv')
        #connectome_figure_path = os.path.join(connectome_output,f'{subject}_serts.png')
    elif con_type =='func_FC':
        connectome_subject_path = os.path.join(connectome_func_path, f'time_serFC_{subject}.csv')
        #connectome_figure_path = os.path.join(connectome_output, f'{subject}_serFC.png')
    elif con_type =='FA':
        connectome_subject_path = os.path.join(connectome_path, f'{subject}_mean_FA_connectome.csv')

    connectome_figure_path = os.path.join(connectome_output, f'{subject}_{con_type}.png')

    if not os.path.exists(connectome_subject_path):
        print(f'Could not find {os.path.basename(connectome_subject_path)}, skipping {subject}')
        continue

    df = pd.read_csv(connectome_subject_path,header = None)
    connectivity_matrix = df.to_numpy()
    connectivity_matrix[np.isnan(connectivity_matrix)] = 0

    plt.figure(figsize=(6, 6),dpi=1200)

    try:
        if color =='bwr':
            plt.imshow(connectivity_matrix, cmap=color, interpolation='nearest',vmin = -1 * np.max(connectivity_matrix))
        else:
            plt.imshow(connectivity_matrix, cmap=color, interpolation='nearest',vmax = np.percentile(connectivity_matrix,90))
    except:
        print(f'Color {color} is invalid')
    if colorbars:
        plt.colorbar(label='Connectivity Strength')
    if full_title:
        txt = f'Connectivity Subject {subject}'
        for group in group_columns:
            txt+=f', {group_vals[group_vals[subj_column]==subject][group].iloc[0]}'
            #print(txt)
        plt.title(txt)
    else:
        plt.title(f'Connectivity Subject {subject}')
    #plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45)
    #plt.yticks(np.arange(len(df.index)), df.index)
    plt.tight_layout()

    # Display the figure
    plt.savefig(connectome_figure_path)
    print(f'Saved at {connectome_figure_path}')
    plt.close()

    connectome_subj_list[subject] = connectivity_matrix

    if make_group_connectomes:
        for group in group_columns:
            group_col = group_vals[group_vals[subj_column]==subject][group].iloc[0]
            if not group_col in connectome_group.keys():
                connectome_group[group_col] = connectivity_matrix
                connectome_group_list[group_col] = [connectivity_matrix]
                num_group[group_col] = 1
            else:
                connectome_group[group_col] = connectome_group[group_col] + connectivity_matrix
                connectome_group_list[group_col].append(connectivity_matrix)
                num_group[group_col] = num_group[group_col] + 1

new_cols = list(group_vals.columns)
new_cols.remove('RUNNO')
new_cols = new_cols + ['ID1','ID2','KS','canberra']


new_df = pd.DataFrame(columns=new_cols)
counter = 1
for i in group_vals['RUNNO']:
    for j in group_vals['RUNNO']:
        if i == j:
            continue
        else:
            new_df.at[counter, 'ID1'] = i
            new_df.at[counter, 'ID2'] = j

            df_subj1 = group_vals[group_vals['RUNNO']==i]
            df_subj2 = group_vals[group_vals['RUNNO']==j]
            #new_df[counter,'Lesion'] = 1*(new_df[i,"Lesion"] != new_df[j,"Lesion"])

            new_df.at[counter, 'Lesion'] = 1*(df_subj1['Lesion'].values[0] != df_subj2['Lesion'].values[0])
            new_df.at[counter, 'Genotype'] = 1*(df_subj1['Genotype'].values[0] != df_subj2['Genotype'].values[0])
            new_df.at[counter, 'Ambulatory'] = 1*(df_subj1['Ambulatory'].values[0] != df_subj2['Ambulatory'].values[0])
            #new_df[counter,'Lesion'] = 1*(df_subj1['Lesion'].values[0] == df_subj2['Lesion'].values[0])
            #new_df[counter,'Genotype'] = 1*(df_subj1['Genotype'].values[0] == df_subj2['Genotype'].values[0])
            #new_df[counter,'Ambulatory'] = 1*(df_subj1['Ambulatory'].values[0] == df_subj2['Ambulatory'].values[0])

            new_df.at[counter, 'canberra'] = scipy.spatial.distance.canberra(connectome_subj_list[i].flatten(), connectome_subj_list[j].flatten())
            new_df.at[counter,'KS'] = np.round(ks_2samp(connectome_subj_list[i].flatten(),connectome_subj_list[j].flatten())[0],3)
            counter += 1


formula = f'{dist_type} ~ C(Genotype) + Lesion + Ambulatory'
#formula = 'KS ~ ' + ' + '.join(new_df.columns.difference(['KS']))

pd.to_numeric(new_df['KS'])
new_df['Genotype'] = pd.to_numeric(new_df['Genotype'])
new_df['Ambulatory'] = pd.to_numeric(new_df['Ambulatory'])
new_df['Lesion'] = pd.to_numeric(new_df['Lesion'])
new_df['KS'] = pd.to_numeric(new_df['KS'])
new_df['canberra'] = pd.to_numeric(new_df['canberra'])

model = sm.OLS.from_formula(formula, data=new_df)
result = model.fit()

diff_pval = result.pvalues[1]

round_up = True

#ks_results_WM = result.params

# df.loc[df['R'] == 'ID1', 'L'].values[0]
if make_group_connectomes:
    for key in connectome_group.keys():

        group_type = group_types[key]

        connectome_figure_path = os.path.join(connectome_output,f'{group_type}_{key}_{con_type}_average.png')

        connectome_group_f = connectome_group[key]/num_group[key]

        plt.figure(figsize=(8, 6),dpi=1200)

        try:
            if color =='bwr':
                plt.imshow(connectome_group_f, cmap=color, interpolation='nearest', vmin = -1 * np.max(connectome_group_f))
            else:
                plt.imshow(connectome_group_f, cmap=color, interpolation='nearest')
        except:
            print(f'Color {color} is invalid')
        if colorbars:
            plt.colorbar(label='Connectivity Strength')


        # Generate example data
        connectome_this_list = connectome_group_list[key]

        # Flatten each array within the observations
        #connectome_group_flat = [obs.flatten() for obs in connectome_this_list]

        connectome_group_flat = np.concatenate([arr.flatten().reshape(7056,1) for arr in connectome_this_list], axis=1)

        sparse_cutoff = np.median(connectome_group_f)/10


        # Calculate Pearson correlation between each pair of observations
        sparsity_sum = 0


        #correlation_coefficients = []
        dist_vals = []

        for i in range(len(connectome_this_list)):
            sparsity_sum += np.sum(connectome_group_f<sparse_cutoff)
            for j in range(i + 1, len(connectome_this_list)):
                #corr, _ = spearmanr(connectome_group_flat[i], connectome_group_flat[j])
                #dist = mahalanobis(connectome_group_flat[i].flatten(),connectome_group_flat[j].flatten())
                #dist = scipy.spatial.distance.canberra(connectome_this_list[0].flatten(),connectome_this_list[1].flatten())
                #correlation_coefficients.append(corr)
                if dist_type == 'canberra':
                    dist = scipy.spatial.distance.canberra(connectome_this_list[i].flatten(),
                                                           connectome_this_list[j].flatten())
                if dist_type == 'KS':
                    dist = (ks_2samp(connectome_this_list[i].flatten(),connectome_this_list[j].flatten()))
                dist_vals.append(dist)


        #average_correlation = np.round(np.mean(correlation_coefficients),3)
        if round_up:
            average_dist = np.round(np.mean(dist_vals),3)

        sparsity_sum /= len(connectome_group_list)
        sparsity_val = np.sum(connectome_group_f<sparse_cutoff)

        num_vals = np.shape(connectome_group_f)[0]*np.shape(connectome_group_f)[1]

        #title_txt = f'Average Connectivity, {group_type} {key}, avg corr = {average_correlation}'
        title_txt = f'Average Connectivity, {group_type} {key}, avg distance = {average_dist}'
        if sparsity_val>0 or sparsity_sum>0:
            title_txt+=f'\ngroup sparsity = {np.round((100*sparsity_val)/num_vals,1)}%, average sparsity = {np.round((100*sparsity_sum)/num_vals,1)}%'
        plt.title(title_txt)

        #plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45)
        #plt.yticks(np.arange(len(df.index)), df.index)
        plt.tight_layout()

        # Display the figure
        plt.savefig(connectome_figure_path)
        #plt.show()
        plt.close()

    #connectome_dif = (connectome_group['APOE3']/num_group['APOE3'] - connectome_group['APOE4']/num_group['APOE4'])/(connectome_group['APOE3']+connectome_group['APOE4'])
    connectome_dif = (connectome_group['APOE4'] / num_group['APOE4'] - connectome_group['APOE3'] / num_group['APOE3'])
    connectome_figure_path = os.path.join(connectome_output, f'APOE3_APOE4_diff_{con_type}.png')

    #model = sm.OLS.from_formula('KS ~ .' + '- EUC - JACCARD - Mink_14 - 1', data=new_df)
    #result = model.fit()

    # Get coefficients from the summary
    #ks_results_WM = result.params

    """
    #Spearman correlation
    #spearman_cor = spearman_correlation_matrix(connectome_group['APOE3'] / num_group['APOE3'], connectome_group['APOE4'] / num_group['APOE4'])

    connectomes_APOE3 = [arr.flatten() for arr in connectome_group_list['APOE3']]
    connectomes_APOE4 = [arr.flatten() for arr in connectome_group_list['APOE4']]

    # Calculate Spearman correlation


    correlations = []
    for arr1 in connectomes_APOE3:
        for arr2 in connectomes_APOE4:
            rho, p_value = np.corrcoef(arr1, arr2)
            if rho is not np.nan:
                correlations.append(rho)

    average_correlation, p_value = spearmanr(connectome_group['APOE3'].flatten()/num_group['APOE3'],connectome_group['APOE4'].flatten()/num_group['APOE4'])

    average_correlation = np.round(average_correlation,3)
    #    average_correlation, p_value = spearmanr(connectome_group_list['APOE3'][0].flatten(),connectome_group_list['APOE3'][1].flatten())
    """

    """
    group1_corr_matrices = [np.corrcoef(observation) for observation in connectome_group_list['APOE3']]
    group2_corr_matrices = [np.corrcoef(observation) for observation in connectome_group_list['APOE4']]

    # Step 2: Average correlation matrices
    avg_group1_corr_matrix = np.mean(group1_corr_matrices, axis=0)
    avg_group2_corr_matrix = np.mean(group2_corr_matrices, axis=0)

    # Step 3: Conduct hypothesis testing (e.g., t-test) to compare the groups
    # Reshape correlation matrices for the t-test
    group1_corr_flat = avg_group1_corr_matrix.flatten()
    group2_corr_flat = avg_group2_corr_matrix.flatten()

    # Perform independent t-test
    t_statistic, p_value = ttest_ind(group1_corr_flat, group2_corr_flat)
    """

    # Average the correlations
    #average_correlation = np.round(np.mean(correlations),3)

    #print("Spearman correlation:", average_correlation)
    (connectome_group['APOE3'] / num_group['APOE3'] - connectome_group['APOE4'] / num_group['APOE4'])
    if dist_type == 'canberra':
        mean_dist = scipy.spatial.distance.canberra((connectome_group['APOE3']/num_group['APOE3']).flatten(),(connectome_group['APOE4']/num_group['APOE4']).flatten())
    if dist_type == 'KS':
        mean_dist = (ks_2samp((connectome_group['APOE3']/num_group['APOE3']).flatten(),
                              (connectome_group['APOE4']/num_group['APOE4']).flatten()))[0]

    if round_up:
        mean_dist = np.round(mean_dist,3)
        #diff_pval = np.round(diff_pval,3)
        diff_pval = "{:.2e}".format(diff_pval)

    color = 'bwr'

    plt.figure(figsize=(8, 6),dpi=1200)

    mymax = np.max(np.abs(connectome_dif))
    try:
        plt.imshow(connectome_dif, cmap=color, interpolation='nearest',vmax = mymax, vmin= -1 * mymax)
    except:
        print(f'Color {color} is invalid')

    if colorbars:
        plt.colorbar(label='Connectivity Strength')

    txt = f'Diff Connectivity, APOE3 versus APOE4\ndistance between means = {mean_dist}\np-value of difference = {diff_pval}'
    #txt = f'Diff Connectivity, APOE3 versus APOE4\nsp-cor = {average_correlation}'
    plt.title(txt)

    plt.tight_layout()

    # Display the figure
    plt.savefig(connectome_figure_path)
    # plt.show()
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