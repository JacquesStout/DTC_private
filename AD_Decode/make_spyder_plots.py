from scipy.stats import zscore
import pandas as pd
import numpy as np
import os

import numpy as np
import nibabel as nib
from scipy import stats
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from DTC.file_manager.file_tools import mkcdir, check_files
import shutil, socket
import networkx as nx
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import pickle
import tempfile
import statsmodels.api as sm
import warnings
from sklearn.metrics import r2_score


def capitalize_words(input_string):
    # Split the string into words
    words = input_string.split()

    # Capitalize each word
    capitalized_words = [word.capitalize() for word in words]

    # Join the words back into a string
    output_string = ' '.join(capitalized_words)

    return output_string


def get_group(subject, excel_path, subj_column = 'RUNNO', group_column = 'Risk'):
    df = pd.read_excel(excel_path)
    columns_subj = []
    for column_name in df.columns:
        if subj_column in column_name:
            columns_subj.append(column_name)

    columns_group = []
    for column_name in df.columns:
        if group_column in column_name:
            columns_group.append(column_name)

    #subjects = database[columns_subj[0]]

    try:
        result = df[df[columns_subj[0]] == subject][columns_group[0]].values[0]
    except IndexError:
        result = None

    return result


def filter_strings_with_substrings(first_list, second_list):
    filtered_list = []

    for string1 in first_list:
        for string2 in second_list:
            if string2 in string1:
                filtered_list.append(string1)
                break  # Once a match is found, no need to check further

    return filtered_list


def standardize_database(db, subjects):
    all_subj = db.columns.values
    all_subj_trunc = [subj.split('_')[0] for subj in all_subj]
    for i,subj in enumerate(subjects):
        indices = [i for i, x in enumerate(all_subj_trunc) if x == subj]
        if np.size(indices)>1:
            test1 = np.all(abs(db[all_subj[indices[0]]]-db[all_subj[indices[1]]])<1e-1)
            if not test1:
                found=0
                for indice in indices:
                    if 'master' in all_subj[indice]:
                        indices = [indice]
                        found=1
                if not found:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices)>2:
                test2 = np.all(abs(db[all_subj[indices[1]]]-db[all_subj[indices[2]]])<1e-1)
                if not test2:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices)>3:
                test3 = np.all(abs(db[all_subj[indices[2]]]-db[all_subj[indices[3]]])<1e-1)
                if not test3:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices)>4:
                test4 = np.all(abs(db[all_subj[indices[3]]]-db[all_subj[indices[4]]])<1e-1)
                if not test4:
                    print(f'{subj} has 2 dissimilar instances')
            for drop_indice in np.arange(1,np.size(indices)):
                db = db.drop(all_subj[indices[drop_indice]], axis=1)
            db = db.rename(columns={all_subj[indices[0]]: all_subj[indices[0]].split('_')[0]})

    col_ordered = ['ROI']+subjects
    col_ordered = [item for item in col_ordered if item in list(db.columns.values)]

    db = db.drop(columns=[col for col in db if col not in col_ordered])
    db = db[col_ordered + [c for c in db.columns if c not in col_ordered]]
    return db


def degree_connectivity(connectome, method = 'sum'):
    if method=='sum':
        return np.sum(connectome,1)
    if method=='mean':
        return np.mean(connectome,1)


def clustering_coefficient(matrix):
    # Local clustering coefficient for each node
    local_clustering = np.sum(matrix @ matrix @ matrix, axis=1) / 2
    # Average clustering coefficient for the entire network
    return np.mean(local_clustering / (np.sum(matrix, axis=1) * (np.sum(matrix, axis=1) - 1)))


def eigenvector_centrality(matrix):
    _, vectors = eigsh(matrix, k=1, which='LM')
    centrality = np.abs(vectors.flatten())
    return centrality / np.sum(centrality)


def newLegend(fig, newNames):
    for item in newNames:
        for i, elem in enumerate(fig.data[0].labels):
            if elem == item:
                fig.data[0].labels[i] = newNames[item]
    return(fig)


excel_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx' #The excel path for the AD_Decode data
excel_path_zscores = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3_zscores.xlsx' #The output excel path for the AD_Decode data with zscores

rewrite_cog = True  #If true, go ahead and rewrite excel_path_zscores, otherwise just load it
fontsize_txt = 24 #Font size for the text of the graphs (r2, pvalues)
labelsize = 30  #Font size for the labels
myfontsize_title = 30 #Font size for the title
ticksize = 24 #Font size for the label ticks

if not os.path.exists(excel_path_zscores) or rewrite_cog:       #Begin writing the zscores
    cog_df = pd.read_excel(excel_path)

    cog_df = cog_df[~cog_df['Risk'].isin(['MCI', 'AD'])]
    cog_df = cog_df[~cog_df['genotype'].isna()]

    cog_df['MRI_Exam'] = cog_df['MRI_Exam'].astype(int)
    cog_df['MRI_Exam'] = 'S0' + cog_df['MRI_Exam'].astype(str)
    cog_df['MRI_Exam'] = cog_df['MRI_Exam'].str.replace('S0775', 'S00775')

    cog_cols = cog_df.columns[16:]

    cog_df = cog_df.dropna(axis=0,how='any',subset='MRI_Exam')

    reversed_vals = ['trailA','trailB','ufov2','ufov3','RAVLT_FORGETTING']          #These zscores are swapped, as the lower value means higher skill

    for cog_col in cog_cols:
        if cog_col in reversed_vals:
            sig = -1
        else:
            sig = 1
        if cog_df[cog_col].isna().any():                #If any scores are Not Applicable, rewrite them as the minimum value for the test
            min_value = cog_df[cog_col].min()
            cog_df[cog_col].fillna(min_value, inplace=True)
        try:
            cog_df[cog_col+'_zscore'] = zscore(cog_df[cog_col]*sig)
        except TypeError: #Strange problem with composite intensity needing to be converted to float
            cog_col_list = [float(i) for i in cog_df[cog_col]]
            cog_df[cog_col + '_zscore'] = zscore(cog_col_list*sig)

    """
    ['MOCA_TOTAL'] = 'None'
    ['Composite_Familiarity','Composite_Nameability','PrecentCorrectRecall_outof3','Recognized_outof6'] = 'Olfactive_Mem_Mean'
    ['AVLT_Trial6','AVLT_Trial7','RAVLT_LEARNING','RAVLT_FORGETTING'] = 'Verbal_Mem_Mean'
    ['Story_Immediate_verbatim, Story_Immediate_paraphrase','Delayed_verbatim', 'Delayed_paraphrase'] = 'Story_Mean' #Verbal_Mem_Mean
    ['fwd_total_correct','fwd_max_length'] = 'fwd_mean' #Verbal_Mem_Mean
    ['bckwds_total_correct','bckwds_max_length'] = 'Working_Mem_Mean'
    ['fluency_4x','letter_fluency'] = 'Verbal_Fluency_Mean'
    ['Digit Symbol'] = 'Cognition_Mean'
    ['trailA','trailB'] = 'Visual_attention_Mean'
    ['ufov2','ufov3'] = 'Visuospatial_Mean'
    """

    columns_toavg = {}
    #columns_toavg['Olfactive_Mem_Mean'] = ['Composite_Familiarity','Composite_Nameability','PrecentCorrectRecall_outof3','Recognized_outof6']
    columns_toavg['Olfactory_Memory'] = ['Composite_Familiarity', 'Composite_Nameability', 'Recognized_outof6']
    columns_toavg['Word_List_learning_and_Verbal_Memory'] = ['AVLT_Trial6','AVLT_Trial7','RAVLT_LEARNING','RAVLT_FORGETTING','RAVLT_IMMEDIATE']
    columns_toavg['Narrative_Learning_and_Verbal_attention'] = ['Story_Immediate_verbatim', 'Story_Immediate_paraphrase','Delayed_verbatim', 'Delayed_paraphrase']
    columns_toavg['Verbal_Attention'] = ['fwd_total_correct','fwd_max_length']
    columns_toavg['Working_Memory_and_Mental_Flexibility']= ['bckwds_total_correct','bckwds_max_length','trailB']
    columns_toavg['Verbal_Fluency'] = ['fluency_4x','letter_fluency']
    columns_toavg['Graphomotor_and_Visual_scanning_speed'] = ['trailA','num']
    columns_toavg['Visual_Attention'] = ['ufov2','ufov3']

    for key in columns_toavg.keys():
        # Calculate the average along the columns and store it in a new column
        zscore_cols = [column+'_zscore' for column in columns_toavg[key]]
        cog_df[key] = cog_df[zscore_cols].mean(axis=1)

    cog_df.to_excel(excel_path_zscores, index=False)

else:
    cog_df = pd.read_excel(excel_path_zscores)


col_tests = ['Olfactory_Memory', 'Word_List_learning_and_Verbal_Memory', 'Narrative_Learning_and_Verbal_attention',
             'Verbal_Attention', 'Working_Memory_and_Mental_Flexibility', 'Verbal_Fluency',
             'Graphomotor_and_Visual_scanning_speed', 'Visual_Attention','MOCA_TOTAL']

#Specify the main group cofactor
group_columns = ['genotype']

subjects = cog_df['MRI_Exam']

main_output_path = f'/Users/jas/jacques/AD_Decode_bundles_figures' #root output path of figures
mkcdir(main_output_path)

p_value_sig = 0.05   #pvalue cutoff

bundle_sub_select = None #If bundle_sub_select has a specified bundle, ex:'bundle_4', run over the bundle and/or all its descendants only.
# If None, run over all bundles

make_spider_graph = True    #Make the radar plots
group_corr_compare = False  #If True, on top of getting group influence, also compare the plots of groupA vs groupB directly
quadratic = False

#Specify the main dependent variables to cover: cog_mean is the average cognitive values, cog_cols is all the cognitive values, phys_cols is all the physical values
#To note that for FDR purposes, all dependent variables in a group are grouped together.

reg_stat_types = ['meanfa','num_sl','vol_sl','sdfa','BUAN','meanfa_assym'] #All independent variables to cover. These are treated separately for FDR purposes.
reg_stat_types = ['len_sl']

#'cog_mean' or 'cog_cols' or 'phys_cols'

sub_bundling_levels = [1,2,3,4]
sub_bundling_levels = [1,2,3,4]  #The sub bundling levels to cover. FDR-wise. for each level,
# all bundles of that level are grouped together unless specified otherwise byf bundle_sub_select

formula_wgroup = False

dpi_value = 1200    #Resolution of significant figures

full_stat_db = cog_df

group_column = 'genotype'

output_path = os.path.join(main_output_path, group_column)
mkcdir(output_path)
lm_path = os.path.join(output_path,
                       'lm_switched')  # previous test had cog and stat swapped, so here it is named 'switched'
mkcdir(lm_path)

categories = col_tests
categories.remove('MOCA_TOTAL')

mid_age = np.quantile(full_stat_db['age'], (1 / 3))
old_age = np.quantile(full_stat_db['age'], (2 / 3))

spiderfig_groups = ['young', 'mid', 'old', 'all']  # All radar plot groupings

spider_graph_folder = os.path.join(lm_path, f'spider_graph_newnames')  # radar plot output path
mkcdir(spider_graph_folder)

full_stat_db[group_column] = full_stat_db[group_column].replace(
    {'APOE23': 'APOE3', 'APOE24': 'APOE4', 'APOE34': 'APOE4', 'APOE33': 'APOE3', 'APOE44': 'APOE4'}, regex=True)

for spiderfig_group in spiderfig_groups:

    if spiderfig_group == 'all':
        temp_db = full_stat_db
    elif spiderfig_group == 'young':
        temp_db = full_stat_db[full_stat_db['age'] < mid_age]
    elif spiderfig_group == 'mid':
        temp_db = full_stat_db[(full_stat_db['age'] > mid_age) & (full_stat_db['age'] < old_age)]
    elif spiderfig_group == 'old':
        temp_db = full_stat_db[full_stat_db['age'] > old_age]
    else:
        raise Exception('Option unrecolnized')

    rvals_dic = {}

    max = 0
    min = 10
    fig = go.Figure()

    for group in np.unique(temp_db[group_column]):
        rvals_dic[group] = []
        for category in categories:
            category_val = np.mean(temp_db[temp_db[group_column] == group][category])
            rvals_dic[group].append(category_val)

        if group == 'APOE3':
            color = 'blue'
        elif group == 'APOE4':
            color = 'red'
        elif group == 'APOE2':
            color = 'green'
        else:
            color = None

        # color = None

        fig['layout']['yaxis']['autorange'] = "reversed"

        # r0 = math.ceil(np.max(rvals_dic[group])*10)/10
        # rend = math.ceil(np.abs(np.min(rvals_dic[group]))*10)/10
        # dr = np
        categories_clean = []
        for category in categories:
            categories_clean.append(
                category.replace('_Mean', '').replace('_', ' ').replace('short term',
                                                                                                 'short').replace('and','&').
                    replace('Working Memory & Mental Flexibility','WMM').replace('Word List learning & Verbal Memory','WLV'))

        if color is None:
            fig.add_trace(go.Scatterpolar(
                r=rvals_dic[group],
                theta=categories_clean,
                fill='toself',
                name=group,
            ))
        else:
            fig.add_trace(go.Scatterpolar(
                r=rvals_dic[group],
                theta=categories_clean,
                fill='toself',
                name=group,
                fillcolor=color,
                opacity=0.5
            ))

        if max < np.max(rvals_dic[group]):
            max = np.max(rvals_dic[group])
        if min > np.min(rvals_dic[group]):
            min = np.min(rvals_dic[group])

    if spiderfig_group == 'all':
        showlegend = True
    else:
        showlegend = False

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                # range=[max, min]
                range=[0.5, -0.9],
                # This range helps specify that low score makes radar plotting bigger, not smaller
                ticktext=categories_clean
            ),
            angularaxis=dict(
                tickfont=dict(size=18),  # Change font size for category labels
                categoryarray=categories_clean,  # Specify the category array
                categoryorder='array'  # Use the order of categories provided
            )
        ),
        showlegend=showlegend
    )

    fig.write_image(os.path.join(spider_graph_folder, f'spider_{spiderfig_group}.png'))