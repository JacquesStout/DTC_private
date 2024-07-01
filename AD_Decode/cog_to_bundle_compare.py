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

    columns_toavg = {}
    """
    #columns_toavg['Olfactive_Mem_Mean'] = ['Composite_Familiarity','Composite_Nameability','PrecentCorrectRecall_outof3','Recognized_outof6']
    columns_toavg['Olfactive_Mem_Mean'] = ['Composite_Familiarity', 'Composite_Nameability', 'Recognized_outof6']
    columns_toavg['Verbal_Mem_Mean'] = ['AVLT_Trial6','AVLT_Trial7','RAVLT_LEARNING','RAVLT_FORGETTING','RAVLT_IMMEDIATE']
    columns_toavg['Story_Mean'] = ['Story_Immediate_verbatim', 'Story_Immediate_paraphrase','Delayed_verbatim', 'Delayed_paraphrase']
    columns_toavg['Verbal_short_term_Mem_Mean'] = ['fwd_total_correct','fwd_max_length']
    columns_toavg['Working_Mem_Mean']= ['bckwds_total_correct','bckwds_max_length','trailB']
    columns_toavg['Verbal_Fluency_Mean'] = ['fluency_4x','letter_fluency']
    #columns_toavg['Cognition_Mean'] = ['Digit Symbol']
    columns_toavg['Visual_attention_Mean'] = ['trailA','Digit_Symbol']
    columns_toavg['Visuospatial_Mean'] = ['ufov2','ufov3']
    """
    columns_toavg['Olfactory_Memory'] = ['Composite_Familiarity', 'Composite_Nameability', 'Recognized_outof6']
    columns_toavg['Word_List_learning_and_Verbal_Memory'] = ['AVLT_Trial6','AVLT_Trial7','RAVLT_LEARNING','RAVLT_FORGETTING','RAVLT_IMMEDIATE']
    columns_toavg['Narrative_Learning_and_Verbal_attention'] = ['Story_Immediate_verbatim', 'Story_Immediate_paraphrase','Delayed_verbatim', 'Delayed_paraphrase']
    columns_toavg['Verbal_Attention'] = ['fwd_total_correct','fwd_max_length']
    columns_toavg['Working_Memory_and_Mental_Flexibility']= ['bckwds_total_correct','bckwds_max_length','trailB']
    columns_toavg['Verbal_Fluency'] = ['fluency_4x','letter_fluency']
    columns_toavg['Graphomotor_and_Visual_scanning_speed'] = ['trailA','Digit_Symbol']
    columns_toavg['Visual_Attention'] = ['ufov2','ufov3']


    for key in columns_toavg.keys():
        # Calculate the average along the columns and store it in a new column
        zscore_cols = [column+'_zscore' for column in columns_toavg[key]]
        try:
            cog_df[key] = cog_df[zscore_cols].mean(axis=1)
        except:
            print('hi')

    cog_df.to_excel(excel_path_zscores, index=False)

else:
    cog_df = pd.read_excel(excel_path_zscores)


#Specify the main group cofactor
group_columns = ['genotype']

subjects = cog_df['MRI_Exam']

main_output_path = f'/Users/jas/jacques/AD_Decode_bundles_figures' #root output path of figures
mkcdir(main_output_path)

p_value_sig = 0.05   #pvalue cutoff

bundle_sub_select = None #If bundle_sub_select has a specified bundle, ex:'bundle_4', run over the bundle and/or all its descendants only.
# If None, run over all bundles

group_corr_compare = False  #If True, on top of getting group influence, also compare the plots of groupA vs groupB directly
quadratic = False

#Specify the main dependent variables to cover: cog_mean is the average cognitive values, cog_cols is all the cognitive values, phys_cols is all the physical values
#To note that for FDR purposes, all dependent variables in a group are grouped together.
col_col_types = ['cog_mean','cog_cols','phys_cols']
col_col_types = ['cog_mean']
#col_col_types = ['blood_pressure']


reg_stat_types = ['meanfa','num_sl','vol_sl','sdfa','BUAN','meanfa_assym'] #All independent variables to cover. These are treated separately for FDR purposes.
reg_stat_types = ['len_sl']

#'cog_mean' or 'cog_cols' or 'phys_cols'

sub_bundling_levels = [1,2,3,4]  #The sub bundling levels to cover. FDR-wise. for each level,
# all bundles of that level are grouped together unless specified otherwise byf bundle_sub_select

formula_wgroup = False

dpi_value = 1200    #Resolution of significant figures

if bundle_sub_select is None:
    bundle_sub_select_txt = ''
else:
    bundle_sub_select_txt = '_'+bundle_sub_select

for sub_bundling_level in sub_bundling_levels:

    if True:
        sub_bund_txt = ''
        for i in np.arange(sub_bundling_level):
            sub_bund_txt+='_all'

        #Specified stats file, generated by /Users/jas/bass/gitfolder/DTC_private/AD_Decode/combine_allstats_df.R
        stats_excel_path = f'/Users/jas/jacques/AD_Decode_bundles_figures/bundle_split_results/bundles_6_100_excels' \
            f'/master_df{sub_bund_txt}_combined.xlsx'

        #Bad warning complaining about no default style, this is done to suppress that useless warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat_db = pd.read_excel(stats_excel_path)


        #combining stats and cognition, also cleans up some of the columns post merge
        full_stat_db = pd.merge(cog_df, stat_db, on='MRI_Exam', how='inner')

        x_cols_rename = {}
        y_cols_del = []

        for col in full_stat_db:
            if '_x' in col:
                x_cols_rename[col] = col.replace('_x','')
            if '_y' in col:
                y_cols_del.append(col)

        full_stat_db.rename(columns=x_cols_rename, inplace=True)
        full_stat_db.drop(columns=y_cols_del, inplace=True)

    for group_column in group_columns:

        group_column_txt = group_column     #Specify if you ever want the name of group to be different from name of group column

        output_path = os.path.join(main_output_path,group_column)
        mkcdir(output_path)
        lm_path = os.path.join(output_path,'lm_switched')  #previous test had cog and stat swapped, so here it is named 'switched'
        mkcdir(lm_path)

        if group_column == 'genotype':      #combining all genotypes into APOE3 or APOE4
            full_stat_db[group_column] = full_stat_db[group_column].replace(
                {'APOE23': 'APOE3', 'APOE24': 'APOE4', 'APOE34': 'APOE4', 'APOE33': 'APOE3', 'APOE44': 'APOE4'}, regex=True)
            # full_stat_db[group_column] = full_stat_db[group_column].replace(
            #    {'APOE23':'APOE2','APOE24': 'APOE2', 'APOE33': 'APOE3','APOE34': 'APOE4','APOE44':'APOE4'}, regex=True)

        for col_col_type in col_col_types:      #The full columns for each specified col type

            if col_col_type == 'cog_mean':
                col_tests = ['Olfactive_Mem_Mean', 'Verbal_Mem_Mean', 'Story_Mean', 'Verbal_short_term_Mem_Mean', 'Working_Mem_Mean',
                             'Verbal_Fluency_Mean', 'Visual_attention_Mean', 'Visuospatial_Mean', 'MOCA_TOTAL']
            elif col_col_type == 'cog_cols':
                col_tests = [col for col in cog_df.columns if 'zscore' in col]
            elif col_col_type == 'phys_cols':
                col_tests = ['age', 'Systolic', 'Diastolic', 'Pulse', 'Height', 'Weight', 'BMI']
            elif col_col_type == 'olfactive':
                col_tests = ['Olfactive_Mem_Mean']
            elif col_col_type == 'blood_pressure':
                col_tests = ['Diastolic','Systolic']

            output_type = ''

            for stat_type in reg_stat_types:

                print(f'Running the statistic {stat_type}')

                stats_folder_results = os.path.join(lm_path, f'lm_results_all')
                stats_folder_results_sig = os.path.join(lm_path, f'lm_results_sig_all')
                stats_folder_results_fsig = os.path.join(lm_path, f'lm_results_fsig_all{bundle_sub_select_txt}')
                stats_folder_results_fsig_interact = os.path.join(lm_path, f'lm_results_fsig_onlyinteract{bundle_sub_select_txt}')

                stat_type_folder = os.path.join(stats_folder_results, stat_type)
                stat_type_folder_sig = os.path.join(stats_folder_results_sig, stat_type)

                stat_type_folder_fsig = os.path.join(stats_folder_results_fsig, stat_type)
                stat_type_folder_fsig_interact = os.path.join(stats_folder_results_fsig_interact, stat_type)

                if bundle_sub_select is not None:
                    bundles = [col.split('_'+stat_type)[0] for col in full_stat_db.columns if stat_type in col and bundle_sub_select in col]
                else:
                    bundles = [col.split('_' + stat_type)[0] for col in full_stat_db.columns if stat_type in col]

                #mkcdir([stats_folder_results, stat_type_folder,stat_type_folder_sig,stat_type_folder_fsig,stats_folder_results_fsig_interact,stat_type_folder_fsig_interact])
                mkcdir([stats_folder_results,stats_folder_results_sig,stats_folder_results_fsig,
                        stats_folder_results_fsig_interact,stat_type_folder,stat_type_folder_sig,
                        stat_type_folder_fsig,stat_type_folder_fsig_interact])


                #Initiating lists of relevant p-values for later fdr correction
                p_values_group = []
                p_values_stat = []
                p_values_interact = []
                p_values_group_compare = []
                pickle_paths = []
                group_corrs = []

                group_data = {}

                for group in np.unique(full_stat_db[group_column]):
                    group_data[group] = full_stat_db[full_stat_db[group_column] == group]
                    #APOE3_data = full_stat_db[full_stat_db['genotype'] == 'APOE3']
                    #APOE4_data = full_stat_db[full_stat_db['genotype'] == 'APOE4']

                groups = sorted(np.unique(full_stat_db[group_column]))
                group_1 = groups[0]
                group_2 = groups[1]

                #iterating through dependent variables
                for col_type in col_tests:

                    stat_ROIs = {}
                    #iterating through independent variable from each bu ndle
                    for i, bundle in enumerate(bundles):
                        stat_path = os.path.join(stat_type_folder,
                                                 f'lm_scatterplot_{stat_type}_{col_type}_{bundle}.png')


                        bundle_stat = f'{bundle}_{stat_type}'
                        colnames = [col_type, bundle_stat,'genotype']
                        stat_bundle = full_stat_db[colnames]

                        #if including genotype or sex or other group as co-factor or not, different model
                        if formula_wgroup:
                            formula = f'{col_type} ~ {bundle_stat} * {group_column}'
                        else:
                            formula = f'{col_type} ~ {bundle_stat}'

                        model = sm.formula.ols(formula=formula, data=full_stat_db).fit()

                        stat_key = [key for key in model.pvalues.keys() if group_column not in key and bundle_stat in key][0]


                        if formula_wgroup:
                            interact_key = [key for key in model.pvalues.keys() if group_column in key and bundle_stat in key][0]
                            group_key = [key for key in model.pvalues.keys() if group_column in key and bundle_stat not in key][0]

                        correlation_col = model.params[stat_key]

                        group_corr = {}

                        #individual group comparison
                        if group_corr_compare:
                            if np.unique(full_stat_db[group_column]) == 2:

                                formula = f'{col_type} ~ {bundle_stat}'

                                model_1 = sm.formula.ols(formula=formula, data=group_data[group_1]).fit()
                                model_2 = sm.formula.ols(formula=formula, data=group_data[group_2]).fit()

                                corr_1, p_value_group_1 = spearmanr(group_data[group_1][col_type],
                                                                    group_data[group_1][bundle_stat])
                                corr_2, p_value_group_2 = spearmanr(group_data[group_2][col_type],
                                                                    group_data[group_2][bundle_stat])

                                group_corr[group_1] = corr_1
                                group_corr[group_2] = corr_2

                                n1 = np.size(group_data[group_1][col_type])
                                n2 = np.size(group_data[group_2][col_type])

                                z1 = .5 * np.log((1 + corr_1) / (1 - corr_1))

                                z2 = .5 * np.log((1 + corr_2) / (1 - corr_2))

                                sezdiff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

                                ztest = (z1 - z2) / sezdiff

                                alpha = 2 * (1 - norm.cdf(abs(ztest), loc=0, scale=1))
                                p_values_group_compare.append(alpha)

                        group_corr['stat'] = model.params[stat_key]

                        group_corr['r2score'] = model.rsquared

                        p_values_stat.append(model.pvalues[stat_key])

                        if formula_wgroup:

                            p_values_group.append(model.pvalues[group_key])
                            p_values_interact.append(model.pvalues[interact_key])

                            group_corr['group'] = model.params[group_key]
                            group_corr['interact'] = model.params[interact_key]

                            group_corr['r2score_group'] = sm.stats.anova_lm(model)['sum_sq'][group_column] / model.ess
                            group_corr['r2score_interact'] = sm.stats.anova_lm(model)['sum_sq'][
                                                                 f'{col_type}:{group_column}'] / model.ess
                        else: #if there is no group taken into account, write p-values as one to easily ignore them in results
                            p_values_group.append(1)
                            p_values_interact.append(1)

                            group_corr['group'] = 0
                            group_corr['interact'] = 0

                            group_corr['r2score_group'] = 0
                            group_corr['r2score_interact'] = 0
                            group_corr['r2score_stat'] = model.rsquared

                        if np.abs(group_corr['group']>1):
                            a = 1

                        group_corrs.append(group_corr)

                        fig_handle = plt.figure()

                        #these values determine the limits of the graph and are useful for text positioning
                        x_min = np.min(stat_bundle[bundle_stat])
                        x_max = np.max(stat_bundle[bundle_stat])
                        y_min = np.min(stat_bundle[col_type])
                        y_max = np.max(stat_bundle[col_type])

                        #counter looking at where the least amount of points are located, for text positioning
                        dic_count = {}
                        dic_count['bottom_left'] = len(stat_bundle[(stat_bundle[col_type] < (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] < (x_min + (x_max-x_min)/2))])
                        dic_count['top_left'] = len(stat_bundle[(stat_bundle[col_type] > (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] < (x_min + (x_max-x_min)/2))])
                        dic_count['bottom_right_count'] = len(stat_bundle[(stat_bundle[col_type] < (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] > (x_min + (x_max-x_min)/2))])
                        dic_count['top_right_count'] = len(stat_bundle[(stat_bundle[col_type] > (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] > (x_min + (x_max-x_min)/2))])
                        min = 1000

                        #txt positioning values based on counter and graph limits, seen above
                        for key in dic_count.keys():
                            if dic_count[key]<min:
                                loc = key
                                min = dic_count[key]
                        if 'bottom' in loc:
                            yloc = y_min
                        if 'top' in loc:
                            yloc = y_max - (y_max - y_min) / 5
                        if 'right' in loc:
                            xloc = x_max  - (x_max - x_min) / 3
                        if 'left' in loc:
                            xloc = x_min

                        #plt.ylabel(bundle_stat)

                        if 'volume' in stat_type or 'Volume' in stat_type or 'vol' in stat_type:
                            #plt.xlabel(f'Volume for {bundle} (mm³)', fontsize=labelsize)
                            plt.xlabel(f'Volume (mm³)', fontsize=labelsize)
                        elif 'meanfa' in stat_type:
                            plt.xlabel(f'μ FA', fontsize=labelsize)
                            #plt.xlabel(f'Mean of FA for {bundle}', fontsize=labelsize)
                        elif 'sdfa' in stat_type:
                            plt.xlabel(f'sd FA', fontsize=labelsize)
                            #plt.xlabel(f'Mean of FA for {bundle}', fontsize=labelsize)
                        elif 'volume_prop' in stat_type:
                            #plt.xlabel(f'Proportional volume for {bundle} (%)', fontsize=labelsize)
                            plt.xlabel(f'Volume (%)', fontsize=labelsize)
                        else:
                            plt.xlabel(f'{stat_type}', fontsize=labelsize)

                        # Set labels and title
                        if col_col_type == 'cog_mean':
                            ylabel_txt = capitalize_words(' '.join(col_type.split('_Mean')[0].split('_'))+' Zscore')
                        else:
                            ylabel_txt = capitalize_words(' '.join(col_type.split('_')))

                        plt.ylabel(ylabel_txt, fontsize=labelsize)

                        #plt.title(f'Scatter Plot of {capitalize_words(bundle)} \nfor {stat_type} vs {col_type}', fontsize=myfontsize_title)
                        plt.title(f'{capitalize_words(bundle)}', fontsize=myfontsize_title)

                        # Save the figure object using pickle
                        pickle_file_path = tempfile.NamedTemporaryFile(delete=False)
                        pickle_paths.append(pickle_file_path.name)

                        with open(pickle_file_path.name, 'wb') as f:
                            pickle.dump(fig_handle, f)

                        sns.regplot(x=bundle_stat, y=col_type, data=stat_bundle, color='purple')

                        plt.scatter(x=bundle_stat, y=col_type, data=group_data[group_1], color='blue', label=group_1)
                        plt.scatter(x=bundle_stat, y=col_type, data=group_data[group_2], color='red', label=group_2)

                        txt = f'Correlation: {group_corr["stat"]:.4f}\n' \
                            f'p-value for {stat_type}: {model.pvalues[stat_key]:.4f}\n'

                        if formula_wgroup:
                            txt += f'p-value for {group_column}: {model.pvalues[group_key]:.4f}\n' \
                            f'p-value for interaction: {model.pvalues[interact_key]:.4f}\n'

                        plt.text(xloc, yloc,txt,fontsize=fontsize_txt)

                        plt.savefig(stat_path)

                        plt.close()

                _, p_values_stat_corrected, _, _ = multipletests(p_values_stat, method='fdr_bh')
                _, p_values_group_corrected, _, _ = multipletests(p_values_group, method='fdr_bh')
                _, p_values_interact_corrected, _, _ = multipletests(p_values_interact, method='fdr_bh')


                for l, col_type in enumerate(col_tests):

                    for i,bundle in enumerate(bundles):

                        stat_path = os.path.join(stat_type_folder,
                                                 f'lm_{stat_type}_{col_type}_{bundle}.png')
                        bundle_stat = f'{bundle}_{stat_type}'

                        p_value_stat = p_values_stat[l * np.size(bundles) + i]
                        p_value_group = p_values_group[l * np.size(bundles) + i]
                        p_value_interact = p_values_interact[l * np.size(bundles) + i]

                        p_value_stat_corrected = p_values_stat_corrected[l * np.size(bundles) + i]
                        p_value_group_corrected = p_values_group_corrected[l * np.size(bundles) + i]
                        p_value_interact_corrected = p_values_interact_corrected[l * np.size(bundles) + i]

                        colnames = [col_type, bundle_stat, 'genotype']
                        stat_bundle = full_stat_db[colnames]

                        x_min = np.min(stat_bundle[bundle_stat])
                        x_max = np.max(stat_bundle[bundle_stat])
                        y_min = np.min(stat_bundle[col_type])
                        y_max = np.max(stat_bundle[col_type])

                        group_corr = group_corrs[l * np.size(bundles) + i]
                        correlation_col = group_corr['stat']

                        """
                        if not correlation_col/y_max < 0.5:
                            if correlation_col <0:
                                yloc = y_min + (y_max - y_min) / 5
                            else:
                                yloc = y_max - (y_max - y_min) / 5
                            xloc = x_min
                        """
                        #else:
                        dic_count = {}
                        dic_count['bottom_left'] = len(stat_bundle[(stat_bundle[col_type] < (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] < (x_min + (x_max-x_min)/2))])
                        dic_count['top_left'] = len(stat_bundle[(stat_bundle[col_type] > (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] < (x_min + (x_max-x_min)/2))])
                        dic_count['bottom_right_count'] = len(stat_bundle[(stat_bundle[col_type] < (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] > (x_min + (x_max-x_min)/2))])
                        dic_count['top_right_count'] = len(stat_bundle[(stat_bundle[col_type] > (y_min + (y_max-y_min)/2)) & (stat_bundle[bundle_stat] > (x_min + (x_max-x_min)/2))])
                        min = 1000
                        for key in dic_count.keys():
                            if dic_count[key]<min:
                                loc = key
                                min = dic_count[key]
                        if 'bottom' in loc:
                            #yloc = y_min + (y_max - y_min) / 10
                            yloc = y_min
                        if 'top' in loc:
                            yloc = y_max - (y_max - y_min) / 5
                        if 'right' in loc:
                            xloc = x_max - (x_max - x_min) / 3
                        if 'left' in loc:
                            xloc = x_min


                        if p_value_stat_corrected < p_value_sig or p_value_group_corrected < p_value_sig or p_value_interact_corrected < p_value_sig:

                            pickle_file_path = pickle_paths[l * np.size(bundles) + i]

                            mkcdir([stats_folder_results_sig, stats_folder_results_fsig,stat_type_folder_fsig])
                            stat_path_fsig = os.path.join(stat_type_folder_fsig,
                                                         f'lm_{stat_type}_{col_type}_{bundle}.png')
                            stat_path_fsig_interact = os.path.join(stat_type_folder_fsig_interact,
                                                         f'lm_{stat_type}_{col_type}_{bundle}.png')
                            with open(pickle_file_path, 'rb') as f:
                                fig_handle = pickle.load(f)

                            txt = ''

                            # \
                            #f'p-value for {col_type}: {model.pvalues[stat_key]:.4f}\n' \
                            ##    f'p-value for {group_type}: {model.pvalues[group_key]}\n' \
                            #    f'p-value for interaction: {model.pvalues[interact_key]}\n'

                            if 'volume' in stat_type or 'Volume' in stat_type or 'vol' in stat_type:
                                # plt.xlabel(f'Volume for {bundle} (mm³)', fontsize=labelsize)
                                #plt.xlabel(f'Volume (mm³)', fontsize=labelsize)
                                label_txt = f'Volume (mm³)'
                                stat_type_txt = 'Vol'
                            elif 'meanfa' in stat_type:
                                #plt.xlabel(f'μ FA', fontsize=labelsize)
                                label_txt = f'μ FA'
                                stat_type_txt = 'FA'
                            elif 'sdfa' in stat_type:
                                label_txt = f'sd FA'
                                stat_type_txt = 'sd FA'
                                #plt.xlabel(f'Mean of FA for {bundle}', fontsize=labelsize)
                            elif 'volume_prop' in stat_type:
                                # plt.xlabel(f'Proportional volume for {bundle} (%)', fontsize=labelsize)
                                #plt.xlabel(f'Volume (%)', fontsize=labelsize)
                                label_txt = f'Volume (%)'
                                stat_type_txt = 'Volp'
                            else:
                                #plt.xlabel(f'{stat_type}', fontsize=labelsize)
                                label_txt = f'{stat_type}'
                                stat_type_txt = stat_type


                            if p_value_stat_corrected < p_value_sig:

                                #txt+= f'Correlation for {stat_type}: {group_corr["stat"]:.2f}\nfdr p-value for {stat_type}: {p_value_stat_corrected:.4f}\n'
                                #txt += f'R\u00b2 {stat_type_txt}= {group_corr["r2score_stat"]:.2f}\nf-pval {stat_type_txt}= {p_value_stat_corrected:.4f}\n'
                                txt += f'R\u00b2= {group_corr["r2score_stat"]:.2f}\npval = {p_value_stat_corrected:.4f}\n'
                                sns.regplot(x=bundle_stat, y=col_type, data=stat_bundle, color='purple')

                            elif p_value_group_corrected < p_value_sig or p_value_interact_corrected < p_value_sig:

                                if p_value_group_corrected < p_value_sig and p_value_interact_corrected < p_value_sig: #adjustment for too many lines
                                    if yloc>((y_max + y_min)/2):
                                        yloc = yloc - (y_max - y_min) / 10
                                    if xloc > ((x_max + x_min) / 2):
                                        xloc = xloc - (x_max - x_min) / 5

                                #txt+= f'Correlation for {group_column}: {group_corr["group"]:.2f}\nfdr p-value for {group_column}: {p_value_group_corrected:.4f}\n'
                                if p_value_group_corrected < p_value_sig:
                                    txt += f'R\u00b2 {group_column_txt}= {group_corr["r2score_group"]:.2f}\n' \
                                        f'pval {group_column_txt}= {p_value_group_corrected:.4f}\n'
                                    if xloc>((x_max + x_min)/2):
                                        xloc = xloc - (x_max - x_min) / 6

                                if p_value_interact_corrected < p_value_sig:
                                    txt += f'R\u00b2 {stat_type_txt}:{group_column_txt}= {group_corr["r2score_interact"]:.2f}\n' \
                                        f'pval {stat_type_txt}:{group_column_txt}= {p_value_interact_corrected:.4f}\n'
                                    if xloc>((x_max + x_min)/2):
                                        xloc = xloc - (x_max - x_min) / 5

                                sns.regplot(x=bundle_stat, y=col_type, data=group_data[group_1], color='blue')
                                sns.regplot(x=bundle_stat, y=col_type, data=group_data[group_2], color='red')


                            plt.text(xloc, yloc, txt, fontsize=fontsize_txt)
                            plt.scatter(x=bundle_stat, y=col_type, data=group_data[group_1], color='blue', label=group_1)
                            plt.scatter(x=bundle_stat, y=col_type, data=group_data[group_2], color='red', label=group_2)

                            if col_col_type == 'cog_mean':
                                ylabel_txt = capitalize_words(
                                    ' '.join(col_type.split('_Mean')[0].split('_')) + ' Zscore')
                            else:
                                ylabel_txt = capitalize_words(' '.join(col_type.split('_')))

                            plt.ylabel(ylabel_txt, fontsize=labelsize)

                            plt.xlabel(label_txt, fontsize=labelsize)

                            plt.xticks(fontsize=ticksize)
                            plt.yticks(fontsize=ticksize)
                            plt.gcf().set_dpi(1200)

                            plt.savefig(stat_path_fsig)
                            if p_value_interact_corrected < p_value_sig:
                                shutil.copy(stat_path_fsig, stat_path_fsig_interact)
                            plt.close('all')

                        if p_value_stat < p_value_sig or p_value_group < p_value_sig or p_value_interact < p_value_sig:

                            pickle_file_path = pickle_paths[l * np.size(bundles) + i]

                            mkcdir([stats_folder_results_sig, stat_type_folder_sig])
                            stat_path_sig = os.path.join(stat_type_folder_sig,
                                                         f'lm_{stat_type}_{col_type}_{bundle}.png')

                            with open(pickle_file_path, 'rb') as f:
                                fig_handle = pickle.load(f)

                            txt = ''
                            plt.xticks(fontsize=ticksize)
                            plt.yticks(fontsize=ticksize)

                            if 'volume' in stat_type or 'Volume' in stat_type or 'vol' in stat_type:
                                # plt.xlabel(f'Volume for {bundle} (mm³)', fontsize=labelsize)
                                #plt.xlabel(f'Volume (mm³)', fontsize=labelsize)
                                label_txt = f'Volume (mm³)'
                                stat_type_txt = 'Vol'
                            elif 'fa' in stat_type or 'meanfa' in stat_type:
                                #plt.xlabel(f'μ FA', fontsize=labelsize)
                                label_txt = f'μ FA'
                                stat_type_txt = 'FA'
                                #plt.xlabel(f'Mean of FA for {bundle}', fontsize=labelsize)
                            elif 'volume_prop' in stat_type:
                                # plt.xlabel(f'Proportional volume for {bundle} (%)', fontsize=labelsize)
                                #plt.xlabel(f'Volume (%)', fontsize=labelsize)
                                label_txt = f'Volume (%)'
                                stat_type_txt = 'Volp'
                            else:
                                #plt.xlabel(f'{stat_type}', fontsize=labelsize)
                                label_txt = f'{stat_type}'
                                stat_type_txt = stat_type


                            if p_value_stat < p_value_sig:
                                #txt += f'Correlation for {stat_type}: {group_corr["stat"]:.2f}\np-value for {stat_type}: {p_value_stat:.4f}\n'
                                txt += f'R\u00b2 {stat_type_txt}= {group_corr["r2score_stat"]:.2f}\npval = {p_value_stat:.4f}\n'

                                sns.regplot(x=bundle_stat, y=col_type, data=stat_bundle, color='purple')

                            elif p_value_group < p_value_sig or p_value_interact < p_value_sig:

                                if p_value_interact < p_value_sig:
                                    txt += f'R\u00b2 {stat_type}:{group_column}= {group_corr["r2score_interact"]:.2f}\npval {stat_type}= {p_value_interact:.4f}\n'

                                if p_value_group < p_value_sig:
                                    txt += f'R\u00b2 {group_column_txt}= {group_corr["r2score_group"]:.2f}\npval {group_column}= {p_value_group:.4f}\n'

                                sns.regplot(x=bundle_stat, y=col_type, data=group_data[group_1], color='blue')
                                sns.regplot(x=bundle_stat, y=col_type, data=group_data[group_2], color='red')

                            plt.text(xloc, yloc, txt, fontsize=fontsize_txt)
                            plt.xticks(fontsize=ticksize)
                            plt.yticks(fontsize=ticksize)
                            plt.scatter(x=bundle_stat, y=col_type, data=group_data[group_1], color='blue', label=group_1)
                            plt.scatter(x=bundle_stat, y=col_type, data=group_data[group_2], color='red', label=group_2)

                            plt.xlabel(label_txt, fontsize=labelsize)

                            plt.savefig(stat_path_sig)
                            plt.close('all')
