from scipy.stats import zscore
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from DTC.file_manager.file_tools import mkcdir, check_files
import shutil
from statsmodels.stats.multitest import multipletests
import pandas as pd
import pickle
import tempfile
import statsmodels.api as sm
import warnings


def capitalize_words(input_string):
    # Split the string into words
    words = input_string.split()

    # Capitalize each word
    capitalized_words = [word.capitalize() for word in words]

    # Join the words back into a string
    output_string = ' '.join(capitalized_words)

    return output_string


excel_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx'
excel_path_zscores = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3_zscores.xlsx'

rewrite_cog = True
fontsize_txt = 26
labelsize = 30
myfontsize_title = 30
ticksize = 24

if not os.path.exists(excel_path_zscores) or rewrite_cog:
    cog_df = pd.read_excel(excel_path)

    cog_df = cog_df[~cog_df['Risk'].isin(['MCI', 'AD'])]
    cog_df = cog_df[~cog_df['genotype'].isna()]

    cog_df['MRI_Exam'] = cog_df['MRI_Exam'].astype(int)
    cog_df['MRI_Exam'] = 'S0' + cog_df['MRI_Exam'].astype(str)
    cog_df['MRI_Exam'] = cog_df['MRI_Exam'].str.replace('S0775', 'S00775')

    cog_cols = cog_df.columns[16:]

    cog_df = cog_df.dropna(axis=0,how='any',subset='MRI_Exam')

    reversed_vals = ['trailA','trailB','ufov2','ufov3','RAVLT_FORGETTING']

    for cog_col in cog_cols:
        if cog_col in reversed_vals:
            sig = -1
        else:
            sig = 1
        if cog_df[cog_col].isna().any():
            min_value = cog_df[cog_col].min()
            cog_df[cog_col].fillna(min_value, inplace=True)
        try:
            cog_df[cog_col+'_zscore'] = zscore(cog_df[cog_col]*sig)
        except:
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
    columns_toavg['Olfactive_Mem_Mean'] = ['Composite_Familiarity', 'Composite_Nameability', 'Recognized_outof6']
    columns_toavg['Verbal_Mem_Mean'] = ['AVLT_Trial6','AVLT_Trial7','RAVLT_LEARNING','RAVLT_FORGETTING','RAVLT_IMMEDIATE']
    columns_toavg['Story_Mean'] = ['Story_Immediate_verbatim', 'Story_Immediate_paraphrase','Delayed_verbatim', 'Delayed_paraphrase']
    columns_toavg['Verbal_short_term_Mem_Mean'] = ['fwd_total_correct','fwd_max_length']
    columns_toavg['Working_Mem_Mean']= ['bckwds_total_correct','bckwds_max_length']
    columns_toavg['Verbal_Fluency_Mean'] = ['fluency_4x','letter_fluency']
    #columns_toavg['Cognition_Mean'] = ['Digit Symbol']
    columns_toavg['Visual_attention_Mean'] = ['trailA','trailB']
    columns_toavg['Visuospatial_Mean'] = ['ufov2','ufov3']

    for key in columns_toavg.keys():
        # Calculate the average along the columns and store it in a new column
        zscore_cols = [column+'_zscore' for column in columns_toavg[key]]
        cog_df[key] = cog_df[zscore_cols].mean(axis=1)

    cog_df.to_excel(excel_path_zscores, index=False)

else:
    cog_df = pd.read_excel(excel_path_zscores)

#cog_types_trimmed = ['MoCA Total']

group_columns = ['sex']

subjects = cog_df['MRI_Exam']

main_output_path = f'/Users/jas/jacques/AD_Decode_bundles_figures'
mkcdir(main_output_path)

p_value_sig = 0.05
#bundle_sub_select = 'bundle_2' #If bundle_sub_select is None, run over all bundles
bundle_sub_select = None

make_spider_graph = True
group_corr_compare = False
quadratic = False

#bundle_sub_select = None

#col_col_types = ['age']
col_col_types = ['age']
#col_col_types = ['cog_mean']
#col_col_types = ['cog_cols','phys_cols']
#col_col_types = ['blood_pressure']
#col_col_types = ['phys_cols']


reg_stat_types = ['meanfa','num_sl','vol_sl','sdfa','len_sl']
reg_stat_types = ['num_sl','vol_sl','len_sl']
reg_stat_types = ['meanfa','sdfa','num_sl','vol_sl','len_sl','BUAN']
#reg_stat_types = ['meanfa']
#reg_stat_types = ['num_sl','vol_sl','sdfa','len_sl']
#reg_stat_types = ['num_sl']
#reg_stat_types = ['BUAN']

#reg_stat_types = ['sdfa']
#reg_stat_types = ['len3_sl', 'vol_sl']
#reg_stat_types = ['num_sl']
#reg_stat_types = ['BUAN']

#'cog_mean' or 'cog_cols' or 'phys_cols'

sub_bundling_levels = [1,2,3,4]
#sub_bundling_levels = [4]

combined = True

#sub_bundling_levels = [4]


dpi_value = 1200

if bundle_sub_select is None:
    bundle_sub_select_txt = ''
else:
    bundle_sub_select_txt = '_'+bundle_sub_select

for sub_bundling_level in sub_bundling_levels:

    if sub_bundling_level==1:
        num_lowest=2
    if sub_bundling_level==2:
        num_lowest=4
    if sub_bundling_level == 3:
        num_lowest=6
    if sub_bundling_level == 4:
        num_lowest=8

    if True:
        sub_bund_txt = ''
        for i in np.arange(sub_bundling_level):
            sub_bund_txt+='_all'

        if combined:
            stats_excel_path = f'/Users/jas/jacques/AD_Decode_bundles_figures/bundle_split_results/bundles_6_100_excels' \
                f'/master_df{sub_bund_txt}_combined.xlsx'
        else:
            stats_excel_path = f'/Users/jas/jacques/AD_Decode_bundles_figures/bundle_split_results/bundles_6_100_excels' \
                f'/master_df{sub_bund_txt}.xlsx'

        #Bad warning complaining about no default style, this is done to suppress that useless warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat_db = pd.read_excel(stats_excel_path)

        full_stat_db = pd.merge(cog_df, stat_db, on='MRI_Exam', how='inner')

        x_cols_rename = {}
        y_cols_del = []

        for col in full_stat_db:
            if '_x' in col:
                x_cols_rename[col] = col.replace('_x','')
            if '_y' in col:
                y_cols_del.append(col)

        #full_stat_db.rename(columns={'genotype_x': 'genotype','MOCA_TOTAL_x':'MOCA_TOTAL'}, inplace=True)
        #full_stat_db.drop(columns=['genotype_y'], inplace=True)

        full_stat_db.rename(columns=x_cols_rename, inplace=True)
        full_stat_db.drop(columns=y_cols_del, inplace=True)


    for group_column in group_columns:

        if group_column == 'genotype':
            group_column_txt = 'genotype'
        else:
            group_column_txt = group_column

        output_path = os.path.join(main_output_path,group_column)
        mkcdir(output_path)
        if combined:
            lm_path = os.path.join(output_path,'lm_age_combined')
        else:
            lm_path = os.path.join(output_path,'lm_age')
        mkcdir(lm_path)

        if group_column == 'genotype':
            full_stat_db[group_column] = full_stat_db[group_column].replace(
                {'APOE23': 'APOE3', 'APOE24': 'APOE4', 'APOE34': 'APOE4', 'APOE33': 'APOE3', 'APOE44': 'APOE4'}, regex=True)
            # full_stat_db[group_column] = full_stat_db[group_column].replace(
            #    {'APOE23':'APOE2','APOE24': 'APOE2', 'APOE33': 'APOE3','APOE34': 'APOE4','APOE44':'APOE4'}, regex=True)

        for col_col_type in col_col_types:

            if col_col_type == 'cog_mean':
                col_tests = ['Olfactive_Mem_Mean', 'Verbal_Mem_Mean', 'Story_Mean', 'Verbal_short_term_Mem_Mean', 'Working_Mem_Mean',
                             'Verbal_Fluency_Mean', 'Visual_attention_Mean', 'Visuospatial_Mean', 'MOCA_TOTAL']
            elif col_col_type == 'cog_cols':
                col_tests = [col for col in cog_df.columns if 'zscore' in col]
            elif col_col_type == 'phys_cols':
                col_tests = ['Systolic', 'Diastolic', 'Height', 'Weight']
            elif col_col_type == 'olfactive':
                col_tests = ['Olfactive_Mem_Mean']
            elif col_col_type == 'blood_pressure':
                col_tests = ['Diastolic','Systolic']
            elif col_col_type == 'age':
                col_tests = ['age']

            output_type = ''

            for stat_type in reg_stat_types:

                print(f'Running the statistic {stat_type}')

                if stat_type == 'meanfa' and col_col_type=='age':
                    formula_type = 'quadratic'
                    add_folder = '_quadratic_x'
                else:
                    formula_type = ''
                    add_folder = ''

                stats_folder_results = os.path.join(lm_path, f'lm_results_all{add_folder}')
                stats_folder_results_sig = os.path.join(lm_path, f'lm_results_sig_all{add_folder}')
                stats_folder_results_fsig = os.path.join(lm_path, f'lm_results_fsig_all{bundle_sub_select_txt}{add_folder}')
                stats_folder_results_fsig_interact = os.path.join(lm_path, f'lm_results_fsig_onlyinteract{bundle_sub_select_txt}{add_folder}')

                stat_type_folder = os.path.join(stats_folder_results, stat_type)
                stat_type_folder_sig = os.path.join(stats_folder_results_sig, stat_type)

                #if bundle_sub_select is None:
                stat_type_folder_fsig = os.path.join(stats_folder_results_fsig, stat_type)
                stat_type_folder_fsig_interact = os.path.join(stats_folder_results_fsig_interact, stat_type)

                if combined:
                    if bundle_sub_select is not None:
                        bundles = [col.split('_' + stat_type)[0] for col in full_stat_db.columns if
                                   stat_type in col and bundle_sub_select in col]
                    else:
                        bundles = [col.split('_' + stat_type)[0] for col in full_stat_db.columns if stat_type in col]
                else:
                    if bundle_sub_select is not None:
                        bundles = [col.split('_'+stat_type)[0] for col in full_stat_db.columns if stat_type in col and
                                   bundle_sub_select in col and ('right' in col or 'left' in col)]
                    else:
                        bundles = [col.split('_' + stat_type)[0] for col in full_stat_db.columns if stat_type in col
                                   and ('right' in col or 'left' in col)]

                #mkcdir([stats_folder_results, stat_type_folder,stat_type_folder_sig,stat_type_folder_fsig,stats_folder_results_fsig_interact,stat_type_folder_fsig_interact])
                mkcdir([stats_folder_results,stats_folder_results_sig,stats_folder_results_fsig,
                        stats_folder_results_fsig_interact,stat_type_folder,stat_type_folder_sig,
                        stat_type_folder_fsig,stat_type_folder_fsig_interact])

                p_values_group = []
                p_values_col = []
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

                for col_type in col_tests:

                    stat_ROIs = {}
                    for i, bundle in enumerate(bundles):
                        stat_path = os.path.join(stat_type_folder,
                                                 f'lm_scatterplot_{stat_type}_{col_type}_{bundle}.png')


                        bundle_stat = f'{bundle}_{stat_type}'
                        colnames = [col_type, bundle_stat,'genotype']
                        stat_bundle = full_stat_db[colnames]
                        #formula = f'{bundle_stat} ~ {col_type}*{col_type} * {group_column}'
                        #formula = f'{bundle_stat} ~ {col_type}*{col_type} + {col_type}'
                        if formula_type == 'quadratic':
                            formula = f'{bundle_stat} ~ I({col_type}**2)'
                            #formula = f'{bundle_stat} ~ I({col_type}**2) + {col_type}'
                            #formula = f'{bundle_stat} ~ {col_type}*{col_type} + {col_type}'
                        else:
                            formula = f'{bundle_stat} ~ {col_type} * {group_column}'

                        model = sm.formula.ols(formula=formula, data=full_stat_db).fit()

                        """
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        poly_features = poly.fit_transform(np.array(full_stat_db[col_type].tolist()).reshape(-1, 1))
                        poly_reg_model = LinearRegression()
                        poly_reg_model.fit(poly_features, full_stat_db[bundle_stat])
                        y_predicted = poly_reg_model.predict(poly_features)
                        """

                        #formula = f'{bundle_stat} ~ {col_type} * {col_type} * {col_type}'
                        #print(sm.formula.ols(formula=formula, data=full_stat_db).fit().pvalues['age'])

                        if formula_type != 'quadratic':
                            group_key = [key for key in model.pvalues.keys() if group_column in key and bundle_stat not in key][0]
                            interact_key = [key for key in model.pvalues.keys() if group_column in key and col_type in key][0]

                        col_key = [key for key in model.pvalues.keys() if group_column not in key and col_type in key][0]

                        correlation_col = model.params[col_key]

                        group_corr = {}

                        p_values_col.append(model.pvalues[col_key])

                        group_corr['col'] = model.params[col_key]

                        #group_corr['r2score'] = r2_score(full_stat_db[col_type], model.predict(full_stat_db))
                        group_corr['r2score'] = model.rsquared

                        if formula_type != 'quadratic':
                            group_corr['r2score_stat'] = sm.stats.anova_lm(model)['sum_sq'][col_key] / model.ess

                            p_values_group.append(model.pvalues[group_key])
                            p_values_interact.append(model.pvalues[interact_key])

                            group_corr['group'] = model.params[group_key]
                            group_corr['interact'] = model.params[interact_key]

                            group_corr['r2score_group'] = sm.stats.anova_lm(model)['sum_sq'][group_column] / model.ess
                            group_corr['r2score_interact'] = sm.stats.anova_lm(model)['sum_sq'][
                                                                 f'{col_type}:{group_column}'] / model.ess
                        else:

                            p_values_group.append(1)
                            p_values_interact.append(1)

                            group_corr['group'] = 0
                            group_corr['interact'] = 0

                            group_corr['r2score_group'] = 0
                            group_corr['r2score_interact'] = 0
                            group_corr['r2score_stat'] = model.rsquared

                        group_corrs.append(group_corr)

                        fig_handle = plt.figure()

                        x_min = np.min(stat_bundle[col_type])
                        x_max = np.max(stat_bundle[col_type])
                        y_min = np.min(stat_bundle[bundle_stat])
                        y_max = np.max(stat_bundle[bundle_stat])

                        """
                        if not correlation_col/y_max < 0.5:
                            if model.params[col_key] <0:
                                yloc = y_min + (y_max - y_min) / 5
                            else:
                                yloc = y_max - (y_max - y_min) / 5
                                xloc = x_min
                        """
                        #else:
                        dic_count = {}
                        dic_count['bottom_left'] = len(stat_bundle[(stat_bundle[bundle_stat] < (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] < (x_min + (x_max-x_min)/2))])
                        dic_count['top_left'] = len(stat_bundle[(stat_bundle[bundle_stat] > (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] < (x_min + (x_max-x_min)/2))])
                        dic_count['bottom_right_count'] = len(stat_bundle[(stat_bundle[bundle_stat] < (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] > (x_min + (x_max-x_min)/2))])
                        dic_count['top_right_count'] = len(stat_bundle[(stat_bundle[bundle_stat] > (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] > (x_min + (x_max-x_min)/2))])
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
                            xloc = x_max  - (x_max - x_min) / 3
                            #xloc = x_max - (x_max - x_min) / 2
                        if 'left' in loc:
                            xloc = x_min

                        #plt.ylabel(bundle_stat)

                        if 'volume' in stat_type or 'Volume' in stat_type or 'vol' in stat_type:
                            #plt.xlabel(f'Volume for {bundle} (mm³)', fontsize=labelsize)
                            ylabel_txt = f'Volume (mm³)'
                        elif 'meanfa_assym' in stat_type:
                            ylabel_txt = f'FA assymetry %'
                        elif 'meanfa' in stat_type:
                            # plt.xlabel(f'μ FA', fontsize=labelsize)
                            ylabel_txt = f'μ FA'
                        elif 'sdfa' in stat_type:
                            ylabel_txt = f'sd FA'
                            #plt.xlabel(f'Mean of FA for {bundle}',
                        elif 'volume_prop' in stat_type:
                            #plt.xlabel(f'Proportional volume for {bundle} (%)',
                            ylabel_txt = f'Volume (%)'
                        elif 'len_sl' in stat_type:
                            ylabel_txt = f'Streamline length (mm)'
                        elif 'num_sl' in stat_type:
                            ylabel_txt = f'Streamline number'
                        else:
                            ylabel_txt = f'{stat_type}'

                        # Set labels and title
                        if col_col_type == 'cog_mean':
                            xlabel_txt = capitalize_words(' '.join(col_type.split('_Mean')[0].split('_'))+' Zscore')
                        else:
                            xlabel_txt = capitalize_words(' '.join(col_type.split('_')))

                        if 'Age' in xlabel_txt:
                            xlabel_txt += ' (years)'

                        plt.xlabel(xlabel_txt, fontsize=labelsize)
                        plt.ylabel(ylabel_txt, fontsize=labelsize)

                        #plt.title(f'Scatter Plot of {capitalize_words(bundle)} \nfor {stat_type} vs {col_type}', fontsize=myfontsize_title)
                        plt.title(f'{capitalize_words(bundle)}', fontsize=myfontsize_title)

                        # Save the figure object using pickle
                        pickle_file_path = tempfile.NamedTemporaryFile(delete=False)
                        pickle_paths.append(pickle_file_path.name)

                        with open(pickle_file_path.name, 'wb') as f:
                            pickle.dump(fig_handle, f)

                        if formula_type == 'quadratic':
                            x_pred = np.linspace(full_stat_db[col_type].min(), full_stat_db[col_type].max(), 100)
                            y_pred = model.predict(pd.DataFrame({col_type: x_pred, f'I({col_type}**2)': x_pred ** 2}))

                            #x_pred = np.linspace(full_stat_db[col_type].min(), full_stat_db[col_type].max(), 100)
                            df_pred = pd.DataFrame({
                                'age': x_pred,
                                'age2': x_pred ** 2
                            })

                            predictions = model.get_prediction(pd.DataFrame({col_type: x_pred, f'I({col_type}**2)': x_pred ** 2}))
                            prediction_summary = predictions.summary_frame()

                            #plt.plot(x_pred, y_pred, color='purple', label='Quadratic fit')
                            plt.plot(x_pred, prediction_summary['mean'], color='purple', label='Quadratic fit')

                            plt.fill_between(x_pred,
                                             prediction_summary['mean_ci_lower'],
                                             prediction_summary['mean_ci_upper'],
                                             color='purple', alpha=0.2, label='95% Confidence Interval')

                            predictions = model.get_prediction(df_pred)
                            prediction_summary = predictions.summary_frame()
                        else:
                            sns.regplot(x=col_type, y=bundle_stat, data=stat_bundle, color='purple')


                        plt.scatter(x=col_type, y=bundle_stat, data=group_data[group_1], color='blue', label=group_1)
                        plt.scatter(x=col_type, y=bundle_stat, data=group_data[group_2], color='red', label=group_2)

                        txt = f'Correlation: {group_corr["col"]:.4f}\n' \
                            f'p-value for {col_type}: {model.pvalues[col_key]:.4f}\n'

                        if formula_type != 'quadratic':
                            txt += f'p-value for {group_column}: {model.pvalues[group_key]:.4f}\n' \
                            f'p-value for interaction: {model.pvalues[interact_key]:.4f}\n'
                        plt.text(xloc, yloc,txt,fontsize=fontsize_txt)

                        plt.savefig(stat_path)

                        plt.close()

                #_, p_values_stat_corrected, _, _ = multipletests(p_values_col, method='fdr_bh')
                _, p_values_stat_corrected, _, _ = multipletests(p_values_col, method='bonferroni')
                _, p_values_group_corrected, _, _ = multipletests(p_values_group, method='bonferroni')
                _, p_values_interact_corrected, _, _ = multipletests(p_values_interact, method='bonferroni')


                def select_top_significant(pvals, num, sig_cutoff = 0.05):
                    pvals_sig = pvals[pvals < sig_cutoff]
                    num_pvals_sig = np.size(pvals_sig)
                    if num_pvals_sig<=num:
                        return pvals_sig,sig_cutoff
                    else:
                        proportion = num/num_pvals_sig
                        cutoff = np.quantile(pvals_sig,proportion)
                        pvals_picked = pvals_sig[pvals_sig<cutoff]
                        return pvals_picked,cutoff


                _,stat_cutoff = select_top_significant(p_values_stat_corrected,num_lowest,sig_cutoff = p_value_sig)
                _,group_cutoff = select_top_significant(p_values_group_corrected,num_lowest,sig_cutoff = p_value_sig)
                _,interact_cutoff = select_top_significant(p_values_interact_corrected,num_lowest,sig_cutoff = p_value_sig)


                for l, col_type in enumerate(col_tests):

                    for i,bundle in enumerate(bundles):

                        stat_path = os.path.join(stat_type_folder,
                                                 f'lm_{stat_type}_{col_type}_{bundle}.png')
                        bundle_stat = f'{bundle}_{stat_type}'

                        p_value_col = p_values_col[l * np.size(bundles) + i]
                        p_value_group = p_values_group[l * np.size(bundles) + i]
                        p_value_interact = p_values_interact[l * np.size(bundles) + i]

                        p_value_stat_corrected = p_values_stat_corrected[l * np.size(bundles) + i]
                        p_value_group_corrected = p_values_group_corrected[l * np.size(bundles) + i]
                        p_value_interact_corrected = p_values_interact_corrected[l * np.size(bundles) + i]

                        colnames = [col_type, bundle_stat, 'genotype']
                        stat_bundle = full_stat_db[colnames]

                        x_min = np.min(stat_bundle[col_type])
                        x_max = np.max(stat_bundle[col_type])
                        y_min = np.min(stat_bundle[bundle_stat])
                        y_max = np.max(stat_bundle[bundle_stat])

                        group_corr = group_corrs[l * np.size(bundles) + i]
                        correlation_col = group_corr['col']

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
                        dic_count['bottom_left'] = len(stat_bundle[(stat_bundle[bundle_stat] < (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] < (x_min + (x_max-x_min)/2))])
                        dic_count['top_left'] = len(stat_bundle[(stat_bundle[bundle_stat] > (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] < (x_min + (x_max-x_min)/2))])
                        dic_count['bottom_right_count'] = len(stat_bundle[(stat_bundle[bundle_stat] < (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] > (x_min + (x_max-x_min)/2))])
                        dic_count['top_right_count'] = len(stat_bundle[(stat_bundle[bundle_stat] > (y_min + (y_max-y_min)/2)) & (stat_bundle[col_type] > (x_min + (x_max-x_min)/2))])
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
                            #xloc = x_max - (x_max - x_min) / 2
                        if 'left' in loc:
                            xloc = x_min


                        if p_value_stat_corrected < p_value_sig or p_value_group_corrected < p_value_sig or p_value_interact_corrected < p_value_sig:

                            pickle_file_path = pickle_paths[l * np.size(bundles) + i]

                            marker = ''
                            lowermarker_name = '_lowest'
                            if p_value_interact_corrected < p_value_sig:
                                sig_type = f'{stat_type}:{group_column}'
                                if p_value_interact_corrected < interact_cutoff:
                                    marker = lowermarker_name
                            elif p_value_group_corrected < p_value_sig:
                                sig_type = f'{group_column}'
                                if p_value_group_corrected < group_cutoff:
                                    marker = lowermarker_name
                            elif p_value_stat_corrected < p_value_sig:
                                sig_type = f'{stat_type}'
                                if p_value_stat_corrected < stat_cutoff:
                                    marker = lowermarker_name

                            mkcdir([stats_folder_results_sig, stats_folder_results_fsig,stat_type_folder_fsig])
                            stat_path_fsig = os.path.join(stat_type_folder_fsig,
                                                         f'lm_{stat_type}_{col_type}_{bundle}_{sig_type}{marker}.png')
                            stat_path_fsig_interact = os.path.join(stat_type_folder_fsig_interact,
                                                         f'lm_{stat_type}_{col_type}_{bundle}{marker}.png')
                            with open(pickle_file_path, 'rb') as f:
                                fig_handle = pickle.load(f)

                            txt = ''

                            # \
                            #f'p-value for {col_type}: {model.pvalues[col_key]:.4f}\n' \
                            ##    f'p-value for {group_type}: {model.pvalues[group_key]}\n' \
                            #    f'p-value for interaction: {model.pvalues[interact_key]}\n'

                            if 'volume' in stat_type or 'Volume' in stat_type or 'vol' in stat_type:
                                # plt.xlabel(f'Volume for {bundle} (mm³)', fontsize=labelsize)
                                #plt.xlabel(f'Volume (mm³)', fontsize=labelsize)
                                label_txt = f'Volume (mm³)'
                                stat_type_txt = 'Vol'
                            elif 'meanfa_assym' in stat_type:
                                label_txt = f'FA assymetry %'
                                stat_type_txt = 'assym FA'
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

                                #txt+= f'Correlation for {stat_type}: {group_corr["col"]:.2f}\nfdr p-value for {stat_type}: {p_value_stat_corrected:.4f}\n'
                                #txt += f'R\u00b2 {stat_type_txt}= {group_corr["r2score_stat"]:.2f}\nf-pval {stat_type_txt}= {p_value_stat_corrected:.4f}\n'
                                txt += f'R\u00b2= {group_corr["r2score_stat"]:.2f}\npval = {p_value_stat_corrected:.4f}\n'

                                if formula_type == 'quadratic':
                                    formula = f'{bundle_stat} ~ I({col_type}**2)'
                                    model = sm.formula.ols(formula=formula, data=full_stat_db).fit()

                                    x_pred = np.linspace(full_stat_db[col_type].min(), full_stat_db[col_type].max(),
                                                         100)
                                    y_pred = model.predict(
                                        pd.DataFrame({col_type: x_pred, f'I({col_type}**2)': x_pred ** 2}))

                                    # x_pred = np.linspace(full_stat_db[col_type].min(), full_stat_db[col_type].max(), 100)
                                    df_pred = pd.DataFrame({
                                        'age': x_pred,
                                        'age2': x_pred ** 2
                                    })

                                    predictions = model.get_prediction(
                                        pd.DataFrame({col_type: x_pred, f'I({col_type}**2)': x_pred ** 2}))
                                    prediction_summary = predictions.summary_frame()

                                    # plt.plot(x_pred, y_pred, color='purple', label='Quadratic fit')
                                    plt.plot(x_pred, prediction_summary['mean'], color='purple', label='Quadratic fit')

                                    plt.fill_between(x_pred,
                                                     prediction_summary['mean_ci_lower'],
                                                     prediction_summary['mean_ci_upper'],
                                                     color='purple', alpha=0.2, label='95% Confidence Interval')

                                    predictions = model.get_prediction(df_pred)
                                    prediction_summary = predictions.summary_frame()
                                else:
                                    sns.regplot(x=col_type, y=bundle_stat, data=stat_bundle, color='purple')

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

                                sns.regplot(x=col_type, y=bundle_stat, data=group_data[group_1], color='blue')
                                sns.regplot(x=col_type, y=bundle_stat, data=group_data[group_2], color='red')

                            plt.text(xloc, yloc, txt, fontsize=fontsize_txt)
                            plt.scatter(x=col_type, y=bundle_stat, data=group_data[group_1], color='blue', label=group_1)
                            plt.scatter(x=col_type, y=bundle_stat, data=group_data[group_2], color='red', label=group_2)

                            if 'volume' in stat_type or 'Volume' in stat_type or 'vol' in stat_type:
                                # plt.xlabel(f'Volume for {bundle} (mm³)', fontsize=labelsize)
                                ylabel_txt = f'Volume (mm³)'
                            elif 'meanfa_assym' in stat_type:
                                ylabel_txt = f'FA assymetry %'
                            elif 'meanfa' in stat_type:
                                # plt.xlabel(f'μ FA', fontsize=labelsize)
                                ylabel_txt = f'μ FA'
                            elif 'sdfa' in stat_type:
                                ylabel_txt = f'sd FA'
                                # plt.xlabel(f'Mean of FA for {bundle}', fontsize=labelsize)
                            elif 'volume_prop' in stat_type:
                                # plt.xlabel(f'Proportional volume for {bundle} (%)', fontsize=labelsize)
                                ylabel_txt = f'Volume (%)'
                            elif 'len_sl' in stat_type:
                                ylabel_txt = f'Streamline length (mm)'
                            elif 'num_sl' in stat_type:
                                ylabel_txt = f'Streamline number'
                            else:
                                ylabel_txt = f'{stat_type}'

                            plt.ylabel(ylabel_txt, fontsize=labelsize)

                            if col_col_type == 'cog_mean':
                                xlabel_txt = capitalize_words(
                                    ' '.join(col_type.split('_Mean')[0].split('_')) + ' Zscore')
                            else:
                                xlabel_txt = capitalize_words(' '.join(col_type.split('_')))

                            if 'Age' in xlabel_txt:
                                xlabel_txt += ' (years)'

                            plt.xlabel(xlabel_txt, fontsize=labelsize)

                            plt.xticks(fontsize=ticksize)
                            plt.yticks(fontsize=ticksize)

                            #plt.legend(prop={'size': 26})
                            plt.gcf().set_dpi(1200)

                            plt.savefig(stat_path_fsig)
                            if p_value_interact_corrected < p_value_sig:
                                shutil.copy(stat_path_fsig, stat_path_fsig_interact)
                            plt.close('all')

                        if p_value_col < p_value_sig or p_value_group < p_value_sig or p_value_interact < p_value_sig:

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


                            if p_value_col < p_value_sig:
                                #txt += f'Correlation for {stat_type}: {group_corr["col"]:.2f}\np-value for {stat_type}: {p_value_col:.4f}\n'
                                txt += f'R\u00b2 {stat_type_txt}= {group_corr["r2score_stat"]:.2f}\npval = {p_value_col:.4f}\n'

                                if formula_type == 'quadratic':

                                    formula = f'{bundle_stat} ~ I({col_type}**2)'
                                    model = sm.formula.ols(formula=formula, data=full_stat_db).fit()

                                    x_pred = np.linspace(full_stat_db[col_type].min(), full_stat_db[col_type].max(),
                                                         100)
                                    y_pred = model.predict(
                                        pd.DataFrame({col_type: x_pred, f'I({col_type}**2)': x_pred ** 2}))

                                    # x_pred = np.linspace(full_stat_db[col_type].min(), full_stat_db[col_type].max(), 100)
                                    df_pred = pd.DataFrame({
                                        'age': x_pred,
                                        'age2': x_pred ** 2
                                    })

                                    predictions = model.get_prediction(
                                        pd.DataFrame({col_type: x_pred, f'I({col_type}**2)': x_pred ** 2}))
                                    prediction_summary = predictions.summary_frame()

                                    # plt.plot(x_pred, y_pred, color='purple', label='Quadratic fit')
                                    plt.plot(x_pred, prediction_summary['mean'], color='purple', label='Quadratic fit')

                                    plt.fill_between(x_pred,
                                                     prediction_summary['mean_ci_lower'],
                                                     prediction_summary['mean_ci_upper'],
                                                     color='purple', alpha=0.2, label='95% Confidence Interval')

                                    predictions = model.get_prediction(df_pred)
                                    prediction_summary = predictions.summary_frame()
                                else:
                                    sns.regplot(x=col_type, y=bundle_stat, data=stat_bundle, color='purple')

                            elif p_value_group < p_value_sig or p_value_interact < p_value_sig:

                                if p_value_interact < p_value_sig:
                                    txt += f'R\u00b2 {stat_type}:{group_column}= {group_corr["r2score_interact"]:.2f}\npval {stat_type}= {p_value_interact:.4f}\n'

                                if p_value_group < p_value_sig:
                                    txt += f'R\u00b2 {group_column_txt}= {group_corr["r2score_group"]:.2f}\npval {group_column}= {p_value_group:.4f}\n'

                                sns.regplot(x=col_type, y=bundle_stat, data=group_data[group_1], color='blue')
                                sns.regplot(x=col_type, y=bundle_stat, data=group_data[group_2], color='red')

                            plt.text(xloc, yloc, txt, fontsize=fontsize_txt)
                            plt.xticks(fontsize=ticksize)
                            plt.yticks(fontsize=ticksize)
                            plt.scatter(x=col_type, y=bundle_stat, data=group_data[group_1], color='blue', label=group_1)
                            plt.scatter(x=col_type, y=bundle_stat, data=group_data[group_2], color='red', label=group_2)

                            plt.xlabel(label_txt, fontsize=labelsize)

                            plt.savefig(stat_path_sig)
                            plt.close('all')
