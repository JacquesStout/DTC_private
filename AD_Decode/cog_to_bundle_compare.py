from scipy.stats import zscore
import pandas as pd

excel_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx'
excel_path_zscores = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3_zscores.xlsx'

cog_df = pd.read_excel(excel_path)

cog_cols = cog_df.columns[16:]

cog_df = cog_df.dropna(axis=0,how='any',subset='MRI_Exam')

for cog_col in cog_cols:
    try:
        cog_df[cog_col+'_zscore'] = zscore(cog_df[cog_col])
    except:
        cog_col_list = [float(i) for i in cog_df[cog_col]]
        cog_df[cog_col + '_zscore'] = cog_col_list

cog_df.to_excel(excel_path_zscores, index=False)

#['MOCA_TOTAL'] = 'None'
#['lm_BensonTotal', 'Delay_BensonTotal'] = 'Visual_Mem_Mean'
#['Composite_PLeasantness','Composite_Intensity','Composite_Familiarity','Composite_Nameability','Composite_Intensity']
['ufov2','ufov3'] = 'Visuospatial_Mean'
['Composite_Familiarity','Composite_Nameability','PrecentCorrectRecall_outof3','Recognized_outof6'] = 'Olfactive_Mem_Mean'
['AVLT_Trial6','AVLT_Trial7','RAVLT_LEARNING','RAVLT_FORGETTING'] = 'Verbal_Mem_Mean'
['Story_Immediate_verbatim, Story_Immediate_paraphrase','Delayed_verbatim', 'Delayed_paraphrase'] = 'Verbal_Mem_Mean'
['fwd_total_correct','fwd_max_length','bckwds_total_correct','bckwds_max_length'] = 'Working_Mem_Mean'
['fluency_4x','letter_fluency'] = 'Verbal_Fluency_Mean'
['Digit Symbol'] = 'Cognition_Mean'
['trailA','trailB'] = 'Visual_attention_Mean'
#,'trailDiff'

