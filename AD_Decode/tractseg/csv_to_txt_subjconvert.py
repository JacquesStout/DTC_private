import pandas as pd
import numpy as np

#data_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
data_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx'

text_path = '/Volumes/Data/Badea/ADdecode.01/Analysis/AD_DECODE_subjects_v2.txt'
text_path_y = '/Volumes/Data/Badea/ADdecode.01/Analysis/AD_DECODE_subjects_young.txt'
text_path_m = '/Volumes/Data/Badea/ADdecode.01/Analysis/AD_DECODE_subjects_middlea.txt'
text_path_o = '/Volumes/Data/Badea/ADdecode.01/Analysis/AD_DECODE_subjects_old.txt'

subj_to_elim = ['S02745','S02230','S02490','S02523','S02654','S02666','S02227','S02386','S02421','S02410','S02877','S02987','S03890']


if data_path.split('.')[1]=='csv':
    df = pd.read_csv(data_path)
elif data_path.split('.')[1]=='xlsx':
    df = pd.read_excel(data_path)
else:
    txt = f'Unidentifiable data file path {data_path}'


# Replace the name of the column 'MRI_Exam' to 'S'
df = df.dropna(subset=['genotype'])

df = df[df['Risk'].isin([np.nan, 'Familial'])]

df.rename(columns={'MRI_Exam': 'subject_id','genotype': 'group'}, inplace=True)

# Add a 'S0' at the beginning of all values in the new column 'S'
df['subject_id'] = 'S0' + (df['subject_id']).astype(int).astype(str)

df = df.replace({'APOE24': 'APOE4', 'APOE34': 'APOE4', 'APOE33': 'APOE3', 'APOE44':'APOE4','APOE23':'APOE3'}, regex=True)

df = df[['subject_id','group','sex','Risk','age']]

df['Risk'].fillna('None', inplace=True)
# Print out the modified DataFrame
#print(df)

df = df.replace({'APOE3': '0', 'APOE4': '1'}, regex=True)
df['sex'] = df['sex'].replace({'M': '0', 'F': '1'}, regex=True)
df = df.replace({'None': '0', 'Familial': '1','MCI':'2','AD':'3'}, regex=True)
df['age'] = df['age'].round(1)

# Eliminate specific rows
df = df[~df['subject_id'].isin(subj_to_elim)]

# Sort the DataFrame by age
df_sorted = df.sort_values(by='age')

# Calculate the lengths for each subset
total_length = len(df_sorted)
lower_third_length = total_length // 3
upper_third_length = total_length - lower_third_length * 2

# Split the DataFrame into three subsets
df_y = df_sorted.head(lower_third_length)
df_o = df_sorted.tail(upper_third_length)
df_m = df_sorted.iloc[lower_third_length: lower_third_length + upper_third_length]

# Save the DataFrame as a text file
df.to_csv(text_path, index=False, sep=' ')
df_y.to_csv(text_path_y, index=False, sep=' ')
df_m.to_csv(text_path_m, index=False, sep=' ')
df_o.to_csv(text_path_o, index=False, sep=' ')

