import pandas as pd

#data_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data_stripped.csv'
data_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx'

text_path = '/Users/jas/jacques/AD_Decode_excels/AD_DECODE_subjects.txt'

if data_path.split('.')[1]=='csv':
    df = pd.read_csv(data_path)
elif data_path.split('.')[1]=='xlsx':
    df = pd.read_excel(data_path)
else:
    txt = f'Unidentifiable data file path {data_path}'


# Replace the name of the column 'MRI_Exam' to 'S'
df = df.dropna(subset=['genotype'])

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


# Save the DataFrame as a text file
df.to_csv(text_path, index=False, sep=' ')

