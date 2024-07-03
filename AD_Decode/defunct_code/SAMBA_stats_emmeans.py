import numpy as np
import copy
import glob, os
import pandas as pd
from scipy import stats
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
import socket, time
from DTC.file_manager.file_tools import mkcdir, check_files
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from emmeans import emmeans
"""
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from plotnine import ggplot
from DTC.file_manager.file_tools import mkcdir, check_files
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy import stats
import shutil
"""


def standardize_database(db, subjects):
    db = db.dropna(axis=1, how='all')
    all_subj = db.columns.values
    all_subj_trunc = [subj.split('_')[0] for subj in all_subj]
    for i, subj in enumerate(subjects):
        indices = [i for i, x in enumerate(all_subj_trunc) if x == subj]
        if np.size(indices) > 1:
            test1 = np.all(abs(db[all_subj[indices[0]]] - db[all_subj[indices[1]]]) < 1e-1)
            if not test1:
                print(f'{subj} has 2 dissimilar instances')
            if np.size(indices) > 2:
                test2 = np.all(abs(db[all_subj[indices[1]]] - db[all_subj[indices[2]]]) < 1e-1)
                if not test2:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices) > 3:
                test3 = np.all(abs(db[all_subj[indices[2]]] - db[all_subj[indices[3]]]) < 1e-1)
                if not test3:
                    print(f'{subj} has 2 dissimilar instances')
            if np.size(indices) > 4:
                test4 = np.all(abs(db[all_subj[indices[3]]] - db[all_subj[indices[4]]]) < 1e-1)
                if not test4:
                    print(f'{subj} has 2 dissimilar instances')
            for drop_indice in np.arange(1, np.size(indices)):
                db = db.drop(all_subj[indices[drop_indice]], axis=1)
            db = db.rename(columns={all_subj[indices[0]]: all_subj[indices[0]].split('_')[0]})

    col_ordered = ['ROI'] + subjects
    col_ordered = [item for item in col_ordered if item in list(db.columns.values)]

    db = db.drop(columns=[col for col in db if col not in col_ordered])
    db = db[col_ordered + [c for c in db.columns if c not in col_ordered]]
    return db


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


def get_large_folders(folder_list, min_size_bytes):
    large_folders = []
    folder_sizes = []
    for folder_path in folder_list:
        folder_size = get_folder_size(folder_path)
        if folder_size > min_size_bytes:
            large_folders.append(folder_path)
            folder_sizes.append(folder_size)
    return large_folders, folder_sizes


def file_recent(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        return False

    # Get the file's creation timestamp
    file_stat = os.stat(file_path)
    creation_time = file_stat.st_ctime

    # Get the current timestamp
    current_time = time.time()

    # Calculate the time difference in seconds
    time_difference = current_time - creation_time

    # Check if the file was created less than a day ago (86400 seconds in a day)
    return time_difference < 86400


def detection_outlier(x, data, outlier_col):
    quartiles = np.percentile(x[outlier_col], [25, 75])
    iqr = np.percentile(data[outlier_col], 75) - np.percentile(data[outlier_col], 25)
    lower = quartiles[0] - 1 * iqr
    upper = quartiles[1] + 1 * iqr
    index = list(x[((x[outlier_col] < lower) | (x[outlier_col] > upper))].index)
    bounds = [lower, upper]
    return index, bounds


#### Important paths
ROI_legends = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx'
VBM_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/'
stat_folder = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/stats_by_region/labels/pre_rigid_native_space/IITmean_RPI/stats/studywide_label_statistics/'
root = '/Volumes/Data/Badea/Lab'

if socket.gethostname().split('.')[0] == 'santorini':
    metadata_path = '/Users/jas/jacques/AD_Decode/AD_DECODE_data2.csv'
    output_path = '/Users/jas/jacques/AD_Decode/ANOVA_results/pre_analysis_outputs'


datapath = '/Volumes/Data/Badea/ADdecode.01/Data/Anat/'
connectome_folder = os.path.join(root, 'mouse/mrtrix_ad_decode/connectome')
subject_text = os.path.join(output_path, 'subjects.txt')

outpath='/Users/jas/jacques/AD_Decode/R_outputs/test_folder/'
mkcdir(outpath)

#### Getting the ROI information from the Excel file
index1_to_2, _, index2_to_struct, _ = atlas_converter(ROI_legends)
index1_to_2.pop(0, None)
index_to_struct = {}
for key in index1_to_2.keys():
    index_to_struct[key] = index2_to_struct[index1_to_2[key]]

##### Get the true list of subjects
if file_recent(subject_text):
    subjects = []
    with open(subject_text, "r") as file:
        for line in file:
            subject = line.strip()  # Remove newline character
            subjects.append(subject)
else:
    subjects_folders_path = glob.glob(os.path.join(datapath, '*/'))
    min_size_bytes = 250 * 1024 * 1024
    subjects_folders_path_true, _ = get_large_folders(subjects_folders_path, min_size_bytes)
    subjects_folders = [os.path.split(subject_folder)[0].split('/')[-1] for subject_folder in
                        subjects_folders_path_true]
    subjects = ['S' + subj.split('_')[1] for subj in subjects_folders]

    with open(subject_text, "w") as file:
        for subject in subjects:
            file.write(subject + "\n")


Project = 'AD_Decode'
count_path = os.path.join(outpath, f'{Project}_count.csv')
txt_path = os.path.join(outpath, f'{Project}_correlations.txt')
anova_xlsx_path = os.path.join(outpath, f'{Project}_anova.xlsx')

#### The column with subject identifiers
subj_column = 'MRI_Exam'

txt_cors = ''

p_value_sig = 0.05

subj_val_base = {}
subj_cov_base = {}

metadata = pd.read_csv(metadata_path)
metadata['MRI_Exam'] = 'S0' + metadata['MRI_Exam'].astype(str)

countA = metadata.groupby(['genotype', 'sex', 'Risk',]).size().reset_index(name='count')
countB = metadata.groupby(['genotype', 'sex']).size().reset_index(name='count')
countB['Risk'] = 'All'

count = pd.concat([countA, countB], axis=0)

#print(count)
count.to_csv(count_path)

metadata_M = metadata[metadata['sex'] == 'M']
metadata_M = metadata[metadata['sex'] == 'F']

index_toremove = []
for gen in np.unique(metadata['genotype']):
    for sex in np.unique(metadata['sex']):
        temp = metadata[(metadata['genotype'] == gen) & (metadata['sex'] == sex)]
        index,bounds = detection_outlier(temp,metadata, 'age')
        index_toremove+=index

metadata_cleaned = metadata.drop(index_toremove)

data_f = copy.deepcopy(metadata_cleaned)

#### This segment is to get the correlation levels between a categorical and numerical continuous variable
### The first one should be categorical/binary, the second one needs to be continuous
cor_cols = ('Diet','Mass')
cor = data_f[cor_cols[0]].corr(data_f[cor_cols[1]])
correlation_coefficient, p_value = stats.pointbiserialr(cor_cols[0], cor_cols[1])
txt_cors+=f'The Biserial correlation between Diet and Mass is {correlation_coefficient} with p-value {p_value}'

"""
diet_num  = data_f[columns_tocor[0]]
diet_num[diet_num=="HFD"] = 1
diet_num[diet_num=="CTRL"] = 0
#diet_num = as.numeric(diet_num)
cor(as.numeric(data$Mass) , diet_num )
cor = biserial.cor(as.numeric(data$Mass), as.factor(data$Diet))
test_cor = cor.test(as.numeric(data$Mass), diet_num)
print(paste("The Biserial correlation between Diet and Mass is" ,cor ))
 print(paste(" with p-value",  test_cor$p.value))
"""

#### Create a correlation matrix of the columns you want to observe, then plot it out
cols_cormat = ['Diastolic_LV_Volume','Systolic_LV_Volume','Heart_Rate','Stroke_Volume','Ejection_Fraction','Cardiac_Output','cardiac_index']
cor_matrix = data_f[cols_cormat].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(cor_matrix.pivot("Var1", "Var2", "value"), annot=True, cmap="coolwarm")
plt.show()


#### Create a partially corrected correlation matrix of the columns you want to observe, then plot it out
#cols_cormat = ['Diastolic_LV_Volume','Systolic_LV_Volume','Heart_Rate','Stroke_Volume','Ejection_Fraction','Cardiac_Output','cardiac_index']
cor_matrix = pg.partial_corr(data=data_f[cols_cormat])
plt.figure(figsize=(8, 6))
sns.heatmap(cor_matrix.pivot("Var1", "Var2", "value"), annot=True, cmap="coolwarm")
plt.show()

#### Run a PCA on the data

# Standardize the selected data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_f[cols_cormat])

# Create a PCA object with the desired number of components
num_components = 2  # You can adjust the number of components
pca = PCA(n_components=num_components)

# Fit and transform the data
principal_components = pca.fit_transform(data_standardized)
principal_component_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
sns.scatterplot(data=principal_component_df, x='PC1', y='PC2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Results')
plt.grid(True)
plt.show()


cols_majorvar_all= ['age','sex','Risk','ethnicity']
#cols_tolm = ['Diastolic_LV_Volume','Systolic_LV_Volume','Heart_Rate','Stroke_Volume','Ejection_Fraction','Cardiac_Output','cardiac_index']
cols_tolm = []

formula_all = f'Y ~ '
for col in cols_majorvar_all:
    formula_all+=f'{col} * '
formula_all = formula_all[:-2]


major_covar = 'Age'
formulas = {'Geno3*Diet + Age':'' }

for col in cols_tolm:

    model = sm.OLS.from_formula(formula_all, data=data_f).fit()

    anova_result = anova_lm(model)
    anova_result.to_csv(anova_xlsx_path, sheet=col, append=True)

    cohen_f = pg.cohens_f(model, dv='your_dependent_variable', between='your_independent_variable')
    cohen_f.to_csv(anova_xlsx_path, sheet=f'{col}_cohen', append=True)


    model2 = sm.OLS.from_formula('Diet*Geno3*HN+Age', data=data_f).fit()
    anova_result = anova_lm(model2)
    pairwise_list = [('Diet', 'Geno3')]
    emmeans_result = emmeans(anova_result, pairwise_list, adjust="tukey")
    emmeans_result.to_csv(anova_xlsx_path, sheet=f'{col}_emmeans', append=True)




stat_types = ['mrtrixfa']

geno_strip = True
risk_strip = True