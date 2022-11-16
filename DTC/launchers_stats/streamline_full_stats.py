from dipy.io.streamline import load_trk
import os
import pickle
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from DTC.tract_manager.tract_handler import ratio_to_str
from itertools import compress
import numpy as np
import nibabel as nib, socket
from DTC.file_manager.file_tools import mkcdir
from DTC.tract_manager.streamline_nocheck import load_trk as load_trk_spe
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature, AveragePointwiseEuclideanMetric
import warnings
from dipy.align.streamlinear import StreamlineLinearRegistration
import copy
import pandas as pd
from dipy.tracking import utils
from dipy.segment.bundles import bundle_shape_similarity
from scipy import stats
import dill  # pip install dill --user
import csv
from math import nan

computer_name = socket.gethostname().split('.')[0]
mainpath = getremotehome(computer_name)
if 'os' in computer_name:
    ROI_legends = "/mnt/paros_MRI/jacques/atlases/IITmean_RPI/IITmean_RPI_index.xlsx"
elif 'rini' in computer_name:
    ROI_legends = "/Volumes/Data/Badea/ADdecode.01/Analysis/atlases/IITmean_RPI/IITmean_RPI_index.xlsx"
elif 'de' in computer_name:
    ROI_legends = "/mnt/munin6/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx"
else:
    raise Exception('No other computer name yet')


project = 'AMD'

fixed = True
record = ''

inclusive = False
symmetric = True
write_txt = True
ratio = 1
top_percentile = 100
num_bundles = 10
distance = 3
num_points = 50

if project == 'AD_Decode':
    #genotype_noninclusive
    #target_tuples = [(9, 1), (24, 1), (58, 57), (64, 57), (22, 1)]
    #genotype_noninclusive_volweighted_fa
    target_tuples = [(9, 1), (57, 9), (61, 23), (84, 23), (80, 9)]

    #sex_noninclusive
    #target_tuples = [(64, 57), (58, 57), (9, 1), (64, 58), (80,58)]
    #target_tuples = [(64,57)]
    #sex_noninclusive_volweighted_fa
    #target_tuples = [(58, 24), (58, 30), (64, 30), (64, 24), (58,48)]

    #target_tuples = [(9,1)]
    #groups = ['Male', 'Female']
    groups = ['APOE3', 'APOE4']

    mainpath = os.path.join(mainpath, project, 'Analysis')
    anat_path = os.path.join(mainpath,'/../mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_b0.nii.gz')
    space_param = '_MDT'

    control = groups[0]
    non_control = groups[1]

elif project == 'AMD':
    groups_all = ['Paired 2-YR Control','Paired 2-YR AMD','Paired Initial Control','Paired Initial AMD',
                  'Initial AMD', 'Initial Control']
    groups_set = {'Initial':[2,3],'2Year':[0,1]}
    target_tuples_all = {'Initial':[(62, 28), (58, 45)],'2Year':[(28, 9), (62, 1)]}
    target_tuples_all = {'Initial': [(62, 28), (58, 45),(77, 43), (61, 29)], '2Year': [(28, 9), (62, 1),(77, 43), (61, 29)]}
    group_select = '2Year'
    groups = [groups_all[x] for x in groups_set[group_select]]
    target_tuples = target_tuples_all[group_select]
    #groups = ['Paired Initial Control', 'Paired Initial AMD']
    #groups = ['Paired 2-YR Control', 'Paired 2-YR AMD']
    target_tuples = [(62, 28), (58, 45), (28, 9), (62, 1), (77, 43), (61, 29)]
    #target_tuples = [(58, 45)]
    mainpath = os.path.join(mainpath, project)
    anat_path = os.path.join(mainpath,'../../mouse/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_Control_n72_i6/median_images/MDT_dwi.nii.gz')

    space_param = '_affinerigid'

    control = groups[0]
    non_control = groups[1]

    if 'AMD' in control:
        raise Exception('Again with this nonsense!')


fixed = True
record = ''

selection = 'num_streams'
coloring = 'bundles_coloring'
references = ['length','fa','md','ad','rd']
#references = ['fa']

inclusive = False
symmetric = True
write_txt = True
ratio = 1
cutoffref = 0
top_percentile = 100
min_ref = 0

write_stats = False
registration = False
overwrite = True

changewindow_eachtarget = False

if inclusive:
    inclusive_str = '_inclusive'
else:
    inclusive_str = '_non_inclusive'

if symmetric:
    symmetric_str = '_symmetric'
else:
    symmetric_str = '_non_symmetric'

# if fixed:
#    fixed_str = '_fixed'
# else:
#    fixed_str = ''



ratio_str = ratio_to_str(ratio)
print(ratio_str)
if ratio_str == '_all':
    folder_ratio_str = ''
else:
    folder_ratio_str = ratio_str.replace('_ratio', '')

_, _, index_to_struct, _ = atlas_converter(ROI_legends)

# figures_path = '/Volumes/Data/Badea/Lab/human/AMD/Figures_MDT_non_inclusive/'
# centroid_folder = '/Volumes/Data/Badea/Lab/human/AMD/Centroids_MDT_non_inclusive/'
figures_path = os.path.join(mainpath, f'Figures{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
centroid_folder = os.path.join(mainpath, f'Centroids{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
trk_folder = os.path.join(mainpath, f'Centroids{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
stats_folder = os.path.join(mainpath, f'Statistics_allregions{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}',f'stats_allstreams_min{str(min_ref)}_intervals')

mkcdir([figures_path, centroid_folder, stats_folder])

# groups = ['Initial AMD', 'Paired 2-YR AMD', 'Initial Control', 'Paired 2-YR Control', 'Paired Initial Control',
#          'Paired Initial AMD']

# anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_Control_n72_i6/median_images/MDT_dwi.nii.gz'


# superior frontal right to cerebellum right

#set parameter
feature = ResampleFeature(nb_points=num_points)
metric = AveragePointwiseEuclideanMetric(feature=feature)

scene = None
selection = 'num_streams'

test_mode = False
add_bcoherence = True
add_weighttobundle = True
make_stats = True

for target_tuple in target_tuples:

    interactive = True

    print(target_tuple[0], target_tuple[1])
    region_connection = index_to_struct[target_tuple[0]] + '_to_' + index_to_struct[target_tuple[1]]
    print(region_connection)

    if write_txt:
        text_path = os.path.join(figures_path, region_connection + '_stats.txt')
        testfile = open(text_path, "w")
        testfile.write("Parameters for groups\n")
        testfile.close()

    if changewindow_eachtarget:
        firstrun = True

    selected_bundles = {}
    selected_centroids = {}
    selected_sizes = {}
    streamlines = {}
    num_bundles_group = {}

    ref_lines = {}
    ref_points = {}

    for group in groups:

        selected_bundles[group] = []
        selected_centroids[group] = []
        selected_sizes[group] = []

        print(f'Setting up group {group}')
        group_str = group.replace(' ', '_')

        centroid_file_path = os.path.join(centroid_folder,
                                          group_str + space_param + ratio_str + '_' + region_connection + '_centroid.py')

        trk_path = os.path.join(trk_folder,
                                group_str + space_param + ratio_str + '_' + region_connection + '_streamlines.trk')

        # '/Volumes/Data/Badea/Lab/human/AD_Decode/Analysis/Centroids_MDT_non_inclusive_symmetric_100/APOE4_MDT_ratio_100_ctx-lh-inferiorparietal_left_to_ctx-lh-inferiortemporal_left_streamlines.trk'


        if os.path.exists(trk_path):
            try:
                streamlines_data = load_trk(trk_path, 'same')
            except:
                streamlines_data = load_trk_spe(trk_path, 'same')
        streamlines[group] = streamlines_data.streamlines

        for ref in references:
            if ref != 'length':
                ref_path_lines = os.path.join(centroid_folder,
                                       group_str + space_param + ratio_str + '_' + region_connection + f'_{ref}_lines.py')
                ref_path_points = os.path.join(centroid_folder,
                                       group_str + space_param + ratio_str + '_' + region_connection + f'_{ref}_points.py')

                if os.path.exists(ref_path_points):
                    with open(ref_path_points, 'rb') as f:
                        ref_points[group,ref] = pickle.load(f)
                else:
                    txt = f'Could not find file {ref_path_points} for group {group} reference {ref}'
                    raise Exception(txt)

                if os.path.exists(ref_path_lines):
                    with open(ref_path_lines, 'rb') as f:
                        ref_lines[group,ref] = pickle.load(f)
                else:
                    txt = f'Could not find file {ref_path_lines} for group {group} reference {ref}'
                    raise Exception(txt)
            else:
                ref_lines[group,ref] = []
                ref_points[group,ref] = []
                for i,streamline in enumerate(streamlines[group]):
                    ref_lines[group, ref].append(list(utils.length([streamline]))[0])
                    ref_points[group,ref].append(list(utils.length([streamline])) * np.ones((num_points, 1)))


        if min_ref>0:
            select_streams = ref_lines[group,references[cutoffref]] > np.float64(min_ref)
            streamlines[group] = list(compress(streamlines[group], select_streams))
            streamlines[group] = nib.streamlines.ArraySequence(streamlines[group])

            for ref in references:
                ref_lines[group,ref] = list(compress(ref_lines[group,ref], select_streams))

        if top_percentile<100:
            cutoff = np.percentile(ref_lines[group,references[cutoffref]], 100 - top_percentile)
            select_streams = ref_lines[group,references[cutoffref]] > cutoff
            streamlines[group] = list(compress(streamlines[group], select_streams))
            streamlines[group] = nib.streamlines.ArraySequence(streamlines[group])

            for ref in references:
                ref_lines[group,ref] = list(compress(ref_lines[group,ref], select_streams))



    streamlinestats_path = os.path.join(stats_folder, group_select + '_' + region_connection + ratio_str + f'_allstreamline_stats.csv')
    print(np.shape(streamlines[control]))
    print(np.shape(ref_lines[control,'length']))

    if make_stats and (not os.path.exists(streamlinestats_path) or overwrite):
        if overwrite and os.path.exists(streamlinestats_path):
            os.remove(streamlinestats_path)

        csv_columns = {}

        first_elements = ['Number of streamlines']
        first_element_size = np.size(first_elements)
        elements = first_elements
        for ref in references:
            elements = elements + [f'Mean {ref}',f'STD {ref}',f'Median {ref}',f'Mean lower range {ref}', f'Mean higher range {ref}',f'Cohen {ref}',f'Statistic {ref}',f'P-value {ref}']
        num_stats = 8

        col = 0
        for col,element in enumerate(elements):
            csv_columns.update({col: element})


        streamline_stats = np.zeros([2, np.size(elements)])

        functions = [lambda x,: np.mean(x), lambda x: np.std(x), lambda x: np.median(x)]
        print(np.shape(streamlines[control]))
        streamline_stats[0, 0] = np.shape(streamlines[control])[0]
        streamline_stats[1, 0] = np.shape(streamlines[non_control])[0]
        for i, ref in enumerate(references):
            # csv_columns.update({col: element})
            ref_control = ref_lines[control,ref]
            ref_nocontrol = ref_lines[non_control, ref]            
            statistic, pvalue = stats.ttest_ind(np.ravel(ref_lines[control,ref]),
                                                np.ravel(ref_lines[non_control,ref]))
            cohen = (np.mean(ref_lines[control,ref]) - np.mean(ref_lines[non_control,ref]))/np.sqrt((np.var(ref_lines[control,ref])+np.var(ref_lines[non_control,ref]))/2)
            for j, function in enumerate(functions):
                streamline_stats[0, first_element_size + i * num_stats + j] = function(ref_lines[control,ref])
                streamline_stats[1, first_element_size + i * num_stats + j] = function(ref_lines[non_control,ref])

            j+=1
            streamline_stats[0, first_element_size + i * num_stats + j] = np.mean(ref_control) - 1.96*(np.std(ref_control)/np.sqrt(streamline_stats[0,0]))
            streamline_stats[1, first_element_size + i * num_stats + j] =  np.mean(ref_nocontrol) - 1.96*(np.std(ref_nocontrol)/np.sqrt(streamline_stats[0,0])) 
            j+=1
            streamline_stats[0, first_element_size + i * num_stats + j] = np.mean(ref_control) + 1.96*(np.std(ref_control)/np.sqrt(streamline_stats[0,0]))
            streamline_stats[1, first_element_size + i * num_stats + j] =  np.mean(ref_nocontrol) + 1.96*(np.std(ref_nocontrol)/np.sqrt(streamline_stats[0,0]))
            j += 1
            streamline_stats[0, first_element_size + i * num_stats + j] = cohen
            streamline_stats[1, first_element_size + i * num_stats + j] = nan
            j += 1
            streamline_stats[0, first_element_size + i * num_stats + j] = statistic
            streamline_stats[1, first_element_size + i * num_stats + j] = nan
            j += 1
            streamline_stats[0, first_element_size + i * num_stats + j] = pvalue
            streamline_stats[1, first_element_size + i * num_stats + j] = nan
            j+=1

        streamline_statsDF = pd.DataFrame(streamline_stats)
        streamline_statsDF.rename(index=str, columns=csv_columns)

        header_streamlinestats = elements
        streamline_statsDF.index = [control, non_control]
        streamline_statsDF.to_csv(streamlinestats_path, header=header_streamlinestats)
