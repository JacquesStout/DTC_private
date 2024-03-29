from dipy.io.streamline import load_trk
import os
import pickle
from nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
from tract_manager.tract_handler import ratio_to_str
from itertools import compress
import numpy as np
import nibabel as nib, socket
from file_manager.file_tools import mkcdir
from tract_manager.streamline_nocheck import load_trk as load_trk_spe
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
num_bundles = 1
distance = 100
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

    #target_tuples = target_tuples_all[group_select]
    #groups = ['Paired Initial Control', 'Paired Initial AMD']
    #groups = ['Paired 2-YR Control', 'Paired 2-YR AMD']
    target_tuples = [(62, 28), (58, 45), (28, 9), (62, 1), (77, 43), (61, 29)]
    target_tuples = [(61, 29)]
    #target_tuples = [(58, 45)]
    mainpath = os.path.join(mainpath, project)
    anat_path = os.path.join(mainpath,'../../mouse/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_Control_n72_i6/median_images/MDT_dwi.nii.gz')

    space_param = '_affinerigid'
    groupings = ['Initial', '2Year']

fixed = True
record = ''

inclusive = False
symmetric = True
write_txt = True
ratio = 1
top_percentile = 100

selection = 'num_streams'
coloring = 'bundles_coloring'
references = ['fa','md','ad','rd']
#references = ['fa']
cutoffref = 0

write_stats = False
registration = False
overwrite = False

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
if distance==3:
    stats_folder = os.path.join(mainpath, f'Statistics_allregions{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
else:
    stats_folder = os.path.join(mainpath, f'Statistics_allregions_distance_{str(distance)}{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')

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


if add_bcoherence:
    from dipy.tracking.fbcmeasures import FBCMeasures
    #from fbcmeasures import FBCMeasures
    from dipy.denoise.enhancement_kernel import EnhancementKernel
    D33 = 1
    D44 = 0.02
    t = 1
    k = EnhancementKernel(D33, D44, t)

interactive = True

for target_tuple in target_tuples:

    centroids = {}

    for group_select in groupings:

        groups = [groups_all[x] for x in groups_set[group_select]]
        control = groups[0]
        non_control = groups[1]

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

            stats_path = os.path.join(stats_folder,
                                      group_str + space_param + ratio_str + '_' + region_connection + '_bundle_stats.xlsx')

            if not os.path.exists(stats_path) or overwrite:
                if os.path.exists(stats_path):
                    os.remove(stats_path)
                import xlsxwriter
                workbook = xlsxwriter.Workbook(stats_path)
                worksheet = workbook.add_worksheet()
                l=1
                worksheet.write(0,l,'Number streamlines')
                l+=1
                for ref in references:
                    worksheet.write(0,l, ref + ' mean')
                    worksheet.write(0,l+1, ref + ' min')
                    worksheet.write(0,l+2, ref + ' max')
                    worksheet.write(0,l+3, ref + ' std')
                    l=l+4
            else:
                print(f'The file {stats_path} already exists and no overwrite enabled: skipping')
                #continue

            centroid_file_path = os.path.join(centroid_folder,
                                              group_str + space_param + ratio_str + '_' + region_connection + '_centroid.py')

            trk_path = os.path.join(trk_folder,
                                    group_str + space_param + ratio_str + '_' + region_connection + '_streamlines.trk')

            # '/Volumes/Data/Badea/Lab/human/AD_Decode/Analysis/Centroids_MDT_non_inclusive_symmetric_100/APOE4_MDT_ratio_100_ctx-lh-inferiorparietal_left_to_ctx-lh-inferiortemporal_left_streamlines.trk'

            for ref in references:
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

            if os.path.exists(trk_path):
                try:
                    streamlines_data = load_trk(trk_path, 'same')
                except:
                    streamlines_data = load_trk_spe(trk_path, 'same')
            streamlines[group] = streamlines_data.streamlines

            if top_percentile<100:
                cutoff = np.percentile(ref_lines[group,references[cutoffref]], 100 - top_percentile)
                select_streams = ref_lines[group,references[cutoffref]] > cutoff
                streamlines[group] = list(compress(streamlines[group], select_streams))
                streamlines[group] = nib.streamlines.ArraySequence(streamlines[group])

                for ref in references:
                    if np.shape(streamlines[group])[0] != np.shape(ref_lines[group][ref])[0]:
                        raise Exception('Inconsistency between streamlines and fa lines')
                    ref_lines[group,ref] = list(compress(ref_lines[group,ref], select_streams))

            group_qb = QuickBundles(threshold=distance, metric=metric)
            group_clusters = group_qb.cluster(streamlines[group])
            #group2_qb = QuickBundles(threshold=distance2, metric=metric2)
            #group2_clusters = group2_qb.cluster(groupstreamlines2)

            num_bundles_group[group]=0
            if selection =='num_streams':
                num_streamlines = [np.shape(cluster)[0] for cluster in group_clusters.clusters]
                num_streamlines = group_clusters.clusters_sizes()
                top_bundles = sorted(range(len(num_streamlines)), key=lambda i: num_streamlines[i], reverse=True)[:]
            for bundle in top_bundles:
                selected_bundles[group].append(group_clusters.clusters[bundle])
                selected_centroids[group].append(group_clusters.centroids[bundle])
                selected_sizes[group].append(group_clusters.clusters_sizes()[bundle])
                num_bundles_group[group]+=1
            bun_num = 0

            bundles_ref = {}
            bundles_ref_mean = {}
            for ref in references:
                bundles_ref[ref] = []
                bundles_ref_mean[ref] = []

            for bundle in selected_bundles[group]:
                for ref in references:
                    bundle_ref = []
                    for idx in bundle.indices:
                        bundle_ref.append(ref_lines[group,ref][idx])
                    bundles_ref[ref].append(bundle_ref)
                    bundles_ref_mean[ref].append(np.mean(bundle_ref))

            empty_bundles = {}
            for ref in references:
                empty_bundles[ref] = 0

            if write_stats:
                bun_num=0
                for bundle in top_bundles:
                    l=0
                    worksheet.write(bun_num+1, l, bun_num+1)
                    l+=1
                    worksheet.write(bun_num + 1, l, np.shape(group_clusters.clusters[bundle])[0])
                    l+=1
                    for ref in references:
                        if np.mean(bundles_ref[ref][bun_num])==0:
                            empty_bundles[ref] += 1
                        worksheet.write(bun_num+1, l+0, np.mean(bundles_ref[ref][bun_num]))
                        worksheet.write(bun_num+1, l+1, np.min(bundles_ref[ref][bun_num]))
                        worksheet.write(bun_num+1, l+2, np.max(bundles_ref[ref][bun_num]))
                        worksheet.write(bun_num+1, l+3, np.std(bundles_ref[ref][bun_num]))
                        l = l + 4
                    bun_num+=1
                workbook.close()
            for ref in references:
                if empty_bundles[ref]>0:
                    print(f'Found {empty_bundles} empty bundles out of {np.size(top_bundles)} for {ref} in group {group} for {region_connection}')

        #if test_mode:
        #    filename = '/Users/jas/jacques/globalsave.pkl'
        #    dill.dump_session(filename)

        if registration:
            srr = StreamlineLinearRegistration()
            for streamline,i in enumerate(selected_centroids[non_control]):
                srm = srr.optimize(static=selected_centroids[control], moving=streamline)
                streamlines[control][i] = srm.transform(streamline)

        from dipy.segment.metric import mdf

        #dist_all = np.zeros((np.size(selected_bundles[control]), np.size(selected_bundles[non_control])))
        dist_all = np.zeros((num_bundles, num_bundles))

        print(np.shape(num_bundles))
        print(np.shape(selected_bundles[group]))
        print(np.shape(selected_sizes[control]))
        print(np.arange(num_bundles))

        if test_mode:
            top_idx_group_control = sorted(range(len(selected_sizes[control])),
                                    key=lambda i: selected_sizes[group][i], reverse=True)[:num_bundles]
            top_idx_group_noncontrol = sorted(range(len(selected_sizes[non_control])),
                                    key=lambda i: selected_sizes[group][i], reverse=True)[:num_bundles]

            if not np.all(top_idx_group_control == np.arange(num_bundles)) or not np.all(top_idx_group_noncontrol == np.arange(num_bundles)):
                warnings.warn('There is indeed a difference between the two')
            else:
                print('no difference between the two')

        groupbundlesizes = np.zeros((num_bundles, 2))
        groupbundlesizes[:,0] = [np.shape(selected_bundles[control][bundle])[0] for bundle in np.arange(num_bundles)]
        groupbundlesizes[:,1] = [np.shape(selected_bundles[non_control][bundle])[0] for bundle in np.arange(num_bundles)]

        num_streamlines_control = np.shape(streamlines[control])[0]
        num_streamlines_noncontrol = np.shape(streamlines[non_control])[0]

        if groupbundlesizes[:,0]<num_streamlines_control:
            print(f'Bundle was created using {str(groupbundlesizes[:,0]/num_streamlines_control)}')
        if groupbundlesizes[:,1]<num_streamlines_noncontrol:
            print(f'Bundle was created using {str(groupbundlesizes[:,0]/num_streamlines_noncontrol)}')
        streamline_spread = {}
        streamline_spread[control] = np.zeros(num_streamlines_control)
        streamline_spread[non_control] = np.zeros(num_streamlines_noncontrol)
        #streamline_spread[non_control] = np.zeros(np.size(streamlines[group]))

        for group in groups:
            for i,streamline in enumerate(streamlines[group]):
                streamline_spread[group][i] = mdf(selected_bundles[group][0].centroid, streamline)

        csv_streamlinedistance = os.path.join(stats_folder,
                                          group_select + '_' + region_connection + ratio_str + f'_allstreamlines_distancetocenter.csv')

        overwrite = True
        if not os.path.exists(csv_streamlinedistance) or overwrite:
            csv_columns = {}
            singlegroup_elements = []
            elements = ['Mean streamline distance', 'Low streamline distance', 'High streamline distance',
                        'Std streamline distance', 'Median streamline distance', 'Minimum streamline distance', 'Maximum streamline distance']

            dualgroup_elements = ['Cohen Effect streamline distance', 'Statistic ttest streamline distance', 'Pvalue streamline distance']

            functions = [lambda x: np.mean(x), lambda x: np.mean(x) - 1.96 * (np.std(x) / np.sqrt(np.size(x))),
                         lambda x: np.mean(x) + 1.96 * (np.std(x) / np.sqrt(np.size(x))), lambda x: np.std(x),
                         lambda x: np.median(x), lambda x: np.min(x), lambda x: np.max(x)]

            streamlinedistance = np.zeros(
                [2, np.size(singlegroup_elements) + np.size(elements) + np.size(dualgroup_elements)])
            col = 0

            for element, function in zip(elements, functions):
                csv_columns.update({col: element})
                for row, group in enumerate(groups):
                    try:
                        streamlinedistance[row, col] = function(streamline_spread[group])
                    except:
                        print('hi')
                col += 1

            statistic, pvalue = stats.ttest_ind(np.ravel(streamline_spread[control]),
                                                np.ravel(streamline_spread[non_control]))
            cohen = (np.mean(np.ravel(streamline_spread[control])) -
                     np.mean(np.ravel(streamline_spread[non_control]))) / \
                    np.sqrt((np.var(np.ravel(streamline_spread[control])) + np.var(
                        np.ravel(streamline_spread[non_control]))) / 2)

            csv_columns.update({col: dualgroup_elements[col - np.size(elements) - np.size(singlegroup_elements)]})
            streamlinedistance[0, col] = cohen
            streamlinedistance[1, col] = nan
            col += 1
            csv_columns.update({col: dualgroup_elements[col - np.size(elements) - np.size(singlegroup_elements)]})
            streamlinedistance[0, col] = statistic
            streamlinedistance[1, col] = nan
            col += 1
            csv_columns.update({col: dualgroup_elements[col - np.size(elements) - np.size(singlegroup_elements)]})
            streamlinedistance[0, col] = pvalue
            streamlinedistance[1, col] = nan
            col += 1

            streamlinedistanceDF = pd.DataFrame(streamlinedistance)
            streamlinedistanceDF.rename(index=str, columns=csv_columns)
            header_bundle = singlegroup_elements + elements + dualgroup_elements
            streamlinedistanceDF.index = [control, non_control]
            streamlinedistanceDF.to_csv(csv_streamlinedistance, header=header_bundle)

        centroids[group_select] = [selected_bundles[control][0].centroid, selected_bundles[non_control][0].centroid]

    import fury

    #connection_str = group_select + space_param + ratio_str + '_' + region_connection
    connection_str = 'allgroups' + space_param + ratio_str + '_' + region_connection

    trkobject = nib.streamlines.ArraySequence(centroids[groupings[0]] + centroids[groupings[1]])
    #trkobject = selected_bundles[control][0].centroid, selected_bundles[non_control][0].centroid]
    #trkobject = nib.streamlines.ArraySequence(trkobject)
    lut_cmap = None
    anat_path = os.path.join(mainpath,'../../mouse/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_Control_n72_i6/median_images/MDT_dwi.nii.gz')
    coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=4)
    colorbar = False
    figures_centroids_path = os.path.join(mainpath,
                 f'Figures_distance_{str(distance)}{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
    record_path = os.path.join(stats_folder,
                               connection_str + f'_allstreamlines.png')
    plane = 'x'
    from DTC.visualization_tools.tract_visualize import setup_view
    scene = setup_view(trkobject, colors=lut_cmap, ref=anat_path, world_coords=True, objectvals=coloring_vals,
                       colorbar=colorbar, record=record_path, scene=scene, plane=plane, interactive=interactive)

    interactive = False

    trkobject = nib.streamlines.ArraySequence([centroids[groupings[0]][0],centroids[groupings[1]][0]])
    coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=2)
    connection_str = 'Ctrl' + space_param + ratio_str + '_' + region_connection
    record_path = os.path.join(stats_folder,
                               connection_str + f'_allstreamlines.png')
    scene = setup_view(trkobject, colors=lut_cmap, ref=anat_path, world_coords=True, objectvals=coloring_vals,
                       colorbar=colorbar, record=record_path, scene=scene, plane=plane, interactive=interactive)

    trkobject = nib.streamlines.ArraySequence([centroids[groupings[0]][1],centroids[groupings[1]][1]])
    connection_str = 'AMD' + space_param + ratio_str + '_' + region_connection
    record_path = os.path.join(stats_folder,
                               connection_str + f'_allstreamlines.png')
    scene = setup_view(trkobject, colors=lut_cmap, ref=anat_path, world_coords=True, objectvals=coloring_vals,
                       colorbar=colorbar, record=record_path, scene=scene, plane=plane, interactive=interactive)



