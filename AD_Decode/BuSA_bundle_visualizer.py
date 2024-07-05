import fury, os
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest #setup_view_test
import numpy as np
from DTC.tract_manager.tract_handler import ratio_to_str
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, \
    read_parameters_from_ini
from dipy.tracking.streamline import set_number_of_points
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence

#project = 'V0_9_10template_100_6_interhe_majority'
#project = 'V0_9_reg_precuneusleft_precuneus_right_split_3'
#project = 'V0_9_reg_precuneusleft_superiorparietalleft_split_3'
project = 'V0_9_reg_precuneusright_superiorparietalright_split_3'
#project = 'V_1_0_10template_100_6_interhe_majority'
project_summary_file = f'/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/{project}.ini'

proj_path = f'/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/{project}/'
figures_proj_path = os.path.join(proj_path, 'Figures')
figures_viz_path = os.path.join(figures_proj_path, 'Bundle_viz')
stats_proj_path = os.path.join(proj_path, 'stats')
mkcdir([figures_proj_path,figures_viz_path])

if not os.path.exists(project_summary_file):
    txt = f'Could not find configuration file at {project_summary_file}'
    raise Exception(txt)
else:
    params = read_parameters_from_ini(project_summary_file)

ratio = int(params['ratio'])
num_bundles = int(params['num_bundles'])

ratiostr = ratio_to_str(ratio, spec_all=False)

trk_proj_path = os.path.join(proj_path, 'trk_roi' + ratiostr)

contrast_background = 'fa'
subj = 'S01912'

# num_bundles = 6

distance = int(params['distance'])
points_resample = int(params['points_resample'])
bundle_points = int(params['bundle_points'])

feature2 = ResampleFeature(nb_points=bundle_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)

qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)

bundle_ids = np.arange(num_bundles)

# anat_path = f'/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_{contrast_background}.nii.gz'
anat_path = f'/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images/{subj}_{contrast_background}_to_MDT.nii.gz'
#added_mask = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/labels_MDT/atlas_MDT_masks_combined/precuneus_MDT.nii.gz'
added_mask = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/labels_MDT/atlas_MDT_masks_combined/lingual_MDT.nii.gz'
#added_mask = None

centroid_path = os.path.join(proj_path, 'Figures', 'Centroids')
lut_cmap = None

streamline_bundle = {}

bundle_data_dic = {}
bundle_data_twoside = {}
centroid_data_dic = {}

#bundle_full_list = [[0,1,2,3,4,5]]

#bundle_full_list = [[5]]
#bundle_full_list = [['0_split_3','1_split_3','2_split_3']]
#bundle_full_list = [['0_split_3']]
#bundle_full_list = [['4']]
#quad_genotype_sig = ['4','3_0','4_0_2','5_2_0_1','4_0_2_0','3_1_1_0']
#quad_age_sig = ['4','4_0','4_1','4_2','4_2_0','4_0_1','4_2_2','4_2_2_0','4_0_0_0','4_1_1_2','4_0_1_1','4_2_0_2','4_0_1_0','4_2_0_1']
bundle_full_list = [['4'],['3_0'],['4_0_2'],['5_2_0_1'],['4_0_2_0'],['3_1_1_0'],['4_0'],['4_1'],['4_2'],['4_2_0'],['4_0_1'],['4_2_2'],['4_2_2_0'],['4_0_0_0'],['4_1_1_2'],['4_0_1_1'],['4_2_0_2'],['4_0_1_0'],['4_2_0_1']]
bundle_full_list = [['3_1_1_0'],['4_0_2_0'],['1_0_1_2']]
bundle_full_list = [['1_2_1_0']]
bundle_full_list = [['5_1_2'],['3_0_2']]
#bundle_full_list = [['4'],['4_0_1'],['4_1_2'],['4_2_0'],['4_2_2'],['4_2_2_0']]

bundle_full_list = [['4_2_0'],['4_2_2_0']]
bundle_full_list = [['2_2_1_2'],['2'],['2_1'],['2_1_2'],['2_1_2_0'],['2_2'],['2_2_1'],['2_2_1_1'],['2_2_2']]
bundle_full_list = [['4_2_0_1'],['4_2_0'],['4_2_2_0'],['4_2_2']]
bundle_full_list = [['4_1_1_1'],['4_1_2_1']]
bundle_full_list = [['0_1_0_2'],['0_1_1_2'],['0_2_1_2']]
bundle_full_list = [['4'],['4_0'],['4_1'],['4_2'],['4_0_0'],['4_0_1'],['4_0_2'],['4_1_0'],['4_1_1'],['4_1_2'],['4_2_0'],
                    ['4_2_1'],['4_2_2'],['4_0_0_0'],['4_0_0_1'],['4_0_0_2'],['4_0_1_0'],['4_0_1_1'],['4_0_1_2'],
                    ['4_0_2_0'], ['4_0_2_1'], ['4_0_2_2'],['4_1_0_0'],['4_1_0_1'],['4_1_0_2'],['4_1_1_0'],['4_1_1_1'],['4_1_1_2'],
                    ['4_1_2_0'], ['4_1_2_1'], ['4_1_2_2'],['4_2_0_0'],['4_2_0_1'],['4_2_0_2'],['4_2_1_0'],['4_2_1_1'],['4_2_1_2'],
                    ['4_2_2_0'], ['4_2_2_1'], ['4_2_2_2']]
bundle_full_list = [['4_0'],['4_0_0'],['4_0_1'],['4_0_2'],['4_0_0_0'],['4_0_0_1'],['4_0_0_2'],['4_0_1_0'],['4_0_1_1'],['4_0_1_2'],
                    ['4_0_2_0'], ['4_0_2_1'], ['4_0_2_2']]
bundle_full_list = [['4_1'],['4_1_0'],['4_1_1'],['4_1_2'],['4_1_0_0'],['4_1_0_1'],['4_1_0_2'],['4_1_1_0'],['4_1_1_1'],['4_1_1_2'],
                    ['4_1_2_0'], ['4_1_2_1'], ['4_1_2_2']]
bundle_full_list = [['4_2'],['4_2_0'],['4_2_1'],['4_2_2'],['4_2_0_0'],['4_2_0_1'],['4_2_0_2'],['4_2_1_0'],['4_2_1_1'],['4_2_1_2'],
                    ['4_2_2_0'], ['4_2_2_1'], ['4_2_2_2']]
bundle_full_list = [['2_1_0_2']]
bundle_full_list = [['5_2_2']]
bundle_full_list = [['0','1','2','3','4','5']]
bundle_full_list = [['4']]
#bundle_full_list = [['4_0_0_0'],['4_0_1_1'],['4_0_1_0'],['4_1_1_2']]
bundle_full_list = [['4_0_0_0','4_0_1_1','4_0_1_0','4_1_1_2']]
bundle_full_list = [['0']]

pattern_lvl = '_4'

list1 = ['0','1','2','3','4','5']
list1 = ['4']
list2 = ['_0','_1','_2']

if pattern_lvl=='_1':
    bundle_full_list = [[x for x in list1]]
elif pattern_lvl == '_2':
    bundle_full_list = [[(x + y) for x in list1 for y in list2]]
elif pattern_lvl == '_3':
    bundle_full_list = [[(x + y + z) for x in list1 for y in list2 for z in list2]]
elif pattern_lvl == '_4':
    bundle_full_list = [[(x + y + z + k) for x in list1 for y in list2 for z in list2 for k in list2]]
elif pattern_lvl == '_4':
    bundle_full_list = [[x for x in list1] + [(x + y) for x in list1 for y in list2] + [(x + y + z) for x in list1
                                                                                       for y in list2 for z in list2] \
                       + [(x + y + z + k) for x in list1 for y in list2 for z in list2 for k in list2]]
    pattern_lvl = ''
else:
    bundle_full_list = [['0', '1', '2', '3', '4', '5']]


bundle_full_list = [['2_0_2']]

scene_dic = {}
interactive_dic = {}

scene_dic['x'] = None
interactive_dic['x'] = True
scene_dic['y'] = None
interactive_dic['y'] = True
scene_dic['z'] = None
interactive_dic['z'] = True
scene_dic['all'] = None
interactive_dic['all'] = True

sides = ['left', 'right']
#sides = ['all']

colorbar = False

planes = ['all']
#planes = ['x','y','z','all']
#planes = ['y']
toview = 'centroids'
toview = 'streamlines'

colortype = 'bundle' #['bundle' or 'fa' or 'bundle_lr']
#colortype = 'bundle_lr'

for bundle_ids in bundle_full_list:

    value_range = (0, 0.73)


    #planes = ['x','y','z']

    if colortype == 'bundle':
        color_contrast = 'bundle_coloring'
    elif colortype == 'bundle_lr':
        color_contrast = 'bundle_lr_coloring'
    elif colortype == 'fa':
        fa_scale_range = (0, 0.5)
        color_contrast = 'mrtrixfa'

    for bundle_id in bundle_ids:
        streamlines_bothsides = []
        for side in sides:
            if side != 'all':
                side_str = f'_{side}'
            else:
                side_str = ''
            filepath_bundle = os.path.join(trk_proj_path, f'{subj}_bundle{side_str}_{bundle_id}.trk')
            bundle_data = load_trk_remote(filepath_bundle, 'same', None)
            streamlines_numpoints = set_number_of_points(bundle_data.streamlines, nb_points=points_resample)
            streamlines_bothsides += streamlines_numpoints

            from dipy.segment.clustering import ClusterCentroid

            #centroid_file_path = os.path.join(centroid_path, f'centroid{side_str}_bundle_{bundle_id.split("_split")[0]}.trk') #eventually we'll make the centroids and bundle ids consistent again
            centroid_file_path = os.path.join(centroid_path,
                                              f'centroid_bundle{side_str}_{bundle_id.split("_split")[0]}.trk')
            #instead of having centroid_bundle_0.trk and {subj}_bundle_0_split_3.trk
            centroid_data = load_trk_remote(centroid_file_path, 'same', None)
            centroid_data_dic[side, bundle_id] = qb_test.cluster(centroid_data.streamlines)[0]

            # bundle_data_dic[side, bundle_id] = bundle_data.streamlines

            try:
                # bundle_data_dic[side, bundle_id] = qb_test.cluster(streamlines_numpoints)[0]
                streamlines_added = streamlines_numpoints
            except IndexError:
                streamlines_added = qb_test.cluster(centroid_data.streamlines)[0]
            bundle_data_dic[side, bundle_id] = streamlines_added
            if colortype == 'bundle':
                try:
                    bundle_data_twoside[bundle_id] = ArraySequence(list(bundle_data_twoside[bundle_id])+list(streamlines_added))
                except KeyError:
                    bundle_data_twoside[bundle_id] = streamlines_added
            elif colortype == 'bundle_lr':
                bundle_data_twoside[bundle_id,side] = streamlines_added

    for plane in planes:

        if plane == 'x':
            plane_txt = 'sagittal'
        elif plane == 'y':
            plane_txt = 'coronal'
        elif plane == 'z':
            plane_txt = 'axial'
        else:
            plane_txt = ''

        if toview == 'streamlines':

            linewidth = 2

            #list_bundles = list(bundle_data_dic.values())
            list_bundles = []

            if colortype=='bundle':
                for key in bundle_data_twoside.keys():
                    if key in bundle_ids:
                        list_bundles.append(bundle_data_twoside[key])
            elif colortype=='bundle_lr':
                for key in bundle_data_twoside.keys():
                    if key[0] in bundle_ids:
                        list_bundles.append(bundle_data_twoside[key])
            elif colortype=='fa':
                for key in bundle_data_dic.keys():
                    if key[1] in bundle_ids:
                        list_bundles.append(bundle_data_dic[key])

            from DTC.file_manager.computer_nav import glob_remote, load_df_remote
            import glob
            from dipy.viz import window, actor

            column_names = []
            for i in range(0, points_resample):
                column_names.append(f"point_{i}_{color_contrast}")

            if colortype == 'fa':
                bundle_fa_dic = {}
                for bundle_id in bundle_ids:
                    for side in sides:
                        if side != 'all':
                            side_str = f'_{side}'
                        else:
                            side_str = ''
                        bundles_fa = []
                        stats_subj_path = glob.glob(os.path.join(stats_proj_path, f'{subj}_bundle{side_str}_{bundle_id.split("_split")[0]}.xlsx'))[0]
                        stats_df = load_df_remote(stats_subj_path, None)
                        bundle_fa_dic[side, bundle_id] = stats_df[column_names].values


            record_path = os.path.join(figures_viz_path,
                                       f'{subj}_{np.size(bundle_ids)}_bundles{ratiostr}_{color_contrast}_{plane_txt}{bundle_ids[0]}.png')

            record_path = os.path.join(figures_viz_path,f'{subj}_bundles{ratiostr}_{color_contrast}_id_{bundle_ids[0]}_{plane_txt}.png')

            if colortype == 'bundle':
                lut_cmap = fury.colormap.distinguishable_colormap(nb_colors=np.size(bundle_ids))
                objectvals = None
            elif colortype == 'bundle_lr':
                lut_cmap = fury.colormap.distinguishable_colormap(nb_colors=2*np.size(bundle_ids))
                objectvals = None
            elif colortype == 'fa':
                lut_cmap = actor.colormap_lookup_table(
                    scale_range=fa_scale_range)
                objectvals = list(bundle_fa_dic.values())

            """
            scene_dic[plane] = setup_view_test(list_bundles, colors=lut_cmap, ref=anat_path, world_coords=True,
                                          objectvals=list_bundles_fa,
                                          colorbar=colorbar, record=record_path, scene=scene_dic[plane], plane=plane,
                                          interactive=interactive_dic[plane], value_range=value_range,
                                          linewidth=linewidth, addedmask = added_mask)
            """
            scene_dic[plane] = setup_view(list_bundles, colors=lut_cmap, ref=anat_path, world_coords=True,
                                               objectvals=objectvals,
                                               colorbar=colorbar, record=record_path, scene=scene_dic[plane],
                                               plane=plane,
                                               interactive=interactive_dic[plane], value_range=value_range,
                                               linewidth=linewidth, addedmask=added_mask)

            interactive_dic[plane] = False

        elif toview == 'centroids':

            linewidth = 20

            list_bundles = list(centroid_data_dic.values())

            record_path = os.path.join(figures_viz_path,
                                       f'{subj}_bundles{ratiostr}_{color_contrast}_{plane_txt}_{bundle_ids[0]}_centroids.png')

            scene_dic[plane] = setup_view(list_bundles, colors=lut_cmap, ref=anat_path, world_coords=True,
                               objectvals=None,
                               colorbar=colorbar, record=record_path, scene=scene_dic[plane], plane=plane,
                               interactive=interactive_dic[plane], value_range=value_range, linewidth=linewidth)
            interactive_dic[plane] = False

            print(f'Saved at {record_path}')

        """
        bundles_streamlines = []
        bundles_fa = []
        bundles_fa_mean = []
        for bundle_streamlines in list_bundles:
            bundle_fa = []
            for idx in bundle_streamlines.indices:
                bundle_fa.append(fa_lines_idx[idx])
            bundles_fa.append(bundle_fa)
            bundles_fa_mean.append(np.mean(bundle_fa))
        coloring_vals = bundles_fa
        trkobject = bundles_streamlines
        lut_cmap = actor.colormap_lookup_table(
            scale_range=fa_scale_range)




        scene = setup_view(trkobject, colors=lut_cmap, ref=anat_path, world_coords=True, objectvals=coloring_vals,
                           colorbar=colorbar, record=record_path, scene=scene, plane=plane, interactive=interactive)
        """

