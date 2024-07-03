import fury,os
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
import numpy as np
from DTC.tract_manager.tract_handler import ratio_to_str
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, read_parameters_from_ini
from dipy.tracking.streamline import set_number_of_points

project = 'V0_9_10template_100_6_interhe_majority'
project_summary_file = f'/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/{project}.ini'

proj_path = f'/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/{project}/'
figures_proj_path = os.path.join(proj_path, 'Figures')
figures_viz_path = os.path.join(figures_proj_path, 'Bundle_viz')
stats_proj_path = os.path.join(proj_path, 'stats')


if not os.path.exists(project_summary_file):
    txt = f'Could not find configuration file at {project_summary_file}'
    raise Exception(txt)
else:
    params = read_parameters_from_ini(project_summary_file)


ratio = int(params['ratio'])
num_bundles = int(params['num_bundles'])

ratiostr = ratio_to_str(ratio,spec_all=False)

trk_proj_path = os.path.join(proj_path, 'trk_roi'+ratiostr)

contrast_background = 'fa'
subj = 'S01912'

#num_bundles = 6

distance= int(params['distance'])
points_resample = int(params['points_resample'])
bundle_points = int(params['bundle_points'])

feature2 = ResampleFeature(nb_points=bundle_points)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)

qb_test = QuickBundles(threshold=distance, metric=metric2, max_nb_clusters=1)

bundle_ids = np.arange(num_bundles)

#anat_path = f'/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_{contrast_background}.nii.gz'
anat_path = f'/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_images/{subj}_{contrast_background}_to_MDT.nii.gz'

centroid_path = os.path.join(proj_path,'Figures','Centroids')
lut_cmap = None
lut_cmap = fury.colormap.distinguishable_colormap(nb_colors=np.size(bundle_ids)*2)

streamline_bundle = {}

bundle_data_dic = {}
bundle_data_twoside = {}
centroid_data_dic = {}

bundle_full_list = [[0],[1],[3],[4],[5]]
bundle_full_list = [[5]]
bundle_full_list = [[0,1,2,3,4,5]]
scene_dic = {}
interactive_dic = {}

scene_dic['x'] = None
interactive_dic['x'] = True
scene_dic['y'] = None
interactive_dic['y'] = True
scene_dic['z'] = None
interactive_dic['z'] = True

for bundle_ids in bundle_full_list:

    value_range = (0.1,0.4)
    plane = 'y'
    toview = 'streamlines'
    #toview = 'centroids'

    planes = ['x','y','z']


    fa_scale_range = (0, 0.5)
    color_contrast = 'fa'

    for bundle_id in bundle_ids:
        streamlines_bothsides = []
        for side in ['left', 'right']:
            filepath_bundle = os.path.join(trk_proj_path, f'{subj}_{side}_bundle_{bundle_id}.trk')
            bundle_data = load_trk_remote(filepath_bundle, 'same', None)
            streamlines_numpoints = set_number_of_points(bundle_data.streamlines, nb_points=points_resample)
            streamlines_bothsides += streamlines_numpoints

            from dipy.segment.clustering import ClusterCentroid

            centroid_file_path = os.path.join(centroid_path, f'centroid_{side}_bundle_{bundle_id}.trk')
            centroid_data = load_trk_remote(centroid_file_path, 'same', None)
            centroid_data_dic[side, bundle_id] = qb_test.cluster(centroid_data.streamlines)[0]

            #bundle_data_dic[side, bundle_id] = bundle_data.streamlines
            try:
                #bundle_data_dic[side, bundle_id] = qb_test.cluster(streamlines_numpoints)[0]
                bundle_data_dic[side, bundle_id] = streamlines_numpoints
            except IndexError:
                bundle_data_dic[side, bundle_id] = qb_test.cluster(centroid_data.streamlines)[0]


    for plane in planes:

        if plane =='x':
            plane_txt = 'sagittal'
        if plane=='y':
            plane_txt = 'coronal'
        if plane =='z':
            plane_txt = 'axial'


        if toview == 'streamlines':
            colorbar = False

            linewidth = 2

            list_bundles = list(bundle_data_dic.values())

            from DTC.file_manager.computer_nav import glob_remote, load_df_remote
            import glob
            from dipy.viz import window, actor

            column_names = []
            for i in range(0, points_resample):
                column_names.append(f"point_{i}_{color_contrast}")

            bundle_fa_dic = {}
            for bundle_id in bundle_ids:
                for side in ['left', 'right']:
                    bundles_fa = []
                    stats_subj_path = glob.glob(os.path.join(stats_proj_path, f'{subj}_{side}_bundle_{bundle_id}.xlsx'))[0]
                    stats_df = load_df_remote(stats_subj_path, None)
                    bundle_fa_dic[side, bundle_id] = stats_df[column_names].values

            list_bundles_fa = list(bundle_fa_dic.values())


            record_path = os.path.join(figures_viz_path,
                                       f'{subj}_{np.size(bundle_ids)}_bundles{ratiostr}_{color_contrast}_{plane_txt}_{bundle_ids[0]}.png')

            lut_cmap = actor.colormap_lookup_table(
                scale_range=fa_scale_range)

            scene_dic[plane] = setup_view(list_bundles, colors=lut_cmap, ref=anat_path, world_coords=True,
                               objectvals=list_bundles_fa,
                               colorbar=colorbar, record=record_path, scene=scene_dic[plane], plane=plane,
                               interactive=interactive_dic[plane], value_range=value_range, linewidth=linewidth)

            interactive_dic[plane] = False

        elif toview == 'centroids':
            colorbar = False

            linewidth = 5

            list_bundles = list(centroid_data_dic.values())

            record_path = os.path.join(figures_viz_path,
                                       f'{subj}_{side}_side_{np.size(bundle_ids)}_bundles_distance_'
                                       f'{str(distance)}{ratiostr}_centroids.png')

            scene = setup_view(list_bundles, colors=lut_cmap, ref=anat_path, world_coords=True,
                               objectvals=None,
                               colorbar=colorbar, record=record_path, scene=scene, plane=plane,
                               interactive=interactive, value_range=value_range, linewidth=5)

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

