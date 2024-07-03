import fury,os
from DTC.visualization_tools.tract_visualize import show_bundles, setup_view, view_test, setup_view_colortest
import numpy as np
from DTC.tract_manager.tract_handler import ratio_to_str
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from DTC.file_manager.computer_nav import checkfile_exists_remote, get_mainpaths, load_nifti_remote, load_trk_remote, read_parameters_from_ini
from dipy.tracking.streamline import set_number_of_points

project = 'V0_9_10template_100_36_interhe_majority'
project_summary_file = f'/Volumes/Data/Badea/Lab/jacques/BuSA_headfiles/{project}.ini'

proj_path = f'/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/{project}/'
figures_proj_path = os.path.join(proj_path, 'Figures')

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
coloring_vals = fury.colormap.distinguishable_colormap(nb_colors=np.size(bundle_ids)*2)
colorbar = False

streamline_bundle = {}

toview = 'streamlines'
toview = 'centroids'

bundle_data_dic = {}
bundle_data_twoside = {}
centroid_data_dic = {}

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
            bundle_data_dic[side, bundle_id] = qb_test.cluster(streamlines_numpoints)[0]
        except IndexError:
            bundle_data_dic[side, bundle_id] = qb_test.cluster(centroid_data.streamlines)[0]

        """
        if bundle_id in bundle_data_twoside:
            bundle_data_twoside[bundle_id] += bundle_data.streamlines
        else:
            bundle_data_twoside[bundle_id] = bundle_data.streamlines
        """
    """
    try:
        bundle_data_twoside[side, bundle_id] = qb_test.cluster(streamlines_bothsides)[0]
    except IndexError:
        from dipy.segment.clustering import ClusterCentroid

        centroid_file_path = os.path.join(centroid_path, f'centroid_{side}_bundle_{bundle_id}.trk')
        bundle_data = load_trk_remote(centroid_file_path, 'same', None)
        bundle_data_dic[side, bundle_id] = qb_test.cluster(bundle_data.streamlines)[0]
    """
if toview == 'streamlines':
    list_bundles = list(bundle_data_dic.values())
elif toview == 'centroids':
    list_bundles = list(centroid_data_dic.values())

"""
for bundle_id in np.arange(num_bundles):
    new_bundle = qb_test.cluster(streamline_bundle[side, bundle_id])[0]
    streamlines_side.append(new_bundle)
"""
record_path = os.path.join(figures_proj_path,
                           f'{subj}_{side}_side_{np.size(bundle_ids)}_bundles_distance_'
                           f'{str(distance)}{ratiostr}_figure_test.png')
scene =None
interactive=True
value_range = (0,1)
plane = 'all'

scene = setup_view(list_bundles, colors=coloring_vals, ref=anat_path, world_coords=True,
                   objectvals=None,
                   colorbar=colorbar, record=record_path, scene=scene, plane=plane,
                   interactive=interactive, value_range= value_range, linewidth = 5)

print(f'Saved at {record_path}')