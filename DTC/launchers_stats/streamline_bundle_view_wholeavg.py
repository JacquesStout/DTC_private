from dipy.io.streamline import load_trk, save_trk
from dipy.viz import window, actor
import os
import pickle
from visualization_tools.visualization_tools.tract_visualize import show_bundles, setup_view
from nifti_handlers.atlas_handlers.convert_atlas_mask import convert_labelmask, atlas_converter
from tract_manager.tract_handler import ratio_to_str, gettrkpath
from itertools import compress
import numpy as np
import nibabel as nib, socket
from file_manager.file_tools import mkcdir
from tract_manager.streamline_nocheck import load_trk as load_trk_spe
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature, AveragePointwiseEuclideanMetric
import warnings


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
write_txt = False
ratio = 1
top_percentile = 100
num_bundles = 20

if project == 'AD_Decode':
    # ,(23,30)
    target_tuples = [(9, 1), (24, 1), (22, 1), (58, 57), (64, 57)]
    target_tuples = [(9, 1), (24, 1), (22, 1), (58, 57), (23, 24), (64, 57)]
    target_tuples = [(58, 57), (9, 1), (24, 1), (22, 1), (64, 57), (23, 24), (24, 30), (23, 30)]
    target_tuples = [(24, 30), (23, 24)]
    target_tuples = [(80, 58)]
    target_tuples = [(58, 57)]
    target_tuples = [(64,57)]
    #genotype_noninclusive
    target_tuples = [(9, 1), (24, 1), (58, 57), (64, 57), (22, 1)]
    #target_tuples = [(24, 1)]
    #genotype_noninclusive_volweighted_fa
    #target_tuples = [(9, 1), (57, 9), (61, 23), (84, 23), (80, 9)]

    #sex_noninclusive
    #target_tuples = [(64, 57), (58, 57), (9, 1), (64, 58), (80,58)]
    #sex_noninclusive_volweighted_fa
    #target_tuples = [(58, 24), (58, 30), (64, 30), (64, 24), (58,48)]

    groups = ['APOE4', 'APOE3']
    #groups = ['Male','Female']

    mainpath = os.path.join(mainpath, project, 'Analysis')
    anat_path = os.path.join(mainpath,'../../mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_b0.nii.gz')
    space_param = '_MDT'

if project == 'AMD':
    #target_tuples = [(9, 1),(24, 1), (76, 42), (76, 64), (77, 9), (43, 9)]
    groups_all = ['Paired 2-YR AMD','Initial AMD','Initial Control','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']
    groups = ['Paired Initial Control', 'Paired Initial AMD', 'Paired 2-YR Control', 'Paired 2-YR AMD']
    groups = ['Paired Initial Control', 'Paired Initial AMD']
    groups = ['Paired 2-YR Control', 'Paired 2-YR AMD']
    mainpath = os.path.join(mainpath, project)
    anat_path = os.path.join(mainpath,'../mouse/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_Control_n72_i6/median_images/MDT_b0.nii.gz')

    space_param = '_affinerigid'

    #TN-PCA
    #target_tuples = [(62, 28), (56, 45), (77, 43), (58, 45), (79, 45), (56, 50), (28, 9), (62, 1), (28, 1), (62, 9), (22, 9), (56, 1),(77, 43), (76, 43), (61, 29), (63, 27), (73, 43), (53, 43)]
    #VBA
    #target_tuples = [(27,29), (61,63),(30, 16), (24, 16),(28, 31), (28, 22),(22, 31)]
    #TN-PCA / VBA combination
    target_tuples = [(62, 28), (58, 45), (28, 9), (62, 1), (77, 43), (61, 29)]
    target_tuples = [(77, 43)]
    plane = 'x'

write_stats = False

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



# target_tuple = (24,1)
# target_tuple = [(58, 57)]
# target_tuples = [(64, 57)]


ratio_str = ratio_to_str(ratio)
print(ratio_str)
if ratio_str == '_all':
    folder_ratio_str = ''
else:
    folder_ratio_str = ratio_str.replace('_ratio', '')
# target_tuple = (9,77)

_, _, index_to_struct, _ = atlas_converter(ROI_legends)

# figures_path = '/Volumes/Data/Badea/Lab/human/AMD/Figures{space_param}_non_inclusive/'
# centroid_folder = '/Volumes/Data/Badea/Lab/human/AMD/Centroids{space_param}_non_inclusive/'

figures_path = os.path.join(mainpath, f'Figures{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
centroid_folder = os.path.join(mainpath, f'Centroids{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
trk_folder = os.path.join(mainpath, f'Centroids{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')
stats_folder = os.path.join(mainpath, f'Statistics{space_param}{inclusive_str}{symmetric_str}{folder_ratio_str}')

mkcdir([figures_path, centroid_folder, stats_folder])

# groups = ['Initial AMD', 'Paired 2-YR AMD', 'Initial Control', 'Paired 2-YR Control', 'Paired Initial Control',
#          'Paired Initial AMD']

# anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_19BrainChAMD01_IITmean_RPI_with_2yr-work/dwi/SyN_0p5_3_0p5_dwi/dwiMDT_Control_n72_i6/median_images/MDT_dwi.nii.gz'


# superior frontal right to cerebellum right

#set parameter
num_points1 = 50
distance1 = 1
feature1 = ResampleFeature(nb_points=num_points1)
metric1 = AveragePointwiseEuclideanMetric(feature=feature1)

#group cluster parameter
num_points2 = 50
distance2 = 2
feature2 = ResampleFeature(nb_points=num_points2)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)


scene = None
selection = 'num_streams'

coloring = 'bundles_coloring'

references = ['fa']

for target_tuple in target_tuples:

    interactive = True

    print(target_tuple[0], target_tuple[1])
    print(index_to_struct[target_tuple[0]] + '_to_' + index_to_struct[target_tuple[1]])

    if changewindow_eachtarget:
        firstrun = True

    for group in groups:



        print(f'Setting up group {group}')
        group_str = group.replace(' ', '_')


        centroid_file_path = os.path.join(centroid_folder,
                                          group_str + space_param + ratio_str + '_' + index_to_struct[
                                              target_tuple[0]] + '_to_' +
                                          index_to_struct[target_tuple[1]] + '_centroid.py')
        fa_path = os.path.join(centroid_folder,
                               group_str + space_param + ratio_str + '_' + index_to_struct[target_tuple[0]] + '_to_' +
                               index_to_struct[target_tuple[1]] + '_fa_lines.py')
        md_path = os.path.join(centroid_folder,
                               group_str + space_param + ratio_str + '_' + index_to_struct[target_tuple[0]] + '_to_' +
                               index_to_struct[target_tuple[1]] + '_md_lines.py')
        trk_path = os.path.join(trk_folder,
                                group_str + space_param + ratio_str + '_' + index_to_struct[target_tuple[0]] + '_to_' +
                                index_to_struct[target_tuple[1]] + '_streamlines.trk')

        if os.path.exists(fa_path):
            with open(fa_path, 'rb') as f:
                fa_lines = pickle.load(f)
        if os.path.exists(md_path):
            with open(md_path, 'rb') as f:
                md_lines = pickle.load(f)
        # '/Volumes/Data/Badea/Lab/human/AD_Decode/Analysis/Centroids_MDT_non_inclusive_symmetric_100/APOE4_MDT_ratio_100_ctx-lh-inferiorparietal_left_to_ctx-lh-inferiortemporal_left_streamlines.trk'
        if os.path.exists(trk_path):
            try:
                streamlines_data = load_trk(trk_path, 'same')
            except:
                streamlines_data = load_trk_spe(trk_path, 'same')
        streamlines = streamlines_data.streamlines

        if 'fa_lines' in locals():
            cutoff = np.percentile(fa_lines, 100 - top_percentile)
            select_streams = fa_lines > cutoff
            fa_lines = list(compress(fa_lines, select_streams))
            streamlines = list(compress(streamlines, select_streams))
            streamlines = nib.streamlines.ArraySequence(streamlines)
            if np.shape(streamlines)[0] != np.shape(fa_lines)[0]:
                raise Exception('Inconsistency between streamlines and fa lines')
        else:
            txt = f'Cannot find {fa_path}, could not select streamlines based on fa'
            warnings.warn(txt)
            fa_lines = [None]

        lut_cmap = actor.colormap_lookup_table(
            scale_range=(0.1, 0.25))

        record_path = os.path.join(figures_path, group_str + space_param + ratio_str + '_' + index_to_struct[target_tuple[0]] + '_to_' +
                                          index_to_struct[target_tuple[1]] + '_allstreams_figure.png')
        #scene = None
        #interactive = False
        #record_path = None
        anat_path = '/Volumes/Data/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_b0.nii.gz'
        scene = setup_view(streamlines, colors = lut_cmap,ref = anat_path, world_coords = True, objectvals = fa_lines, colorbar=True, record = record_path, scene = scene, plane = plane, interactive = interactive)
        del(fa_lines,streamlines)
        interactive = False

        """
        # color by line-average fa
        renderer = window.Renderer()
        renderer.clear()
        renderer = window.Renderer()
        stream_actor3 = actor.line(group_clusters.clusters[bundle_id], np.array(bundle_fa), lookup_colormap=cmap)
        renderer.add(stream_actor3)
        bar = actor.scalar_bar(cmap)
        renderer.add(bar)
        # Uncomment the line below to show to display the window
        window.show(renderer, size=(600, 600), reset_camera=False)
        """