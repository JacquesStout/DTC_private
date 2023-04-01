import os
from DTC.tract_manager.tract_handler import ratio_to_str, gettrkpath, gettrkpath_testsftp
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
import pandas as pd
from DTC.file_manager.file_tools import file_rename, mkcdir

distance = 20

other_spec = '_mrtrix'
other_spec = ''

space_param = '_MDT'
space_param = '_affinerigid'

inpath = '/Volumes/dusom_mousebrains/All_Staff/Data/AMD/'
excel_subjID_path = '/Volumes/dusom_mousebrains/All_Staff/Analysis/AMD/temp_subjID_unwrangler'
#excel_subjID_path = f'/Volumes/dusom_mousebrains/All_Staff/Analysis/AMD/Centroids{space_param}_non_inclusive_symmetric{other_spec}'
bundlestats_folder = '/Volumes/dusom_mousebrains/All_Staff/Analysis/AMD/Statistics_allregions_affinerigid_non_inclusive_symmetric/bundle_stats/'
bundlestats_folder = '/Volumes/dusom_mousebrains/All_Staff/Analysis/AMD/Statistics_allregions_distance_30_affinerigid_non_inclusive_symmetric'
bundlestats_folder = f'/Volumes/dusom_mousebrains/All_Staff/Analysis/AMD/Statistics_allregions_distance_{distance}{space_param}_non_inclusive_symmetric{other_spec}/'
atlas_legends = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx'
outpath = os.path.join(bundlestats_folder, 'bundle_stats_withid')
mkcdir(outpath)

_, _, index_to_struct, _ = atlas_converter(atlas_legends)
ratio=1
ratio_str = ratio_to_str(ratio)

#group = 'Paired 2-YR AMD'

target_tuples = [(62, 28),(58,45),(28,9), (62, 1)]
#target_tuples = [(62, 28),(28,9), (62, 1)]

groups = ['Paired 2-YR AMD','Paired 2-YR Control','Paired Initial Control','Paired Initial AMD']
#groups = ['Paired Initial Control','Paired Initial AMD']
#groups = ['Paired 2-YR AMD','Paired 2-YR Control']

rename = True
test = False
if rename:
    print(rename)
    file_rename(bundlestats_folder, ' ', '_', test=test)
    file_rename(bundlestats_folder, 'Paired_', '', test=test)
    file_rename(bundlestats_folder, '2-YR', '2Year', test=test)


for group in groups:
    for target_tuple in target_tuples:

        group_str = group.replace(' ', '_')

        streamline_IDs_excel = os.path.join(excel_subjID_path, group_str+f'_MDT_all_'+index_to_struct[target_tuple[0]] + '_to_' +
                                   index_to_struct[target_tuple[1]]+'_streamlineID_subj.xlsx')

        group_str = group_str.replace('Paired_','')
        group_str = group_str.replace('2-YR', '2Year')

        bundle_stats_path = os.path.join(bundlestats_folder, group_str+'_'+index_to_struct[target_tuple[0]] + '_to_' +
                                   index_to_struct[target_tuple[1]] + '_all_bundle_stats.csv')

        bundle_stats_path_new = os.path.join(outpath,group_str+'_'+index_to_struct[target_tuple[0]] + '_to_' +
                                   index_to_struct[target_tuple[1]] + '_all_bundle_stats.csv')

        if not os.path.exists(bundle_stats_path_new):

            bundles_excel = os.path.join(bundle_stats_path)

            streamline_IDs_df = pd.read_excel(streamline_IDs_excel)
            bundle_stats_df = pd.read_csv(bundle_stats_path)

            streamline_IDs_df.rename(columns = {'Streamline_ID':'Streamlines ID'}, inplace = True)

            bundle_stats_df_new = bundle_stats_df.merge(streamline_IDs_df, how='inner', on = 'Streamlines ID')

            bundle_stats_df_new.to_csv(bundle_stats_path_new)
            print(f'Wrote {bundle_stats_path_new}')
        else:
            print(f'Already wrote {bundle_stats_path_new}')
