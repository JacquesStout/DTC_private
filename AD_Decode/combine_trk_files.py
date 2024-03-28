import glob, os
import numpy as np
from DTC.file_manager.computer_nav import load_trk_remote
from DTC.tract_manager.tract_save import save_trk_header

"""
trk_folder_1 = '/Volumes/Data/Badea/Lab/human/AD_Decode/Streamlines_tupled_MDT_mrtrix_act_non_inclusive_symmetric/' \
               'ctx-lh-precuneus_left_to_left-thalamus-proper_left_MDT_all'
trk_folder_2 = '/Volumes/Data/Badea/Lab/human/AD_Decode/Streamlines_tupled_MDT_mrtrix_act_non_inclusive_symmetric/' \
               'ctx-rh-precuneus_right_to_right-thalamus-proper_right_MDT_all'
trk_folder_output = '/Volumes/Data/Badea/Lab/human/AD_Decode/Streamlines_tupled_MDT_mrtrix_act_non_inclusive_symmetric/' \
                    'thalamus_precuneus_intrahemispheric'
"""
trk_folder_1 = '/Volumes/Data/Badea/Lab/human/AD_Decode/Streamlines_tupled_MDT_mrtrix_act_non_inclusive_symmetric/' \
               'ctx-lh-superiorparietal_left_to_ctx-lh-precuneus_left_MDT_all'
trk_folder_2 = '/Volumes/Data/Badea/Lab/human/AD_Decode/Streamlines_tupled_MDT_mrtrix_act_non_inclusive_symmetric/' \
               'ctx-rh-superiorparietal_right_to_ctx-rh-precuneus_right_MDT_all'
trk_folder_output = '/Volumes/Data/Badea/Lab/human/AD_Decode/Streamlines_tupled_MDT_mrtrix_act_non_inclusive_symmetric/' \
                    'superior_parietal_precuneus_intrahemispheric'


files = glob.glob(os.path.join(trk_folder_1,'*.trk'))

verbose = False
sftp_out = None

for file_path_1 in files:
    file_name = os.path.basename(file_path_1)
    file_path_2 = os.path.join(trk_folder_2,file_name)
    file_output = os.path.join(trk_folder_output,file_name)
    if not os.path.exists(file_output) and os.path.exists(file_path_2):
        streamlines_data = load_trk_remote(file_path_1, 'same', None)
        streamlines_1 = list(streamlines_data.streamlines)
        streamlines_2 = list(load_trk_remote(file_path_2, 'same', None).streamlines)
        header = streamlines_data.space_attributes

        streamlines_all = streamlines_1 + streamlines_2

        sg = lambda: (s for i, s in enumerate(streamlines_all))

        save_trk_header(filepath=file_output, streamlines=sg, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=sftp_out)

