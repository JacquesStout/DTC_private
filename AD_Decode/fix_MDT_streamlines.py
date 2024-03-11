import os, glob
from DTC.file_manager.computer_nav import load_trk_remote
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from DTC.tract_manager.tract_save import save_trk_header


def check_streamline(streamline):
    for i in range(len(streamline) - 1):
        point1 = streamline[i]
        point2 = streamline[i + 1]

        distance = np.linalg.norm(point2 - point1)

        if distance > 10:

            firstpart = streamline[:i + 1]
            secondpart = streamline[i + 1:]
            if np.linalg.norm(firstpart - firstpart[0]) < 0.5:
                if len(firstpart)==1:
                    secondpart = secondpart[1:]
                    continue
                else:
                    return(secondpart)
            elif np.linalg.norm(secondpart - secondpart[0]) < 0.5:
                return(firstpart)
            else:
                streamline_1 = check_streamline(firstpart)
                streamline_2 = check_streamline(secondpart)
                if len(streamline_1)>len(streamline_2):
                    return(streamline_1)
                else:
                    return(streamline_2)

    return(streamline)


def fix_badwarp_streamlines(streamlines):
    streamlines_pruned = []
    streamlines_onlypruned = []

    for id, streamline in enumerate(streamlines):
        streamline = check_streamline(streamline)
        streamlines_pruned.append(streamline)
    return streamlines_pruned

TRK_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/ADDECODE/TRK_MDT'
TRK_folder_new = '/Volumes/dusom_mousebrains/All_Staff/Data/ADDECODE/TRK_MDT_fixed'

trk_files = glob.glob(os.path.join(TRK_folder,'*trk'))
verbose = False

for trk_path in trk_files:

    trk_path_new = os.path.join(TRK_folder_new,os.path.basename(trk_path))
    if not os.path.exists(trk_path_new):
        streamlines_data = load_trk_remote(trk_path, 'same')
        streamlines_unfixed = ArraySequence(streamlines_data.streamlines)
        header = streamlines_data.space_attributes

        streamlines_fixed = fix_badwarp_streamlines(streamlines_unfixed)

        sg_f = lambda: (s for i, s in enumerate(streamlines_fixed))
        save_trk_header(filepath=trk_path_new, streamlines=sg_f, header=header,
                        affine=np.eye(4), verbose=verbose, sftp=None)