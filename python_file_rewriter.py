

i=0
thelist = []
thelist.append(['dif_to_trk', 'tract_manager.dif_to_trk'])
thelist.append(['downsample_TRK_folder', 'tract_manager.downsample_TRK_folder'])
thelist.append(['DTC_manager', 'tract_manager.DTC_manager'])
thelist.append(['fix_downsample_trk_files', 'tract_manager.fix_downsample_trk_files'])
thelist.append(['fix_trk_files', 'tract_manager.fix_trk_files'])
thelist.append(['rush_create_tracts', 'tract_manager.rush_create_tracts'])
thelist.append(['streamline_nocheck', 'tract_manager.streamline_nocheck'])
thelist.append(['tract_eval', 'tract_manager.tract_eval'])
thelist.append(['tract_handler', 'tract_manager.tract_handler'])
thelist.append(['tract_save', 'tract_manager.tract_save'])
thelist.append(['trk_stats', 'tract_manager.trk_stats'])

thelist.append(['tract_visualize', 'visualization_tools.tract_visualize'])

thelist.append(['bvec_converter', 'diff_handlers.bvec_converter'])
thelist.append(['bvec_handler', 'diff_handlers.bvec_handler'])
thelist.append(['denoise_processes', 'diff_handlers.denoise_processes'])
thelist.append(['diff_preprocessing', 'diff_handlers.diff_preprocessing'])
thelist.append(['DTC_bvecorientchecker', 'diff_handlers.DTC_bvecorientchecker'])

thelist.append(['connectome_handler', 'diff_handlers.connectome_handlers.connectome_handler'])
thelist.append(['excel_management', 'diff_handlers.connectome_handlers.excel_management'])

thelist.append(['argument_tools', 'file_manager.argument_tools'])
thelist.append(['BIAC_tools', 'file_manager.BIAC_tools'])
thelist.append(['computer_nav', 'file_manager.computer_nav'])
thelist.append(['Daemonprocess', 'file_manager.Daemonprocess'])
thelist.append(['easycython', 'file_manager.easycython'])
thelist.append(['file_tools', 'file_manager.file_tools'])
thelist.append(['mac_aliashandling', 'file_manager.mac_aliashandling'])
thelist.append(['trktotck', 'file_manager.trktotck'])
thelist.append(['Update_allcsv', 'file_manager.Update_allcsv'])

thelist.append(['basic_LPCA_denoise', 'gunnies.basic_LPCA_denoise'])

thelist.append(['nifti_handler', 'nifti_handlers.nifti_handler'])
thelist.append(['transform_handler', 'nifti_handlers.transform_handler'])
thelist.append(['convert_atlas_mask', 'nifti_handlers.atlas_handlers.convert_atlas_mask'])
thelist.append(['create_backported_labels', 'nifti_handlers.atlas_handlers.create_backported_labels'])
thelist.append(['labels_converter', 'nifti_handlers.atlas_handlers.labels_converter'])
thelist.append(['mask_handler', 'nifti_handlers.atlas_handlers.mask_handler'])

thelist.append(['figures_handler', 'visualization_tools.figures_handler'])
thelist.append(['tract_visualize', 'visualization_tools.tract_visualize'])

import glob
files = glob.glob('C:\\Users\\JacquesStout\\Documents\\Work\\DTC\\*\\*py')

for pyfile in files:
    fin = open(pyfile, "rt")
    pyfileout = pyfile.replace('Work\\DTC','Work\\DTC_2')
    fout = open(pyfileout, "wt")
    for line in fin:
        newline = line
        for replace in thelist:
            newline = newline.replace(replace[0], replace[1])
        fout.write(newline)




