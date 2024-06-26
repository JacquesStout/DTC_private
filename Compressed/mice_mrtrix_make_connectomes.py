from DTC.nifti_handlers.transform_handler import img_transform_exec
from DTC.nifti_handlers.transform_handler import img_transform_exec, space_transpose, header_superpose
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import atlas_converter
import os
import nibabel as nib
import numpy as np
import pandas as pd
from DTC.diff_handlers.connectome_handlers.connectome_handler import mrtrixcsv_addatlas


orient_SAMBA_preproc = 'ALS'
orient_subj = 'RAS'
img_labels_preprocess = '/Volumes/Data/Badea/Lab/mouse/burn_after_reading/M22090514_chass_symmetric3_labels_preprocess_labels.nii.gz'
img_labels_reoriented = f'/Volumes/Data/Badea/Lab/mouse/burn_after_reading/M22090514_chass_symmetric3_labels_preprocess_labels_{orient_subj}.nii.gz'
labels_subj_path = f'/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/M22090514_labels.nii.gz'

subj_ref = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/11_CS_DWI_bart_recon.nii.gz'
subj_ref = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/trkscutdirs_improved/20220905_14_fa.nii.gz'

if not os.path.exists(labels_subj_path):
    img_transform_exec(img_labels_preprocess, orient_SAMBA_preproc, orient_subj, output_path=img_labels_reoriented, recenter_test=True)
    header_superpose(subj_ref, img_labels_reoriented, outpath=labels_subj_path)

try:
    label_nii = nib.load(labels_subj_path)
except:
    print(f'Could not find {labels_subj_path}, skipping the connectomes creation')
    make_connectomes = False

cut = True

if cut:
    input_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/trkscutdirs_improved/'
    connectomes_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/connectomes_improvedcut/'
else:
    input_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/trks/'
    connectomes_folder = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/connectomes_all/'


fa_mif = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/trkscutdirs_improved/20220905_14_fa.mif'

subj = '20220905_14'

overwrite=False
bvec_string = ''
denoise = True


distances_csv = connectomes_folder +subj+'_distances.csv'

mean_FA_connectome =  connectomes_folder+subj+'_mean_FA_connectome.csv'

parcels_csv = connectomes_folder+subj+'_conn_sift_node.csv'
assignments_parcels_csv = connectomes_folder +subj+f'_assignments_con_sift_node.csv'

parcels_csv_2 = connectomes_folder+subj+'_conn_plain.csv'
assignments_parcels_csv2 = connectomes_folder +subj+f'_assignments_con_plain.csv'

parcels_csv_3 = connectomes_folder+subj+'_conn_sift.csv'
assignments_parcels_csv3 = connectomes_folder +subj+f'_assignments_con_sift.csv'

verbose = True

if denoise:
    denoise_str = '_denoised'
else:
    denoise_str = ''

if not os.path.exists(parcels_csv_3) or True:

    wmfod_norm_mif = os.path.join(input_folder,'20220905_14_wmfod_norm.mif')
    smallerTracks = os.path.join(input_folder, f'{subj}_smallerTracks2mill{bvec_string}{denoise_str}.tck')

    if not os.path.exists(wmfod_norm_mif):

        den_unbiased_mif = den_preproc_mif  # bypassing

        mask_mif = path_perm + subj + '_mask.mif'
        if not os.path.exists(mask_mif) or overwrite:
            os.system('dwi2mask ' + den_unbiased_mif + ' ' + mask_mif + ' -force')
        # os.system('mrview '+fa_mif + ' -overlay.load '+ mask_mif )
        mask_mrtrix_nii = subj_path + subj + '_mask_mrtrix.nii.gz'

        if not os.path.exists(mask_mrtrix_nii) or overwrite:
            os.system('mrconvert ' + mask_mif + ' ' + mask_mrtrix_nii + ' -force')

        # making mask

        mask_nii_gz = path_perm + subj + '_mask.nii.gz'
        csf_nii_gz = subj_path + subj + '_csf.nii.gz'

        if not os.path.exists(csf_nii_gz):
            os.system('ImageMath 3 ' + csf_nii_gz + ' ThresholdAtMean ' + b0_orig + ' 10')
            os.system('ImageMath 3 ' + csf_nii_gz + ' MD ' + csf_nii_gz + ' 1')
            os.system('ImageMath 3 ' + csf_nii_gz + ' ME ' + csf_nii_gz + ' 1')

        if not os.path.exists(mask_mif) or overwrite:
            os.system('ImageMath 3 ' + mask_nii_gz + ' - ' + mask_mrtrix_nii + ' ' + csf_nii_gz)
            os.system('ThresholdImage 3 ' + mask_nii_gz + ' ' + mask_nii_gz + ' 0.0001 1 1 0')
            os.system('ImageMath 3 ' + mask_nii_gz + ' ME ' + mask_nii_gz + ' 1')
            os.system('ImageMath 3 ' + mask_nii_gz + ' MD ' + mask_nii_gz + ' 1')

            os.system('mrconvert ' + mask_nii_gz + ' ' + mask_mif + ' -force')

        ########### making a mask out of labels

        label_path_orig = orig_subj_path + subj + '_labels.nii.gz'
        # label_path = path_perm +subj+'_labels.nii.gz'
        # os.system("/Applications/Convert3DGUI.app/Contents/bin/c3d "+label_path_orig+" -orient RAS -o "+label_path)

        label_path = label_path_orig  # if not doing the rotation seen above

        """
        #mask_output = subj_path +subj+'_mask_of_label.nii.gz'
        #mask_labels_data = label_nii.get_fdata()
        #mask_labels = np.unique(mask_labels_data)
        #mask_labels=np.delete(mask_labels, 0)
        #mask_of_label =label_nii.get_fdata()*0
        """

        path_atlas_legend = root + 'IIT/IITmean_RPI_index.xlsx'
        legend = pd.read_excel(path_atlas_legend)

        """
        #new_bval_path = path_perm+subj+'_new_bvals.txt' 
        #new_bvec_path = path_perm+subj+'_new_bvecs.txt' 
        #os.system('dwigradcheck ' + out_mif +  ' -fslgrad '+bvec_path+ ' '+ bval_path +' -mask '+ mask_mif + ' -number 100000 -export_grad_fsl '+ new_bvec_path + ' '  + new_bval_path  +  ' -force' )
        #bvec_temp=np.loadtxt(new_bvec_path)
        """

        # Estimating the Basis Functions:
        wm_txt = subj_path + subj + '_wm.txt'
        gm_txt = subj_path + subj + '_gm.txt'
        csf_txt = subj_path + subj + '_csf.txt'
        voxels_mif = subj_path + subj + '_voxels.mif' + index_gz
        if not os.path.exists(wm_txt) or not os.path.exists(gm_txt) or not os.path.exists(csf_txt) or overwrite:
            os.system(
                'dwi2response dhollander ' + den_unbiased_mif + ' ' + wm_txt + ' ' + gm_txt + ' ' + csf_txt + ' -voxels ' + voxels_mif + ' -mask ' + mask_mif + ' -scratch ' + subj_path + ' -fslgrad ' + bvec_path + ' ' + bval_path + '  -force')

        # Viewing the Basis Functions:
        # os.system('mrview '+den_unbiased_mif+ ' -overlay.load '+ voxels_mif)
        # os.system('shview '+wm_txt)
        # os.system('shview '+gm_txt)
        # os.system('shview '+csf_txt)

        # Applying the basis functions to the diffusion data:
        wmfod_mif = subj_path + subj + '_wmfod.mif' + index_gz
        gmfod_mif = subj_path + subj + '_gmfod.mif' + index_gz
        csffod_mif = subj_path + subj + '_csffod.mif' + index_gz

        # os.system('dwi2fod msmt_csd ' +den_unbiased_mif+ ' -mask '+mask_mif+ ' ' +wm_txt+ ' ' + wmfod_mif+ ' ' +gm_txt+ ' ' + gmfod_mif+ ' ' +csf_txt+ ' ' + csffod_mif + ' -force' )
        if not os.path.exists(wmfod_mif):
            os.system(
                'dwi2fod msmt_csd ' + den_unbiased_mif + ' -mask ' + mask_mif + ' ' + wm_txt + ' ' + wmfod_mif + ' -force')

        # combine to single image to view them
        # Concatenating the FODs:
        ##vf_mif =   subj_path+subj+'_vf.mif'
        # os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat '+csffod_mif+ ' ' +gmfod_mif+ ' - ' + vf_mif+' -force' )
        ##os.system('mrconvert -coord 3 0 ' +wmfod_mif+ ' -| mrcat ' +gmfod_mif+ ' - ' + vf_mif+' -force' ) # without csf

        # Viewing the FODs:
        # os.system('mrview ' +fa_mif+ ' -odf.load_sh '+wmfod_mif )

        # Normalizing the FODs:
        wmfod_norm_mif = subj_path + subj + '_wmfod_norm.mif' + index_gz
        # gmfod_norm_mif = subj_path+subj+'_gmfod_norm.mif'
        # csffod_norm_mif = subj_path+subj+'_csffod_norm.mif'
        if not os.path.exists(wmfod_norm_mif) or overwrite:
            os.system('mtnormalise ' + wmfod_mif + ' ' + wmfod_norm_mif + ' -mask ' + mask_mif + '  -force')

    #Sifting the tracks with tcksift2: bc some wm tracks are over or underfitted
    sift_mu_txt = input_folder+subj+'_sift_mu.txt'
    sift_coeffs_txt = input_folder+subj+'_sift_coeffs.txt'
    sift_1M_txt = input_folder+subj+'_sift_1M.txt'

    if not os.path.exists(sift_mu_txt) or overwrite:
        os.system('tcksift2  -out_mu '+ sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif+ ' ' + sift_1M_txt  + ' -force')

    if not os.path.exists(sift_coeffs_txt) or overwrite:
        os.system('tcksift2  -out_mu '+ sift_mu_txt + ' -out_coeffs ' + sift_coeffs_txt + ' ' + smallerTracks + ' ' + wmfod_norm_mif+ ' ' + sift_1M_txt  + ' -force')

    path_atlas_legend = '/Volumes/Data/Badea/Lab/atlases/chass_symmetric3/CHASSSYMM3AtlasLegends.xlsx'
    legend = pd.read_excel(path_atlas_legend)

    #convert subj labels to mif
    parcels_mif = input_folder + subj + '_parcels.mif.gz'

    if not os.path.exists(parcels_mif) or overwrite:
        labels_data = label_nii.get_fdata()
        labels = np.unique(labels_data)
        labels=np.delete(labels, 0)
        label_nii_order = labels_data*0.0

        #sum(legend['index2'] == labels)
        for i in labels:
            leg_index = np.where(legend['index2'] == np.round(i) )
            leg_index = leg_index [0][0]
            ordered_num = legend['index'][leg_index]
            label3d_index = np.where( labels_data == i )
            label_nii_order[label3d_index] = ordered_num


        file_result= nib.Nifti1Image(label_nii_order, label_nii.affine, label_nii.header)
        new_label = connectomes_folder +subj+'_new_label.nii.gz'
        nib.save(file_result, new_label)

        #new_label = label_path
        os.system('mrconvert '+new_label+ ' ' +parcels_mif + ' -force' )


    #Creating the connectome without coregistration:
    ### connectome folders :

    mean_FA_per_streamline =  connectomes_folder+subj+'_per_strmline_mean_FA.csv'

    if not os.path.isdir(connectomes_folder) : os.mkdir(connectomes_folder)

    #This is the standard symmetric connectome, without sift weights
    if not os.path.exists(parcels_csv_2) or not os.path.exists(assignments_parcels_csv2) or overwrite:
        print(f'Getting standard connectomes, without weights')
        cmd = 'tck2connectome -symmetric -zero_diagonal '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv_2 + ' -out_assignment ' + assignments_parcels_csv2 + ' -force'
        if verbose:
            print(cmd)
        os.system(cmd)

    #This is the standard symmetric connectome, with sift weights
    if not os.path.exists(parcels_csv_3) or not os.path.exists(assignments_parcels_csv3) or overwrite:
        print(f'Getting standard connectomes, with weights')
        cmd = 'tck2connectome -symmetric -zero_diagonal -tck_weights_in '+ sift_1M_txt+ ' '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv_3 + ' -out_assignment ' + assignments_parcels_csv3 + ' -force'
        if verbose:
            print(cmd)
        os.system(cmd)


    #This is the connectome for the length of the streamlines
    if not os.path.exists(distances_csv) or overwrite:
        os.system('tck2connectome ' + smallerTracks + ' ' + parcels_mif+ ' ' + distances_csv + ' -zero_diagonal -symmetric -scale_length -stat_edge  mean' + ' -force')

    #This is the connectome for the average FA of the streamlines
    if not os.path.exists(mean_FA_per_streamline) or not os.path.exists(mean_FA_connectome) or overwrite:
        os.system('tcksample '+ smallerTracks+ ' '+ fa_mif + ' ' + mean_FA_per_streamline + ' -stat_tck mean ' + ' -force')
        os.system('tck2connectome '+ smallerTracks + ' ' + parcels_mif + ' '+ mean_FA_connectome + ' -zero_diagonal -symmetric -scale_file ' + mean_FA_per_streamline + ' -stat_edge mean '+ ' -force')


    #This is the connectome scaled by the volume of the nodes, with weights, and assignment of streamlines
    if not os.path.exists(parcels_csv) or not os.path.exists(assignments_parcels_csv) or overwrite:
        os.system('tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in '+ sift_1M_txt+ ' '+ smallerTracks + ' '+ parcels_mif + ' '+ parcels_csv + ' -out_assignment ' + assignments_parcels_csv + ' -force')


    ROI_legends = "/Volumes/Data/Badea/Lab/atlases/chass_symmetric3/CHASSSYMM3AtlasLegends.xlsx"

    _, _, index_to_struct, _ = atlas_converter(ROI_legends)

    connectome_path = parcels_csv_2
    connectome_outpath = parcels_csv_2.replace('.csv','_labels.csv')
    if not os.path.exists(connectome_outpath) or overwrite:
        mrtrixcsv_addatlas(connectome_path, index_to_struct, connectome_outpath)