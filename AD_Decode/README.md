# AD_DECODE folder

## Description

Contains all code pertaining to AD_Decode data handling, preprocessing, analysis, the BuSA bundle splitting, and figure makers for the 'Mapping the impact of age and APOE risk factors for late onset Alzheimer’s disease on long range brain connections through multiscale bundle analysis' manuscript

### Folder Details

#### BuSA_split_regions_into_bundles

Folder containing code pertaining to bundle Splitting work. Works best with cluster allowing for batch jobs. More details in folder README

#### tractseg

Folder containing code pertaining to usage of TractSeg with AD_Decode data. At time of writing, did not generate final figures. More details in folder README


### Code description

#### AD_Deode_T1_SAMBAprep.py

This file serves to register the T1 files to the dwi files. This allows them to be used as contrast drivers of SAMBA or to have the SAMBA transforms applied to them, depending.

#### AD_Decode_T1_extractor.py

This file will go through the data files of AD_Decode, take the anatomical acquistions and place them together in the input Analysis folder defined by outpath
(at time of writing, /Volumes/Data/Badea/ADdecode.01/Analysis/DWI/)

#### AD_Decode_makemrtrixstats.py

Goes through the preprocessed AD_Decode datat found in /Volumes/Data/Badea/Lab/human/AD_Decode/diffusion_prep_locale/ (see AD_Decode_SAMBA_prep.py) and generates mrtrix diffusion files (fa, ad, rd, etc)

##### AD_Decode_mrtrixstats_wrapper.py
Wrapper for the above code


#### AD_Decode_mrtrixstats_SAMBAprep.py

older file meant to convert original mrtrix preprocessed images from Nariman's pipeline to a usable state for SAMBA where images have the right orientation for SAMBA compared to other images

#### AD_Decode_subj_to_MDT_clustered.py

converter of files in subject space to that of MDT space associated with VBM AD_Decode folder using appropriate transform files

##### AD_Decode_subj_to_MDT_clusterwrapper.py

Wrapper for the above code

#### AD_Decode_tcktotrk_wrapper.py

A useful wrapper for converting all tck files in a specific folder to trk, used to convert the mrtrix output folder files into trk files usable by other pipelines such as AD_Decode_subj_to_MDT_clustered.py

#### AD_Decode_tractdensity_extractor_MDT.py

This file goes through the results of the bundle splitting protocol, and extracts much of the important bundle statistics, specifically:
the number of bundle streamlines, the volume of streamlines (creates the density maps), the length of streamlines, the BUAN of bundles (left-right in subject)
If the flag summarize_label_tract_interact is True, it will also go through the different splitting levels and create a summary of which bundles interact with which IIT label structures overall
defined in full_txt_path and full_excel_path, default location: '/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/V_1_0_10template_100_6_interhe_majority/stats/excel_summary/label_summary_plain_updated.txt'

#### AD_Decode_tractdensity_extractor_tractseg.py

Similar as above, but applied to results of the TractSeg analysis.

#### AD_Decode_trc_generation.py

This code uses the results of the AD_Decode preprocessing (also used as inputs for SAMBA) and generates the tract files via use of the mrtrix program.

##### AD_Decode_trc_generation_wrapper.py

This code serves as a wrapper for the mrtrix pipeline and trc generation.

#### BuSA_MDT_masks_all.py

Creates masks for all IIT structures in MDT space (useful for individual analysis when observing streamlines with BuSA_bundle_visualizer.py)

#### BuSA_bundle_visualizer.py

This code is used for visualization of centroids or bundle streamlines with an MDT reference space and possible addition of regional masks for reference.
(This whole code warrants more explanation if time allows)
Created the anatomical images used for Manuscript: Mapping the impact of age and APOE risk factors for late onset Alzheimer’s disease on long range brain connections through multiscale bundle analysis


#### MDT_masking.py

Applies a mask to all MDT space images found in a folder. Used for QSM data (/Users/jas/jacques/AD_Decode/QSM_MDT/smoothed_1_5/)


#### SAMBA_prep_ADDecode.py

The official 'diffusion preprocessing and preparing for SAMBA code' for AD_Decode data.
Will go through the main AD_Decode folder and organize them into folders and the input folder used by SAMBA.
This version runs locally and does not make the symbolic links, as Mac symbolic links are not readable by SAMBA.
If this ever changes, simply change shortcuts_all_folder from None to 
the SAMBA input folder (at time of writing:"/mnt/munin6/Badea/Lab/human/ADDeccode_symlink_pool_allfiles/")

#### SAMBA_prep_ADDecode.py

The version meant for linux that will create symbolic links. Functionally a very similar code, with the shortcuts_folder changed accordingly.

#### bundlestat_to_age_compare.py

This code is one of the most important figure generators for Manuscript: Mapping the impact of age and APOE risk factors for late onset Alzheimer’s disease on long range brain connections through multiscale bundle analysis
This code creates the figures comparing the bundle statistics to age for all subjects based on the outputs of the BuSA project.
Creates the statistics for figure 3 and 4

#### bundlestat_to_age_compare_sex.py

Variant of the code above where we compare based on Sex instead of Genotype (note: did not create new figures)
  
#### cog_to_bundle_compare.py

This code is one of the most important figure generators for Manuscript: Mapping the impact of age and APOE risk factors for late onset Alzheimer’s disease on long range brain connections through multiscale bundle analysis.
This code creates the figures comparing the cognition or physical metrics to age for all subjects based on the outputs of the BuSA project.


#### combine_trk_files.py

This code is used to combine on a per subject basis trk files from one folder to those in another folder, outputting them to a third folder
This is used after usage of the streamline_extract_tuples.py code that extracts tracts based on which edge connection they refer to.


#### fix_MDT_streamlines.py




### Defunct Code

#### BuSA_analysis_2.py and other BuSA_analysis files

Serie of codes done in cooperation with Ali Mahzarnia so as to implement lP Norm comparison of FA of streamlines based on skfda package.
Was not used for final paper.

##### 

BuSA_analysis_1.py and 3 were creating other statistical outputs that are now mostly defunct. (3 makes a histogram of length of streamlines in different bundles, 1 makes boxplots based on streamline length)
BuSA_analysis_allboxplots.py was an attempt to reconcile these different files together, BuSA_analysis_4_BUAN_compare.py would compare BUAN values

#### DTC_launcher_ADDecode.py

Old generator of connectomes, was rendered defunct by implementation of mrtrix over time.

#### SAMBA_stats_ANCOVAS_2plots.py

partial ancestor to bundlestat_to_age_compare.py, version that runs a corrected ANCOVA, uses seaborn to create box or violin plots 

#### SAMBA_stats_emmeans.py

Older code meant to run PCA on metadata associated with AD_Decode




