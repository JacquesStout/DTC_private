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