# Compressed Sensing preprocessing and handling

## Contains all codes pertaining to compressed sensing conversion to images and analysis

## Details

Details on Compressed Sensing data and project:

CS_mse_compare.py
This file was used to determined the optimized values for TV_Val and L1_val for different compressed sensing levels, the original data can be found at:
/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs
Simply prints out the optimal values as output, please refer to specific CS_value in 'CS_val' variable line 25
Original data designation is:
'Need to be filled out'

get_method_summary.py
Simple py file that extracts much of relevant information out of all scans of a specified Bruker Subject.
Simply specify the root path of your data, and give a list of subjects to iterate through
Output will look something like:
For pathofsubject/1, the method is FLASH, the spat_resol is ['0.25', '0.25', '0.75']
For pathofsubject/2, the method is FISP, the spat_resol is ['0.25', '0.25', '0.75']

get_method_summary_all.py
More complex but detailed version of get_method_summary. More useful when looking for specific methods in a large number of subjects and obtain specific information.
Will look through a data path, and generate compressed sensing results and nifti reconstructions in specified folders if it does not already exist.
Will also provide specific scan information similar to get_method_summary, with more details. Can be told to go through a large number of subjects but by default will only give information for 
specific method scans, as specified by variable 'allowed_methods' (takes the form of a list of strings, where each string is a type scan that can be included on readout).
If you do want to go through ALL methods, simply specify read_all_methods=True (this does simplify the print readout, as it will then only output method name, to avoid errors caused by any possible variation of data encoding for certain types of sequences).
If you have CS data but do not want to use automatic reconstruction, specify run_recon = False.
If you do not want the code to generate niftis for each subject upon reaching specified subject, simply specify run_bruker_cmd = False

life_compare.py
Half defunct file, but essentially this file is designed to go through different tract results that resulted from different compression levels found in
/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/ and compare said tract results by:
looking at stremaline statistics: number of streamlines, and min/max/mean/standard deviation of the length of all streamlines
comparing the LiFE value (Linear Fascicle Evaluation), which essentially compares how closely the streamlines align with their original data: https://workshop.dipy.org/documentation/1.7.0/examples_built/13_fiber_tracking/linear_fascicle_evaluation/

make_mrtrix_trks.py
Code used to generate mrtrix tract files from
/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/February_runs/ 
where inputs have different CS values

make_mrtrix_trks_DTI.py
Code used to generate mrtrix tract files from
/Volumes/dusom_mousebrains/All_Staff/jacques/CS_project/DTI_testzone' which contains varied diffusion
images from which to generate mrtrix tract files. This code is partially defunct as the data structure of files found here has been reorganized

mice_mrtrix_make_trks.py
Code used to generate mrtrix tract files from
/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/,
which contains diffusion images obtained from Manisha.

mice_mrtrix_make_trks_cutdirs.py
Same as above, except only selection diffusion files based on optimized cut directions determined from optimize_viz_bvectors.py
Ought to be combined with mice_mrtrix_make_trks.py if time allows.

mice_mrtrix_make_trks_allfunc.py
Same code as above, modified to make it easier to share as all functions are contained in file rather than referring to DTC package

optimize_viz_bvectors.py
Going through the original 40 directions from Manisha acquisition, determine which 21 (cut_number variable) are the most distant from each other, making for a more equilibrated selection




