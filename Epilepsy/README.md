Main code for combining brains and splitting them into large bundles separating the brain into different large regions is found at:
-https://github.com/JacquesStout/DTC_private/blob/main/Epilepsy/01_split_streams_left_right.py 
Goes through all the subjects and splits them into the left right streams and left side streams, saves them to TBD by default. Default format is: {subj}_roi_lstream.trk, {subj}_roi_rstream.trk, {subj}_roi_rstream_flipped.trk, where rstream_flipped is rstream flipped around to the left side.

- https://github.com/JacquesStout/DTC_private/blob/main/Epilepsy/02_bundle_combinedsubj.py 
Combines all subjects into a single trk for each specified side (left side, right side, left_flipped, right_flipped, combined). Also creates a pickle file for a dictionary that keeps track which tracts correspond to which subjects. These are both saved at TBD
Warning! This takes a considerable amount of time. Optimization/parallelization is in order, but would demand some decent amount of code rewrites (biggest problem is to recombine the different dictionaries after the fact, best way would be to parallelize the trk loading and the basic calculations of the number of subjects, then once ALL are loaded, recombine them in a specific order and use saved number of streamlines to reconstruct the dictionary)

- https://github.com/JacquesStout/DTC_private/blob/main/Epilepsy/03_savesplitbundles.py 
Splits the combined brains into a number of bundles described by user, then saves the trks for each subject associated with this particular bundle. This is done for all ‘sides’ (right, left, right flipped, etc) specified by the user. These are saved on a per subject basis. The default setting is ‘bundle_lr_combined=True’ where we create the bundles for ‘combined’, then ‘left’ then ‘right but flipped’ (can always recreate right from this, and it is easily comparable to left). Then all the bundles are saved to:
TBD/trk_roi/{subject}_bundle_lr_{totalnumber of bundles}max

- https://github.com/JacquesStout/DTC_private/blob/main/Epilepsy/04_statssplitbundles.py
Not fully developed, but goes through all those bundles, and for EACH SUBJECT, gets similar statistics to that found in AMD paper, also compares the similarity of the left bundle with the right flipped bundle, possibly other things too.

Code usable results

What all this then, is very good at doing, is separating the brain tracts into bundles that are all equivalent to each other across different subjects, and clearly differentiating the tracts that are associated with the left side and right side, and flipping the right side so that they are directly comparable.
From there, we can compare:
	-How similar is the spatial organization of the tracts are of one side of the brain to the right
	-Possibly if there are any general differences in pattern between different subjects and one subject in particular breaks the mold
	-Obviously comparing standard statistics found, for example, in subject 1/bundle 1, to subject 2/bundle2

There is a bit of an inherent weakness is that these bundles just generally split the whole brain into bundles, but those bundles are not linked to any specific structure, making specific links to brain pathways hard to pin down. 
What it is fairly good at is being able to break down the brain into larger segments and compare whether the general organization in one part of the brain for one subject is particularly different to another. 
Additionally, it can fairly well compare left to right, and therefore is quite usable for figuring out asymmetry. This is relevant to the epilepsy project and Chiari malformation project.


Code speed and possible accelerations

A SMARTER way to do all this would be to change the Bundles feature, so instead of combining all tracts like that into single files, we would instead create a certain number of points in space, then for each subject find all the tracts that are similar enough to this point in space and each other to be affirmed as a bundle.

Additional Notes 

Note: The reason that we don’t individually create the same number of bundles for each subject is that the bundle centroids would be … similar, but still quite a bit different for different subjects, given that there is a decent amount of variance in tract creation between all subjects.

Note: Multiple of these runs also create figures, these are for now saved at ‘/Users/jas/Jacques/Figures_ADDecode’
