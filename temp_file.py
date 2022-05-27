
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature, AveragePointwiseEuclideanMetric
import warnings
from dipy.align.streamlinear import StreamlineLinearRegistration
import copy


#making bundles out of streamlines (subselect streamlines of specific connections
num_points2 = 50
distance2 = 2
feature2 = ResampleFeature(nb_points=num_points2)
metric2 = AveragePointwiseEuclideanMetric(feature=feature2)
qb = QuickBundles(threshold=distance2, metric=metric2)
clusters = qb.cluster(streamlines)

#Order the bundles in order of most streamlines
if selection == 'num_streams':
    num_streamlines = [np.shape(cluster)[0] for cluster in group_clusters.clusters]
    num_streamlines = clusters.clusters_sizes()
    top_bundles = sorted(range(len(num_streamlines)), key=lambda i: num_streamlines[i], reverse=True)[:]

#define the number of bundles you want to look at
num_bundles = 20

#basic way to extract the bundles, centroids, sizes of the top {num_bundles} clusters (could probably be improved tbh)
for bundle in top_bundles:
    selected_bundles.append(group_clusters.clusters[bundle])
    selected_centroids.append(group_clusters.centroids[bundle])
    selected_sizes.append(group_clusters.clusters_sizes()[bundle])
    num_bundles_group += 1


#from here on you want to have two bundle groups, depending on which subject you are looking at
#get the distance between centroids of different bundles between the two groups
for g3 in np.arange(num_bundles):
    for g4 in np.arange(num_bundles):
        dist_all[g3, g4] = (mdf(selected_centroids_subj1[g3], selected_centroids_subj2[g4]))

dist_all_fix = copy.copy(dist_all)
dist_all_idx = []

#creates the dist_all_idx which associates one bundle with
for i in np.arange(num_bundles):
    idx = np.argmin(dist_all_fix[i, :])
    dist_all_idx.append([i, idx])
    #set to 100000 to avoid the same bundle to be chosen multiple times, prioritizes the biggest bundles and descends from there
    dist_all_fix[:, idx] = 100000