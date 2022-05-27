



if registration:
    srr = StreamlineLinearRegistration()
    for streamline, i in enumerate(selected_centroids[non_control]):
        srm = srr.optimize(static=selected_centroids[control], moving=streamline)
        streamlines[control][i] = srm.transform(streamline)

from dipy.segment.metric import mdf

# dist_all = np.zeros((np.size(selected_bundles[control]), np.size(selected_bundles[non_control])))
# dist_all = np.zeros((num_bundles, num_bundles))
dist_all_w = np.zeros((num_bundles, num_bundles))
dist_all_j = np.zeros((num_bundles, num_bundles))

if test_mode:
    top_idx_group_control = sorted(range(len(selected_sizes[control])),
                                   key=lambda i: selected_sizes[group][i], reverse=True)[:num_bundles]
    top_idx_group_noncontrol = sorted(range(len(selected_sizes[non_control])),
                                      key=lambda i: selected_sizes[group][i], reverse=True)[:num_bundles]

    if not np.all(top_idx_group_control == np.arange(num_bundles)) or not np.all(
            top_idx_group_noncontrol == np.arange(num_bundles)):
        warnings.warn('There is indeed a difference between the two')
    else:
        print('no difference between the two')

# wenlin version based on distance of streamlines from centroid
for g3 in np.arange(num_bundles):
    for g4 in np.arange(num_bundles):
        dist_all_w[g3, g4] = (mdf(selected_centroids[control][g3], selected_centroids[non_control][g4]))

rng = np.random.RandomState()
clust_thr = [0]
threshold = 10
for g3 in np.arange(num_bundles):
    for g4 in np.arange(num_bundles):
        dist_all_j[g3, g4] = bundle_shape_similarity(selected_bundles[control][g3], selected_bundles[non_control][g4],
                                                     rng, clust_thr, threshold)

dist_all_fix = copy.copy(dist_all_j)
dist_all_idx = []
# for i in range(len(selected_centroids[group])):


for i in np.arange(num_bundles):
    idx = np.argmin(dist_all_fix[i, :])
    dist_all_idx.append([i, idx])
    dist_all_fix[:, idx] = 100000

dist_group3_idx = [dist_all_idx[iii][0] for iii in range(num_bundles)]  # size id
dist_group4_idx = [dist_all_idx[iii][1] for iii in range(num_bundles)]  # size id

group_list = {}
dist_idx = {}
for j, group in enumerate(groups):
    dist_idx[group] = [dist_all_idx[iii][j] for iii in range(num_bundles)]
    group_list[group] = ([np.arange(num_bundles)[dist_all_idx[i][j]] for i in range(num_bundles)])

num_bundles_full_stats = 10

if calculate_weights:
    import dipy.tracking.streamline as dts
    import dipy.stats.analysis as dsa

    for bundle1, bundle2 in zip(selected_bundles[groups[0]][idbundle], selected_bundles[groups[1]][idbundle]):
        oriented_group1 = dts.orient_by_streamline(bundle1.indices, bundle1.centroids[0])
        oriented_group2 = dts.orient_by_streamline(bundle2.indices, bundle2.centroids[0])
        w_group1 = dsa.gaussian_weights(oriented_group1)
        w_group2 = dsa.gaussian_weights(oriented_group2)

for group in groups:
    groupcsv = np.zeros((1, 5 + np.size(references)))
    references_string = "_".join(references)
    csv_summary = os.path.join(stats_folder,
                               group + '_' + region_connection + ratio_str + f'_bundle_stats_{references_string}.csv')
    if not os.path.exists(csv_summary) or overwrite:
        for i in range(num_bundles_full_stats):
            idsize = dist_idx[group][i]
            idbundle = group_list[group][i]
            fa = []
            for s in selected_bundles[group][idbundle].indices:
                # temp = np.hstack((idsize * np.ones((num_points, 1)),
                #                  idbundle * np.ones((num_points, 1)),
                #                  s * np.ones((num_points, 1)),
                #                  np.array(range(num_points)).reshape(num_points, 1),
                #                  list(utils.length([streamlines[group][s]])) * np.ones((num_points, 1)),
                #                  np.array(ref_points[group, ref][s]).reshape(num_points, 1)))
                temp = np.hstack((idsize * np.ones((num_points, 1)),
                                  idbundle * np.ones((num_points, 1)),
                                  s * np.ones((num_points, 1)),
                                  np.array(range(num_points)).reshape(num_points, 1),
                                  list(utils.length([streamlines[group][s]])) * np.ones((num_points, 1))))
                for ref in references:
                    temp = np.hstack((temp, np.array(ref_points[group, ref][s]).reshape(num_points, 1)))
                groupcsv = np.vstack((groupcsv, temp))
                if add_bcoherence:
                    fbc = FBCMeasures(s, k)
                    fbc_sl_orig, clrs_orig, rfbc_orig = \
                        fbc.get_points_rfbc_thresholded(0, emphasis=0.01)
                groupcsv = groupcsv[1:, :]
        groupcsvDF = pd.DataFrame(groupcsv)
        groupcsvDF.rename(index=str, columns={0: "Bundle Size Rank", 1: "Bundle ID", 2: "Steamlines ID",
                                              3: "Point ID", 4: "length"})
        for i, ref in enumerate(references):
            groupcsvDF.rename(index=str, columns={5 + i: ref})
        print('writing')
        groupcsvDF.to_csv(csv_summary,
                          header=["Bundle Size Rank", "Bundle ID", "Streamlines ID", "Point ID", "Length"] + references)
        print(f'Writing bundle stats for {group} and {region_connection} to {csv_summary}')
    else:
        print(f'The file {csv_summary} already exists and no overwrite enabled: skipping')
