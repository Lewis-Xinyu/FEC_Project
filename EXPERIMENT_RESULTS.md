# Experiment Results

All results below are measured on the datasets in `data/`.

Notes:

- `046.ply` on `FEC_Union` was re-run after fixing the timer to use `steady_clock`.
- Time units are milliseconds.

## Summary

### Overall trend

From the full set of experiments, the main performance pattern is stable:

- `FEC` is consistently the slowest algorithm within the FEC family.
- `FEC1` and `FEC_Union` are much faster than `FEC`, showing that the main bottleneck of the original version is the label-merging strategy rather than KD-tree search itself.
- The block-based parallel versions, especially `FEC_Block`, `FEC1_Block`, and `FEC_Union_Block`, are consistently the fastest group on most datasets.
- `EC` is slower than the optimized FEC-family methods, while `EC_Block` improves it significantly but still usually remains behind the best FEC block variants.
- `RG` is the slowest overall because it performs normal estimation first and then region growing with geometric constraints instead of simple Euclidean connectivity.

### Algorithm differences

#### FEC

`FEC` is the original label-propagation version.

- For each point, it performs a radius search.
- If multiple labels collide, it resolves them by scanning the whole label array and rewriting old labels to the new minimum label.

This means the algorithm spends a large amount of time in `merge`, especially on large datasets. That is exactly what the timing tables show: the `merge` stage dominates almost all runtime in `FEC`.

#### FEC1

`FEC1` keeps the same basic clustering logic as `FEC`, but replaces the expensive full-array relabel operation with a hash map from tag to point indices.

- Instead of scanning all points to find which ones belong to an old tag,
- it directly accesses the point list for that tag and updates only those entries.

This is why `FEC1` dramatically reduces `merge` time compared with `FEC`, while `build` and `search` remain in a similar range.

#### FEC_Union

`FEC_Union` changes the merging strategy again.

- It no longer treats cluster merging as repeated tag rewriting.
- It uses a disjoint-set union structure to maintain connectivity directly.

Compared with `FEC1`, it avoids explicit tag-management logic and instead uses DSU unions. In practice, `FEC1` and `FEC_Union` are usually close. On some datasets `FEC_Union` is slightly faster, and on others `FEC1` is slightly faster. This indicates that after removing the original full-array relabel bottleneck, the remaining differences are mostly constant-factor effects from the chosen data structure.

#### EC

`EC` is PCL's standard Euclidean clustering implementation.

- It serves as a strong baseline.
- Its internal process is dominated by cluster extraction logic inside the PCL implementation.

This is why most of its time appears in `search` in the current timing split. Relative to the FEC-family optimizations, `EC` is usually slower because it does not use your specialized point-wise label or DSU scheme.

#### RG

`RG` uses a different segmentation principle.

- It first estimates normals.
- Then it grows regions using smoothness and curvature constraints.

Because of that, it is not directly comparable to Euclidean clustering only in algorithmic structure. Its `search` cost includes normal estimation, and its `merge` cost includes the region-growing segmentation itself. This makes it consistently the slowest group, but it is also solving a more geometry-constrained problem.

#### FEC_Block

`FEC_Block` keeps the original `FEC` local logic but moves it into a block-parallel framework.

- The point cloud is spatially partitioned into blocks.
- Each block processes a local expanded neighborhood.
- Blocks run in parallel with OpenMP.
- Local clusters are merged globally afterward.

Its improvement comes from two sources:

- the relabel scan is no longer over the full global cloud, only over local block data,
- and multiple blocks run in parallel.

This is why `FEC_Block` is much faster than `FEC` despite keeping the original relabel-style local algorithm.

#### FEC1_Block

`FEC1_Block` combines the `FEC1` hash-map local strategy with the same block-parallel framework.

- Local clustering uses the optimized tag-to-points structure,
- and the global workload is reduced further through partitioning and parallel execution.

In theory this should often outperform `FEC_Block`, but in practice the difference is sometimes small. That is because once the workload is already localized and parallelized, the relative benefit of the hash-map optimization may shrink depending on the block size and point distribution.

#### FEC_Union_Block

`FEC_Union_Block` combines the block-parallel framework with a DSU-based local clustering strategy.

- Inside each block, unions are performed directly.
- Across blocks, another DSU-based merge is used at the cluster level.

This version is the most complete combination of your optimization ideas:

- DSU instead of tag rewriting,
- block partitioning,
- parallel execution,
- global merge only after local work is finished.

It is the fastest or near-fastest variant on most datasets, which suggests that this combination is the strongest candidate for overall efficiency in your current implementation.

#### EC_Block

`EC_Block` applies the same block-parallel skeleton to standard Euclidean clustering.

- It keeps `EC` as the local cluster extractor,
- but accelerates execution through spatial partitioning and parallel block processing.

Its results show that the block framework itself is highly effective, because `EC_Block` is much faster than `EC`. However, it is still usually slower than the best FEC block variants, which suggests that your local FEC-family clustering logic is more efficient than standard EC once both are placed inside the same parallel framework.

### Why the time components differ

#### Build

`build` mainly includes KD-tree construction and, for block methods, also includes partition-related setup in the reported combined value.

- In non-block algorithms, `build` is mostly the cost of building one global KD-tree.
- In block algorithms, `build` includes block partitioning, expanded neighborhood preparation, and the maximum local KD-tree build time among threads.

That is why block methods do not always have a smaller `build` value, even when total runtime is much lower.

#### Search

`search` is dominated by neighborhood queries.

- In `FEC`, `FEC1`, and `FEC_Union`, this is repeated global radius search.
- In `EC`, most extraction work is absorbed into the cluster extraction stage and appears under `search` in this timing split.
- In `RG`, `search` includes normal estimation, which is inherently expensive.
- In block methods, local neighborhoods are much smaller, so search often drops sharply.

This is why the block algorithms usually show very small `search` time compared with their non-block counterparts.

#### Merge

`merge` is where the biggest algorithmic differences appear.

- In `FEC`, `merge` is dominated by repeated full-array relabeling.
- In `FEC1`, `merge` drops sharply because only the relevant label lists are updated.
- In `FEC_Union`, `merge` becomes DSU union work.
- In block algorithms, `merge` includes local merging plus the final global merge across blocks.
- In `RG`, `merge` corresponds to the region-growing extraction stage itself.

This is the main reason `FEC` scales poorly while `FEC1`, `FEC_Union`, and block variants scale much better.

#### Final

`final` mostly reflects output organization:

- sorting labels,
- grouping indices,
- deduplicating owner points in block methods,
- preparing final cluster outputs.

Its value is usually much smaller than `merge`, but it still grows with dataset size and cluster count.

### Main conclusion

The experiments show two clear optimization axes:

1. Data-structure optimization:
   `FEC -> FEC1 -> FEC_Union`

- `FEC1` improves `FEC` by replacing global relabel scans with hash-based tag management.
- `FEC_Union` replaces label management with DSU connectivity merging.

2. Parallel/block optimization:
   `FEC / FEC1 / FEC_Union / EC -> *_Block`

- Block partitioning reduces local problem size.
- OpenMP parallelism improves CPU utilization.
- Global merging is postponed until after local clustering.

Taken together, the best-performing family in these experiments is the block-based FEC family, with `FEC_Union_Block` being the strongest overall candidate for efficiency.

## 001.ply | 49,740 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 28.923 | 4.757 | 10.540 | 10.775 | 2.048 | 45 |
| FEC1 | 18.713 | 4.103 | 10.526 | 1.313 | 1.834 | 45 |
| FEC_Union | 18.137 | 4.129 | 10.052 | 1.156 | 2.134 | 45 |
| EC | 56.229 | 4.086 | 52.143 | 0.000 | 0.000 | 40 |
| RG | 304.987 | 0.004 | 138.965 | 166.018 | 0.000 | 40 |
| FEC_Block | 16.151 | 7.710 | 3.002 | 4.338 | 3.583 | 42 |
| FEC1_Block | 14.591 | 7.433 | 2.789 | 4.440 | 1.488 | 42 |
| FEC_Union_Block | 14.150 | 4.673 | 3.478 | 4.397 | 1.776 | 42 |
| EC_Block | 22.557 | 4.570 | 3.649 | 4.301 | 1.694 | 40 |

## 015.ply | 750,752 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 2252.911 | 98.117 | 217.897 | 1893.881 | 33.211 | 243 |
| FEC1 | 366.197 | 99.630 | 186.984 | 28.134 | 33.827 | 243 |
| FEC_Union | 348.620 | 87.934 | 184.344 | 39.035 | 30.772 | 243 |
| EC | 1290.054 | 86.364 | 1203.690 | 0.000 | 0.000 | 225 |
| RG | 6683.824 | 0.005 | 2773.464 | 3910.355 | 0.000 | 370 |
| FEC_Block | 201.759 | 64.285 | 8.814 | 79.527 | 30.332 | 239 |
| FEC1_Block | 201.670 | 53.350 | 7.516 | 91.529 | 29.056 | 239 |
| FEC_Union_Block | 184.148 | 54.694 | 6.423 | 73.233 | 24.437 | 239 |
| EC_Block | 338.279 | 53.371 | 15.311 | 69.375 | 20.547 | 225 |

## 028.ply | 1,397,202 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 6776.903 | 173.700 | 425.217 | 6113.027 | 50.534 | 373 |
| FEC1 | 595.955 | 163.961 | 312.194 | 39.749 | 51.483 | 373 |
| FEC_Union | 599.664 | 166.116 | 303.742 | 57.183 | 60.773 | 373 |
| EC | 2427.680 | 165.128 | 2262.552 | 0.000 | 0.000 | 362 |
| RG | 10344.384 | 0.006 | 3449.845 | 6894.533 | 0.000 | 651 |
| FEC_Block | 329.094 | 93.753 | 4.436 | 126.355 | 41.714 | 368 |
| FEC1_Block | 325.786 | 95.086 | 4.796 | 126.228 | 34.992 | 368 |
| FEC_Union_Block | 334.838 | 100.541 | 11.687 | 127.056 | 43.221 | 368 |
| EC_Block | 603.286 | 92.783 | 16.514 | 124.798 | 35.789 | 362 |

## 046.ply | 2,288,885 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 17416.847 | 271.190 | 755.812 | 16279.744 | 84.625 | 639 |
| FEC1 | 1007.924 | 277.396 | 530.405 | 68.010 | 83.865 | 639 |
| FEC_Union | 1108.126 | 363.834 | 502.311 | 102.216 | 114.322 | 639 |
| EC | 3997.188 | 272.484 | 3724.703 | 0.000 | 0.000 | 609 |
| RG | 20313.504 | 0.006 | 8356.514 | 11956.983 | 0.000 | 1124 |
| FEC_Block | 586.421 | 158.880 | 8.005 | 239.117 | 61.508 | 627 |
| FEC1_Block | 542.214 | 151.759 | 4.472 | 208.785 | 59.708 | 627 |
| FEC_Union_Block | 540.367 | 155.477 | 7.107 | 208.650 | 59.342 | 627 |
| EC_Block | 982.172 | 159.924 | 19.378 | 210.910 | 54.439 | 609 |

## 066.ply | 3,301,629 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 37135.044 | 401.000 | 1248.444 | 35295.263 | 134.505 | 928 |
| FEC1 | 1448.094 | 401.627 | 755.544 | 97.443 | 124.409 | 928 |
| FEC_Union | 1553.166 | 406.756 | 798.841 | 166.944 | 151.884 | 928 |
| EC | 5768.198 | 401.632 | 5366.565 | 0.000 | 0.000 | 882 |
| RG | 26494.072 | 0.005 | 9244.412 | 17249.655 | 0.000 | 1573 |
| FEC_Block | 798.754 | 232.765 | 4.233 | 304.594 | 85.371 | 914 |
| FEC1_Block | 802.327 | 224.063 | 5.891 | 305.921 | 84.241 | 914 |
| FEC_Union_Block | 786.833 | 216.878 | 9.284 | 302.805 | 83.235 | 914 |
| EC_Block | 1465.142 | 229.304 | 20.166 | 281.435 | 78.726 | 882 |

## 084.ply | 4,208,486 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 57234.417 | 519.107 | 1700.177 | 54732.647 | 196.871 | 1249 |
| FEC1 | 2067.956 | 560.088 | 1071.749 | 146.822 | 190.742 | 1249 |
| FEC_Union | 1950.733 | 543.839 | 986.070 | 191.206 | 193.054 | 1249 |
| EC | 5223.757 | 530.842 | 4692.914 | 0.000 | 0.000 | 1172 |
| RG | 36197.684 | 0.007 | 15803.660 | 20394.017 | 0.000 | 1930 |
| FEC_Block | 1075.190 | 298.224 | 12.614 | 436.028 | 110.853 | 1222 |
| FEC1_Block | 1083.482 | 289.311 | 56.309 | 396.742 | 127.202 | 1222 |
| FEC_Union_Block | 1026.732 | 296.531 | 10.210 | 382.758 | 104.254 | 1222 |
| EC_Block | 2057.185 | 312.932 | 33.873 | 393.353 | 111.943 | 1172 |

## car.ply | 55,048 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 21.124 | 5.488 | 11.453 | 2.771 | 1.111 | 12 |
| FEC1 | 18.412 | 4.605 | 11.259 | 0.798 | 1.125 | 12 |
| FEC_Union | 19.212 | 4.680 | 11.169 | 1.652 | 1.482 | 12 |
| EC | 230.515 | 4.612 | 225.903 | 0.000 | 0.000 | 12 |
| RG | 420.074 | 0.005 | 191.724 | 228.345 | 0.000 | 36 |
| FEC_Block | 13.237 | 3.706 | 1.547 | 4.635 | 1.963 | 12 |
| FEC1_Block | 14.199 | 4.227 | 3.177 | 4.971 | 2.086 | 12 |
| FEC_Union_Block | 12.544 | 4.806 | 2.914 | 3.599 | 2.637 | 12 |
| EC_Block | 64.856 | 5.547 | 53.292 | 5.184 | 1.945 | 12 |

## street.ply | 1,129,267 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 2442.621 | 139.865 | 295.313 | 1962.693 | 34.500 | 220 |
| FEC1 | 437.357 | 122.777 | 240.397 | 23.491 | 32.406 | 220 |
| FEC_Union | 450.775 | 120.827 | 237.747 | 44.579 | 40.210 | 220 |
| EC | 3242.393 | 119.393 | 3123.000 | 0.000 | 0.000 | 207 |
| RG | 10121.482 | 0.006 | 4282.791 | 5838.684 | 0.000 | 372 |
| FEC_Block | 292.389 | 106.098 | 7.141 | 102.515 | 30.049 | 216 |
| FEC1_Block | 297.751 | 105.649 | 5.338 | 106.103 | 31.791 | 216 |
| FEC_Union_Block | 306.386 | 118.417 | 7.788 | 98.895 | 31.576 | 216 |
| EC_Block | 706.127 | 106.097 | 63.586 | 91.942 | 28.103 | 207 |
