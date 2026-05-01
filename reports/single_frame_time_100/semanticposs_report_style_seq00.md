# SemanticPOSS Report-Style Benchmark

## Setup

- Dataset root: `/mnt/d/semanticposs`
- Sequence: `00`
- Ground removal: `semantic_id=22 removed`
- Parameters: `tol=0.2`, `min_cluster_size=100`, `max_n=50`, `min_gt_points=30`
- FPS definition: `FPS = 1000 / frame_cluster_wall_ms`, then averaged over frames

## Summary

| Algorithm | Frames | Raw Points | Removed Ground | Points | GT | Pred | TP | FP | FN | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FECunion|100|6582595|1942695|4639900|2585|4651|1161|3490|1424|17.95|55.79|26.61|82.92|32.09|44.91|39.77|
|FEC|100|6582595|1942695|4639900|2585|4651|1161|3490|1424|23.59|42.43|26.61|82.92|32.09|44.91|39.77|

## Metric Notes

- `PQ = SQ * RQ`
- `SQ` is the mean IoU of one-to-one matched instances with `IoU >= 0.5`.
- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.
- `RC50 = TP / (TP + FN)` with `IoU >= 0.5`.
- `mIoU` is the mean best-overlap IoU over all GT instances.
