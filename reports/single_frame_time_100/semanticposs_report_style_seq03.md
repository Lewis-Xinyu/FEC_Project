# SemanticPOSS Report-Style Benchmark

## Setup

- Dataset root: `/mnt/d/semanticposs`
- Sequence: `03`
- Ground removal: `semantic_id=22 removed`
- Parameters: `tol=0.2`, `min_cluster_size=100`, `max_n=50`, `min_gt_points=30`
- FPS definition: `FPS = 1000 / frame_cluster_wall_ms`, then averaged over frames

## Summary

| Algorithm | Frames | Raw Points | Removed Ground | Points | GT | Pred | TP | FP | FN | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FECunion|100|6931312|1583682|5347630|0|4683|0|4683|0|21.79|45.93|0.00|0.00|0.00|0.00|0.00|
|FEC|100|6931312|1583682|5347630|0|4683|0|4683|0|29.63|33.87|0.00|0.00|0.00|0.00|0.00|

## Metric Notes

- `PQ = SQ * RQ`
- `SQ` is the mean IoU of one-to-one matched instances with `IoU >= 0.5`.
- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.
- `RC50 = TP / (TP + FN)` with `IoU >= 0.5`.
- `mIoU` is the mean best-overlap IoU over all GT instances.
