# SemanticPOSS Report-Style Benchmark

## Setup

- Dataset root: `/mnt/d/semanticposs`
- Sequence: `01`
- Ground removal: `semantic_id=22 removed`
- Parameters: `tol=0.2`, `min_cluster_size=100`, `max_n=50`, `min_gt_points=30`
- FPS definition: `FPS = 1000 / frame_cluster_wall_ms`, then averaged over frames

## Summary

| Algorithm | Frames | Raw Points | Removed Ground | Points | GT | Pred | TP | FP | FN | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FECunion|100|6589551|1031162|5558389|2668|4367|1247|3120|1421|21.24|47.17|29.85|84.19|35.45|46.74|40.96|
|FEC|100|6589551|1031162|5558389|2668|4367|1247|3120|1421|27.64|36.27|29.85|84.19|35.45|46.74|40.96|

## Metric Notes

- `PQ = SQ * RQ`
- `SQ` is the mean IoU of one-to-one matched instances with `IoU >= 0.5`.
- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.
- `RC50 = TP / (TP + FN)` with `IoU >= 0.5`.
- `mIoU` is the mean best-overlap IoU over all GT instances.
