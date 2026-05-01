# SemanticPOSS Report-Style Benchmark

## Setup

- Dataset root: `/mnt/d/semanticposs`
- Sequence: `05`
- Ground removal: `semantic_id=22 removed`
- Parameters: `tol=0.2`, `min_cluster_size=100`, `max_n=50`, `min_gt_points=30`
- FPS definition: `FPS = 1000 / frame_cluster_wall_ms`, then averaged over frames

## Summary

| Algorithm | Frames | Raw Points | Removed Ground | Points | GT | Pred | TP | FP | FN | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FECunion|100|6597826|1666706|4931120|2055|4856|810|4046|1245|17.88|56.34|19.91|84.93|23.44|39.42|36.14|
|FEC|100|6597826|1666706|4931120|2055|4856|810|4046|1245|24.60|40.92|19.91|84.93|23.44|39.42|36.14|

## Metric Notes

- `PQ = SQ * RQ`
- `SQ` is the mean IoU of one-to-one matched instances with `IoU >= 0.5`.
- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.
- `RC50 = TP / (TP + FN)` with `IoU >= 0.5`.
- `mIoU` is the mean best-overlap IoU over all GT instances.
