# SemanticPOSS Report-Style Benchmark

## Setup

- Dataset root: `/mnt/d/semanticposs`
- Sequence: `02`
- Ground removal: `semantic_id=22 removed`
- Parameters: `tol=0.2`, `min_cluster_size=100`, `max_n=50`, `min_gt_points=30`
- FPS definition: `FPS = 1000 / frame_cluster_wall_ms`, then averaged over frames

## Summary

| Algorithm | Frames | Raw Points | Removed Ground | Points | GT | Pred | TP | FP | FN | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FECunion|100|6869394|1701887|5167507|2022|5282|721|4561|1301|19.34|51.74|16.29|82.50|19.74|35.66|31.22|
|FEC|100|6869394|1701887|5167507|2022|5282|721|4561|1301|26.24|38.15|16.29|82.50|19.74|35.66|31.22|

## Metric Notes

- `PQ = SQ * RQ`
- `SQ` is the mean IoU of one-to-one matched instances with `IoU >= 0.5`.
- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.
- `RC50 = TP / (TP + FN)` with `IoU >= 0.5`.
- `mIoU` is the mean best-overlap IoU over all GT instances.
