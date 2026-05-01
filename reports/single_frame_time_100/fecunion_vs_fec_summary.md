# SemanticPOSS Single-Frame No-Ground Timing: FECunion vs FEC

## Setup

- Dataset root: `/mnt/d/semanticposs`
- Sequences: `00, 01, 02, 03, 04, 05`
- Frames per sequence: first `100`
- Ground removal: remove `semantic_id=22` before clustering
- Clustering input: whole remaining frame
- Algorithms: `FECunion`, `FEC`
- Parameters: `tol=0.2`, `min_cluster_size=100`, `max_n=50`, `min_gt_points=30`
- FPS definition: `FPS = 1000 / frame_cluster_wall_ms`, then averaged over frames

## Summary

| Sequence | Frames | Avg Raw Points / Frame | Avg Ground Removed / Frame | Avg Clustering Points / Frame | FECunion Time(ms) | FECunion FPS | FEC Time(ms) | FEC FPS | FEC/FECunion Time Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 100 | 65,826 | 19,427 | 46,399 | 17.95 | 55.79 | 23.59 | 42.43 | 1.31x |
| 01 | 100 | 65,896 | 10,312 | 55,584 | 21.24 | 47.17 | 27.64 | 36.27 | 1.30x |
| 02 | 100 | 68,694 | 17,019 | 51,675 | 19.34 | 51.74 | 26.24 | 38.15 | 1.36x |
| 03 | 100 | 69,313 | 15,837 | 53,476 | 21.79 | 45.93 | 29.63 | 33.87 | 1.36x |
| 04 | 100 | 66,897 | 9,764 | 57,133 | 20.35 | 49.16 | 29.13 | 34.46 | 1.43x |
| 05 | 100 | 65,978 | 16,667 | 49,311 | 17.88 | 56.34 | 24.60 | 40.92 | 1.38x |
| Overall | 600 | 67,101 | 14,838 | 52,263 | 19.76 | 51.02 | 26.80 | 37.68 | 1.36x |

## Notes

- `Time(ms)` is average clustering wall time per frame.
- `FPS` is the average of per-frame FPS values reported by the evaluator.
- `FECunion` and `FEC` have identical clustering outputs under this setup, so the speed comparison is the main signal here.
- Sequence `03` has `GT=0` under `min_gt_points=30` for these first 100 frames, but timing and point-count statistics remain valid.
