# Experiment Results

Source: half-strict batch run through `./cpp/build/fec_run`.

Method: for each dataset and algorithm, run `7` times, use `1` warmup run, then drop the max and min by `total`, and average the remaining `4` runs. This removes cold-start influence and reduces single-run fluctuation.

Algorithms included: `FEC`, `FEC_Union`, `EC`, `FEC_Block`, `FEC_Union_Block`, `FEC_Union_Block_new`, `FEC_Union_Block_new2`, `EC_Block`.

## 001.ply | 49,740 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 27.398 | 3.943 | 9.818 | 11.251 | 1.677 | 45 |
| FEC_Union | 16.368 | 3.966 | 8.917 | 0.980 | 1.851 | 45 |
| EC | 53.035 | 3.806 | 49.228 | 0.000 | 0.000 | 40 |
| FEC_Block | 13.972 | 3.382 | 0.757 | 5.157 | 2.465 | 42 |
| FEC_Union_Block | 12.104 | 4.092 | 0.934 | 4.164 | 2.125 | 42 |
| FEC_Union_Block_new | 12.713 | 4.177 | 0.862 | 3.565 | 2.478 | 42 |
| FEC_Union_Block_new2 | 11.099 | 3.969 | 0.424 | 3.167 | 1.648 | 42 |
| EC_Block | 18.105 | 3.972 | 3.266 | 4.185 | 1.643 | 40 |

Fastest: `FEC_Union_Block_new2` with `total = 11.099 ms`, `clusters = 42`.

## 015.ply | 750,752 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 2264.269 | 79.354 | 200.768 | 1951.126 | 25.753 | 243 |
| FEC_Union | 283.694 | 79.856 | 144.012 | 24.504 | 29.335 | 243 |
| EC | 1198.734 | 79.264 | 1119.470 | 0.000 | 0.000 | 225 |
| FEC_Block | 163.679 | 48.212 | 4.520 | 64.371 | 19.149 | 239 |
| FEC_Union_Block | 180.082 | 50.973 | 5.973 | 76.039 | 21.714 | 239 |
| FEC_Union_Block_new | 153.990 | 49.009 | 5.309 | 53.675 | 19.061 | 239 |
| FEC_Union_Block_new2 | 158.325 | 50.013 | 4.793 | 57.941 | 19.618 | 239 |
| EC_Block | 308.589 | 49.123 | 16.266 | 64.478 | 18.238 | 225 |

Fastest: `FEC_Union_Block_new` with `total = 153.990 ms`, `clusters = 239`.

## 028.ply | 1,397,202 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 7076.524 | 151.071 | 389.792 | 6474.139 | 48.052 | 373 |
| FEC_Union | 525.286 | 151.802 | 262.230 | 44.975 | 55.541 | 373 |
| EC | 2243.414 | 150.648 | 2092.765 | 0.000 | 0.000 | 362 |
| FEC_Block | 297.322 | 83.523 | 3.866 | 115.460 | 33.407 | 368 |
| FEC_Union_Block | 299.811 | 87.416 | 5.213 | 115.028 | 33.837 | 368 |
| FEC_Union_Block_new | 268.472 | 83.855 | 3.326 | 88.559 | 32.142 | 368 |
| FEC_Union_Block_new2 | 265.271 | 83.817 | 3.338 | 86.810 | 29.497 | 368 |
| EC_Block | 517.547 | 84.434 | 13.829 | 104.975 | 30.424 | 362 |

Fastest: `FEC_Union_Block_new2` with `total = 265.271 ms`, `clusters = 368`.

## 046.ply | 2,288,885 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 18366.530 | 256.092 | 683.407 | 17324.798 | 79.168 | 639 |
| FEC_Union | 868.316 | 251.774 | 432.169 | 75.171 | 91.559 | 639 |
| EC | 3750.278 | 253.572 | 3496.706 | 0.000 | 0.000 | 609 |
| FEC_Block | 474.559 | 138.823 | 3.187 | 177.423 | 52.941 | 627 |
| FEC_Union_Block | 477.822 | 139.819 | 3.487 | 179.982 | 51.406 | 627 |
| FEC_Union_Block_new | 446.444 | 139.746 | 4.025 | 150.575 | 53.396 | 627 |
| FEC_Union_Block_new2 | 437.066 | 139.850 | 3.482 | 140.552 | 50.462 | 627 |
| EC_Block | 903.193 | 138.196 | 19.385 | 175.372 | 49.947 | 609 |

Fastest: `FEC_Union_Block_new2` with `total = 437.066 ms`, `clusters = 627`.

## 066.ply | 3,301,629 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 39599.318 | 368.657 | 1148.477 | 37918.534 | 119.140 | 928 |
| FEC_Union | 1304.972 | 377.048 | 647.364 | 117.450 | 137.111 | 928 |
| EC | 5386.730 | 361.710 | 5025.020 | 0.000 | 0.000 | 882 |
| FEC_Block | 714.668 | 204.619 | 5.496 | 267.774 | 75.117 | 914 |
| FEC_Union_Block | 693.416 | 198.268 | 6.406 | 259.847 | 74.936 | 914 |
| FEC_Union_Block_new | 683.880 | 201.692 | 7.809 | 240.722 | 75.631 | 914 |
| FEC_Union_Block_new2 | 656.865 | 200.096 | 4.837 | 224.585 | 71.010 | 914 |
| EC_Block | 1293.628 | 200.513 | 16.782 | 264.892 | 76.503 | 882 |

Fastest: `FEC_Union_Block_new2` with `total = 656.865 ms`, `clusters = 914`.

## 084.ply | 4,208,486 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 64931.736 | 487.959 | 1655.011 | 62539.617 | 172.262 | 1249 |
| FEC_Union | 1642.982 | 473.578 | 816.476 | 144.347 | 175.712 | 1249 |
| EC | 6881.356 | 478.025 | 6403.331 | 0.000 | 0.000 | 1172 |
| FEC_Block | 883.920 | 259.556 | 4.286 | 327.887 | 96.538 | 1222 |
| FEC_Union_Block | 890.839 | 261.684 | 4.064 | 334.225 | 93.731 | 1222 |
| FEC_Union_Block_new | 831.697 | 262.315 | 6.202 | 276.258 | 92.016 | 1222 |
| FEC_Union_Block_new2 | 831.766 | 261.751 | 6.673 | 277.300 | 90.248 | 1222 |
| EC_Block | 1590.145 | 260.822 | 17.050 | 321.565 | 90.228 | 1172 |

Fastest: `FEC_Union_Block_new` with `total = 831.697 ms`, `clusters = 1222`.

## car.ply | 55,048 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 18.063 | 4.065 | 9.562 | 3.145 | 1.025 | 12 |
| FEC_Union | 16.706 | 4.013 | 9.719 | 1.473 | 1.291 | 12 |
| EC | 206.132 | 4.268 | 201.864 | 0.000 | 0.000 | 12 |
| FEC_Block | 11.561 | 4.056 | 1.817 | 3.762 | 1.885 | 12 |
| FEC_Union_Block | 10.903 | 3.728 | 2.295 | 3.320 | 1.578 | 12 |
| FEC_Union_Block_new | 9.347 | 3.706 | 1.901 | 2.312 | 1.632 | 12 |
| FEC_Union_Block_new2 | 9.391 | 3.713 | 1.924 | 2.057 | 1.553 | 12 |
| EC_Block | 48.663 | 3.268 | 38.731 | 4.730 | 1.632 | 12 |

Fastest: `FEC_Union_Block_new` with `total = 9.347 ms`, `clusters = 12`.

## street.ply | 1,129,267 points

| Algorithm | Total (ms) | Build (ms) | Search (ms) | Merge (ms) | Final (ms) | Clusters |
|---|---:|---:|---:|---:|---:|---:|
| FEC | 2400.165 | 120.108 | 266.753 | 1970.430 | 33.346 | 220 |
| FEC_Union | 463.287 | 118.997 | 247.081 | 47.829 | 41.904 | 220 |
| EC | 2994.870 | 113.685 | 2881.185 | 0.000 | 0.000 | 207 |
| FEC_Block | 258.904 | 90.191 | 4.621 | 91.586 | 24.969 | 216 |
| FEC_Union_Block | 253.059 | 90.408 | 4.604 | 84.872 | 24.512 | 216 |
| FEC_Union_Block_new | 233.530 | 91.777 | 4.011 | 65.973 | 26.386 | 216 |
| FEC_Union_Block_new2 | 231.134 | 90.292 | 3.828 | 65.197 | 23.695 | 216 |
| EC_Block | 622.193 | 91.791 | 66.298 | 84.022 | 33.911 | 207 |

Fastest: `FEC_Union_Block_new2` with `total = 231.134 ms`, `clusters = 216`.

