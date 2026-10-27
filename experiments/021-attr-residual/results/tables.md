# 021-attr-residual: H-A2 tables

Labels: (i) **INCONCLUSIVE-i**, (ii) **PASS-ii**, overall **PASS**. M1 margin (CI lower bound of bacc minus M1 >= 0.10, clean): False.

## (i) Generators, K=8, 256x256 top-left crop, chance 0.125
Train 717 / test 303 images; families 379 / 163; test images per generator {'dalle2': 40, 'dalle3': 30, 'firefly': 36, 'midjourney-v5': 36, 'stable-diffusion-2': 43, 'stable-diffusion-xl': 29, 'sd3_large': 50, 'flux': 39}.

| condition | fingerprint from | bacc [95% CI] | spectral bacc | M1 bacc | bacc - M1 [CI] | bacc w/o Firefly (K=7) | pairwise [CI] | perm mean (max) | split-seed mean+-sd (10) |
|---|---|---|---|---|---|---|---|---|---|
| clean | clean | 0.522 [0.478, 0.567] | 0.254 [0.223, 0.289] | 0.530 [0.484, 0.577] | -0.008 [-0.072, 0.057] | 0.479 (chance 0.143) | 0.840 [0.776, 0.894] | 0.140 (0.317) | 0.506 +- 0.018 |
| jpeg75_resize05 | clean | 0.114 [0.080, 0.151] | 0.193 [0.165, 0.220] | 0.278 [0.243, 0.314] | -0.163 [-0.215, -0.111] | 0.149 (chance 0.143) | 0.459 [0.393, 0.536] | 0.125 (0.185) | 0.117 +- 0.014 |
| jpeg75_resize05 | matched | 0.523 [0.476, 0.567] | 0.290 [0.243, 0.334] | 0.454 [0.404, 0.508] | 0.069 [-0.006, 0.139] | 0.466 (chance 0.143) | 0.835 [0.777, 0.897] | 0.135 (0.256) | 0.537 +- 0.021 |
| social_sim | clean | 0.164 [0.125, 0.204] | 0.205 [0.176, 0.234] | 0.305 [0.256, 0.356] | -0.141 [-0.208, -0.072] | 0.170 (chance 0.143) | 0.479 [0.394, 0.563] | 0.126 (0.165) | 0.134 +- 0.017 |
| social_sim | matched | 0.211 [0.169, 0.257] | 0.234 [0.199, 0.269] | 0.426 [0.380, 0.468] | -0.215 [-0.276, -0.149] | 0.209 (chance 0.143) | 0.613 [0.533, 0.692] | 0.125 (0.184) | 0.219 +- 0.020 |

M0 (raw-file metadata, random forest): bacc 0.674 [0.628, 0.716]; M0 on the preprocessed 256x256 tensors: 0.125 (constant features -> chance). Pairwise chance 0.5.

### Per-generator recall (clean, clean fingerprints)
| dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl | sd3_large | flux |
|---|---|---|---|---|---|---|---|
| 0.175 | 0.367 | 1.000 | 0.472 | 0.419 | 0.345 | 0.400 | 1.000 |

### Confusion matrix, clean (rows true, cols predicted; order as above)
- dalle2: [7, 4, 2, 4, 6, 3, 6, 8]
- dalle3: [3, 11, 4, 3, 0, 3, 3, 3]
- firefly: [0, 0, 36, 0, 0, 0, 0, 0]
- midjourney-v5: [0, 5, 4, 17, 5, 1, 2, 2]
- stable-diffusion-2: [2, 1, 1, 8, 18, 8, 2, 3]
- stable-diffusion-xl: [3, 1, 7, 1, 1, 10, 0, 6]
- sd3_large: [5, 2, 5, 1, 6, 3, 20, 8]
- flux: [0, 0, 0, 0, 0, 0, 0, 39]

## (ii) VISION devices, 14-way, full resolution (native), PRNU/PCE, leave-one-photo-out

| arm | photos | top-1 [CI] (chance 0.071) | within-model [CI] (chance 0.445) | TPR@PCE60 [CI] | FPR@PCE60 all [CI] | FPR same-model | FPR diff-model | peak at 0 shift (true dev; PCE>=60) | median PCE true / false |
|---|---|---|---|---|---|---|---|---|---|
| native -> native | 197 | 0.985 [0.969, 1.000] | 1.000 [1.000, 1.000] | 0.964 [0.939, 0.985] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.989 (n=190); all true: 0.959; false PCE>=60: n=0 | 1091.0 / 1.7 |
| WA -> WA | 197 | 0.930 [0.895, 0.964] | 0.977 [0.947, 1.000] | 0.371 [0.320, 0.421] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 1.000 (n=73); all true: 0.629; false PCE>=60: n=0 | 37.7 / 0.3 |
| FBH -> FBH | 197 | 0.871 [0.829, 0.911] | 0.977 [0.947, 1.000] | 0.381 [0.330, 0.431] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 1.000 (n=75); all true: 0.563; false PCE>=60: n=0 | 35.5 / 0.3 |
| native fingerprint resized -> WA | 197 | 0.915 [0.880, 0.951] | 0.969 [0.938, 0.993] | 0.452 [0.401, 0.503] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 1.000 (n=89); all true: 0.695; false PCE>=60: n=0 | 52.9 / 0.3 |

### Native arm: per-device top-1 and within-model accuracy
| device | top-1 | within-model |
|---|---|---|
| D02 | 1.000 | 1.000 |
| D04 | 1.000 | n/a |
| D05 | 0.857 | 1.000 |
| D06 | 1.000 | 1.000 |
| D10 | 1.000 | 1.000 |
| D11 | 1.000 | n/a |
| D14 | 1.000 | 1.000 |
| D15 | 1.000 | 1.000 |
| D18 | 1.000 | 1.000 |
| D19 | 1.000 | n/a |
| D21 | 1.000 | n/a |
| D29 | 1.000 | 1.000 |
| D33 | 0.929 | n/a |
| D34 | 1.000 | 1.000 |

Within-model by set (native): {"iPhone4s": 1.0, "iPhone5c": 1.0, "iPhone6": 1.0, "iPhone5": 1.0}

## Permutation control (devices)
zero-shift NCC on centre 1024x1024 crop of the native, LOPO: true labels 0.974; 200 shuffles of train labels: mean 0.074, 95% interval [0.015, 0.161], max 0.179 (chance 0.071); p(perm >= real) = 0.0050.
