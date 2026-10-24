# E3 PILOT: RAID transfer attacks vs canonicalization

Device cuda, preprocessing p79d04831. 95% CIs: bootstrap over ids (2000 resamples, seed 0), reals and each generator's fakes resampled independently.
Pooled AUROC = real vs the generators pooled. Higher score = more fake. Pilot: does not count toward the stop rule.

## Counts

- PRIMARY (labelled): unpadded reals vs unpadded fakes (both sides >= 256 px in the original): 48 reals, 400 fakes {'gen_0': 100, 'gen_1': 100, 'gen_2': 100, 'gen_3': 100}.
- SENSITIVITY (never labelled): all 100 reals (52 padded, with black borders after RAID's zero-pad crop) vs 400 fakes.

- Pairing check (clean control vs attacked, mean |diff| in grey levels per item): eps8 max 7.24 (limit 10), eps16 max 10.96 (limit 18); max pixel diff 9 / 17.

## B0 label (Amendment 9, primary set)

**UNINFORMATIVE**  (AUROC clean 0.643, drop 0.083, recovery 0.419); bootstrap label shares: UNINFORMATIVE 1.00, PASS 0.00, FAIL 0.00, INCONCLUSIVE 0.00

## PRIMARY (unpadded): pooled AUROC by input set

| detector | clean | clean_canon | eps8 | eps8_canon | eps16 | eps16_canon |
|---|---|---|---|---|---|---|
| b0 | 0.643 [0.560, 0.723] | 0.602 [0.516, 0.683] | 0.559 [0.470, 0.646] | 0.594 [0.509, 0.677] | 0.573 [0.483, 0.657] | 0.578 [0.490, 0.661] |
| communityforensics | 0.952 [0.928, 0.972] | 0.797 [0.728, 0.857] | 0.719 [0.643, 0.787] | 0.736 [0.654, 0.814] | 0.643 [0.566, 0.721] | 0.686 [0.609, 0.760] |
| rine | 0.898 [0.862, 0.933] | 0.817 [0.755, 0.874] | 0.754 [0.687, 0.823] | 0.786 [0.719, 0.847] | 0.651 [0.561, 0.736] | 0.739 [0.665, 0.811] |
| clipdet | 0.821 [0.758, 0.879] | 0.730 [0.650, 0.798] | 0.735 [0.661, 0.803] | 0.691 [0.606, 0.771] | 0.724 [0.646, 0.795] | 0.693 [0.606, 0.773] |

### PRIMARY (unpadded): drop / recovery / canonicalization cost, eps 8/255

| detector | drop | gain (canon on attacked) | recovery | canon clean cost |
|---|---|---|---|---|
| b0 | 0.083 [0.042, 0.127] | 0.035 [-0.009, 0.081] | 0.419 [-0.146, 0.831] (n=1999) | 0.041 [0.016, 0.066] |
| communityforensics | 0.233 [0.175, 0.296] | 0.018 [-0.060, 0.096] | 0.075 [-0.297, 0.368] (n=2000) | 0.155 [0.105, 0.211] |
| rine | 0.144 [0.084, 0.206] | 0.031 [-0.040, 0.103] | 0.218 [-0.353, 0.614] (n=2000) | 0.081 [0.028, 0.142] |
| clipdet | 0.086 [0.025, 0.149] | -0.043 [-0.109, 0.024] | -0.502 [-2.277, 0.213] (n=1964) | 0.091 [0.027, 0.160] |

### PRIMARY (unpadded): drop / recovery / canonicalization cost, eps 16/255 (descriptive)

| detector | drop | gain (canon on attacked) | recovery | canon clean cost |
|---|---|---|---|---|
| b0 | 0.069 [0.021, 0.116] | 0.005 [-0.034, 0.044] | 0.069 [-0.837, 0.499] (n=1955) | 0.041 [0.016, 0.066] |
| communityforensics | 0.308 [0.234, 0.385] | 0.043 [-0.058, 0.139] | 0.138 [-0.227, 0.390] (n=2000) | 0.155 [0.105, 0.211] |
| rine | 0.248 [0.175, 0.325] | 0.089 [0.004, 0.170] | 0.359 [0.022, 0.602] (n=2000) | 0.081 [0.028, 0.142] |
| clipdet | 0.097 [0.031, 0.167] | -0.031 [-0.095, 0.037] | -0.320 [-1.850, 0.295] (n=1976) | 0.091 [0.027, 0.160] |

## SENSITIVITY (all 100 reals, unlabelled): pooled AUROC by input set

| detector | clean | clean_canon | eps8 | eps8_canon | eps16 | eps16_canon |
|---|---|---|---|---|---|---|
| b0 | 0.673 [0.612, 0.733] | 0.663 [0.601, 0.724] | 0.605 [0.541, 0.669] | 0.645 [0.579, 0.705] | 0.606 [0.537, 0.667] | 0.634 [0.567, 0.696] |
| communityforensics | 0.954 [0.934, 0.971] | 0.775 [0.723, 0.824] | 0.689 [0.628, 0.745] | 0.701 [0.642, 0.760] | 0.625 [0.566, 0.679] | 0.657 [0.600, 0.716] |
| rine | 0.886 [0.853, 0.914] | 0.837 [0.794, 0.878] | 0.686 [0.628, 0.738] | 0.782 [0.732, 0.830] | 0.589 [0.524, 0.652] | 0.743 [0.691, 0.795] |
| clipdet | 0.829 [0.786, 0.869] | 0.747 [0.698, 0.796] | 0.784 [0.735, 0.833] | 0.697 [0.640, 0.755] | 0.772 [0.724, 0.821] | 0.660 [0.600, 0.724] |

### SENSITIVITY (all 100 reals, unlabelled): drop / recovery / canonicalization cost, eps 8/255

| detector | drop | gain (canon on attacked) | recovery | canon clean cost |
|---|---|---|---|---|
| b0 | 0.068 [0.041, 0.094] | 0.040 [0.012, 0.066] | 0.583 [0.229, 0.955] (n=1999) | 0.010 [-0.009, 0.030] |
| communityforensics | 0.265 [0.215, 0.319] | 0.012 [-0.044, 0.068] | 0.045 [-0.185, 0.238] (n=2000) | 0.179 [0.138, 0.224] |
| rine | 0.200 [0.154, 0.250] | 0.096 [0.040, 0.153] | 0.481 [0.231, 0.693] (n=2000) | 0.049 [0.009, 0.090] |
| clipdet | 0.045 [0.006, 0.086] | -0.087 [-0.132, -0.040] | -1.913 [-4.613, -0.557] (n=1773) | 0.081 [0.037, 0.129] |

### SENSITIVITY (all 100 reals, unlabelled): drop / recovery / canonicalization cost, eps 16/255 (descriptive)

| detector | drop | gain (canon on attacked) | recovery | canon clean cost |
|---|---|---|---|---|
| b0 | 0.067 [0.034, 0.101] | 0.028 [-0.001, 0.057] | 0.418 [-0.004, 0.767] (n=1994) | 0.010 [-0.009, 0.030] |
| communityforensics | 0.329 [0.271, 0.388] | 0.033 [-0.031, 0.101] | 0.099 [-0.109, 0.275] (n=2000) | 0.179 [0.138, 0.224] |
| rine | 0.297 [0.238, 0.354] | 0.154 [0.084, 0.222] | 0.518 [0.332, 0.680] (n=2000) | 0.049 [0.009, 0.090] |
| clipdet | 0.056 [0.010, 0.101] | -0.113 [-0.167, -0.059] | -1.997 [-5.554, -0.680] (n=1884) | 0.081 [0.037, 0.129] |

## PRIMARY per-generator AUROC (point values; CIs in results.json)

| detector | set | gen_0 | gen_1 | gen_2 | gen_3 |
|---|---|---|---|---|---|
| b0 | clean | 0.687 | 0.598 | 0.626 | 0.659 |
| b0 | clean_canon | 0.661 | 0.564 | 0.583 | 0.599 |
| b0 | eps8 | 0.598 | 0.546 | 0.577 | 0.516 |
| b0 | eps8_canon | 0.657 | 0.577 | 0.589 | 0.555 |
| b0 | eps16 | 0.600 | 0.572 | 0.595 | 0.526 |
| b0 | eps16_canon | 0.636 | 0.569 | 0.570 | 0.537 |
| communityforensics | clean | 0.839 | 0.994 | 0.987 | 0.988 |
| communityforensics | clean_canon | 0.629 | 0.854 | 0.794 | 0.910 |
| communityforensics | eps8 | 0.561 | 0.802 | 0.780 | 0.732 |
| communityforensics | eps8_canon | 0.575 | 0.797 | 0.753 | 0.820 |
| communityforensics | eps16 | 0.491 | 0.753 | 0.709 | 0.620 |
| communityforensics | eps16_canon | 0.505 | 0.732 | 0.709 | 0.798 |
| rine | clean | 0.872 | 0.977 | 0.826 | 0.918 |
| rine | clean_canon | 0.747 | 0.839 | 0.798 | 0.884 |
| rine | eps8 | 0.717 | 0.765 | 0.735 | 0.800 |
| rine | eps8_canon | 0.758 | 0.790 | 0.750 | 0.845 |
| rine | eps16 | 0.638 | 0.667 | 0.619 | 0.678 |
| rine | eps16_canon | 0.739 | 0.744 | 0.696 | 0.778 |
| clipdet | clean | 0.837 | 0.806 | 0.787 | 0.854 |
| clipdet | clean_canon | 0.732 | 0.695 | 0.713 | 0.781 |
| clipdet | eps8 | 0.758 | 0.682 | 0.718 | 0.780 |
| clipdet | eps8_canon | 0.733 | 0.640 | 0.676 | 0.716 |
| clipdet | eps16 | 0.744 | 0.688 | 0.722 | 0.744 |
| clipdet | eps16_canon | 0.744 | 0.639 | 0.672 | 0.720 |

## PRIMARY balanced accuracy at logit 0 (descriptive)

| detector | clean | clean_canon | eps8 | eps8_canon | eps16 | eps16_canon |
|---|---|---|---|---|---|---|
| b0 | 0.585 | 0.547 | 0.545 | 0.576 | 0.549 | 0.530 |
| communityforensics | 0.829 | 0.557 | 0.555 | 0.516 | 0.528 | 0.510 |
| rine | 0.821 | 0.716 | 0.641 | 0.708 | 0.501 | 0.687 |
| clipdet | 0.734 | 0.533 | 0.602 | 0.515 | 0.617 | 0.532 |

Anchor: RAID Table 4 (arXiv 2506.03988, via kb/datasets/raid.md): DINOv2 accuracy 83.1 -> 54.6 at eps 16/255. Our probe, preprocessing and balanced-accuracy-at-0 differ; compare only qualitatively.

Caveats: see notes.md (D1 canon upsample, clean control = RAID center crop, originals already JPEG ~q75, RAID generator mapping unverified).
