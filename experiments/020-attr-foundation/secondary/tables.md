# 020 secondary (pre-declared DESCRIPTIVE) arms

No labels. All CIs: cluster bootstrap (1000, seed 0) over caption families (SD) or RAID ids. Source: results/secondary.json. Frozen 020 verdicts unchanged.

## 0. Pipeline checks

D-008 (GPU cache vs CPU recompute, 32 new images per universe; scaled max feature diff and max |dlogit| of the 020 b8 probe, tol 1e-3):

| universe/condition | max scaled dF | max abs dlogit (b8) | ok |
|---|---|---|---|
| fov224/clean | 1.82e-05 | 1.14e-05 | True |
| fov224/jpeg(75) | 3.44e-05 | 1.17e-05 | True |
| fov224/jpeg(90) | 1.09e-05 | 8.11e-06 | True |
| fov512/clean | 3.73e-05 | 6.68e-05 | True |
| fov512/jpeg75_resize05 | 1.93e-05 | 2.57e-05 | True |
| fov512/mavali_medium | 1.21e-05 | 2.71e-05 | True |
| fov512/screenshot_sim | 2.42e-05 | 5.46e-05 | True |
| fov512/social_sim | 1.72e-05 | 2.38e-05 | True |

Parity (16 existing images re-extracted via this script vs the existing cache; pixel parity = raw file re-processed equals the existing processed PNG):

| universe | n | raw->PNG pixel-identical | per-condition max scaled dF |
|---|---|---|---|
| fov512 | 16 | 16/16 | clean: 5.3e-06, jpeg75_resize05: 6.0e-06, social_sim: 6.4e-06, screenshot_sim: 4.5e-06, mavali_medium: 5.9e-06 |
| fov224 | 16 | 16/16 | clean: 5.4e-06, jpeg(75): 5.9e-06 |

New images: {'stable-diffusion-1-3': 300, 'stable-diffusion-1-4': 300, 'glide': 300}; 300 caption families, 157 also used by the old 1020 fakes. GLIDE raw 256 px, upscaled 2x BICUBIC to 512 before the fov512 pipeline (fov512 never upscales); SD1.3/1.4 are native 512.

Raw-size top (old classes): dalle2 {'1024x1024': 120}; dalle3 {'1024x1024': 100, '1792x1024': 13, '1024x1792': 7}; firefly {'2304x1792': 36, '2688x1536': 28, '2048x2048': 28, '1792x2304': 28}; midjourney-v5 {'1360x896': 72, '1344x896': 20, '896x1360': 14, '896x1344': 14}; stable-diffusion-2 {'1472x704': 9, '832x1152': 8, '896x1152': 8, '1344x768': 7}; stable-diffusion-xl {'1344x768': 8, '960x1024': 8, '1088x960': 7, '896x1088': 6}; sd3_large {'1024x1024': 150}; flux {'1024x1024': 150}; stable-diffusion-1-3 {'512x512': 300}; stable-diffusion-1-4 {'512x512': 300}; glide {'256x256': 300}

## (a) Q2-near-twin: SD1.3 vs SD1.4 (chance 0.5)

600 images, 300 caption families, 5-fold StratifiedGroupKFold by family, inner-CV C (020 code). Training regime T-mix = clean + jpeg75_resize05 + social_sim; 2-AFC = same-caption pair, correct iff s(1.4)-s(1.3) is larger for the SD1.4 image.

| regime/features | eval condition | bacc | AUROC | 2-AFC pairwise |
|---|---|---|---|---|
| tmix/b8 | clean | 0.523 [0.487, 0.558] | 0.507 [0.469, 0.547] | 0.500 [0.440, 0.550] |
| tmix/b8 | jpeg75_resize05 | 0.527 [0.490, 0.562] | 0.535 [0.498, 0.573] | 0.557 [0.500, 0.613] |
| tmix/b8 | social_sim | 0.515 [0.482, 0.550] | 0.517 [0.479, 0.559] | 0.527 [0.467, 0.583] |
| tmix/b8 | screenshot_sim | 0.487 [0.452, 0.520] | 0.494 [0.455, 0.532] | 0.513 [0.457, 0.573] |
| tmix/b8 | mavali_medium | 0.552 [0.518, 0.587] | 0.542 [0.503, 0.580] | 0.550 [0.493, 0.607] |
| tmix/cat | clean | 0.508 [0.472, 0.547] | 0.506 [0.464, 0.551] | 0.507 [0.453, 0.563] |
| tmix/cat | jpeg75_resize05 | 0.508 [0.475, 0.542] | 0.513 [0.475, 0.555] | 0.527 [0.473, 0.583] |
| tmix/cat | social_sim | 0.485 [0.450, 0.520] | 0.492 [0.452, 0.530] | 0.507 [0.447, 0.567] |
| tmix/cat | screenshot_sim | 0.502 [0.463, 0.538] | 0.492 [0.451, 0.534] | 0.527 [0.467, 0.580] |
| tmix/cat | mavali_medium | 0.495 [0.460, 0.532] | 0.507 [0.467, 0.550] | 0.547 [0.493, 0.600] |
| clean/b8 | clean | 0.497 [0.463, 0.528] | 0.499 [0.462, 0.536] | 0.490 [0.430, 0.543] |
| clean/b8 | jpeg75_resize05 | 0.520 [0.492, 0.550] | 0.508 [0.479, 0.540] | 0.570 [0.513, 0.623] |
| clean/b8 | social_sim | 0.508 [0.477, 0.540] | 0.498 [0.462, 0.535] | 0.500 [0.443, 0.553] |
| clean/b8 | screenshot_sim | 0.490 [0.457, 0.523] | 0.483 [0.447, 0.519] | 0.517 [0.460, 0.573] |
| clean/b8 | mavali_medium | 0.505 [0.473, 0.538] | 0.498 [0.460, 0.534] | 0.497 [0.440, 0.553] |
| clean/cat | clean | 0.532 [0.497, 0.567] | 0.529 [0.490, 0.566] | 0.510 [0.450, 0.567] |
| clean/cat | jpeg75_resize05 | 0.505 [0.475, 0.533] | 0.520 [0.488, 0.553] | 0.527 [0.470, 0.580] |
| clean/cat | social_sim | 0.525 [0.490, 0.558] | 0.513 [0.477, 0.548] | 0.503 [0.443, 0.563] |
| clean/cat | screenshot_sim | 0.518 [0.483, 0.553] | 0.517 [0.481, 0.555] | 0.537 [0.480, 0.593] |
| clean/cat | mavali_medium | 0.513 [0.480, 0.548] | 0.513 [0.478, 0.551] | 0.517 [0.460, 0.573] |

Label-swap-within-family null (b8, C=0.01, T-mix, 60 shuffles) bacc: clean mean 0.500, p95 0.533, max 0.545; jpeg75_resize05 mean 0.498, p95 0.540, max 0.560.

## (b) Q3-family

Frozen 020 protocol (T-mix, 8-way, retrained on all 1020 fakes): feature set picked by inner CV = cat (C=0.1, inner bacc 0.846). The argmax can only be one of the 8 trained classes, so any new image is forced into one of them.

### Predicted-class distribution, 8-way probe, feature set cat (fraction of the 300 images [95% CI]; classes with >= 5% in any shown condition)

**stable-diffusion-1-3**

| condition | dalle2 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl | sd3_large | flux |
|---|---|---|---|---|---|---|---|
| clean | 0.17 [0.13, 0.21] | 0.13 [0.09, 0.17] | 0.14 [0.10, 0.17] | 0.34 [0.28, 0.39] | 0.01 [0.00, 0.03] | 0.16 [0.12, 0.20] | 0.04 [0.02, 0.06] |
| jpeg75_resize05 | 0.01 [0.00, 0.03] | 0.05 [0.03, 0.08] | 0.41 [0.36, 0.46] | 0.28 [0.23, 0.33] | 0.05 [0.02, 0.07] | 0.13 [0.09, 0.17] | 0.05 [0.03, 0.08] |
| social_sim | 0.05 [0.03, 0.07] | 0.05 [0.03, 0.08] | 0.33 [0.28, 0.38] | 0.35 [0.30, 0.41] | 0.04 [0.02, 0.06] | 0.12 [0.08, 0.15] | 0.03 [0.01, 0.05] |
| screenshot_sim | 0.29 [0.24, 0.34] | 0.05 [0.03, 0.07] | 0.21 [0.16, 0.25] | 0.34 [0.28, 0.40] | 0.01 [0.00, 0.02] | 0.04 [0.02, 0.07] | 0.03 [0.01, 0.05] |
| mavali_medium | 0.07 [0.04, 0.10] | 0.08 [0.05, 0.11] | 0.28 [0.24, 0.34] | 0.34 [0.29, 0.40] | 0.07 [0.05, 0.10] | 0.13 [0.10, 0.17] | 0.01 [0.00, 0.02] |

**stable-diffusion-1-4**

| condition | dalle2 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl | sd3_large | flux |
|---|---|---|---|---|---|---|---|
| clean | 0.15 [0.11, 0.19] | 0.10 [0.07, 0.14] | 0.13 [0.09, 0.17] | 0.36 [0.31, 0.42] | 0.03 [0.01, 0.05] | 0.17 [0.13, 0.21] | 0.03 [0.01, 0.05] |
| jpeg75_resize05 | 0.03 [0.02, 0.06] | 0.03 [0.01, 0.05] | 0.40 [0.35, 0.45] | 0.24 [0.19, 0.29] | 0.08 [0.05, 0.11] | 0.13 [0.09, 0.16] | 0.06 [0.03, 0.08] |
| social_sim | 0.07 [0.04, 0.10] | 0.04 [0.02, 0.06] | 0.30 [0.24, 0.36] | 0.37 [0.31, 0.42] | 0.07 [0.04, 0.10] | 0.11 [0.08, 0.15] | 0.03 [0.01, 0.05] |
| screenshot_sim | 0.29 [0.24, 0.34] | 0.03 [0.01, 0.05] | 0.27 [0.22, 0.32] | 0.28 [0.23, 0.33] | 0.02 [0.01, 0.04] | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] |
| mavali_medium | 0.06 [0.03, 0.08] | 0.07 [0.04, 0.10] | 0.29 [0.24, 0.34] | 0.32 [0.27, 0.37] | 0.10 [0.06, 0.13] | 0.14 [0.10, 0.18] | 0.01 [0.00, 0.03] |

**glide**

| condition | dalle2 |
|---|---|
| clean | 0.91 [0.88, 0.94] |
| jpeg75_resize05 | 0.92 [0.88, 0.94] |
| social_sim | 0.98 [0.96, 0.99] |
| screenshot_sim | 0.84 [0.80, 0.88] |
| mavali_medium | 0.97 [0.94, 0.98] |

### Predicted-class distribution, 8-way probe, feature set b8 (fraction of the 300 images [95% CI]; classes with >= 5% in any shown condition)

**stable-diffusion-1-3**

| condition | dalle2 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl | sd3_large |
|---|---|---|---|---|---|---|
| clean | 0.19 [0.15, 0.24] | 0.09 [0.06, 0.13] | 0.13 [0.09, 0.17] | 0.33 [0.28, 0.38] | 0.06 [0.03, 0.08] | 0.17 [0.13, 0.22] |
| jpeg75_resize05 | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.07] | 0.45 [0.39, 0.51] | 0.23 [0.18, 0.28] | 0.09 [0.05, 0.12] | 0.11 [0.07, 0.14] |
| social_sim | 0.07 [0.04, 0.10] | 0.04 [0.02, 0.06] | 0.35 [0.30, 0.40] | 0.31 [0.26, 0.36] | 0.08 [0.06, 0.12] | 0.13 [0.09, 0.17] |
| screenshot_sim | 0.24 [0.20, 0.29] | 0.08 [0.05, 0.11] | 0.24 [0.19, 0.28] | 0.27 [0.22, 0.32] | 0.03 [0.01, 0.05] | 0.07 [0.04, 0.10] |
| mavali_medium | 0.14 [0.11, 0.19] | 0.08 [0.06, 0.12] | 0.22 [0.18, 0.26] | 0.26 [0.22, 0.31] | 0.18 [0.14, 0.22] | 0.09 [0.06, 0.13] |

**stable-diffusion-1-4**

| condition | dalle2 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl | sd3_large |
|---|---|---|---|---|---|---|
| clean | 0.18 [0.14, 0.23] | 0.09 [0.06, 0.12] | 0.14 [0.10, 0.18] | 0.35 [0.30, 0.40] | 0.08 [0.05, 0.11] | 0.15 [0.11, 0.19] |
| jpeg75_resize05 | 0.05 [0.03, 0.08] | 0.04 [0.02, 0.07] | 0.45 [0.39, 0.51] | 0.18 [0.14, 0.23] | 0.11 [0.08, 0.15] | 0.11 [0.08, 0.15] |
| social_sim | 0.08 [0.05, 0.12] | 0.03 [0.01, 0.05] | 0.33 [0.27, 0.38] | 0.33 [0.28, 0.38] | 0.11 [0.08, 0.15] | 0.10 [0.07, 0.14] |
| screenshot_sim | 0.24 [0.19, 0.29] | 0.06 [0.03, 0.08] | 0.28 [0.23, 0.33] | 0.21 [0.17, 0.26] | 0.05 [0.02, 0.07] | 0.11 [0.07, 0.14] |
| mavali_medium | 0.11 [0.08, 0.15] | 0.06 [0.04, 0.09] | 0.23 [0.19, 0.28] | 0.27 [0.22, 0.32] | 0.22 [0.17, 0.27] | 0.10 [0.07, 0.14] |

**glide**

| condition | dalle2 | firefly | midjourney-v5 | stable-diffusion-xl | sd3_large |
|---|---|---|---|---|---|
| clean | 0.63 [0.58, 0.69] | 0.07 [0.04, 0.09] | 0.16 [0.12, 0.20] | 0.11 [0.08, 0.15] | 0.03 [0.01, 0.05] |
| jpeg75_resize05 | 0.79 [0.75, 0.83] | 0.03 [0.01, 0.05] | 0.11 [0.08, 0.15] | 0.05 [0.02, 0.07] | 0.01 [0.00, 0.02] |
| social_sim | 0.83 [0.78, 0.87] | 0.02 [0.01, 0.04] | 0.09 [0.06, 0.12] | 0.05 [0.03, 0.08] | 0.00 [0.00, 0.01] |
| screenshot_sim | 0.61 [0.56, 0.66] | 0.08 [0.05, 0.11] | 0.12 [0.09, 0.16] | 0.09 [0.06, 0.13] | 0.05 [0.03, 0.08] |
| mavali_medium | 0.90 [0.87, 0.94] | 0.01 [0.00, 0.02] | 0.01 [0.00, 0.03] | 0.08 [0.05, 0.11] | 0.00 [0.00, 0.00] |

### 11-way closed-set probe (8 + SD1.3 + SD1.4 + GLIDE), T-mix, nested feature set per fold, 5-fold grouped CV (chance 1/11 = 0.091)

| condition | bacc |
|---|---|
| clean | 0.839 [0.823, 0.853] |
| jpeg75_resize05 | 0.765 [0.747, 0.783] |
| social_sim | 0.796 [0.778, 0.812] |
| screenshot_sim | 0.767 [0.748, 0.783] |
| mavali_medium | 0.731 [0.711, 0.751] |

Per-class recall at jpeg75_resize05: dalle2 0.79, dalle3 0.96, firefly 0.81, midjourney-v5 0.64, stable-diffusion-2 0.71, stable-diffusion-xl 0.73, sd3_large 0.88, flux 0.95, stable-diffusion-1-3 0.46, stable-diffusion-1-4 0.49, glide 0.98

Confusion rows for the new classes at jpeg75_resize05 (columns = dalle2, dalle3, firefly, midjourney-v5, stable-diffusion-2, stable-diffusion-xl, sd3_large, flux, stable-diffusion-1-3, stable-diffusion-1-4, glide):

- stable-diffusion-1-3: [1, 1, 1, 5, 3, 4, 0, 3, 138, 144, 0] (SD1.3+SD1.4 pooled correct 558/600 for the two SD1.x classes)
- stable-diffusion-1-4: [2, 0, 1, 10, 3, 1, 4, 3, 128, 148, 0]
- glide: [2, 0, 0, 1, 0, 0, 0, 0, 0, 2, 295]

2-AFC pairwise (jpeg75_resize05, same-caption pairs): dalle2|stable-diffusion-1-3 1.000, dalle2|stable-diffusion-1-4 1.000, dalle2|glide 0.969, dalle3|stable-diffusion-1-3 1.000, dalle3|stable-diffusion-1-4 1.000, firefly|stable-diffusion-1-3 1.000, firefly|stable-diffusion-1-4 1.000, midjourney-v5|stable-diffusion-1-3 1.000, midjourney-v5|stable-diffusion-1-4 1.000, stable-diffusion-2|stable-diffusion-1-3 0.975, stable-diffusion-2|stable-diffusion-1-4 1.000, stable-diffusion-xl|stable-diffusion-1-3 1.000, stable-diffusion-xl|stable-diffusion-1-4 1.000, sd3_large|stable-diffusion-1-3 1.000, sd3_large|stable-diffusion-1-4 1.000, flux|stable-diffusion-1-3 1.000, flux|stable-diffusion-1-4 1.000, stable-diffusion-1-3|stable-diffusion-1-4 0.503, stable-diffusion-1-3|glide 1.000, stable-diffusion-1-4|glide 1.000

## (c) RAID replication, n=500 ids (100 old + 400 new), 2000 fakes, 4-way (chance 0.25), T-mix = clean + jpeg(75) + jpeg(90), nested feature set

Frozen n=100 reference (020 results.json): H-A1 RAID bacc 0.688 [0.650, 0.725] at jpeg(75), T-mix. gen_k -> generator mapping is [unverified].

| regime | eval cond | 4-way bacc | macro recall gen_1-3 (4-way model, 'without gen_0') | 2-AFC pairwise macro | pairwise w/o gen_0 pairs | bacc old 100 ids | bacc new 400 ids |
|---|---|---|---|---|---|---|---|
| tmix | clean | 0.824 [0.809, 0.839] | 0.780 [0.759, 0.799] | 0.969 [0.963, 0.975] | 0.938 [0.926, 0.949] | 0.825 [0.786, 0.860] | 0.824 [0.807, 0.841] |
| tmix | jpeg(75) | 0.772 [0.756, 0.789] | 0.719 [0.698, 0.739] | 0.956 [0.949, 0.963] | 0.915 [0.902, 0.929] | 0.783 [0.750, 0.815] | 0.770 [0.749, 0.789] |
| tmix | jpeg(90) | 0.798 [0.780, 0.814] | 0.753 [0.731, 0.773] | 0.961 [0.954, 0.967] | 0.922 [0.909, 0.933] | 0.808 [0.767, 0.843] | 0.795 [0.776, 0.813] |
| clean | clean | 0.852 [0.836, 0.867] | 0.806 [0.786, 0.825] | 0.976 [0.971, 0.981] | 0.952 [0.942, 0.962] | 0.857 [0.821, 0.892] | 0.851 [0.834, 0.866] |
| clean | jpeg(75) | 0.497 [0.480, 0.516] | 0.332 [0.309, 0.356] | 0.949 [0.942, 0.957] | 0.912 [0.899, 0.925] | 0.492 [0.454, 0.533] | 0.499 [0.479, 0.518] |
| clean | jpeg(90) | 0.768 [0.752, 0.785] | 0.697 [0.676, 0.721] | 0.963 [0.957, 0.969] | 0.928 [0.916, 0.940] | 0.782 [0.747, 0.818] | 0.764 [0.745, 0.784] |

Per-class recall (gen_0..gen_3), T-mix: clean: 0.96, 0.70, 0.70, 0.94; jpeg(75): 0.93, 0.64, 0.62, 0.89; jpeg(90): 0.93, 0.68, 0.65, 0.93

Fresh 3-way probe on gen_1-3 only (no gen_0 rows; chance 0.333):

| regime | eval cond | bacc | 2-AFC macro |
|---|---|---|---|
| tmix | clean | 0.777 [0.757, 0.797] | 0.936 [0.923, 0.947] |
| tmix | jpeg(75) | 0.736 [0.715, 0.756] | 0.913 [0.899, 0.926] |
| tmix | jpeg(90) | 0.771 [0.749, 0.791] | 0.921 [0.907, 0.933] |
| clean | clean | 0.803 [0.784, 0.821] | 0.947 [0.936, 0.957] |
| clean | jpeg(75) | 0.723 [0.703, 0.743] | 0.905 [0.891, 0.918] |
| clean | jpeg(90) | 0.762 [0.743, 0.781] | 0.923 [0.911, 0.935] |

Raw sizes of the new RAID ids: d3_gen_0 {'256x256': 400}; d3_gen_1 {'512x512': 187, '640x480': 109, '640x360': 104}; d3_gen_2 {'512x512': 200, '640x480': 107, '640x360': 93}; d3_gen_3 {'512x512': 197, '640x360': 105, '640x480': 98}
