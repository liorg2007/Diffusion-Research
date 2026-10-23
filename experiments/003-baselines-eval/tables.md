# 003-baselines-eval tables

git HEAD dff201162722ecfd2202b228e415d2466c41d1e3 · seed 0 · AUROC real pool = VISION-native test + RAISE test; fakes = 7 non-FLUX generators + FLUX (all test-split). B0 = DINOv2-base probe (FLUX never in train). 95% stratified bootstrap CIs (1000). Calibrated = threshold at 5% FPR on the calib reals of the SAME condition.

## Headline: AUROC and TPR@1%FPR per condition; FLUX (held-out) AUROC

| scorer | AUROC clean | AUROC jpeg75_resize05 | AUROC social_sim | TPR@1%FPR clean | TPR@1%FPR jpeg75_resize05 | TPR@1%FPR social_sim | FLUX AUROC clean / jpeg75_resize05 / social_sim |
|---|---|---|---|---|---|---|---|
| communityforensics | 0.992 | 0.800 | 0.977 | 0.829 | 0.321 | 0.689 | 0.984 / 0.593 / 0.946 |
| bfree | 0.970 | 0.714 | 0.910 | 0.749 | 0.198 | 0.579 | 0.913 / 0.469 / 0.744 |
| rine | 0.844 | 0.807 | 0.730 | 0.556 | 0.132 | 0.209 | 0.816 / 0.689 / 0.438 |
| clipdet | 0.780 | 0.595 | 0.774 | 0.080 | 0.019 | 0.079 | 0.649 / 0.460 / 0.670 |
| b0_dinov2 | 0.932 | 0.875 | 0.917 | 0.368 | 0.176 | 0.352 | 0.886 / 0.813 / 0.862 |

## Main table: scorer x condition (all fakes incl. FLUX vs VISION-native + RAISE test reals)

| scorer | condition | AUROC [95% CI] | AP | bal.acc @logit0 | bal.acc @5%FPR-calib | FPR reached @calib | FPR@95%TPR | TPR@1%FPR [CI] | AUROC excl. FLUX |
|---|---|---|---|---|---|---|---|---|---|
| communityforensics | clean | 0.992 [0.988, 0.995] | 0.992 | 0.954 | 0.956 | 0.038 | 0.038 | 0.829 [0.679, 0.927] | 0.995 [0.992, 0.997] |
| communityforensics | jpeg75_resize05 | 0.800 [0.774, 0.823] | 0.835 | 0.683 | 0.701 | 0.066 | 0.750 | 0.321 [0.248, 0.369] | 0.871 [0.848, 0.892] |
| communityforensics | social_sim | 0.977 [0.970, 0.983] | 0.978 | 0.904 | 0.914 | 0.048 | 0.120 | 0.689 [0.535, 0.791] | 0.987 [0.982, 0.992] |
| bfree | clean | 0.970 [0.961, 0.977] | 0.976 | 0.900 | 0.886 | 0.158 | 0.187 | 0.749 [0.680, 0.827] | 0.989 [0.985, 0.993] |
| bfree | jpeg75_resize05 | 0.714 [0.686, 0.742] | 0.772 | 0.674 | 0.658 | 0.038 | 0.907 | 0.198 [0.121, 0.256] | 0.798 [0.769, 0.826] |
| bfree | social_sim | 0.910 [0.894, 0.926] | 0.932 | 0.830 | 0.839 | 0.095 | 0.542 | 0.579 [0.537, 0.656] | 0.968 [0.958, 0.977] |
| rine | clean | 0.844 [0.820, 0.866] | 0.898 | 0.799 | 0.815 | 0.032 | 0.950 | 0.556 [0.485, 0.641] | 0.854 [0.823, 0.881] |
| rine | jpeg75_resize05 | 0.807 [0.781, 0.832] | 0.817 | 0.673 | 0.580 | 0.020 | 0.706 | 0.132 [0.084, 0.195] | 0.848 [0.823, 0.871] |
| rine | social_sim | 0.730 [0.699, 0.759] | 0.793 | 0.628 | 0.659 | 0.029 | 0.935 | 0.209 [0.162, 0.320] | 0.830 [0.804, 0.857] |
| clipdet | clean | 0.780 [0.752, 0.806] | 0.784 | 0.660 | 0.656 | 0.070 | 0.736 | 0.080 [0.041, 0.168] | 0.825 [0.799, 0.850] |
| clipdet | jpeg75_resize05 | 0.595 [0.560, 0.627] | 0.574 | 0.557 | 0.513 | 0.077 | 0.817 | 0.019 [0.003, 0.051] | 0.641 [0.608, 0.674] |
| clipdet | social_sim | 0.774 [0.747, 0.801] | 0.772 | 0.659 | 0.633 | 0.065 | 0.706 | 0.079 [0.043, 0.128] | 0.810 [0.785, 0.836] |
| b0_dinov2 | clean | 0.932 [0.918, 0.946] | 0.931 | 0.856 | 0.782 | 0.047 | 0.278 | 0.368 [0.274, 0.434] | 0.947 [0.934, 0.959] |
| b0_dinov2 | jpeg75_resize05 | 0.875 [0.854, 0.895] | 0.870 | 0.797 | 0.703 | 0.048 | 0.458 | 0.176 [0.080, 0.306] | 0.897 [0.878, 0.916] |
| b0_dinov2 | social_sim | 0.917 [0.901, 0.932] | 0.914 | 0.839 | 0.739 | 0.050 | 0.327 | 0.352 [0.250, 0.409] | 0.936 [0.921, 0.951] |

## AUROC by real pool (all fakes vs that pool only)

| scorer | condition | VISION-native vs fakes | RAISE vs fakes |
|---|---|---|---|
| communityforensics | clean | 0.991 [0.986, 0.995] | 0.993 [0.989, 0.996] |
| communityforensics | jpeg75_resize05 | 0.762 [0.732, 0.792] | 0.868 [0.841, 0.891] |
| communityforensics | social_sim | 0.971 [0.961, 0.980] | 0.986 [0.979, 0.992] |
| bfree | clean | 0.957 [0.945, 0.968] | 0.992 [0.986, 0.996] |
| bfree | jpeg75_resize05 | 0.686 [0.656, 0.723] | 0.764 [0.729, 0.798] |
| bfree | social_sim | 0.891 [0.873, 0.911] | 0.945 [0.931, 0.959] |
| rine | clean | 0.823 [0.796, 0.850] | 0.882 [0.855, 0.905] |
| rine | jpeg75_resize05 | 0.893 [0.873, 0.914] | 0.652 [0.610, 0.695] |
| rine | social_sim | 0.747 [0.715, 0.777] | 0.699 [0.661, 0.739] |
| clipdet | clean | 0.752 [0.722, 0.785] | 0.829 [0.795, 0.859] |
| clipdet | jpeg75_resize05 | 0.458 [0.420, 0.497] | 0.841 [0.815, 0.869] |
| clipdet | social_sim | 0.746 [0.715, 0.778] | 0.826 [0.793, 0.855] |
| b0_dinov2 | clean | 0.962 [0.950, 0.973] | 0.876 [0.847, 0.899] |
| b0_dinov2 | jpeg75_resize05 | 0.919 [0.900, 0.936] | 0.798 [0.763, 0.827] |
| b0_dinov2 | social_sim | 0.958 [0.945, 0.969] | 0.843 [0.808, 0.870] |

## Per-generator AUROC (each generator vs the same real pool: VISION-native + RAISE test). FLUX = held out for B0

| scorer | condition | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-xl | stable-diffusion-2 | sd3_large | flux |
|---|---|---|---|---|---|---|---|---|---|
| communityforensics | clean | 0.991 | 0.990 | 0.993 | 0.999 | 1.000 | 1.000 | 0.990 | 0.984 |
| communityforensics | jpeg75_resize05 | 0.971 | 0.784 | 0.985 | 0.839 | 0.959 | 0.955 | 0.656 | 0.593 |
| communityforensics | social_sim | 0.977 | 0.988 | 0.989 | 0.996 | 0.997 | 0.999 | 0.969 | 0.946 |
| bfree | clean | 0.964 | 0.977 | 0.991 | 0.997 | 0.999 | 0.997 | 0.997 | 0.913 |
| bfree | jpeg75_resize05 | 0.869 | 0.801 | 0.956 | 0.825 | 0.915 | 0.783 | 0.511 | 0.469 |
| bfree | social_sim | 0.957 | 0.968 | 0.989 | 0.996 | 0.997 | 0.986 | 0.898 | 0.744 |
| rine | clean | 0.968 | 0.215 | 0.828 | 0.987 | 1.000 | 0.984 | 0.966 | 0.816 |
| rine | jpeg75_resize05 | 0.881 | 0.849 | 0.817 | 0.797 | 0.934 | 0.909 | 0.768 | 0.689 |
| rine | social_sim | 0.808 | 0.714 | 0.733 | 0.880 | 0.916 | 0.884 | 0.868 | 0.438 |
| clipdet | clean | 0.828 | 0.916 | 0.857 | 0.685 | 0.727 | 0.818 | 0.921 | 0.649 |
| clipdet | jpeg75_resize05 | 0.671 | 0.755 | 0.495 | 0.589 | 0.728 | 0.683 | 0.582 | 0.460 |
| clipdet | social_sim | 0.729 | 0.868 | 0.820 | 0.697 | 0.803 | 0.856 | 0.883 | 0.670 |
| b0_dinov2 | clean | 0.916 | 0.963 | 0.947 | 0.913 | 0.978 | 0.960 | 0.953 | 0.886 |
| b0_dinov2 | jpeg75_resize05 | 0.866 | 0.920 | 0.946 | 0.867 | 0.917 | 0.898 | 0.871 | 0.813 |
| b0_dinov2 | social_sim | 0.928 | 0.950 | 0.944 | 0.893 | 0.965 | 0.954 | 0.921 | 0.862 |

## Per-real-source FPR at the calibrated threshold (5% FPR on calib reals of the same condition); k/n shown. WA/FB-HQ reals are platform-shared copies of the test natives

| scorer | condition | thr | VISION-native | RAISE | VISION-WhatsApp | VISION-Facebook-HQ |
|---|---|---|---|---|---|---|
| communityforensics | clean | -0.22 | 0.045 (16/358) | 0.025 (5/199) | 0.011 (4/358) | 0.017 (6/358) |
| communityforensics | jpeg75_resize05 | -1.32 | 0.101 (36/358) | 0.005 (1/199) | 0.050 (18/358) | 0.056 (20/358) |
| communityforensics | social_sim | -1.31 | 0.070 (25/358) | 0.010 (2/199) | 0.020 (7/358) | 0.042 (15/358) |
| bfree | clean | -0.75 | 0.235 (84/358) | 0.020 (4/199) | 0.176 (63/358) | 0.156 (56/358) |
| bfree | jpeg75_resize05 | 2.32 | 0.045 (16/358) | 0.025 (5/199) | 0.047 (17/358) | 0.064 (23/358) |
| bfree | social_sim | 1.26 | 0.131 (47/358) | 0.030 (6/199) | 0.084 (30/358) | 0.103 (37/358) |
| rine | clean | -0.20 | 0.042 (15/358) | 0.015 (3/199) | 0.008 (3/358) | 0.011 (4/358) |
| rine | jpeg75_resize05 | 1.07 | 0.000 (0/358) | 0.055 (11/199) | 0.014 (5/358) | 0.000 (0/358) |
| rine | social_sim | -0.20 | 0.008 (3/358) | 0.065 (13/199) | 0.039 (14/358) | 0.008 (3/358) |
| clipdet | clean | 0.11 | 0.073 (26/358) | 0.065 (13/199) | 0.025 (9/358) | 0.047 (17/358) |
| clipdet | jpeg75_resize05 | 1.06 | 0.120 (43/358) | 0.000 (0/199) | 0.070 (25/358) | 0.095 (34/358) |
| clipdet | social_sim | 0.14 | 0.081 (29/358) | 0.035 (7/199) | 0.036 (13/358) | 0.078 (28/358) |
| b0_dinov2 | clean | 1.63 | 0.022 (8/358) | 0.090 (18/199) | 0.039 (14/358) | 0.036 (13/358) |
| b0_dinov2 | jpeg75_resize05 | 2.00 | 0.036 (13/358) | 0.070 (14/199) | 0.028 (10/358) | 0.031 (11/358) |
| b0_dinov2 | social_sim | 1.96 | 0.014 (5/358) | 0.116 (23/199) | 0.028 (10/358) | 0.025 (9/358) |

### Same, at the native threshold (logit 0)

| scorer | condition | VISION-native | RAISE | VISION-WhatsApp | VISION-Facebook-HQ |
|---|---|---|---|---|---|
| communityforensics | clean | 0.042 | 0.020 | 0.011 | 0.014 |
| communityforensics | jpeg75_resize05 | 0.045 | 0.000 | 0.028 | 0.022 |
| communityforensics | social_sim | 0.047 | 0.005 | 0.006 | 0.020 |
| bfree | clean | 0.173 | 0.015 | 0.128 | 0.109 |
| bfree | jpeg75_resize05 | 0.254 | 0.106 | 0.165 | 0.184 |
| bfree | social_sim | 0.251 | 0.055 | 0.156 | 0.190 |
| rine | clean | 0.022 | 0.010 | 0.006 | 0.006 |
| rine | jpeg75_resize05 | 0.000 | 0.176 | 0.084 | 0.011 |
| rine | social_sim | 0.006 | 0.025 | 0.025 | 0.003 |
| clipdet | clean | 0.101 | 0.065 | 0.036 | 0.056 |
| clipdet | jpeg75_resize05 | 0.559 | 0.085 | 0.383 | 0.486 |
| clipdet | social_sim | 0.106 | 0.045 | 0.061 | 0.103 |
| b0_dinov2 | clean | 0.089 | 0.347 | 0.190 | 0.128 |
| b0_dinov2 | jpeg75_resize05 | 0.173 | 0.462 | 0.293 | 0.218 |
| b0_dinov2 | social_sim | 0.092 | 0.432 | 0.190 | 0.154 |

### Same, at the threshold calibrated on CLEAN calib reals (deployment view: threshold not re-fit per condition); TPR over all fakes

| scorer | condition | thr | VISION-native | RAISE | VISION-WhatsApp | VISION-Facebook-HQ | TPR all fakes |
|---|---|---|---|---|---|---|---|
| communityforensics | clean | -0.22 | 0.045 | 0.025 | 0.011 | 0.017 | 0.949 |
| communityforensics | jpeg75_resize05 | -0.22 | 0.050 | 0.000 | 0.031 | 0.031 | 0.403 |
| communityforensics | social_sim | -0.22 | 0.047 | 0.010 | 0.008 | 0.028 | 0.848 |
| bfree | clean | -0.75 | 0.235 | 0.020 | 0.176 | 0.156 | 0.930 |
| bfree | jpeg75_resize05 | -0.75 | 0.330 | 0.171 | 0.257 | 0.251 | 0.614 |
| bfree | social_sim | -0.75 | 0.349 | 0.090 | 0.235 | 0.277 | 0.882 |
| rine | clean | -0.20 | 0.042 | 0.015 | 0.008 | 0.011 | 0.662 |
| rine | jpeg75_resize05 | -0.20 | 0.006 | 0.276 | 0.145 | 0.034 | 0.513 |
| rine | social_sim | -0.20 | 0.008 | 0.055 | 0.039 | 0.006 | 0.328 |
| clipdet | clean | 0.11 | 0.073 | 0.065 | 0.025 | 0.047 | 0.381 |
| clipdet | jpeg75_resize05 | 0.11 | 0.520 | 0.060 | 0.332 | 0.433 | 0.443 |
| clipdet | social_sim | 0.11 | 0.081 | 0.035 | 0.045 | 0.078 | 0.340 |
| b0_dinov2 | clean | 1.63 | 0.022 | 0.090 | 0.039 | 0.036 | 0.610 |
| b0_dinov2 | jpeg75_resize05 | 1.63 | 0.045 | 0.141 | 0.056 | 0.053 | 0.528 |
| b0_dinov2 | social_sim | 1.63 | 0.020 | 0.146 | 0.039 | 0.031 | 0.588 |

## B0 (DINOv2-base CLS+mean probe) leave-one-generator-out AUROC (probe trained on clean; test g = all its fakes vs VISION-native+RAISE test reals; FLUX is never trained on and is not a LOGO fold)

| held-out generator | clean | jpeg75_resize05 | social_sim |
|---|---|---|---|
| dalle2 | 0.867 [0.831, 0.902] | 0.858 [0.821, 0.891] | 0.886 [0.856, 0.912] |
| dalle3 | 0.933 [0.909, 0.954] | 0.891 [0.861, 0.922] | 0.920 [0.896, 0.942] |
| firefly | 0.938 [0.912, 0.961] | 0.912 [0.888, 0.933] | 0.936 [0.913, 0.958] |
| midjourney-v5 | 0.867 [0.836, 0.894] | 0.807 [0.766, 0.840] | 0.847 [0.813, 0.877] |
| stable-diffusion-xl | 0.974 [0.963, 0.984] | 0.929 [0.905, 0.951] | 0.963 [0.947, 0.977] |
| stable-diffusion-2 | 0.923 [0.894, 0.948] | 0.829 [0.786, 0.869] | 0.900 [0.870, 0.928] |
| sd3_large | 0.912 [0.889, 0.935] | 0.797 [0.759, 0.831] | 0.872 [0.842, 0.900] |
| macro-mean AUROC | 0.916 | 0.861 | 0.903 |
| macro TPR@1%FPR | 0.318 | 0.170 | 0.306 |

(B0 standard probe: n_train=674 of which 239 reals, C=0.01; LOGO folds choose their own C.)

## Format-bias probe (bytes/pixel, quant-table presence + quality, dims of ORIGINAL files, HF energy; logistic regression train split -> test split; fakes = all non-LDM generators incl. FLUX in test)

| real pool | AUROC [95% CI] | n train real | n test real | leakage flag (CI lower > 0.6) |
|---|---|---|---|---|
| vision_native_vs_fakes | 0.989 [0.983, 0.994] | 99 | 358 | True |
| raise_vs_fakes | 0.724 [0.684, 0.762] | 140 | 199 | True |
| both_pools_vs_fakes | 0.839 [0.813, 0.863] | 239 | 557 | True |

Standardized coefficients (bytes_per_pixel, has_quant_table, jpeg_quality, width, height, hf_energy):
- vision_native_vs_fakes: {'bytes_per_pixel': -2.452, 'has_quant_table': -0.875, 'jpeg_quality': -1.939, 'width': -2.844, 'height': -0.962, 'hf_energy': 0.654}
- raise_vs_fakes: {'bytes_per_pixel': 1.313, 'has_quant_table': 0.217, 'jpeg_quality': 0.217, 'width': 0.914, 'height': 1.726, 'hf_energy': -0.0}
- both_pools_vs_fakes: {'bytes_per_pixel': 1.037, 'has_quant_table': 1.738, 'jpeg_quality': 0.066, 'width': -1.803, 'height': -0.215, 'hf_energy': 0.141}

## D-008 guard (max |dlogit| MPS vs CPU on fixed 32-image subset; tol 1e-3; B0 = decision logit of the probe)

| run | max abs dlogit | ok |
|---|---|---|
| communityforensics/clean | 6.20e-05 | True |
| communityforensics/jpeg75_resize05 | 5.29e-05 | True |
| communityforensics/social_sim | 6.48e-05 | True |
| bfree/clean | 5.34e-05 | True |
| bfree/jpeg75_resize05 | 5.72e-05 | True |
| bfree/social_sim | 6.44e-05 | True |
| rine/clean | 7.70e-05 | True |
| rine/jpeg75_resize05 | 8.70e-06 | True |
| rine/social_sim | 7.92e-05 | True |
| clipdet/clean | 4.39e-05 | True |
| clipdet/jpeg75_resize05 | 5.51e-05 | True |
| clipdet/social_sim | 5.91e-05 | True |
| b0_dinov2/clean | 0.00e+00 | True |
| b0_dinov2/jpeg75_resize05 | 0.00e+00 | True |
| b0_dinov2/social_sim | 0.00e+00 | True |
