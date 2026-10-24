# E1 (Amendment 7): zoom sensitivity of published detectors. PILOT, exploratory. Not a Phase 4 verdict.

Data: VISION test natives/WA/FBH (358 reals, 25 devices each) + 6 Synthbuster generators x 60 test fakes. Bootstrap: paired cluster, 2000 resamples, seed 0. Scale-sensitive = |delta| >= 0.03 and the 95% CI excludes 0.

## Primary: macro-over-6-generators AUROC vs test natives, delta = orig - fov512

### clean

| detector | AUROC orig [95% CI] | AUROC fov512 [95% CI] | delta [95% CI] | scale-sensitive |
|---|---|---|---|---|
| communityforensics | 0.995 [0.988,0.999] | 0.995 [0.991,0.997] | +0.000 [-0.007,+0.005] | no (orig higher) |
| rine | 0.813 [0.789,0.838] | 0.813 [0.773,0.849] | -0.000 [-0.037,+0.037] | no (fov512 higher) |
| clipdet | 0.780 [0.728,0.825] | 0.907 [0.879,0.933] | -0.127 [-0.171,-0.089] | YES (fov512 higher) |
| b0 | 0.990 [0.982,0.996] | 0.980 [0.961,0.993] | +0.010 [-0.003,+0.026] | no (orig higher) |
| b0_trained_orig | 0.990 [0.982,0.996] | 0.969 [0.955,0.980] | +0.021 [+0.010,+0.035] | no (orig higher) [extra, not pre-registered] |
| b0_trained_fov512 | 0.969 [0.948,0.984] | 0.980 [0.961,0.993] | -0.011 [-0.020,-0.003] | no (fov512 higher) [extra, not pre-registered] |
| texture_communityforensics | 0.738 [0.671,0.797] | 0.603 [0.514,0.680] | +0.135 [+0.095,+0.178] | YES (orig higher) [texture baseline (descriptive)] |
| texture_rine | 0.778 [0.722,0.825] | 0.607 [0.524,0.680] | +0.171 [+0.127,+0.216] | YES (orig higher) [texture baseline (descriptive)] |
| texture_clipdet | 0.676 [0.596,0.745] | 0.609 [0.530,0.680] | +0.067 [+0.029,+0.105] | YES (orig higher) [texture baseline (descriptive)] |
| texture_b0 | 0.778 [0.722,0.825] | 0.607 [0.524,0.680] | +0.171 [+0.127,+0.216] | YES (orig higher) [texture baseline (descriptive)] |

### jpeg75_resize05

| detector | AUROC orig [95% CI] | AUROC fov512 [95% CI] | delta [95% CI] | scale-sensitive |
|---|---|---|---|---|
| communityforensics | 0.891 [0.861,0.917] | 0.828 [0.795,0.861] | +0.062 [+0.027,+0.097] | YES (orig higher) |
| rine | 0.945 [0.928,0.960] | 0.717 [0.671,0.763] | +0.228 [+0.187,+0.271] | YES (orig higher) |
| clipdet | 0.524 [0.459,0.589] | 0.742 [0.687,0.791] | -0.217 [-0.272,-0.170] | YES (fov512 higher) |
| b0 | 0.981 [0.971,0.990] | 0.980 [0.961,0.993] | +0.001 [-0.014,+0.020] | no (orig higher) |
| b0_trained_orig | 0.981 [0.971,0.990] | 0.923 [0.902,0.942] | +0.059 [+0.041,+0.077] | YES (orig higher) [extra, not pre-registered] |
| b0_trained_fov512 | 0.979 [0.964,0.990] | 0.980 [0.961,0.993] | -0.001 [-0.007,+0.005] | no (fov512 higher) [extra, not pre-registered] |
| texture_communityforensics | 0.684 [0.605,0.752] | 0.619 [0.543,0.688] | +0.065 [+0.029,+0.101] | YES (orig higher) [texture baseline (descriptive)] |
| texture_rine | 0.686 [0.606,0.754] | 0.619 [0.541,0.689] | +0.067 [+0.032,+0.103] | YES (orig higher) [texture baseline (descriptive)] |
| texture_clipdet | 0.680 [0.601,0.749] | 0.619 [0.543,0.688] | +0.062 [+0.025,+0.098] | YES (orig higher) [texture baseline (descriptive)] |
| texture_b0 | 0.686 [0.606,0.754] | 0.619 [0.541,0.689] | +0.067 [+0.032,+0.103] | YES (orig higher) [texture baseline (descriptive)] |

## Per-real-pool macro AUROC (orig / fov512 / delta [CI])

### clean

| detector | native | WA | FBH |
|---|---|---|---|
| communityforensics | 0.995 / 0.995 / +0.000 [-0.007,+0.005] | 0.999 / 0.996 / +0.003 [+0.001,+0.006] | 0.998 / 0.996 / +0.002 [-0.000,+0.004] |
| rine | 0.813 / 0.813 / -0.000 [-0.037,+0.037] | 0.856 / 0.873 / -0.018 [-0.040,+0.007] | 0.834 / 0.916 / -0.082 [-0.106,-0.058] |
| clipdet | 0.780 / 0.907 / -0.127 [-0.171,-0.089] | 0.887 / 0.880 / +0.008 [-0.019,+0.035] | 0.840 / 0.863 / -0.023 [-0.057,+0.006] |
| b0 | 0.990 / 0.980 / +0.010 [-0.003,+0.026] | 0.962 / 0.980 / -0.018 [-0.035,+0.001] | 0.979 / 0.981 / -0.003 [-0.017,+0.016] |
| b0_trained_orig | 0.990 / 0.969 / +0.021 [+0.010,+0.035] | 0.962 / 0.974 / -0.012 [-0.024,-0.000] | 0.979 / 0.974 / +0.005 [-0.006,+0.017] |
| b0_trained_fov512 | 0.969 / 0.980 / -0.011 [-0.020,-0.003] | 0.969 / 0.980 / -0.011 [-0.019,-0.005] | 0.970 / 0.981 / -0.012 [-0.019,-0.004] |
| texture_communityforensics | 0.738 / 0.603 / +0.135 [+0.095,+0.178] | 0.576 / 0.593 / -0.017 [-0.039,+0.004] | 0.700 / 0.603 / +0.098 [+0.070,+0.127] |
| texture_rine | 0.778 / 0.607 / +0.171 [+0.127,+0.216] | 0.576 / 0.595 / -0.019 [-0.042,+0.003] | 0.762 / 0.609 / +0.153 [+0.119,+0.191] |
| texture_clipdet | 0.676 / 0.609 / +0.067 [+0.029,+0.105] | 0.577 / 0.605 / -0.028 [-0.048,-0.009] | 0.629 / 0.608 / +0.021 [-0.005,+0.047] |
| texture_b0 | 0.778 / 0.607 / +0.171 [+0.127,+0.216] | 0.576 / 0.595 / -0.019 [-0.042,+0.003] | 0.762 / 0.609 / +0.153 [+0.119,+0.191] |

### jpeg75_resize05

| detector | native | WA | FBH |
|---|---|---|---|
| communityforensics | 0.891 / 0.828 / +0.062 [+0.027,+0.097] | 0.937 / 0.826 / +0.112 [+0.084,+0.142] | 0.926 / 0.827 / +0.099 [+0.067,+0.132] |
| rine | 0.945 / 0.717 / +0.228 [+0.187,+0.271] | 0.825 / 0.711 / +0.114 [+0.080,+0.149] | 0.924 / 0.722 / +0.202 [+0.160,+0.247] |
| clipdet | 0.524 / 0.742 / -0.217 [-0.272,-0.170] | 0.660 / 0.719 / -0.058 [-0.091,-0.026] | 0.593 / 0.728 / -0.135 [-0.178,-0.096] |
| b0 | 0.981 / 0.980 / +0.001 [-0.014,+0.020] | 0.940 / 0.981 / -0.040 [-0.059,-0.022] | 0.963 / 0.981 / -0.018 [-0.034,+0.000] |
| b0_trained_orig | 0.981 / 0.923 / +0.059 [+0.041,+0.077] | 0.940 / 0.924 / +0.016 [+0.002,+0.030] | 0.963 / 0.924 / +0.039 [+0.024,+0.055] |
| b0_trained_fov512 | 0.979 / 0.980 / -0.001 [-0.007,+0.005] | 0.980 / 0.981 / -0.001 [-0.005,+0.004] | 0.977 / 0.981 / -0.004 [-0.009,+0.001] |
| texture_communityforensics | 0.684 / 0.619 / +0.065 [+0.029,+0.101] | 0.576 / 0.619 / -0.043 [-0.063,-0.024] | 0.631 / 0.620 / +0.011 [-0.015,+0.036] |
| texture_rine | 0.686 / 0.619 / +0.067 [+0.032,+0.103] | 0.576 / 0.619 / -0.042 [-0.062,-0.023] | 0.632 / 0.620 / +0.012 [-0.013,+0.037] |
| texture_clipdet | 0.680 / 0.619 / +0.062 [+0.025,+0.098] | 0.582 / 0.618 / -0.036 [-0.056,-0.017] | 0.630 / 0.620 / +0.010 [-0.017,+0.036] |
| texture_b0 | 0.686 / 0.619 / +0.067 [+0.032,+0.103] | 0.576 / 0.619 / -0.042 [-0.062,-0.023] | 0.632 / 0.620 / +0.012 [-0.013,+0.037] |

## Cross-zoom 1%-FPR threshold transfer (test FPR with Clopper-Pearson 95%; TPR on 360 test fakes)

Threshold from calibration natives (43 reals: the calib max + eps, so nominal FPR is 0 on calib) of the source arm, applied to the target arm. B0 = mean over the 6 LOGO models (CP CI approximate). Within = source arm == target arm. For `b0` (retrained per arm) only within-arm rows are given; the rows `b0_trained_<arm>` (one fixed probe scored on the other arm's features) are an extra, NOT PRE-REGISTERED addition.

### clean, thresholds from calib_natives

| detector | thr -> test | FPR native | FPR WA | FBH | TPR |
|---|---|---|---|---|---|
| communityforensics | thr:orig->test:orig(within) | 0.017 [0.006,0.036] | 0.000 [0.000,0.010] | 0.003 [0.000,0.015] | 0.925 [0.893,0.950] |
| communityforensics | thr:orig->test:fov512(cross) | 0.003 [0.000,0.015] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.786 [0.740,0.827] |
| communityforensics | thr:fov512->test:orig(cross) | 0.064 [0.041,0.095] | 0.022 [0.010,0.044] | 0.031 [0.015,0.054] | 0.989 [0.972,0.997] |
| communityforensics | thr:fov512->test:fov512(within) | 0.017 [0.006,0.036] | 0.014 [0.005,0.032] | 0.011 [0.003,0.028] | 0.917 [0.883,0.943] |
| rine | thr:orig->test:orig(within) | 0.003 [0.000,0.015] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.572 [0.519,0.624] |
| rine | thr:orig->test:fov512(cross) | 0.020 [0.008,0.040] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.333 [0.285,0.385] |
| rine | thr:fov512->test:orig(cross) | 0.036 [0.019,0.061] | 0.008 [0.002,0.024] | 0.011 [0.003,0.028] | 0.692 [0.641,0.739] |
| rine | thr:fov512->test:fov512(within) | 0.053 [0.032,0.082] | 0.011 [0.003,0.028] | 0.006 [0.001,0.020] | 0.550 [0.497,0.602] |
| clipdet | thr:orig->test:orig(within) | 0.003 [0.000,0.015] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.050 [0.030,0.078] |
| clipdet | thr:orig->test:fov512(cross) | 0.003 [0.000,0.015] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.083 [0.057,0.117] |
| clipdet | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.017 [0.006,0.036] |
| clipdet | thr:fov512->test:fov512(within) | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.014 [0.005,0.032] |
| b0 | thr:orig->test:orig(within) | 0.015 [0.006,0.036] | 0.084 [0.057,0.117] | 0.034 [0.017,0.058] | 0.900 [0.864,0.929] |
| b0 | thr:fov512->test:fov512(within) | 0.015 [0.005,0.032] | 0.015 [0.005,0.032] | 0.014 [0.005,0.032] | 0.683 [0.633,0.731] |
| b0_trained_orig (not pre-registered) | thr:orig->test:orig(within) | 0.015 [0.006,0.036] | 0.084 [0.057,0.117] | 0.034 [0.017,0.058] | 0.900 [0.864,0.929] |
| b0_trained_orig (not pre-registered) | thr:orig->test:fov512(cross) | 0.234 [0.192,0.282] | 0.199 [0.158,0.243] | 0.196 [0.156,0.240] | 0.947 [0.919,0.968] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.002 [0.000,0.015] | 0.000 [0.000,0.010] | 0.425 [0.373,0.478] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:fov512(within) | 0.003 [0.000,0.015] | 0.002 [0.000,0.015] | 0.002 [0.000,0.015] | 0.547 [0.494,0.599] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:orig(within) | 0.002 [0.000,0.015] | 0.010 [0.003,0.028] | 0.001 [0.000,0.010] | 0.394 [0.344,0.447] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:fov512(cross) | 0.008 [0.002,0.024] | 0.007 [0.001,0.020] | 0.006 [0.001,0.020] | 0.464 [0.411,0.517] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:orig(cross) | 0.011 [0.003,0.028] | 0.019 [0.008,0.040] | 0.013 [0.005,0.032] | 0.603 [0.550,0.654] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:fov512(within) | 0.015 [0.005,0.032] | 0.015 [0.005,0.032] | 0.014 [0.005,0.032] | 0.683 [0.633,0.731] |

### clean, thresholds from calib_all_reals (NOT PRE-REGISTERED sensitivity variant)

| detector | thr -> test | FPR native | FPR WA | FBH | TPR |
|---|---|---|---|---|---|
| communityforensics | thr:orig->test:orig(within) | 0.022 [0.010,0.044] | 0.011 [0.003,0.028] | 0.011 [0.003,0.028] | 0.944 [0.916,0.966] |
| communityforensics | thr:orig->test:fov512(cross) | 0.003 [0.000,0.015] | 0.003 [0.000,0.015] | 0.000 [0.000,0.010] | 0.831 [0.788,0.868] |
| communityforensics | thr:fov512->test:orig(cross) | 0.145 [0.110,0.186] | 0.061 [0.039,0.092] | 0.115 [0.083,0.152] | 0.997 [0.985,1.000] |
| communityforensics | thr:fov512->test:fov512(within) | 0.050 [0.030,0.078] | 0.039 [0.022,0.065] | 0.036 [0.019,0.061] | 0.975 [0.953,0.989] |
| rine | thr:orig->test:orig(within) | 0.020 [0.008,0.040] | 0.000 [0.000,0.010] | 0.006 [0.001,0.020] | 0.619 [0.567,0.670] |
| rine | thr:orig->test:fov512(cross) | 0.036 [0.019,0.061] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.436 [0.384,0.489] |
| rine | thr:fov512->test:orig(cross) | 0.061 [0.039,0.092] | 0.014 [0.005,0.032] | 0.031 [0.015,0.054] | 0.714 [0.664,0.760] |
| rine | thr:fov512->test:fov512(within) | 0.078 [0.053,0.111] | 0.017 [0.006,0.036] | 0.008 [0.002,0.024] | 0.586 [0.533,0.637] |
| clipdet | thr:orig->test:orig(within) | 0.042 [0.024,0.068] | 0.014 [0.005,0.032] | 0.020 [0.008,0.040] | 0.281 [0.235,0.330] |
| clipdet | thr:orig->test:fov512(cross) | 0.014 [0.005,0.032] | 0.022 [0.010,0.044] | 0.022 [0.010,0.044] | 0.325 [0.277,0.376] |
| clipdet | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.017 [0.006,0.036] |
| clipdet | thr:fov512->test:fov512(within) | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.014 [0.005,0.032] |
| b0 | thr:orig->test:orig(within) | 0.003 [0.000,0.015] | 0.017 [0.006,0.036] | 0.004 [0.000,0.015] | 0.758 [0.711,0.802] |
| b0 | thr:fov512->test:fov512(within) | 0.015 [0.005,0.032] | 0.014 [0.005,0.032] | 0.014 [0.005,0.032] | 0.683 [0.633,0.731] |
| b0_trained_orig (not pre-registered) | thr:orig->test:orig(within) | 0.003 [0.000,0.015] | 0.017 [0.006,0.036] | 0.004 [0.000,0.015] | 0.758 [0.711,0.802] |
| b0_trained_orig (not pre-registered) | thr:orig->test:fov512(cross) | 0.048 [0.028,0.075] | 0.037 [0.019,0.061] | 0.040 [0.022,0.065] | 0.833 [0.791,0.870] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.003 [0.000,0.015] | 0.000 [0.000,0.010] | 0.481 [0.428,0.534] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:fov512(within) | 0.006 [0.001,0.020] | 0.003 [0.000,0.015] | 0.003 [0.000,0.015] | 0.600 [0.547,0.651] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:orig(within) | 0.002 [0.000,0.015] | 0.011 [0.003,0.028] | 0.001 [0.000,0.010] | 0.408 [0.357,0.461] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:fov512(cross) | 0.009 [0.002,0.024] | 0.008 [0.002,0.024] | 0.007 [0.002,0.024] | 0.481 [0.428,0.534] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:orig(cross) | 0.011 [0.003,0.028] | 0.018 [0.006,0.036] | 0.013 [0.003,0.028] | 0.603 [0.550,0.654] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:fov512(within) | 0.015 [0.005,0.032] | 0.014 [0.005,0.032] | 0.014 [0.005,0.032] | 0.683 [0.633,0.731] |

### jpeg75_resize05, thresholds from calib_natives

| detector | thr -> test | FPR native | FPR WA | FBH | TPR |
|---|---|---|---|---|---|
| communityforensics | thr:orig->test:orig(within) | 0.042 [0.024,0.068] | 0.025 [0.012,0.047] | 0.022 [0.010,0.044] | 0.561 [0.508,0.613] |
| communityforensics | thr:orig->test:fov512(cross) | 0.017 [0.006,0.036] | 0.020 [0.008,0.040] | 0.011 [0.003,0.028] | 0.186 [0.147,0.230] |
| communityforensics | thr:fov512->test:orig(cross) | 0.089 [0.062,0.124] | 0.039 [0.022,0.065] | 0.042 [0.024,0.068] | 0.644 [0.593,0.694] |
| communityforensics | thr:fov512->test:fov512(within) | 0.017 [0.006,0.036] | 0.022 [0.010,0.044] | 0.022 [0.010,0.044] | 0.225 [0.183,0.272] |
| rine | thr:orig->test:orig(within) | 0.017 [0.006,0.036] | 0.173 [0.135,0.216] | 0.059 [0.037,0.088] | 0.669 [0.618,0.718] |
| rine | thr:orig->test:fov512(cross) | 0.316 [0.268,0.367] | 0.324 [0.276,0.375] | 0.274 [0.228,0.323] | 0.603 [0.550,0.654] |
| rine | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.022 [0.010,0.044] | 0.000 [0.000,0.010] | 0.281 [0.235,0.330] |
| rine | thr:fov512->test:fov512(within) | 0.039 [0.022,0.065] | 0.034 [0.017,0.058] | 0.025 [0.012,0.047] | 0.222 [0.180,0.269] |
| clipdet | thr:orig->test:orig(within) | 0.017 [0.006,0.036] | 0.034 [0.017,0.058] | 0.042 [0.024,0.068] | 0.039 [0.021,0.064] |
| clipdet | thr:orig->test:fov512(cross) | 0.006 [0.001,0.020] | 0.003 [0.000,0.015] | 0.006 [0.001,0.020] | 0.033 [0.017,0.058] |
| clipdet | thr:fov512->test:orig(cross) | 0.036 [0.019,0.061] | 0.042 [0.024,0.068] | 0.061 [0.039,0.092] | 0.069 [0.045,0.101] |
| clipdet | thr:fov512->test:fov512(within) | 0.014 [0.005,0.032] | 0.022 [0.010,0.044] | 0.017 [0.006,0.036] | 0.067 [0.043,0.098] |
| b0 | thr:orig->test:orig(within) | 0.002 [0.000,0.015] | 0.005 [0.001,0.020] | 0.007 [0.001,0.020] | 0.522 [0.469,0.575] |
| b0 | thr:fov512->test:fov512(within) | 0.014 [0.005,0.032] | 0.013 [0.005,0.032] | 0.013 [0.005,0.032] | 0.606 [0.553,0.656] |
| b0_trained_orig (not pre-registered) | thr:orig->test:orig(within) | 0.002 [0.000,0.015] | 0.005 [0.001,0.020] | 0.007 [0.001,0.020] | 0.522 [0.469,0.575] |
| b0_trained_orig (not pre-registered) | thr:orig->test:fov512(cross) | 0.022 [0.010,0.044] | 0.020 [0.008,0.040] | 0.019 [0.008,0.040] | 0.547 [0.494,0.599] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.267 [0.222,0.316] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:fov512(within) | 0.002 [0.000,0.015] | 0.001 [0.000,0.010] | 0.002 [0.000,0.015] | 0.306 [0.258,0.356] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:orig(within) | 0.013 [0.003,0.028] | 0.011 [0.003,0.028] | 0.014 [0.005,0.032] | 0.583 [0.531,0.635] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:fov512(cross) | 0.014 [0.005,0.032] | 0.014 [0.005,0.032] | 0.014 [0.005,0.032] | 0.594 [0.542,0.646] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:orig(cross) | 0.013 [0.005,0.032] | 0.009 [0.002,0.024] | 0.014 [0.005,0.032] | 0.611 [0.559,0.662] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:fov512(within) | 0.014 [0.005,0.032] | 0.013 [0.005,0.032] | 0.013 [0.005,0.032] | 0.606 [0.553,0.656] |

### jpeg75_resize05, thresholds from calib_all_reals (NOT PRE-REGISTERED sensitivity variant)

| detector | thr -> test | FPR native | FPR WA | FBH | TPR |
|---|---|---|---|---|---|
| communityforensics | thr:orig->test:orig(within) | 0.078 [0.053,0.111] | 0.036 [0.019,0.061] | 0.042 [0.024,0.068] | 0.639 [0.587,0.689] |
| communityforensics | thr:orig->test:fov512(cross) | 0.017 [0.006,0.036] | 0.022 [0.010,0.044] | 0.022 [0.010,0.044] | 0.219 [0.178,0.266] |
| communityforensics | thr:fov512->test:orig(cross) | 0.089 [0.062,0.124] | 0.039 [0.022,0.065] | 0.042 [0.024,0.068] | 0.644 [0.593,0.694] |
| communityforensics | thr:fov512->test:fov512(within) | 0.017 [0.006,0.036] | 0.022 [0.010,0.044] | 0.022 [0.010,0.044] | 0.225 [0.183,0.272] |
| rine | thr:orig->test:orig(within) | 0.000 [0.000,0.010] | 0.022 [0.010,0.044] | 0.000 [0.000,0.010] | 0.281 [0.235,0.330] |
| rine | thr:orig->test:fov512(cross) | 0.039 [0.022,0.065] | 0.034 [0.017,0.058] | 0.025 [0.012,0.047] | 0.222 [0.180,0.269] |
| rine | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.022 [0.010,0.044] | 0.000 [0.000,0.010] | 0.281 [0.235,0.330] |
| rine | thr:fov512->test:fov512(within) | 0.039 [0.022,0.065] | 0.034 [0.017,0.058] | 0.025 [0.012,0.047] | 0.222 [0.180,0.269] |
| clipdet | thr:orig->test:orig(within) | 0.014 [0.005,0.032] | 0.028 [0.013,0.051] | 0.031 [0.015,0.054] | 0.025 [0.011,0.047] |
| clipdet | thr:orig->test:fov512(cross) | 0.006 [0.001,0.020] | 0.000 [0.000,0.010] | 0.006 [0.001,0.020] | 0.028 [0.013,0.050] |
| clipdet | thr:fov512->test:orig(cross) | 0.028 [0.013,0.051] | 0.042 [0.024,0.068] | 0.047 [0.028,0.075] | 0.056 [0.034,0.084] |
| clipdet | thr:fov512->test:fov512(within) | 0.011 [0.003,0.028] | 0.014 [0.005,0.032] | 0.008 [0.002,0.024] | 0.044 [0.026,0.071] |
| b0 | thr:orig->test:orig(within) | 0.002 [0.000,0.015] | 0.007 [0.002,0.024] | 0.007 [0.002,0.024] | 0.544 [0.491,0.597] |
| b0 | thr:fov512->test:fov512(within) | 0.014 [0.005,0.032] | 0.013 [0.005,0.032] | 0.013 [0.005,0.032] | 0.617 [0.564,0.667] |
| b0_trained_orig (not pre-registered) | thr:orig->test:orig(within) | 0.002 [0.000,0.015] | 0.007 [0.002,0.024] | 0.007 [0.002,0.024] | 0.544 [0.491,0.597] |
| b0_trained_orig (not pre-registered) | thr:orig->test:fov512(cross) | 0.024 [0.010,0.044] | 0.023 [0.010,0.044] | 0.023 [0.010,0.044] | 0.569 [0.517,0.621] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:orig(cross) | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.000 [0.000,0.010] | 0.267 [0.222,0.316] |
| b0_trained_orig (not pre-registered) | thr:fov512->test:fov512(within) | 0.002 [0.000,0.015] | 0.001 [0.000,0.010] | 0.002 [0.000,0.015] | 0.308 [0.261,0.359] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:orig(within) | 0.010 [0.003,0.028] | 0.007 [0.002,0.024] | 0.013 [0.003,0.028] | 0.578 [0.525,0.629] |
| b0_trained_fov512 (not pre-registered) | thr:orig->test:fov512(cross) | 0.010 [0.003,0.028] | 0.010 [0.003,0.028] | 0.010 [0.003,0.028] | 0.564 [0.511,0.616] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:orig(cross) | 0.014 [0.005,0.032] | 0.010 [0.003,0.028] | 0.015 [0.006,0.036] | 0.619 [0.567,0.670] |
| b0_trained_fov512 (not pre-registered) | thr:fov512->test:fov512(within) | 0.014 [0.005,0.032] | 0.013 [0.005,0.032] | 0.013 [0.005,0.032] | 0.617 [0.564,0.667] |

## Spearman(score, mean|Laplacian| of the detector input) on test reals (B0: 'full' model)

| cond | detector | arm | native | WA | FBH | all reals | fakes | median Lap native / fakes |
|---|---|---|---|---|---|---|---|---|
| clean | communityforensics | orig | -0.32 | -0.12 | -0.42 | -0.30 | +0.09 | 0.0239 / 0.0517 |
| clean | communityforensics | fov512 | +0.03 | +0.02 | -0.01 | +0.01 | -0.06 | 0.0383 / 0.0510 |
| clean | rine | orig | +0.15 | +0.00 | +0.03 | -0.03 | -0.05 | 0.0307 / 0.0800 |
| clean | rine | fov512 | -0.40 | -0.25 | -0.51 | -0.35 | -0.20 | 0.0510 / 0.0714 |
| clean | clipdet | orig | -0.19 | -0.20 | -0.15 | -0.22 | +0.20 | 0.0395 / 0.0746 |
| clean | clipdet | fov512 | -0.20 | -0.17 | -0.19 | -0.18 | +0.18 | 0.0653 / 0.0830 |
| clean | b0 | orig | +0.37 | +0.24 | +0.34 | +0.38 | +0.24 | 0.0307 / 0.0800 |
| clean | b0 | fov512 | +0.07 | +0.09 | +0.09 | +0.08 | +0.24 | 0.0510 / 0.0714 |
| jpeg75_resize05 | communityforensics | orig | -0.53 | -0.38 | -0.50 | -0.49 | -0.12 | 0.0106 / 0.0221 |
| jpeg75_resize05 | communityforensics | fov512 | -0.28 | -0.26 | -0.28 | -0.27 | -0.25 | 0.0196 / 0.0257 |
| jpeg75_resize05 | rine | orig | -0.15 | -0.02 | -0.08 | -0.03 | +0.11 | 0.0305 / 0.0700 |
| jpeg75_resize05 | rine | fov512 | -0.12 | -0.08 | -0.11 | -0.10 | +0.02 | 0.0612 / 0.0814 |
| jpeg75_resize05 | clipdet | orig | -0.54 | -0.46 | -0.42 | -0.48 | +0.16 | 0.0336 / 0.0680 |
| jpeg75_resize05 | clipdet | fov512 | -0.24 | -0.22 | -0.24 | -0.23 | +0.15 | 0.0597 / 0.0764 |
| jpeg75_resize05 | b0 | orig | +0.36 | +0.14 | +0.27 | +0.30 | +0.08 | 0.0305 / 0.0700 |
| jpeg75_resize05 | b0 | fov512 | -0.05 | -0.04 | -0.03 | -0.04 | +0.16 | 0.0612 / 0.0814 |

## Texture direction (ONE sign per input/condition, set on the orig arm's TRAIN natives vs train fakes, used in both arms; per-arm AUROC of the raw Laplacian shown for information)

| input/arm/cond | AUROC(Lap) on train | sign used |
|---|---|---|
| communityforensics/orig/clean | 0.709 | +1 |
| communityforensics/fov512/clean | 0.562 | +1 |
| rine/orig/clean | 0.745 | +1 |
| rine/fov512/clean | 0.579 | +1 |
| clipdet/orig/clean | 0.654 | +1 |
| clipdet/fov512/clean | 0.566 | +1 |
| b0/orig/clean | 0.745 | +1 |
| b0/fov512/clean | 0.579 | +1 |
| communityforensics/orig/jpeg75_resize05 | 0.663 | +1 |
| communityforensics/fov512/jpeg75_resize05 | 0.575 | +1 |
| rine/orig/jpeg75_resize05 | 0.666 | +1 |
| rine/fov512/jpeg75_resize05 | 0.572 | +1 |
| clipdet/orig/jpeg75_resize05 | 0.656 | +1 |
| clipdet/fov512/jpeg75_resize05 | 0.571 | +1 |
| b0/orig/jpeg75_resize05 | 0.666 | +1 |
| b0/fov512/jpeg75_resize05 | 0.572 | +1 |

## D-008

```
{
 "orig_recheck_vs_007": {
  "communityforensics/clean": {
   "max_abs_diff": 1.7763568394002505e-15,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "communityforensics/jpeg75_resize05": {
   "max_abs_diff": 1.7763568394002505e-15,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "rine/clean": {
   "max_abs_diff": 4.440892098500626e-16,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "rine/jpeg75_resize05": {
   "max_abs_diff": 4.440892098500626e-16,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "clipdet/clean": {
   "max_abs_diff": 2.220446049250313e-16,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "clipdet/jpeg75_resize05": {
   "max_abs_diff": 8.326672684688674e-17,
   "n": 32,
   "tol": 0.001,
   "ok": true
  }
 },
 "fov512_cuda_vs_cpu": {
  "communityforensics/clean": {
   "max_abs_dlogit": 5.817413330078125e-05,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "communityforensics/jpeg75_resize05": {
   "max_abs_dlogit": 8.082389831542969e-05,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "rine/clean": {
   "max_abs_dlogit": 1.5974044799804688e-05,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "rine/jpeg75_resize05": {
   "max_abs_dlogit": 3.56137752532959e-05,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "clipdet/clean": {
   "max_abs_dlogit": 5.5909156799316406e-05,
   "n": 32,
   "tol": 0.001,
   "ok": true
  },
  "clipdet/jpeg75_resize05": {
   "max_abs_dlogit": 4.035234451293945e-05,
   "n": 32,
   "tol": 0.001,
   "ok": true
  }
 }
}
```

B0 (fresh fit through 005/run.py code) vs 005/pilot_v2 cached A0 scores: max |d| = {'orig': 0.0, 'fov512': 0.0}
