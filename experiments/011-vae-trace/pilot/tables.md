# E4 PILOT: SD-VAE decoder-trace detectability

Pair AUROC = pooled AUROC of probe score for the test round-trips vs their own test reals (cluster bootstrap by device, 2000 replicates; a pair shares its cluster). `primary` = probe fit on CLEAN train pairs; `matched` = probe fit on train pairs of the tested condition (sensitivity, not pre-registered).


## Arm `orig` (label arm)

297 train pairs, 1074 test pairs, 25 test devices; transfer on 358 natives vs 720 fakes.


### Pair AUROC, primary training

| family | clean | jpeg75_resize05 | win-rate clean | win-rate laundered | train in-sample clean |
|---|---|---|---|---|---|
| F1 | 0.670 [0.647, 0.695] | 0.560 [0.552, 0.569] | 0.933 | 0.787 | 0.841 |
| F2 | 0.859 [0.827, 0.891] | 0.597 [0.579, 0.620] | 0.991 | 0.843 | 1.000 |
| F3 | 0.625 [0.598, 0.650] | 0.527 [0.510, 0.544] | 0.651 | 0.533 | 0.767 |
| F4 | 0.862 [0.833, 0.894] | 0.598 [0.580, 0.620] | 0.991 | 0.831 | 1.000 |
| texture (Laplacian, sign -1/-1) | 0.554 [0.547, 0.564] | 0.540 [0.534, 0.550] | 0.972 | 0.976 | - |

Per-pool pair AUROC (point):

| family | cond | vision_facebook_hq | vision_native | vision_whatsapp |
|---|---|---|---|---|
| F1 | clean | 0.647 | 0.671 | 0.698 |
| F1 | jpeg75_resize05 | 0.555 | 0.550 | 0.577 |
| F2 | clean | 0.809 | 0.875 | 0.894 |
| F2 | jpeg75_resize05 | 0.600 | 0.570 | 0.625 |
| F3 | clean | 0.595 | 0.649 | 0.628 |
| F3 | jpeg75_resize05 | 0.512 | 0.508 | 0.559 |
| F4 | clean | 0.814 | 0.874 | 0.901 |
| F4 | jpeg75_resize05 | 0.600 | 0.571 | 0.625 |

### Pair AUROC, matched training

| family | clean | jpeg75_resize05 | win-rate clean | win-rate laundered | train in-sample clean |
|---|---|---|---|---|---|
| F1 | 0.670 [0.647, 0.695] | 0.579 [0.566, 0.593] | 0.933 | 0.850 | 0.841 |
| F2 | 0.859 [0.827, 0.891] | 0.670 [0.640, 0.705] | 0.991 | 0.980 | 1.000 |
| F3 | 0.625 [0.598, 0.650] | 0.569 [0.552, 0.589] | 0.651 | 0.619 | 0.767 |
| F4 | 0.862 [0.833, 0.894] | 0.670 [0.641, 0.705] | 0.991 | 0.976 | 1.000 |
| texture (Laplacian, sign -1/-1) | 0.554 [0.547, 0.564] | 0.540 [0.534, 0.550] | 0.972 | 0.976 | - |

Per-pool pair AUROC (point):

| family | cond | vision_facebook_hq | vision_native | vision_whatsapp |
|---|---|---|---|---|
| F1 | clean | 0.647 | 0.671 | 0.698 |
| F1 | jpeg75_resize05 | 0.571 | 0.564 | 0.604 |
| F2 | clean | 0.809 | 0.875 | 0.894 |
| F2 | jpeg75_resize05 | 0.657 | 0.643 | 0.716 |
| F3 | clean | 0.595 | 0.649 | 0.628 |
| F3 | jpeg75_resize05 | 0.577 | 0.546 | 0.587 |
| F4 | clean | 0.814 | 0.874 | 0.901 |
| F4 | jpeg75_resize05 | 0.656 | 0.644 | 0.716 |

### Transfer (primary probe): test natives vs Synthbuster fakes, macro AUROC

| family | cond | all 6 | SD-family (SD2, SDXL) | other (DALL-E 2/3, Firefly, MJ5) |
|---|---|---|---|---|
| F1 | clean | 0.819 [0.769, 0.861] | 0.865 [0.817, 0.906] | 0.796 [0.743, 0.842] |
| F1 | jpeg75_resize05 | 0.778 [0.731, 0.817] | 0.814 [0.763, 0.855] | 0.759 [0.711, 0.802] |
| F2 | clean | 0.776 [0.715, 0.837] | 0.874 [0.825, 0.919] | 0.727 [0.657, 0.797] |
| F2 | jpeg75_resize05 | 0.643 [0.578, 0.707] | 0.697 [0.630, 0.762] | 0.616 [0.546, 0.685] |
| F3 | clean | 0.563 [0.511, 0.617] | 0.719 [0.665, 0.770] | 0.485 [0.428, 0.543] |
| F3 | jpeg75_resize05 | 0.567 [0.522, 0.610] | 0.540 [0.486, 0.594] | 0.580 [0.534, 0.626] |
| F4 | clean | 0.768 [0.707, 0.827] | 0.874 [0.826, 0.917] | 0.715 [0.645, 0.783] |
| F4 | jpeg75_resize05 | 0.649 [0.587, 0.713] | 0.700 [0.635, 0.764] | 0.624 [0.557, 0.692] |
| texture | clean | 0.228 [0.180, 0.282] | 0.171 [0.123, 0.230] | 0.256 [0.205, 0.312] |
| texture | jpeg75_resize05 | 0.311 [0.241, 0.391] | 0.276 [0.206, 0.356] | 0.329 [0.258, 0.410] |

### Transfer (matched probe): test natives vs Synthbuster fakes, macro AUROC

| family | cond | all 6 | SD-family (SD2, SDXL) | other (DALL-E 2/3, Firefly, MJ5) |
|---|---|---|---|---|
| F1 | clean | 0.819 [0.769, 0.861] | 0.865 [0.817, 0.906] | 0.796 [0.743, 0.842] |
| F1 | jpeg75_resize05 | 0.513 [0.464, 0.562] | 0.573 [0.515, 0.631] | 0.483 [0.434, 0.532] |
| F2 | clean | 0.776 [0.715, 0.837] | 0.874 [0.825, 0.919] | 0.727 [0.657, 0.797] |
| F2 | jpeg75_resize05 | 0.494 [0.438, 0.553] | 0.649 [0.586, 0.713] | 0.417 [0.358, 0.479] |
| F3 | clean | 0.563 [0.511, 0.617] | 0.719 [0.665, 0.770] | 0.485 [0.428, 0.543] |
| F3 | jpeg75_resize05 | 0.497 [0.454, 0.540] | 0.531 [0.479, 0.582] | 0.479 [0.437, 0.525] |
| F4 | clean | 0.768 [0.707, 0.827] | 0.874 [0.826, 0.917] | 0.715 [0.645, 0.783] |
| F4 | jpeg75_resize05 | 0.501 [0.443, 0.562] | 0.652 [0.589, 0.717] | 0.426 [0.366, 0.488] |
| texture | clean | 0.228 [0.180, 0.282] | 0.171 [0.123, 0.230] | 0.256 [0.205, 0.312] |
| texture | jpeg75_resize05 | 0.311 [0.241, 0.391] | 0.276 [0.206, 0.356] | 0.329 [0.258, 0.410] |

### Decision rules (primary training, pre-registered reading)

Best family per condition: {'clean': 'F4', 'jpeg75_resize05': 'F4'}

- VAE trace detectable (best clean >= 0.90, CI lo > 0.85): **NO** (F4: 0.862, CI lo 0.833)
- Survives laundering (best laundered >= 0.75, CI lo > 0.70): **NO** (F4: 0.598, CI lo 0.580)
- Trace-specific at clean (best - texture >= 0.10): **YES** (delta +0.308 [+0.279, +0.336], F4)
- Trace-specific at jpeg75_resize05 (best - texture >= 0.10): **NO** (delta +0.058 [+0.043, +0.074], F4)
- Headline trace-specific (clean): **YES**
- VAE-trace methods deprioritized in the Phase 5 design: **YES**

### Decision rules (matched training, sensitivity)

Best family per condition: {'clean': 'F4', 'jpeg75_resize05': 'F4'}

- VAE trace detectable (best clean >= 0.90, CI lo > 0.85): **NO** (F4: 0.862, CI lo 0.833)
- Survives laundering (best laundered >= 0.75, CI lo > 0.70): **NO** (F4: 0.670, CI lo 0.641)
- Trace-specific at clean (best - texture >= 0.10): **YES** (delta +0.308 [+0.279, +0.336], F4)
- Trace-specific at jpeg75_resize05 (best - texture >= 0.10): **YES** (delta +0.130 [+0.104, +0.157], F4)
- Headline trace-specific (clean): **YES**
- VAE-trace methods deprioritized in the Phase 5 design: **YES**

## Arm `fov512` (descriptive)

297 train pairs, 1074 test pairs, 25 test devices; transfer on 358 natives vs 720 fakes.


### Pair AUROC, primary training

| family | clean | jpeg75_resize05 | win-rate clean | win-rate laundered | train in-sample clean |
|---|---|---|---|---|---|
| F1 | 0.699 [0.676, 0.726] | 0.563 [0.549, 0.577] | 0.955 | 0.811 | 1.000 |
| F2 | 0.945 [0.932, 0.958] | 0.599 [0.581, 0.621] | 1.000 | 0.860 | 1.000 |
| F3 | 0.882 [0.841, 0.914] | 0.526 [0.503, 0.549] | 0.907 | 0.535 | 0.967 |
| F4 | 0.962 [0.953, 0.972] | 0.600 [0.583, 0.625] | 1.000 | 0.830 | 1.000 |
| texture (Laplacian, sign -1/-1) | 0.556 [0.545, 0.574] | 0.555 [0.546, 0.572] | 0.885 | 0.993 | - |

Per-pool pair AUROC (point):

| family | cond | vision_facebook_hq | vision_native | vision_whatsapp |
|---|---|---|---|---|
| F1 | clean | 0.693 | 0.706 | 0.699 |
| F1 | jpeg75_resize05 | 0.565 | 0.562 | 0.563 |
| F2 | clean | 0.936 | 0.954 | 0.945 |
| F2 | jpeg75_resize05 | 0.597 | 0.597 | 0.601 |
| F3 | clean | 0.890 | 0.868 | 0.887 |
| F3 | jpeg75_resize05 | 0.528 | 0.529 | 0.520 |
| F4 | clean | 0.952 | 0.971 | 0.962 |
| F4 | jpeg75_resize05 | 0.600 | 0.600 | 0.601 |

### Pair AUROC, matched training

| family | clean | jpeg75_resize05 | win-rate clean | win-rate laundered | train in-sample clean |
|---|---|---|---|---|---|
| F1 | 0.699 [0.676, 0.726] | 0.624 [0.600, 0.651] | 0.955 | 0.890 | 1.000 |
| F2 | 0.945 [0.932, 0.958] | 0.739 [0.697, 0.787] | 1.000 | 0.980 | 1.000 |
| F3 | 0.882 [0.841, 0.914] | 0.579 [0.551, 0.612] | 0.907 | 0.605 | 0.967 |
| F4 | 0.962 [0.953, 0.972] | 0.737 [0.695, 0.784] | 1.000 | 0.967 | 1.000 |
| texture (Laplacian, sign -1/-1) | 0.556 [0.545, 0.574] | 0.555 [0.546, 0.572] | 0.885 | 0.993 | - |

Per-pool pair AUROC (point):

| family | cond | vision_facebook_hq | vision_native | vision_whatsapp |
|---|---|---|---|---|
| F1 | clean | 0.693 | 0.706 | 0.699 |
| F1 | jpeg75_resize05 | 0.626 | 0.623 | 0.622 |
| F2 | clean | 0.936 | 0.954 | 0.945 |
| F2 | jpeg75_resize05 | 0.740 | 0.736 | 0.741 |
| F3 | clean | 0.890 | 0.868 | 0.887 |
| F3 | jpeg75_resize05 | 0.585 | 0.583 | 0.569 |
| F4 | clean | 0.952 | 0.971 | 0.962 |
| F4 | jpeg75_resize05 | 0.738 | 0.735 | 0.738 |

### Transfer (primary probe): test natives vs Synthbuster fakes, macro AUROC

| family | cond | all 6 | SD-family (SD2, SDXL) | other (DALL-E 2/3, Firefly, MJ5) |
|---|---|---|---|---|
| F1 | clean | 0.779 [0.746, 0.809] | 0.837 [0.802, 0.872] | 0.749 [0.712, 0.784] |
| F1 | jpeg75_resize05 | 0.682 [0.640, 0.721] | 0.747 [0.699, 0.791] | 0.649 [0.603, 0.693] |
| F2 | clean | 0.892 [0.864, 0.916] | 0.927 [0.900, 0.948] | 0.875 [0.842, 0.904] |
| F2 | jpeg75_resize05 | 0.665 [0.615, 0.717] | 0.707 [0.651, 0.760] | 0.644 [0.590, 0.698] |
| F3 | clean | 0.594 [0.542, 0.644] | 0.519 [0.460, 0.578] | 0.631 [0.578, 0.682] |
| F3 | jpeg75_resize05 | 0.479 [0.440, 0.521] | 0.430 [0.381, 0.480] | 0.504 [0.461, 0.548] |
| F4 | clean | 0.893 [0.866, 0.917] | 0.926 [0.899, 0.948] | 0.877 [0.845, 0.906] |
| F4 | jpeg75_resize05 | 0.655 [0.599, 0.710] | 0.684 [0.624, 0.742] | 0.641 [0.583, 0.699] |
| texture | clean | 0.394 [0.322, 0.477] | 0.370 [0.294, 0.458] | 0.406 [0.333, 0.487] |
| texture | jpeg75_resize05 | 0.379 [0.310, 0.455] | 0.366 [0.297, 0.443] | 0.386 [0.314, 0.463] |

### Transfer (matched probe): test natives vs Synthbuster fakes, macro AUROC

| family | cond | all 6 | SD-family (SD2, SDXL) | other (DALL-E 2/3, Firefly, MJ5) |
|---|---|---|---|---|
| F1 | clean | 0.779 [0.746, 0.809] | 0.837 [0.802, 0.872] | 0.749 [0.712, 0.784] |
| F1 | jpeg75_resize05 | 0.710 [0.671, 0.747] | 0.729 [0.684, 0.775] | 0.701 [0.659, 0.741] |
| F2 | clean | 0.892 [0.864, 0.916] | 0.927 [0.900, 0.948] | 0.875 [0.842, 0.904] |
| F2 | jpeg75_resize05 | 0.626 [0.575, 0.677] | 0.732 [0.677, 0.784] | 0.572 [0.515, 0.631] |
| F3 | clean | 0.594 [0.542, 0.644] | 0.519 [0.460, 0.578] | 0.631 [0.578, 0.682] |
| F3 | jpeg75_resize05 | 0.515 [0.463, 0.571] | 0.554 [0.494, 0.613] | 0.496 [0.440, 0.555] |
| F4 | clean | 0.893 [0.866, 0.917] | 0.926 [0.899, 0.948] | 0.877 [0.845, 0.906] |
| F4 | jpeg75_resize05 | 0.603 [0.549, 0.655] | 0.712 [0.657, 0.765] | 0.549 [0.491, 0.609] |
| texture | clean | 0.394 [0.322, 0.477] | 0.370 [0.294, 0.458] | 0.406 [0.333, 0.487] |
| texture | jpeg75_resize05 | 0.379 [0.310, 0.455] | 0.366 [0.297, 0.443] | 0.386 [0.314, 0.463] |

### Decision rules (primary training, pre-registered reading)

Best family per condition: {'clean': 'F4', 'jpeg75_resize05': 'F4'}

- VAE trace detectable (best clean >= 0.90, CI lo > 0.85): **YES** (F4: 0.962, CI lo 0.953)
- Survives laundering (best laundered >= 0.75, CI lo > 0.70): **NO** (F4: 0.600, CI lo 0.583)
- Trace-specific at clean (best - texture >= 0.10): **YES** (delta +0.406 [+0.393, +0.417], F4)
- Trace-specific at jpeg75_resize05 (best - texture >= 0.10): **NO** (delta +0.045 [+0.028, +0.061], F4)
- Headline trace-specific (clean): **YES**
- VAE-trace methods deprioritized in the Phase 5 design: **YES**

### Decision rules (matched training, sensitivity)

Best family per condition: {'clean': 'F4', 'jpeg75_resize05': 'F2'}

- VAE trace detectable (best clean >= 0.90, CI lo > 0.85): **YES** (F4: 0.962, CI lo 0.953)
- Survives laundering (best laundered >= 0.75, CI lo > 0.70): **NO** (F2: 0.739, CI lo 0.697)
- Trace-specific at clean (best - texture >= 0.10): **YES** (delta +0.406 [+0.393, +0.417], F4)
- Trace-specific at jpeg75_resize05 (best - texture >= 0.10): **YES** (delta +0.184 [+0.145, +0.219], F2)
- Headline trace-specific (clean): **YES**
- VAE-trace methods deprioritized in the Phase 5 design: **YES**

## D-008
```
{
 "vae": {
  "sd-vae-ft-mse": {
   "n": 32,
   "max_abs_dpixel_x1": 9.194016456604004e-05,
   "max_abs_dpixel_x2": 2.8371810913085938e-05,
   "tol": 0.001,
   "uint8_mismatch_frac_x1": 8.769830067952474e-05,
   "ok": true,
   "reused_from": "experiments/009-vae-aligned/pilot/d008_guard.json (same weights revision, same recon_f/q8; not re-run)",
   "revision": "31f26fdeee1355a5c34592e401dd41e45d25a493"
  }
 },
 "probe": {
  "tol": 0.001,
  "items": {
   "orig/clean": {
    "n": 32,
    "max_abs_feature_diff": 5.817413330078125e-05,
    "max_scaled_feature_diff": 2.0103398128412664e-05,
    "max_abs_dlogit_F1_train_clean": 6.461551223413409e-06,
    "max_abs_dlogit_F2_train_clean": 3.7583885422298735e-05,
    "max_abs_dlogit_F3_train_clean": 0.0,
    "max_abs_dlogit_F4_train_clean": 3.8060803496442475e-05,
    "max_abs_dlogit": 3.8060803496442475e-05,
    "ok": true
   },
   "orig/jpeg75_resize05": {
    "n": 32,
    "max_abs_feature_diff": 0.000102996826171875,
    "max_scaled_feature_diff": 1.56978057930246e-05,
    "max_abs_dlogit_F1_train_clean": 3.134813630678579e-06,
    "max_abs_dlogit_F1_train_jpeg75_resize05": 6.677749721895054e-05,
    "max_abs_dlogit_F2_train_clean": 1.664339872453624e-05,
    "max_abs_dlogit_F2_train_jpeg75_resize05": 1.9672237615786514e-05,
    "max_abs_dlogit_F3_train_clean": 0.0,
    "max_abs_dlogit_F3_train_jpeg75_resize05": 0.0,
    "max_abs_dlogit_F4_train_clean": 1.5862270090982733e-05,
    "max_abs_dlogit_F4_train_jpeg75_resize05": 1.8439478447485058e-05,
    "max_abs_dlogit": 6.677749721895054e-05,
    "ok": true
   },
   "fov512/clean": {
    "n": 32,
    "max_abs_feature_diff": 0.00011801719665527344,
    "max_scaled_feature_diff": 1.7491782273282297e-05,
    "max_abs_dlogit_F1_train_clean": 0.00020621355086092308,
    "max_abs_dlogit_F2_train_clean": 4.505040768343349e-05,
    "max_abs_dlogit_F3_train_clean": 0.0,
    "max_abs_dlogit_F4_train_clean": 2.3861747490805385e-05,
    "max_abs_dlogit": 0.00020621355086092308,
    "ok": true
   },
   "fov512/jpeg75_resize05": {
    "n": 32,
    "max_abs_feature_diff": 5.817413330078125e-05,
    "max_scaled_feature_diff": 1.638523281144444e-05,
    "max_abs_dlogit_F1_train_clean": 0.00024664750189629103,
    "max_abs_dlogit_F1_train_jpeg75_resize05": 1.777045140061162e-05,
    "max_abs_dlogit_F2_train_clean": 4.325866798993161e-05,
    "max_abs_dlogit_F2_train_jpeg75_resize05": 1.1385755768533734e-05,
    "max_abs_dlogit_F3_train_clean": 0.0,
    "max_abs_dlogit_F3_train_jpeg75_resize05": 0.0,
    "max_abs_dlogit_F4_train_clean": 2.3347770877979457e-05,
    "max_abs_dlogit_F4_train_jpeg75_resize05": 1.2216662844677373e-05,
    "max_abs_dlogit": 0.00024664750189629103,
    "ok": true
   }
  },
  "ok": true
 }
}
```


## Checks of 009's cache
```
{
 "orig": {
  "n_train": 297,
  "missing": 0,
  "wrong_format": 0,
  "wrong_qf": 0,
  "wrong_size": 0,
  "regen_n": 16,
  "regen_mismatch_frac_mean": 0.0,
  "regen_max_abs_diff_uint8": 0
 },
 "fov512": {
  "n_train": 297,
  "missing": 0,
  "wrong_format": 0,
  "wrong_qf": 0,
  "wrong_size": 0,
  "regen_n": 16,
  "regen_mismatch_frac_mean": 0.0,
  "regen_max_abs_diff_uint8": 0
 }
}
{
 "orig/clean": {
  "n": 297,
  "max_abs_diff_vs_009": 2.09808349609375e-05
 },
 "orig/jpeg75_resize05": {
  "n": 297,
  "max_abs_diff_vs_009": 1.621246337890625e-05
 },
 "fov512/clean": {
  "n": 297,
  "max_abs_diff_vs_009": 2.288818359375e-05
 },
 "fov512/jpeg75_resize05": {
  "n": 297,
  "max_abs_diff_vs_009": 3.62396240234375e-05
 }
}
```
