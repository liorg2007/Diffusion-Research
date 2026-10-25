# H-D1 PILOT self-ref-trace-contrast

Computed label (fov512 arm, label score B): **UNINFORMATIVE; NOT DIFFUSION-SPECIFIC**

Fake-high scores; macro AUROC over the 6 Synthbuster generators, test natives (25 devices) vs 120 fakes per generator, cluster bootstrap (2000 replicates, reals by device, fakes by image per generator, one plan for all scores). Shift scores: fake-score = -(d(round-trip) - d(original)).


## Arm `fov512` (label arm): UNINFORMATIVE; NOT DIFFUSION-SPECIFIC

297 train reals, 43 calib natives, 1074 test reals (358 natives, 25 devices), 720 fakes.

### Macro AUROC [95% CI]

| score | clean | jpeg75_resize05 | social_sim |
|---|---|---|---|
| A: w_p.[phi(VAE t)-phi(t)] | 0.795 [0.756, 0.836] | 0.434 [0.382, 0.490] | 0.575 [0.531, 0.618] |
| B: w_m.[phi(L0 VAE t)-phi(L0 t)] (LABEL) | 0.533 [0.456, 0.613] | 0.496 [0.431, 0.569] | 0.539 [0.475, 0.610] |
| Bu: B, uniform L0 | 0.557 [0.499, 0.623] | 0.496 [0.431, 0.569] | 0.538 [0.481, 0.603] |
| A2: diff-vector probe | 0.672 [0.622, 0.727] | 0.457 [0.402, 0.519] | 0.559 [0.514, 0.607] |
| A2L: A2 with L0 | 0.615 [0.555, 0.680] | 0.481 [0.412, 0.554] | 0.519 [0.463, 0.582] |
| pooled F2 (E4 primary probe on phi(t)) | 0.892 [0.864, 0.916] | 0.665 [0.615, 0.717] | 0.759 [0.714, 0.801] |
| F2 matched probe on phi(t) | 0.588 [0.517, 0.659] | 0.626 [0.575, 0.677] | 0.645 [0.585, 0.701] |
| Laplacian texture | 0.606 [0.523, 0.678] | 0.621 [0.545, 0.690] | 0.601 [0.519, 0.672] |
| pixel L2 round-trip error | 0.396 [0.311, 0.493] | 0.376 [0.302, 0.456] | 0.397 [0.314, 0.491] |
| A with blur for VAE | 0.838 [0.801, 0.871] | 0.632 [0.573, 0.686] | 0.688 [0.635, 0.734] |
| B with blur for VAE | 0.433 [0.348, 0.531] | 0.628 [0.573, 0.682] | 0.471 [0.417, 0.526] |

Flipped-orientation macro (1 - AUROC; descriptive only): texture: 0.394/0.379/0.399; pixL2: 0.604/0.624/0.603; A: 0.205/0.566/0.425; B: 0.467/0.504/0.461

### SD-family (SD2, SDXL) vs other (DALL-E 2/3, Firefly, MJ5)

| score | cond | SD-family | other |
|---|---|---|---|
| A | clean | 0.831 [0.789, 0.868] | 0.778 [0.734, 0.821] |
| A | jpeg75_resize05 | 0.429 [0.375, 0.490] | 0.436 [0.380, 0.495] |
| A | social_sim | 0.581 [0.530, 0.632] | 0.572 [0.525, 0.617] |
| B | clean | 0.512 [0.436, 0.593] | 0.544 [0.466, 0.626] |
| B | jpeg75_resize05 | 0.426 [0.357, 0.499] | 0.531 [0.465, 0.603] |
| B | social_sim | 0.510 [0.444, 0.583] | 0.553 [0.486, 0.626] |
| Bu | clean | 0.587 [0.522, 0.653] | 0.543 [0.481, 0.609] |
| Bu | jpeg75_resize05 | 0.426 [0.357, 0.499] | 0.531 [0.465, 0.603] |
| Bu | social_sim | 0.513 [0.451, 0.580] | 0.550 [0.489, 0.617] |
| A2 | clean | 0.645 [0.584, 0.707] | 0.686 [0.635, 0.741] |
| A2 | jpeg75_resize05 | 0.415 [0.358, 0.482] | 0.478 [0.420, 0.541] |
| A2 | social_sim | 0.534 [0.479, 0.591] | 0.572 [0.523, 0.621] |
| A2L | clean | 0.594 [0.526, 0.664] | 0.626 [0.566, 0.690] |
| A2L | jpeg75_resize05 | 0.388 [0.320, 0.462] | 0.527 [0.456, 0.602] |
| A2L | social_sim | 0.462 [0.401, 0.530] | 0.547 [0.489, 0.613] |
| F2p | clean | 0.927 [0.900, 0.948] | 0.875 [0.842, 0.904] |
| F2p | jpeg75_resize05 | 0.707 [0.651, 0.760] | 0.644 [0.590, 0.698] |
| F2p | social_sim | 0.800 [0.755, 0.841] | 0.738 [0.689, 0.786] |
| texture | clean | 0.630 [0.542, 0.706] | 0.594 [0.513, 0.667] |
| texture | jpeg75_resize05 | 0.634 [0.557, 0.703] | 0.614 [0.537, 0.686] |
| texture | social_sim | 0.615 [0.531, 0.687] | 0.594 [0.511, 0.667] |
| pixL2 | clean | 0.394 [0.309, 0.490] | 0.396 [0.310, 0.496] |
| pixL2 | jpeg75_resize05 | 0.371 [0.298, 0.450] | 0.379 [0.301, 0.460] |
| pixL2 | social_sim | 0.389 [0.308, 0.480] | 0.401 [0.316, 0.497] |
| blurB | clean | 0.449 [0.365, 0.542] | 0.425 [0.336, 0.526] |
| blurB | jpeg75_resize05 | 0.628 [0.568, 0.683] | 0.628 [0.571, 0.683] |
| blurB | social_sim | 0.494 [0.434, 0.556] | 0.460 [0.403, 0.517] |

### Per generator (AUROC, test natives vs that generator)

| score | cond | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|---|
| A | clean | 0.875 | 0.739 | 0.722 | 0.776 | 0.778 | 0.883 |
| A | jpeg75_resize05 | 0.594 | 0.350 | 0.342 | 0.459 | 0.324 | 0.533 |
| A | social_sim | 0.662 | 0.498 | 0.490 | 0.637 | 0.514 | 0.649 |
| B | clean | 0.526 | 0.533 | 0.466 | 0.653 | 0.390 | 0.633 |
| B | jpeg75_resize05 | 0.549 | 0.507 | 0.492 | 0.577 | 0.311 | 0.541 |
| B | social_sim | 0.522 | 0.536 | 0.545 | 0.608 | 0.392 | 0.629 |
| F2p | clean | 0.922 | 0.900 | 0.824 | 0.853 | 0.878 | 0.976 |
| F2p | jpeg75_resize05 | 0.710 | 0.700 | 0.524 | 0.641 | 0.629 | 0.784 |
| F2p | social_sim | 0.789 | 0.768 | 0.663 | 0.732 | 0.728 | 0.873 |
| blurB | clean | 0.469 | 0.373 | 0.431 | 0.425 | 0.356 | 0.542 |
| blurB | jpeg75_resize05 | 0.653 | 0.700 | 0.526 | 0.632 | 0.546 | 0.709 |
| blurB | social_sim | 0.490 | 0.420 | 0.504 | 0.425 | 0.448 | 0.541 |
| texture | clean | 0.516 | 0.729 | 0.540 | 0.593 | 0.706 | 0.554 |
| texture | jpeg75_resize05 | 0.553 | 0.727 | 0.598 | 0.578 | 0.706 | 0.562 |
| texture | social_sim | 0.512 | 0.721 | 0.560 | 0.582 | 0.696 | 0.533 |
| pixL2 | clean | 0.424 | 0.340 | 0.378 | 0.444 | 0.303 | 0.485 |
| pixL2 | jpeg75_resize05 | 0.420 | 0.296 | 0.376 | 0.423 | 0.300 | 0.443 |
| pixL2 | social_sim | 0.436 | 0.336 | 0.388 | 0.445 | 0.299 | 0.480 |

### Per real pool (macro AUROC of that pool's test reals vs fakes)

| score | cond | native | WhatsApp | Facebook-HQ |
|---|---|---|---|---|
| A | clean | 0.795 [0.756, 0.836] | 0.731 [0.688, 0.776] | 0.721 [0.676, 0.765] |
| A | jpeg75_resize05 | 0.434 [0.382, 0.490] | 0.451 [0.400, 0.507] | 0.436 [0.387, 0.488] |
| A | social_sim | 0.575 [0.531, 0.618] | 0.583 [0.545, 0.619] | 0.590 [0.545, 0.637] |
| B | clean | 0.533 [0.456, 0.613] | 0.545 [0.471, 0.621] | 0.546 [0.472, 0.623] |
| B | jpeg75_resize05 | 0.496 [0.431, 0.569] | 0.502 [0.433, 0.578] | 0.494 [0.422, 0.572] |
| B | social_sim | 0.539 [0.475, 0.610] | 0.553 [0.494, 0.614] | 0.553 [0.491, 0.616] |
| F2p | clean | 0.892 [0.864, 0.916] | 0.873 [0.844, 0.898] | 0.865 [0.833, 0.893] |
| F2p | jpeg75_resize05 | 0.665 [0.615, 0.717] | 0.673 [0.624, 0.723] | 0.671 [0.622, 0.720] |
| F2p | social_sim | 0.759 [0.714, 0.801] | 0.772 [0.734, 0.808] | 0.768 [0.730, 0.805] |
| blurB | clean | 0.433 [0.348, 0.531] | - | - |
| blurB | jpeg75_resize05 | 0.628 [0.573, 0.682] | - | - |
| blurB | social_sim | 0.471 [0.417, 0.526] | - | - |
| texture | clean | 0.606 [0.523, 0.678] | 0.594 [0.515, 0.666] | 0.609 [0.529, 0.678] |
| texture | jpeg75_resize05 | 0.621 [0.545, 0.690] | 0.621 [0.545, 0.690] | 0.622 [0.548, 0.691] |
| texture | social_sim | 0.601 [0.519, 0.672] | 0.601 [0.521, 0.672] | 0.608 [0.529, 0.678] |

### FPR at the 1% threshold (primary: device cross-fit; test natives / WA / FBH of the other half's devices; pooled over both folds)

| score | cond | FPR native | FPR WA | FPR FBH | TPR (fakes) | FPR native/WA/FBH with calib-split threshold | TPR (calib thr) |
|---|---|---|---|---|---|---|---|
| A | clean | 0.008 | 0.022 | 0.045 | 0.133 | 0.034 / 0.053 / 0.078 | 0.221 |
| A | jpeg75_resize05 | 0.008 | 0.008 | 0.008 | 0.018 | 0.003 / 0.003 / 0.000 | 0.010 |
| A | social_sim | 0.011 | 0.011 | 0.036 | 0.018 | 0.008 / 0.008 / 0.025 | 0.017 |
| B | clean | 0.011 | 0.008 | 0.003 | 0.006 | 0.031 / 0.031 / 0.011 | 0.028 |
| B | jpeg75_resize05 | 0.008 | 0.008 | 0.008 | 0.003 | 0.025 / 0.017 / 0.017 | 0.010 |
| B | social_sim | 0.022 | 0.014 | 0.017 | 0.025 | 0.003 / 0.008 / 0.000 | 0.011 |
| Bu | clean | 0.011 | 0.011 | 0.006 | 0.023 | 0.006 / 0.003 / 0.000 | 0.003 |
| Bu | jpeg75_resize05 | 0.008 | 0.008 | 0.008 | 0.003 | 0.025 / 0.017 / 0.017 | 0.010 |
| Bu | social_sim | 0.008 | 0.008 | 0.003 | 0.009 | 0.011 / 0.008 / 0.003 | 0.011 |
| A2 | clean | 0.011 | 0.020 | 0.022 | 0.069 | 0.008 / 0.011 / 0.008 | 0.051 |
| A2 | jpeg75_resize05 | 0.008 | 0.003 | 0.006 | 0.014 | 0.025 / 0.028 / 0.014 | 0.026 |
| A2 | social_sim | 0.014 | 0.003 | 0.014 | 0.018 | 0.028 / 0.031 / 0.047 | 0.057 |
| A2L | clean | 0.006 | 0.014 | 0.006 | 0.026 | 0.031 / 0.034 / 0.025 | 0.049 |
| A2L | jpeg75_resize05 | 0.020 | 0.022 | 0.020 | 0.013 | 0.003 / 0.008 / 0.006 | 0.007 |
| A2L | social_sim | 0.014 | 0.006 | 0.020 | 0.008 | 0.008 / 0.003 / 0.017 | 0.004 |
| F2p | clean | 0.008 | 0.011 | 0.017 | 0.324 | 0.061 / 0.067 / 0.098 | 0.608 |
| F2p | jpeg75_resize05 | 0.011 | 0.008 | 0.011 | 0.033 | 0.020 / 0.008 / 0.011 | 0.061 |
| F2p | social_sim | 0.011 | 0.011 | 0.011 | 0.073 | 0.020 / 0.017 / 0.025 | 0.182 |
| texture | clean | 0.028 | 0.028 | 0.028 | 0.008 | 0.003 / 0.003 / 0.000 | 0.001 |
| texture | jpeg75_resize05 | 0.022 | 0.022 | 0.020 | 0.009 | 0.006 / 0.006 / 0.006 | 0.003 |
| texture | social_sim | 0.025 | 0.028 | 0.022 | 0.010 | 0.000 / 0.000 / 0.000 | 0.000 |
| pixL2 | clean | 0.008 | 0.006 | 0.008 | 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| pixL2 | jpeg75_resize05 | 0.028 | 0.025 | 0.028 | 0.000 | 0.017 / 0.014 / 0.017 | 0.000 |
| pixL2 | social_sim | 0.014 | 0.011 | 0.006 | 0.000 | 0.011 / 0.011 / 0.006 | 0.000 |

### Matched-blur control and other paired differences (macro AUROC, paired bootstrap)

| contrast | cond | delta [95% CI] |
|---|---|---|
| B-texture | clean | -0.073 [-0.218, +0.089] |
| B-pixL2 | clean | +0.138 [+0.108, +0.170] |
| B-F2p | clean | -0.359 [-0.427, -0.285] |
| B-F2m | clean | -0.054 [-0.134, +0.021] |
| B-A | clean | -0.262 [-0.328, -0.194] |
| B-blurB | clean | +0.101 [+0.063, +0.138] |
| B-Bu | clean | -0.024 [-0.057, +0.009] |
| B-texture | jpeg75_resize05 | -0.124 [-0.258, +0.024] |
| B-pixL2 | jpeg75_resize05 | +0.120 [+0.090, +0.150] |
| B-F2p | jpeg75_resize05 | -0.168 [-0.221, -0.112] |
| B-F2m | jpeg75_resize05 | -0.129 [-0.182, -0.074] |
| B-A | jpeg75_resize05 | +0.063 [+0.017, +0.107] |
| B-blurB | jpeg75_resize05 | -0.131 [-0.170, -0.093] |
| B-Bu | jpeg75_resize05 | +0.000 [+0.000, +0.000] |
| B-texture | social_sim | -0.062 [-0.196, +0.089] |
| B-pixL2 | social_sim | +0.141 [+0.109, +0.175] |
| B-F2p | social_sim | -0.220 [-0.274, -0.159] |
| B-F2m | social_sim | -0.106 [-0.171, -0.039] |
| B-A | social_sim | -0.036 [-0.096, +0.031] |
| B-blurB | social_sim | +0.067 [+0.020, +0.118] |
| B-Bu | social_sim | +0.001 [-0.021, +0.023] |
| A-blurA | clean | -0.042 [-0.084, -0.001] |
| A-blurA | jpeg75_resize05 | -0.198 [-0.254, -0.136] |
| A-blurA | social_sim | -0.113 [-0.160, -0.064] |

Blur sigma (chosen on train reals to match the VAE's mean shift):

| cond | VAE mean shift A | sigma_A | status | VAE mean shift B | sigma_B | status |
|---|---|---|---|---|---|---|
| clean | 10.160 | 2.476 | ok | 1.070 | 0.867 | ok |
| jpeg75_resize05 | 3.903 | 2.120 | ok | 0.721 | 1.381 | ok |
| social_sim | 5.874 | 2.162 | ok | 0.739 | 0.966 | ok |

### Gates

- Gate 1 saturation (clean, test natives, w primary): median w.Delta1 = 9.478, median w.Delta2 = -0.276, ratio -0.029 (need <= 0.5): **PASS**
- Gate 2 content cancellation, Spearman(w.phi(t), w.phi(VAE t)) on test natives (need >= 0.8; B is the gated score): clean: A 0.688, B 0.671; jpeg75_resize05: A 0.829, B 0.785; social_sim: A 0.777, B 0.862 -> **FAIL**
- Gate 3 orientation fixed on train pairs: fraction of train reals with positive shift A 1.000 (clean), B 0.970 (laundered); A2 in-sample AUROC Delta1 vs Delta2 1.000 -> **PASS**
- Gate 5 validity (Amendments 6/8): B clean 0.533 [0.456, 0.613] vs texture clean 0.606 -> **FAIL**
- Gate 4 D-008: see the D-008 section.

### Pass/fail criteria

- not evaluated (a validity gate failed -> UNINFORMATIVE)
- B laundered 0.496 [0.431, 0.569]; pooled F2 laundered 0.665; non-SD laundered 0.531; max FPR (B, every cond/pool) 0.022
- Matched-blur: B - blurB at laundered = -0.131 [-0.170, -0.093] -> blur NOT beaten
- Base label UNINFORMATIVE; **computed label: UNINFORMATIVE; NOT DIFFUSION-SPECIFIC**

### Checks
```
{
 "texture_sign": {
  "clean": {
   "sign": 1,
   "auroc_plus_sign_calib_natives_vs_train_fakes": 0.5609173126614987
  },
  "jpeg75_resize05": {
   "sign": 1,
   "auroc_plus_sign_calib_natives_vs_train_fakes": 0.5737080103359173
  },
  "social_sim": {
   "sign": 1,
   "auroc_plus_sign_calib_natives_vs_train_fakes": 0.5556201550387597
  }
 },
 "phi_t_vs_004_cache_max_abs_diff": {
  "clean": 0.00013256072998046875,
  "jpeg75_resize05": 6.401538848876953e-05,
  "social_sim": 1.6689300537109375e-05
 }
}
```


## Arm `orig`: UNINFORMATIVE; NOT DIFFUSION-SPECIFIC

297 train reals, 43 calib natives, 1074 test reals (358 natives, 25 devices), 720 fakes.

### Macro AUROC [95% CI]

| score | clean | jpeg75_resize05 | social_sim |
|---|---|---|---|
| A: w_p.[phi(VAE t)-phi(t)] | 0.851 [0.806, 0.897] | 0.521 [0.463, 0.579] | 0.684 [0.625, 0.739] |
| B: w_m.[phi(L0 VAE t)-phi(L0 t)] (LABEL) | 0.704 [0.663, 0.747] | 0.484 [0.432, 0.543] | 0.588 [0.536, 0.642] |
| Bu: B, uniform L0 | 0.671 [0.626, 0.718] | 0.484 [0.432, 0.543] | 0.587 [0.534, 0.641] |
| A2: diff-vector probe | 0.796 [0.752, 0.843] | 0.548 [0.499, 0.599] | 0.648 [0.608, 0.690] |
| A2L: A2 with L0 | 0.732 [0.686, 0.777] | 0.497 [0.446, 0.550] | 0.605 [0.546, 0.660] |
| pooled F2 (E4 primary probe on phi(t)) | 0.776 [0.715, 0.837] | 0.643 [0.578, 0.707] | 0.783 [0.729, 0.830] |
| F2 matched probe on phi(t) | 0.324 [0.278, 0.374] | 0.494 [0.438, 0.553] | 0.437 [0.391, 0.484] |
| Laplacian texture | 0.772 [0.718, 0.820] | 0.689 [0.609, 0.759] | 0.732 [0.668, 0.787] |
| pixel L2 round-trip error | 0.324 [0.248, 0.409] | 0.313 [0.235, 0.399] | 0.294 [0.220, 0.378] |
| A with blur for VAE | 0.817 [0.769, 0.860] | 0.726 [0.680, 0.770] | 0.809 [0.771, 0.844] |
| B with blur for VAE | 0.337 [0.266, 0.415] | 0.531 [0.464, 0.606] | 0.439 [0.376, 0.506] |

Flipped-orientation macro (1 - AUROC; descriptive only): texture: 0.228/0.311/0.268; pixL2: 0.676/0.687/0.706; A: 0.149/0.479/0.316; B: 0.296/0.516/0.412

### SD-family (SD2, SDXL) vs other (DALL-E 2/3, Firefly, MJ5)

| score | cond | SD-family | other |
|---|---|---|---|
| A | clean | 0.903 [0.863, 0.941] | 0.824 [0.775, 0.875] |
| A | jpeg75_resize05 | 0.418 [0.358, 0.481] | 0.573 [0.513, 0.634] |
| A | social_sim | 0.688 [0.624, 0.751] | 0.683 [0.623, 0.738] |
| B | clean | 0.867 [0.830, 0.904] | 0.623 [0.572, 0.673] |
| B | jpeg75_resize05 | 0.427 [0.369, 0.489] | 0.512 [0.458, 0.574] |
| B | social_sim | 0.619 [0.559, 0.677] | 0.573 [0.520, 0.628] |
| Bu | clean | 0.765 [0.717, 0.814] | 0.623 [0.576, 0.673] |
| Bu | jpeg75_resize05 | 0.427 [0.369, 0.489] | 0.512 [0.458, 0.574] |
| Bu | social_sim | 0.627 [0.570, 0.683] | 0.567 [0.512, 0.623] |
| A2 | clean | 0.890 [0.857, 0.923] | 0.749 [0.697, 0.805] |
| A2 | jpeg75_resize05 | 0.490 [0.432, 0.549] | 0.577 [0.525, 0.631] |
| A2 | social_sim | 0.745 [0.698, 0.787] | 0.600 [0.555, 0.644] |
| A2L | clean | 0.906 [0.873, 0.937] | 0.645 [0.589, 0.700] |
| A2L | jpeg75_resize05 | 0.406 [0.351, 0.462] | 0.543 [0.489, 0.598] |
| A2L | social_sim | 0.659 [0.594, 0.719] | 0.578 [0.518, 0.634] |
| F2p | clean | 0.874 [0.825, 0.919] | 0.727 [0.657, 0.797] |
| F2p | jpeg75_resize05 | 0.697 [0.630, 0.762] | 0.616 [0.546, 0.685] |
| F2p | social_sim | 0.867 [0.822, 0.905] | 0.741 [0.681, 0.795] |
| texture | clean | 0.829 [0.770, 0.877] | 0.744 [0.688, 0.795] |
| texture | jpeg75_resize05 | 0.724 [0.644, 0.794] | 0.671 [0.590, 0.742] |
| texture | social_sim | 0.782 [0.711, 0.839] | 0.707 [0.644, 0.765] |
| pixL2 | clean | 0.327 [0.239, 0.426] | 0.322 [0.250, 0.401] |
| pixL2 | jpeg75_resize05 | 0.287 [0.211, 0.374] | 0.325 [0.247, 0.414] |
| pixL2 | social_sim | 0.269 [0.193, 0.357] | 0.306 [0.232, 0.389] |
| blurB | clean | 0.364 [0.284, 0.449] | 0.323 [0.256, 0.399] |
| blurB | jpeg75_resize05 | 0.541 [0.472, 0.620] | 0.527 [0.456, 0.604] |
| blurB | social_sim | 0.464 [0.396, 0.540] | 0.426 [0.362, 0.492] |

### Per generator (AUROC, test natives vs that generator)

| score | cond | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|---|
| A | clean | 0.680 | 0.778 | 0.878 | 0.961 | 0.968 | 0.838 |
| A | jpeg75_resize05 | 0.608 | 0.493 | 0.654 | 0.537 | 0.327 | 0.510 |
| A | social_sim | 0.554 | 0.693 | 0.722 | 0.761 | 0.692 | 0.683 |
| B | clean | 0.459 | 0.514 | 0.564 | 0.954 | 0.930 | 0.804 |
| B | jpeg75_resize05 | 0.462 | 0.461 | 0.597 | 0.529 | 0.313 | 0.542 |
| B | social_sim | 0.451 | 0.514 | 0.638 | 0.690 | 0.529 | 0.708 |
| F2p | clean | 0.554 | 0.764 | 0.846 | 0.743 | 0.875 | 0.873 |
| F2p | jpeg75_resize05 | 0.610 | 0.624 | 0.633 | 0.596 | 0.665 | 0.728 |
| F2p | social_sim | 0.712 | 0.743 | 0.760 | 0.751 | 0.862 | 0.871 |
| blurB | clean | 0.466 | 0.184 | 0.356 | 0.287 | 0.272 | 0.456 |
| blurB | jpeg75_resize05 | 0.566 | 0.501 | 0.561 | 0.480 | 0.488 | 0.593 |
| blurB | social_sim | 0.536 | 0.288 | 0.519 | 0.361 | 0.376 | 0.552 |
| texture | clean | 0.730 | 0.881 | 0.560 | 0.804 | 0.877 | 0.781 |
| texture | jpeg75_resize05 | 0.615 | 0.791 | 0.589 | 0.687 | 0.784 | 0.665 |
| texture | social_sim | 0.644 | 0.854 | 0.574 | 0.757 | 0.840 | 0.724 |
| pixL2 | clean | 0.296 | 0.163 | 0.400 | 0.429 | 0.362 | 0.292 |
| pixL2 | jpeg75_resize05 | 0.347 | 0.236 | 0.398 | 0.321 | 0.227 | 0.347 |
| pixL2 | social_sim | 0.329 | 0.201 | 0.394 | 0.300 | 0.206 | 0.332 |

### Per real pool (macro AUROC of that pool's test reals vs fakes)

| score | cond | native | WhatsApp | Facebook-HQ |
|---|---|---|---|---|
| A | clean | 0.851 [0.806, 0.897] | 0.871 [0.843, 0.900] | 0.776 [0.735, 0.816] |
| A | jpeg75_resize05 | 0.521 [0.463, 0.579] | 0.585 [0.533, 0.638] | 0.567 [0.511, 0.622] |
| A | social_sim | 0.684 [0.625, 0.739] | 0.703 [0.658, 0.749] | 0.580 [0.532, 0.625] |
| B | clean | 0.704 [0.663, 0.747] | 0.757 [0.715, 0.800] | 0.703 [0.660, 0.747] |
| B | jpeg75_resize05 | 0.484 [0.432, 0.543] | 0.526 [0.470, 0.585] | 0.542 [0.488, 0.597] |
| B | social_sim | 0.588 [0.536, 0.642] | 0.612 [0.558, 0.670] | 0.559 [0.502, 0.615] |
| F2p | clean | 0.776 [0.715, 0.837] | 0.751 [0.697, 0.800] | 0.644 [0.596, 0.692] |
| F2p | jpeg75_resize05 | 0.643 [0.578, 0.707] | 0.688 [0.635, 0.739] | 0.668 [0.611, 0.725] |
| F2p | social_sim | 0.783 [0.729, 0.830] | 0.775 [0.723, 0.821] | 0.713 [0.663, 0.761] |
| blurB | clean | 0.337 [0.266, 0.415] | - | - |
| blurB | jpeg75_resize05 | 0.531 [0.464, 0.606] | - | - |
| blurB | social_sim | 0.439 [0.376, 0.506] | - | - |
| texture | clean | 0.772 [0.718, 0.820] | 0.570 [0.498, 0.636] | 0.758 [0.708, 0.802] |
| texture | jpeg75_resize05 | 0.689 [0.609, 0.759] | 0.579 [0.504, 0.648] | 0.634 [0.562, 0.700] |
| texture | social_sim | 0.732 [0.668, 0.787] | 0.577 [0.504, 0.643] | 0.690 [0.633, 0.743] |

### FPR at the 1% threshold (primary: device cross-fit; test natives / WA / FBH of the other half's devices; pooled over both folds)

| score | cond | FPR native | FPR WA | FPR FBH | TPR (fakes) | FPR native/WA/FBH with calib-split threshold | TPR (calib thr) |
|---|---|---|---|---|---|---|---|
| A | clean | 0.006 | 0.000 | 0.003 | 0.094 | 0.095 / 0.047 / 0.170 | 0.596 |
| A | jpeg75_resize05 | 0.008 | 0.022 | 0.003 | 0.008 | 0.003 / 0.003 / 0.000 | 0.003 |
| A | social_sim | 0.014 | 0.000 | 0.003 | 0.024 | 0.020 / 0.003 / 0.014 | 0.039 |
| B | clean | 0.014 | 0.003 | 0.003 | 0.127 | 0.025 / 0.020 / 0.028 | 0.247 |
| B | jpeg75_resize05 | 0.008 | 0.003 | 0.000 | 0.003 | 0.014 / 0.003 / 0.003 | 0.008 |
| B | social_sim | 0.011 | 0.006 | 0.003 | 0.002 | 0.022 / 0.011 / 0.006 | 0.010 |
| Bu | clean | 0.022 | 0.003 | 0.014 | 0.035 | 0.084 / 0.022 / 0.056 | 0.171 |
| Bu | jpeg75_resize05 | 0.008 | 0.003 | 0.000 | 0.003 | 0.014 / 0.003 / 0.003 | 0.008 |
| Bu | social_sim | 0.014 | 0.006 | 0.008 | 0.006 | 0.028 / 0.014 / 0.014 | 0.018 |
| A2 | clean | 0.014 | 0.011 | 0.014 | 0.358 | 0.028 / 0.017 / 0.025 | 0.379 |
| A2 | jpeg75_resize05 | 0.017 | 0.020 | 0.014 | 0.016 | 0.006 / 0.003 / 0.003 | 0.007 |
| A2 | social_sim | 0.011 | 0.008 | 0.020 | 0.123 | 0.014 / 0.017 / 0.025 | 0.144 |
| A2L | clean | 0.011 | 0.000 | 0.000 | 0.097 | 0.042 / 0.014 / 0.036 | 0.375 |
| A2L | jpeg75_resize05 | 0.008 | 0.006 | 0.000 | 0.000 | 0.008 / 0.006 / 0.003 | 0.000 |
| A2L | social_sim | 0.008 | 0.000 | 0.000 | 0.000 | 0.014 / 0.003 / 0.000 | 0.003 |
| F2p | clean | 0.006 | 0.006 | 0.008 | 0.060 | 0.022 / 0.036 / 0.042 | 0.179 |
| F2p | jpeg75_resize05 | 0.028 | 0.011 | 0.028 | 0.022 | 0.025 / 0.008 / 0.022 | 0.029 |
| F2p | social_sim | 0.028 | 0.036 | 0.045 | 0.178 | 0.031 / 0.053 / 0.050 | 0.235 |
| texture | clean | 0.011 | 0.140 | 0.031 | 0.066 | 0.003 / 0.098 / 0.008 | 0.029 |
| texture | jpeg75_resize05 | 0.011 | 0.017 | 0.011 | 0.006 | 0.003 / 0.003 / 0.000 | 0.001 |
| texture | social_sim | 0.008 | 0.081 | 0.028 | 0.024 | 0.003 / 0.028 / 0.003 | 0.004 |
| pixL2 | clean | 0.025 | 0.006 | 0.045 | 0.005 | 0.025 / 0.011 / 0.047 | 0.007 |
| pixL2 | jpeg75_resize05 | 0.011 | 0.011 | 0.008 | 0.001 | 0.006 / 0.006 / 0.006 | 0.001 |
| pixL2 | social_sim | 0.022 | 0.017 | 0.031 | 0.003 | 0.008 / 0.014 / 0.020 | 0.003 |

### Matched-blur control and other paired differences (macro AUROC, paired bootstrap)

| contrast | cond | delta [95% CI] |
|---|---|---|
| B-texture | clean | -0.068 [-0.146, +0.022] |
| B-pixL2 | clean | +0.381 [+0.315, +0.443] |
| B-F2p | clean | -0.072 [-0.121, -0.018] |
| B-F2m | clean | +0.381 [+0.326, +0.438] |
| B-A | clean | -0.146 [-0.184, -0.111] |
| B-blurB | clean | +0.368 [+0.306, +0.434] |
| B-Bu | clean | +0.034 [+0.005, +0.060] |
| B-texture | jpeg75_resize05 | -0.205 [-0.318, -0.077] |
| B-pixL2 | jpeg75_resize05 | +0.171 [+0.113, +0.229] |
| B-F2p | jpeg75_resize05 | -0.159 [-0.210, -0.105] |
| B-F2m | jpeg75_resize05 | -0.010 [-0.060, +0.041] |
| B-A | jpeg75_resize05 | -0.037 [-0.077, +0.004] |
| B-blurB | jpeg75_resize05 | -0.047 [-0.112, +0.014] |
| B-Bu | jpeg75_resize05 | +0.000 [+0.000, +0.000] |
| B-texture | social_sim | -0.144 [-0.242, -0.032] |
| B-pixL2 | social_sim | +0.295 [+0.244, +0.342] |
| B-F2p | social_sim | -0.195 [-0.252, -0.135] |
| B-F2m | social_sim | +0.152 [+0.086, +0.214] |
| B-A | social_sim | -0.096 [-0.142, -0.045] |
| B-blurB | social_sim | +0.150 [+0.096, +0.200] |
| B-Bu | social_sim | +0.001 [-0.021, +0.025] |
| A-blurA | clean | +0.034 [-0.014, +0.077] |
| A-blurA | jpeg75_resize05 | -0.204 [-0.259, -0.151] |
| A-blurA | social_sim | -0.125 [-0.178, -0.075] |

Blur sigma (chosen on train reals to match the VAE's mean shift):

| cond | VAE mean shift A | sigma_A | status | VAE mean shift B | sigma_B | status |
|---|---|---|---|---|---|---|
| clean | 4.255 | 1.520 | ok | 2.586 | 1.149 | ok |
| jpeg75_resize05 | 3.448 | 1.580 | ok | 1.938 | 1.259 | ok |
| social_sim | 3.469 | 1.499 | ok | 1.882 | 1.114 | ok |

### Gates

- Gate 1 saturation (clean, test natives, w primary): median w.Delta1 = 3.889, median w.Delta2 = -0.547, ratio -0.141 (need <= 0.5): **PASS**
- Gate 2 content cancellation, Spearman(w.phi(t), w.phi(VAE t)) on test natives (need >= 0.8; B is the gated score): clean: A 0.759, B 0.936; jpeg75_resize05: A 0.783, B 0.871; social_sim: A 0.831, B 0.940 -> **PASS**
- Gate 3 orientation fixed on train pairs: fraction of train reals with positive shift A 0.997 (clean), B 0.980 (laundered); A2 in-sample AUROC Delta1 vs Delta2 1.000 -> **PASS**
- Gate 5 validity (Amendments 6/8): B clean 0.704 [0.663, 0.747] vs texture clean 0.772 -> **FAIL**
- Gate 4 D-008: see the D-008 section.

### Pass/fail criteria

- not evaluated (a validity gate failed -> UNINFORMATIVE)
- B laundered 0.484 [0.432, 0.543]; pooled F2 laundered 0.643; non-SD laundered 0.512; max FPR (B, every cond/pool) 0.014
- Matched-blur: B - blurB at laundered = -0.047 [-0.112, +0.014] -> blur NOT beaten
- Base label UNINFORMATIVE; **computed label: UNINFORMATIVE; NOT DIFFUSION-SPECIFIC**

### Checks
```
{
 "texture_sign": {
  "clean": {
   "sign": 1,
   "auroc_plus_sign_calib_natives_vs_train_fakes": 0.7602713178294573
  },
  "jpeg75_resize05": {
   "sign": 1,
   "auroc_plus_sign_calib_natives_vs_train_fakes": 0.664405684754522
  },
  "social_sim": {
   "sign": 1,
   "auroc_plus_sign_calib_natives_vs_train_fakes": 0.7245478036175711
  }
 },
 "phi_t_vs_004_cache_max_abs_diff": {
  "clean": 4.673004150390625e-05,
  "jpeg75_resize05": 0.000102996826171875,
  "social_sim": 2.288818359375e-05
 }
}
```


## B0 LOGO reference (005 pilot)
```
{
 "clean": {
  "auroc": 0.9917287399130975,
  "ap": 0.9812745670723017,
  "tpr_at_1fpr": 0.861111111111111
 },
 "jpeg75_resize05": {
  "auroc": 0.985249844816884,
  "ap": 0.9648236220585923,
  "tpr_at_1fpr": 0.7777777777777778
 },
 "social_sim": {
  "auroc": 0.991286468032278,
  "ap": 0.9803254967899114,
  "tpr_at_1fpr": 0.8597222222222222
 },
 "source": "experiments/005-nuisance-projection/pilot/results.json arms.A0[cond].macro (LOGO-macro, 005 pilot splits)"
}
```


## Refit of E4's probes vs 011/results.json
```
{
 "fov512": {
  "primary/clean": {
   "refit": 0.8921205772811919,
   "e4_results_json": 0.8921205772811919,
   "abs_diff": 0.0
  },
  "primary/jpeg75_resize05": {
   "refit": 0.6648898199875853,
   "e4_results_json": 0.6648898199875853,
   "abs_diff": 0.0
  },
  "matched/jpeg75_resize05": {
   "refit": 0.6255237430167598,
   "e4_results_json": 0.6255237430167598,
   "abs_diff": 0.0
  }
 },
 "orig": {
  "primary/clean": {
   "refit": 0.7760280881440099,
   "e4_results_json": 0.7760280881440099,
   "abs_diff": 0.0
  },
  "primary/jpeg75_resize05": {
   "refit": 0.6425977653631284,
   "e4_results_json": 0.6425977653631284,
   "abs_diff": 0.0
  },
  "matched/jpeg75_resize05": {
   "refit": 0.4941030415890751,
   "e4_results_json": 0.4941030415890751,
   "abs_diff": 0.0
  }
 }
}
```


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
   "fov512/clean": {
    "A": 3.3697084443673475e-05,
    "B": 4.240210231709618e-06,
    "Bu": 4.3366329838256945e-06,
    "A2": 8.384466810174018e-06,
    "A2L": 2.2693468979539944e-05,
    "F2p": 4.379527700137942e-05,
    "F2m": 5.874839341446503e-06,
    "blurA": 6.0224339755166056e-05,
    "blurB": 4.504912013314488e-06,
    "max_scaled_feature_diff": 3.2472980819875374e-05,
    "n_images": 32,
    "n_blur_rows": 32,
    "max_abs_dscore": 6.0224339755166056e-05,
    "ok": true
   },
   "fov512/jpeg75_resize05": {
    "A": 2.8093971619291835e-05,
    "B": 3.9035640684303985e-06,
    "Bu": 3.9035640684303985e-06,
    "A2": 7.380988582494918e-06,
    "A2L": 2.837905678498842e-05,
    "F2p": 3.461484768152445e-05,
    "F2m": 4.193169163668742e-06,
    "blurA": 5.165881811119988e-05,
    "blurB": 5.478664692359914e-06,
    "max_scaled_feature_diff": 2.1887095499550924e-05,
    "n_images": 32,
    "n_blur_rows": 32,
    "max_abs_dscore": 5.165881811119988e-05,
    "ok": true
   },
   "fov512/social_sim": {
    "A": 2.2056649102708548e-05,
    "B": 6.0973674180786475e-06,
    "Bu": 5.607109892613593e-06,
    "A2": 5.525569652942153e-06,
    "A2L": 3.331090470481968e-05,
    "F2p": 3.768562034878187e-05,
    "F2m": 6.382842026475544e-06,
    "blurA": 4.242289314504433e-05,
    "blurB": 4.946532180527541e-06,
    "max_scaled_feature_diff": 1.6101515939226374e-05,
    "n_images": 32,
    "n_blur_rows": 32,
    "max_abs_dscore": 4.242289314504433e-05,
    "ok": true
   },
   "fov512/vv": {
    "n_images": 16,
    "gate1_delta2": 3.320355073732628e-05,
    "a2_delta2": 6.545095621257602e-06,
    "a2l_delta2": 3.2567256140758616e-05,
    "max_abs_dscore": 3.320355073732628e-05,
    "ok": true
   },
   "orig/clean": {
    "A": 1.9396073291932936e-05,
    "B": 2.8086494886281343e-05,
    "Bu": 3.193897788866806e-05,
    "A2": 3.304889045141479e-05,
    "A2L": 7.279946053762032e-05,
    "F2p": 2.140849653464727e-05,
    "F2m": 2.9887924313243275e-05,
    "blurA": 2.768130170194283e-05,
    "blurB": 1.2627519232921713e-05,
    "max_scaled_feature_diff": 1.8994756828760728e-05,
    "n_images": 32,
    "n_blur_rows": 32,
    "max_abs_dscore": 7.279946053762032e-05,
    "ok": true
   },
   "orig/jpeg75_resize05": {
    "A": 1.8542973030388765e-05,
    "B": 4.170446506090997e-05,
    "Bu": 4.170446506090997e-05,
    "A2": 2.116149200670492e-05,
    "A2L": 0.00011261336341839012,
    "F2p": 2.0355797238025275e-05,
    "F2m": 2.53041654638686e-05,
    "blurA": 2.2743024992166738e-05,
    "blurB": 1.6404002324943434e-05,
    "max_scaled_feature_diff": 1.6497097021783702e-05,
    "n_images": 32,
    "n_blur_rows": 32,
    "max_abs_dscore": 0.00011261336341839012,
    "ok": true
   },
   "orig/social_sim": {
    "A": 2.151655436866129e-05,
    "B": 3.224937646706394e-05,
    "Bu": 3.224937646706394e-05,
    "A2": 2.7109735487140796e-05,
    "A2L": 7.916673462382562e-05,
    "F2p": 2.2833776800901262e-05,
    "F2m": 3.1343781632831025e-05,
    "blurA": 2.3391637876457594e-05,
    "blurB": 1.5616023369080523e-05,
    "max_scaled_feature_diff": 2.126749859598931e-05,
    "n_images": 32,
    "n_blur_rows": 32,
    "max_abs_dscore": 7.916673462382562e-05,
    "ok": true
   },
   "orig/vv": {
    "n_images": 16,
    "gate1_delta2": 2.47934713972775e-05,
    "a2_delta2": 5.039466105216661e-05,
    "a2l_delta2": 6.597779488259903e-05,
    "max_abs_dscore": 6.597779488259903e-05,
    "ok": true
   }
  },
  "ok": true
 }
}
```


## Stage times (s)
```
{
 "feat": 3711.1056039333344,
 "vv": 932.8331520557404,
 "probes": 4.769941806793213,
 "blur": 483.59620547294617,
 "guard": 259.6088182926178
}
```
