# 023-attr-detector-emb: H-A5 tables (all numbers from results/results.json)

**H-A5 label: KILL**. Best detector embedding: commfor (bacc 0.740 at jpeg75_resize05, T-mix). H-A1 best fixed layer: cat (bacc 0.858); H-A1 nested-choice bacc 0.855. Rule: PASS if >= best layer + 0.05 and paired CI lower bound > 0; KILL if <= best layer; else INCONCLUSIVE. Gates all ok: True.

n = 1020 fakes, 542 caption families (cluster bootstrap, 1000 reps, 95% percentile CI). Chance bacc = 0.125, pairwise/AUROC chance = 0.5.

## Balanced accuracy, 8-way, T-mix training (headline)

| embedding | clean | jpeg75_resize05 | social_sim | screenshot_sim | mavali_medium |
|---|---|---|---|---|---|
| b4 | 0.893 [0.873, 0.910] | 0.841 [0.817, 0.863] | 0.847 [0.824, 0.867] | 0.817 [0.792, 0.839] | 0.782 [0.757, 0.806] |
| b8 | 0.920 [0.902, 0.937] | 0.816 [0.793, 0.839] | 0.874 [0.854, 0.893] | 0.866 [0.844, 0.886] | 0.817 [0.793, 0.841] |
| b12 | 0.594 [0.564, 0.627] | 0.562 [0.531, 0.590] | 0.577 [0.546, 0.609] | 0.581 [0.553, 0.613] | 0.548 [0.516, 0.577] |
| cat | 0.921 [0.903, 0.937] | 0.858 [0.838, 0.878] | 0.880 [0.861, 0.900] | 0.876 [0.856, 0.896] | 0.825 [0.803, 0.847] |
| nested_dino | 0.913 [0.894, 0.930] | 0.855 [0.834, 0.876] | 0.872 [0.851, 0.894] | 0.866 [0.844, 0.887] | 0.813 [0.790, 0.836] |
| commfor | 0.877 [0.857, 0.897] | 0.740 [0.713, 0.766] | 0.833 [0.810, 0.854] | 0.814 [0.789, 0.838] | 0.735 [0.709, 0.759] |
| bfree | 0.856 [0.833, 0.877] | 0.660 [0.630, 0.688] | 0.790 [0.764, 0.814] | 0.758 [0.731, 0.782] | 0.723 [0.697, 0.747] |
| rine | 0.246 [0.227, 0.266] | 0.157 [0.136, 0.176] | 0.152 [0.131, 0.173] | 0.175 [0.155, 0.196] | 0.145 [0.124, 0.167] |
| clipdet | 0.746 [0.721, 0.773] | 0.709 [0.681, 0.738] | 0.698 [0.670, 0.726] | 0.531 [0.504, 0.562] | 0.590 [0.559, 0.619] |
| chance / majority rate | 0.125 / 0.147 | 0.125 / 0.147 | 0.125 / 0.147 | 0.125 / 0.147 | 0.125 / 0.147 |

## Balanced accuracy, 8-way, clean-train transfer (descriptive)

| embedding | clean | jpeg75_resize05 | social_sim | screenshot_sim | mavali_medium |
|---|---|---|---|---|---|
| b4 | 0.915 [0.896, 0.931] | 0.431 [0.413, 0.451] | 0.557 [0.530, 0.582] | 0.807 [0.782, 0.831] | 0.404 [0.385, 0.423] |
| b8 | 0.936 [0.921, 0.951] | 0.632 [0.609, 0.655] | 0.775 [0.751, 0.798] | 0.835 [0.811, 0.858] | 0.663 [0.640, 0.689] |
| b12 | 0.626 [0.595, 0.659] | 0.505 [0.476, 0.534] | 0.558 [0.528, 0.589] | 0.586 [0.556, 0.616] | 0.517 [0.485, 0.548] |
| cat | 0.942 [0.927, 0.955] | 0.542 [0.521, 0.564] | 0.702 [0.677, 0.728] | 0.877 [0.856, 0.896] | 0.507 [0.484, 0.530] |
| commfor | 0.891 [0.870, 0.909] | 0.468 [0.445, 0.489] | 0.731 [0.704, 0.755] | 0.799 [0.773, 0.824] | 0.524 [0.498, 0.546] |
| bfree | 0.869 [0.845, 0.889] | 0.422 [0.397, 0.447] | 0.682 [0.653, 0.706] | 0.672 [0.645, 0.700] | 0.505 [0.479, 0.529] |
| rine | 0.265 [0.240, 0.288] | 0.158 [0.139, 0.179] | 0.147 [0.129, 0.167] | 0.169 [0.146, 0.192] | 0.137 [0.118, 0.157] |
| clipdet | 0.754 [0.729, 0.782] | 0.394 [0.367, 0.421] | 0.541 [0.512, 0.571] | 0.512 [0.484, 0.540] | 0.425 [0.397, 0.452] |
| chance / majority rate | 0.125 / 0.147 | 0.125 / 0.147 | 0.125 / 0.147 | 0.125 / 0.147 | 0.125 / 0.147 |

## Secondary metrics at jpeg75_resize05 (T-mix)

| embedding | macro-F1 | bacc w/o firefly | pairwise 2-AFC macro | pairwise pooled | pairwise strict macro | n pairs |
|---|---|---|---|---|---|---|
| b4 | 0.842 [0.818, 0.864] | 0.849 [0.825, 0.872] | 0.994 [0.984, 1.000] | 0.997 [0.992, 1.000] | 0.986 [0.972, 0.996] | 695 |
| b8 | 0.815 [0.792, 0.839] | 0.816 [0.792, 0.840] | 0.996 [0.988, 1.000] | 0.999 [0.995, 1.000] | 0.976 [0.960, 0.991] | 695 |
| b12 | 0.564 [0.533, 0.591] | 0.569 [0.537, 0.599] | 0.955 [0.937, 0.973] | 0.957 [0.941, 0.971] | 0.810 [0.774, 0.846] | 695 |
| cat | 0.858 [0.837, 0.878] | 0.858 [0.836, 0.880] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.990 [0.979, 0.997] | 695 |
| nested_dino | 0.856 [0.833, 0.876] | 0.855 [0.833, 0.878] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.988 [0.975, 0.996] | 695 |
| commfor | 0.741 [0.714, 0.766] | 0.753 [0.725, 0.779] | 0.985 [0.970, 0.997] | 0.988 [0.978, 0.997] | 0.929 [0.900, 0.957] | 695 |
| bfree | 0.660 [0.628, 0.688] | 0.663 [0.631, 0.694] | 0.971 [0.952, 0.987] | 0.980 [0.969, 0.989] | 0.890 [0.858, 0.917] | 695 |
| rine | 0.136 [0.115, 0.156] | 0.164 [0.142, 0.185] | 0.603 [0.546, 0.653] | 0.594 [0.545, 0.638] | 0.354 [0.308, 0.398] | 695 |
| clipdet | 0.709 [0.681, 0.735] | 0.716 [0.688, 0.745] | 0.990 [0.978, 0.998] | 0.993 [0.985, 0.998] | 0.927 [0.902, 0.951] | 695 |

Chance for pairwise = 0.5 (2-AFC and strict reading: strict chance 0.25 for independent coin flips).

## Paired comparison with H-A1 best fixed layer (cat, bacc 0.858) at jpeg75_resize05, T-mix

| detector | bacc | diff = detector - H-A1 best layer [95% paired CI] | one-sided p | Holm p |
|---|---|---|---|---|
| commfor | 0.740 | -0.118 [-0.149, -0.087] | 1.0000 | 1.0000 |
| bfree | 0.660 | -0.198 [-0.231, -0.167] | 1.0000 | 1.0000 |
| rine | 0.157 | -0.701 [-0.730, -0.671] | 1.0000 | 1.0000 |
| clipdet | 0.709 | -0.149 [-0.179, -0.120] | 1.0000 | 1.0000 |


### Same diff for every condition and every DINOv2 layer (descriptive, no label)

| detector - layer | clean | jpeg75_resize05 | social_sim | screenshot_sim | mavali_medium |
|---|---|---|---|---|---|
| commfor - b4 | -0.016 [-0.040, 0.008] | -0.100 [-0.132, -0.070] | -0.014 [-0.043, 0.014] | -0.003 [-0.034, 0.028] | -0.047 [-0.079, -0.015] |
| commfor - b8 | -0.043 [-0.066, -0.020] | -0.076 [-0.107, -0.045] | -0.041 [-0.068, -0.015] | -0.052 [-0.082, -0.024] | -0.081 [-0.112, -0.051] |
| commfor - b12 | 0.283 [0.247, 0.315] | 0.179 [0.143, 0.214] | 0.256 [0.221, 0.292] | 0.233 [0.198, 0.266] | 0.188 [0.149, 0.225] |
| commfor - cat | -0.044 [-0.066, -0.021] | -0.118 [-0.149, -0.087] | -0.047 [-0.074, -0.021] | -0.062 [-0.094, -0.033] | -0.090 [-0.121, -0.061] |
| bfree - b4 | -0.037 [-0.062, -0.012] | -0.181 [-0.217, -0.144] | -0.056 [-0.086, -0.027] | -0.059 [-0.093, -0.028] | -0.059 [-0.091, -0.027] |
| bfree - b8 | -0.064 [-0.089, -0.039] | -0.156 [-0.192, -0.123] | -0.084 [-0.112, -0.055] | -0.109 [-0.139, -0.080] | -0.094 [-0.125, -0.063] |
| bfree - b12 | 0.261 [0.225, 0.295] | 0.098 [0.062, 0.136] | 0.214 [0.178, 0.247] | 0.176 [0.142, 0.212] | 0.175 [0.142, 0.208] |
| bfree - cat | -0.065 [-0.089, -0.042] | -0.198 [-0.231, -0.167] | -0.090 [-0.118, -0.063] | -0.119 [-0.148, -0.090] | -0.102 [-0.133, -0.073] |
| rine - b4 | -0.646 [-0.673, -0.618] | -0.684 [-0.714, -0.651] | -0.695 [-0.724, -0.663] | -0.642 [-0.673, -0.610] | -0.637 [-0.670, -0.603] |
| rine - b8 | -0.674 [-0.699, -0.647] | -0.659 [-0.690, -0.629] | -0.723 [-0.750, -0.694] | -0.691 [-0.719, -0.659] | -0.672 [-0.704, -0.639] |
| rine - b12 | -0.348 [-0.386, -0.311] | -0.405 [-0.441, -0.366] | -0.425 [-0.461, -0.388] | -0.406 [-0.445, -0.370] | -0.403 [-0.436, -0.366] |
| rine - cat | -0.675 [-0.699, -0.649] | -0.701 [-0.730, -0.671] | -0.728 [-0.755, -0.700] | -0.701 [-0.732, -0.671] | -0.680 [-0.709, -0.649] |
| clipdet - b4 | -0.146 [-0.175, -0.117] | -0.131 [-0.162, -0.101] | -0.149 [-0.181, -0.116] | -0.285 [-0.318, -0.252] | -0.192 [-0.228, -0.157] |
| clipdet - b8 | -0.174 [-0.201, -0.146] | -0.107 [-0.136, -0.078] | -0.177 [-0.207, -0.147] | -0.335 [-0.368, -0.303] | -0.227 [-0.261, -0.196] |
| clipdet - b12 | 0.152 [0.118, 0.185] | 0.148 [0.113, 0.182] | 0.121 [0.084, 0.155] | -0.050 [-0.083, -0.015] | 0.042 [0.007, 0.079] |
| clipdet - cat | -0.175 [-0.201, -0.148] | -0.149 [-0.179, -0.120] | -0.182 [-0.213, -0.151] | -0.345 [-0.376, -0.315] | -0.235 [-0.267, -0.205] |

## Per-class recall at jpeg75_resize05 (T-mix), with training-set-overlap flags

| generator | b4 | b8 | b12 | cat | nested_dino | commfor | bfree | rine | clipdet |
|---|---|---|---|---|---|---|---|---|---|
| dalle2 | 0.817 | 0.742 | 0.517 | 0.833 | 0.833 | 0.775 | 0.617 | 0.092 | 0.825 |
| dalle3 | 0.942 | 0.967 | 0.750 | 0.983 | 0.975 | 0.875 | 0.825 | 0.092 | 0.883 |
| firefly | 0.783 | 0.817 | 0.508 | 0.858 | 0.858 | 0.650 | 0.633 | 0.108 | 0.658 |
| midjourney-v5 | 0.717 | 0.642 | 0.400 | 0.675 | 0.667 | 0.517 | 0.508 | 0.075 | 0.442 |
| stable-diffusion-2 | 0.825 | 0.842 | 0.625 | 0.842 | 0.833 | 0.792 | 0.683 | 0.075 | 0.733 |
| stable-diffusion-xl | 0.808 | 0.667 | 0.400 | 0.758 | 0.783 | 0.700 | 0.550 | 0.492 | 0.592 |
| sd3_large | 0.907 | 0.893 | 0.593 | 0.927 | 0.933 | 0.807 | 0.720 | 0.273 | 0.800 |
| flux | 0.927 | 0.960 | 0.700 | 0.987 | 0.960 | 0.807 | 0.740 | 0.047 | 0.740 |

Overlap flags (family-level, from kb/papers; [unverified] where the exact checkpoint lists are not public to us):

| generator | commfor | bfree | rine | clipdet |
|---|---|---|---|---|
| dalle2 | no known overlap | none | none known | none known |
| dalle3 | no known overlap (eval-set generator in paper) | none | none known | none known |
| firefly | no known overlap (eval-set generator in paper) | none | none known | none known |
| midjourney-v5 | no known overlap (eval-set generator in paper) | none | none known | none known |
| stable-diffusion-2 | LIKELY family overlap: 4,763 HF latent-diffusion models (SD-family checkpoints/finetunes) [exact SD2 overlap unverified] | OVERLAP: trained on SD 2.1 inpainting regenerations (same decoder/VAE family; SD2 vs SD2.1 inpainting version [unverified]) | FAMILY overlap (latent diffusion), not the same generator | FAMILY overlap (latent diffusion) |
| stable-diffusion-xl | POSSIBLE family overlap: HF diffusers models incl. SDXL derivatives [unverified] | none (different VAE) | FAMILY overlap (latent diffusion) | FAMILY overlap (latent diffusion) |
| sd3_large | no known overlap [unverified] | none (these sd3/flux images are B-Free's own extended-Synthbuster EVAL release) | none known | none known |
| flux | no known overlap (FLUX is an eval generator in the paper) | none (B-Free eval release) | none known | none known |

Notes: rine: LDM checkpoint: trained on LDM data from Corvi et al. (DMimageDetection); CLIP backbone frozen (OpenAI); clipdet: head trained on 10k real/Latent-Diffusion pairs (COCO captions); backbone CommonPool CLIP

## Open set (leave-one-generator-out), T-mix

### AUROC, novelty = -max softmax, jpeg75_resize05

| embedding | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl | sd3_large | flux | mean | min | usable (mean>=0.80, min>=0.65) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b4 | 0.76 [0.71, 0.80] | 0.73 [0.69, 0.77] | 0.83 [0.80, 0.87] | 0.84 [0.81, 0.87] | 0.73 [0.68, 0.77] | 0.72 [0.68, 0.76] | 0.71 [0.67, 0.75] | 0.75 [0.71, 0.78] | 0.758 | 0.706 | False |
| b8 | 0.74 [0.70, 0.78] | 0.78 [0.74, 0.81] | 0.85 [0.82, 0.88] | 0.80 [0.77, 0.84] | 0.65 [0.60, 0.70] | 0.72 [0.68, 0.76] | 0.78 [0.74, 0.81] | 0.72 [0.68, 0.76] | 0.756 | 0.652 | False |
| b12 | 0.64 [0.59, 0.69] | 0.49 [0.44, 0.54] | 0.67 [0.62, 0.71] | 0.68 [0.63, 0.73] | 0.45 [0.40, 0.51] | 0.58 [0.53, 0.64] | 0.64 [0.59, 0.68] | 0.61 [0.56, 0.65] | 0.595 | 0.455 | False |
| cat | 0.78 [0.74, 0.81] | 0.75 [0.70, 0.78] | 0.85 [0.82, 0.88] | 0.83 [0.80, 0.87] | 0.69 [0.64, 0.73] | 0.72 [0.68, 0.75] | 0.76 [0.73, 0.80] | 0.79 [0.75, 0.82] | 0.770 | 0.686 | False |
| commfor | 0.68 [0.63, 0.72] | 0.61 [0.56, 0.66] | 0.72 [0.68, 0.76] | 0.73 [0.68, 0.77] | 0.57 [0.52, 0.62] | 0.57 [0.51, 0.62] | 0.71 [0.67, 0.76] | 0.66 [0.62, 0.70] | 0.656 | 0.567 | False |
| bfree | 0.64 [0.58, 0.69] | 0.56 [0.51, 0.61] | 0.69 [0.65, 0.73] | 0.72 [0.67, 0.76] | 0.58 [0.52, 0.63] | 0.66 [0.61, 0.70] | 0.68 [0.63, 0.72] | 0.63 [0.59, 0.67] | 0.644 | 0.563 | False |
| rine | 0.49 [0.44, 0.55] | 0.53 [0.47, 0.58] | 0.51 [0.46, 0.57] | 0.52 [0.47, 0.57] | 0.43 [0.38, 0.49] | 0.40 [0.35, 0.46] | 0.52 [0.47, 0.56] | 0.53 [0.48, 0.58] | 0.492 | 0.403 | False |
| clipdet | 0.63 [0.57, 0.68] | 0.58 [0.53, 0.64] | 0.71 [0.67, 0.75] | 0.71 [0.66, 0.75] | 0.56 [0.51, 0.62] | 0.65 [0.60, 0.69] | 0.67 [0.63, 0.71] | 0.62 [0.58, 0.67] | 0.641 | 0.565 | False |

### AUROC, novelty = Mahalanobis (Ledoit-Wolf), jpeg75_resize05

| embedding | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl | sd3_large | flux | mean | min | usable (mean>=0.80, min>=0.65) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b4 | 0.69 [0.65, 0.73] | 0.49 [0.44, 0.53] | 0.46 [0.41, 0.51] | 0.44 [0.39, 0.49] | 0.76 [0.72, 0.80] | 0.44 [0.39, 0.49] | 0.70 [0.67, 0.74] | 0.59 [0.55, 0.63] | 0.571 | 0.436 | False |
| b8 | 0.67 [0.63, 0.71] | 0.53 [0.49, 0.58] | 0.51 [0.46, 0.56] | 0.50 [0.45, 0.55] | 0.80 [0.76, 0.84] | 0.53 [0.48, 0.57] | 0.54 [0.50, 0.58] | 0.56 [0.52, 0.60] | 0.579 | 0.496 | False |
| b12 | 0.59 [0.55, 0.64] | 0.53 [0.49, 0.58] | 0.44 [0.38, 0.49] | 0.50 [0.45, 0.55] | 0.62 [0.57, 0.67] | 0.51 [0.46, 0.56] | 0.58 [0.54, 0.61] | 0.52 [0.47, 0.56] | 0.536 | 0.437 | False |
| cat | 0.65 [0.61, 0.70] | 0.54 [0.49, 0.59] | 0.46 [0.40, 0.51] | 0.48 [0.44, 0.54] | 0.73 [0.68, 0.78] | 0.51 [0.46, 0.57] | 0.61 [0.57, 0.64] | 0.55 [0.50, 0.59] | 0.566 | 0.456 | False |
| commfor | 0.71 [0.66, 0.75] | 0.53 [0.49, 0.58] | 0.45 [0.39, 0.50] | 0.61 [0.57, 0.66] | 0.78 [0.73, 0.81] | 0.69 [0.65, 0.73] | 0.48 [0.43, 0.53] | 0.51 [0.46, 0.56] | 0.595 | 0.447 | False |
| bfree | 0.59 [0.54, 0.64] | 0.43 [0.39, 0.49] | 0.40 [0.34, 0.46] | 0.55 [0.50, 0.60] | 0.56 [0.50, 0.62] | 0.48 [0.42, 0.53] | 0.63 [0.58, 0.67] | 0.63 [0.59, 0.67] | 0.533 | 0.401 | False |
| rine | 0.54 [0.49, 0.59] | 0.45 [0.40, 0.50] | 0.46 [0.40, 0.52] | 0.45 [0.40, 0.50] | 0.62 [0.57, 0.67] | 0.62 [0.57, 0.67] | 0.42 [0.36, 0.46] | 0.49 [0.44, 0.53] | 0.506 | 0.415 | False |
| clipdet | 0.71 [0.67, 0.76] | 0.53 [0.48, 0.57] | 0.44 [0.39, 0.48] | 0.49 [0.44, 0.54] | 0.67 [0.62, 0.73] | 0.53 [0.48, 0.58] | 0.62 [0.58, 0.66] | 0.49 [0.44, 0.53] | 0.560 | 0.439 | False |

## Held-out launderings (never in training), bacc T-mix

| embedding | screenshot_sim | mavali_medium |
|---|---|---|
| b4 | 0.817 [0.792, 0.839] | 0.782 [0.757, 0.806] |
| b8 | 0.866 [0.844, 0.886] | 0.817 [0.793, 0.841] |
| b12 | 0.581 [0.553, 0.613] | 0.548 [0.516, 0.577] |
| cat | 0.876 [0.856, 0.896] | 0.825 [0.803, 0.847] |
| nested_dino | 0.866 [0.844, 0.887] | 0.813 [0.790, 0.836] |
| commfor | 0.814 [0.789, 0.838] | 0.735 [0.709, 0.759] |
| bfree | 0.758 [0.731, 0.782] | 0.723 [0.697, 0.747] |
| rine | 0.175 [0.155, 0.196] | 0.145 [0.124, 0.167] |
| clipdet | 0.531 [0.504, 0.562] | 0.590 [0.559, 0.619] |

## Baselines from 020 (same rows and folds)

M0 raw metadata bacc 0.688 [0.665, 0.710]; M1 texture bacc (T-mix) 0.425 [0.397, 0.452] at jpeg75_resize05. M0 on preprocessed tensors (must be near chance): {"clean": 0.125, "jpeg75_resize05": 0.125, "social_sim": 0.13729166666666667, "screenshot_sim": 0.13645833333333332, "mavali_medium": 0.125}.


## Gates

- OK   sha256_dedupe: {"n_rows": 1020, "n_duplicated_rows": 0, "n_cross_class_hashes": 0}
- OK   grouped_splits: {"problems": [], "folds_sha256": "bc2894a41fea5a1b971fe027b530c5884bf16f7becf5dd97c18ba9151fd433c3", "pairwise_computed": true}
- OK   train_conditions: {"bad": []}
- OK   D-008: {"worst_abs_dlogit": 0.0001577138900756836, "worst_scaled_emb_diff": 5.266451626084745e-05, "all_both_criteria_ok": true, "n_subset": 32}
- OK   permutation_200.commfor: {"n": 200, "mean": 0.125546875, "p95": 0.14544791666666665, "chance": 0.125, "mean_tol": 0.01, "p95_excess": 0.03}
- OK   permutation_200.bfree: {"n": 200, "mean": 0.12510416666666666, "p95": 0.14213541666666665, "chance": 0.125, "mean_tol": 0.01, "p95_excess": 0.03}
- OK   permutation_200.rine: {"n": 200, "mean": 0.12678229166666666, "p95": 0.1442708333333333, "chance": 0.125, "mean_tol": 0.01, "p95_excess": 0.03}
- OK   permutation_200.clipdet: {"n": 200, "mean": 0.12550520833333334, "p95": 0.14421874999999998, "chance": 0.125, "mean_tol": 0.01, "p95_excess": 0.03}
- OK   identical_preprocessing: {"bad_paths": 0, "n_exif_or_icc": 0, "n_not_rgb": 0, "n_not_512": 0, "note": "all 1,020 rows go through load_cond (apply_condition keyed by sha256, seed 0) and then the detector's own transform; labels are never an input"}
- OK   holm_and_inner_cv: {"note": "C chosen by inner grouped CV on train folds only (020 inner_table); detector choice is a fixed list; best-of-4 selection is on test results and is reported with Holm-adjusted bootstrap p"}
- OK   no_test_group_in_train: {"note": "assert_disjoint_groups runs inside every 020 closed/LOGO/permutation task"}
- OK   chance_levels_in_tables: {"note": "every table carries chance 0.125 (bacc) / 0.5 (pairwise, AUROC) and the majority-class rate"}
- OK   baselines_M0_M1: {"note": "M0/M1 are computed once in 020 on the same rows and folds (results/results.json); copied into baselines_020 when available. H-A5's own label does not use them (docs/08)", "available": true}

020 results available for cross-check: True; max |bacc diff| vs 020 results.json (primary, T-mix): 0.0; fold identity: {'identical': True, 'rows_identical': True}

## D-008 (GPU vs CPU, 32 images x 5 conditions)

| detector | worst scaled emb diff | worst abs emb diff | worst |dlogit| | all ok |
|---|---|---|---|---|
| commfor | 3.50e-05 | 1.73e-04 | 1.58e-04 | True |
| bfree | 3.19e-05 | 6.07e-05 | 4.79e-05 | True |
| rine | 5.27e-05 | 5.66e-04 | 1.06e-04 | True |
| clipdet | 2.07e-05 | 2.07e-05 | 8.74e-05 | True |

## Runtime

- commfor: extraction 241 s over 5 conditions
- bfree: extraction 316 s over 5 conditions
- rine: extraction 222 s over 5 conditions
- clipdet: extraction 201 s over 5 conditions
