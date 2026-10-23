# Pilot addendum: cross-scale B0 and Laplacian texture baseline (experiment 006)

**DESCRIPTIVE, post-hoc to the pilot, pre-registered for the official run in Amendment 6 (items 5 and 6).** PILOT data (VISION + 6 Synthbuster); not a Phase 4 verdict. Code: `pilot_addendum.py` (CPU only; cached 004 features + the crops). Test split: 358 native reals in 25 device clusters vs 60 test fakes per generator. AUROC = LOGO-macro over generators, cluster-bootstrap 95% CI (2000 resamples; differences are paired: same reals, same fakes).

## Cross-scale B0 (probe trained on arm A, features of arm B), setting (i) = clean, (iii-p) = jpeg(90) both classes, (iv-p) = jpeg75_resize05 both classes

| Firefly | setting | within orig | within fov512 | orig->fov512 | fov512->orig | orig->fov512 - within fov512 | fov512->orig - within orig |
|---|---|---|---|---|---|---|---|
| with_firefly | (i) | 0.990 [0.982,0.996] | 0.980 [0.961,0.993] | 0.969 [0.956,0.979] | 0.969 [0.948,0.984] | -0.012 [-0.028,+0.008] | -0.021 [-0.038,-0.009] |
| with_firefly | (iii-p) | 0.988 [0.979,0.995] | 0.979 [0.958,0.992] | 0.963 [0.951,0.975] | 0.972 [0.952,0.985] | -0.015 [-0.031,+0.005] | -0.017 [-0.031,-0.006] |
| with_firefly | (iv-p) | 0.981 [0.971,0.989] | 0.980 [0.961,0.993] | 0.923 [0.902,0.942] | 0.979 [0.963,0.990] | -0.057 [-0.077,-0.039] | -0.003 [-0.018,+0.010] |
| without_firefly | (i) | 0.994 [0.990,0.998] | 0.979 [0.960,0.992] | 0.965 [0.951,0.977] | 0.973 [0.953,0.986] | -0.014 [-0.032,+0.006] | -0.022 [-0.038,-0.010] |
| without_firefly | (iii-p) | 0.993 [0.987,0.997] | 0.978 [0.958,0.992] | 0.960 [0.945,0.972] | 0.976 [0.958,0.988] | -0.018 [-0.035,+0.002] | -0.017 [-0.031,-0.007] |
| without_firefly | (iv-p) | 0.981 [0.970,0.990] | 0.980 [0.962,0.993] | 0.914 [0.891,0.936] | 0.978 [0.962,0.989] | -0.066 [-0.088,-0.045] | -0.003 [-0.019,+0.010] |

## Laplacian texture baseline, setting (i) (mean |Laplacian| of the grey 224 centre crop; direction from calib natives vs train fakes)

| Firefly | arm | direction (+1 = more texture is fake) | AUROC | AUROC of +Laplacian (fixed direction) | median Lap native test | median Lap fakes test | n calib reals / train fakes |
|---|---|---|---|---|---|---|---|
| with_firefly | orig | +1 | 0.778 [0.723,0.827] | 0.778 [0.723,0.827] | 0.0307 | 0.0800 | 43 / 360 |
| with_firefly | fov512 | +1 | 0.607 [0.526,0.681] | 0.607 [0.526,0.681] | 0.0510 | 0.0714 | 43 / 360 |
| with_firefly | orig - fov512 | | +0.171 [+0.127,+0.216] | | | | |
| without_firefly | orig | +1 | 0.826 [0.774,0.873] | 0.826 [0.774,0.873] | 0.0307 | 0.0800 | 43 / 300 |
| without_firefly | fov512 | +1 | 0.625 [0.541,0.701] | 0.625 [0.541,0.701] | 0.0510 | 0.0714 | 43 / 300 |
| without_firefly | orig - fov512 | | +0.201 [+0.156,+0.247] | | | | |
