# PILOT_V2: experiment 005 (H1 nuisance projection). Not a Phase 4 verdict.

Options: subdir=pilot_fov512, splits=experiments/004-phase4-features/pilot/splits_pilot.csv, grouped_cv=True, balanced=True, group_bootstrap=True, per_pool=True, headroom_rule=True, primary_pools=['vision_native'], generators=['dalle2', 'dalle3', 'firefly', 'midjourney-v5', 'stable-diffusion-2', 'stable-diffusion-xl'], heldout_generators=[], n_boot=1000
Bootstrap: cluster: reals by device (RAISE per image), fakes by image stratified per generator

## LOGO-macro AUROC (6 generators; test reals of pools vision_native; fakes of g)

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.979 | 0.980 | 0.977 | 0.983 | 0.977 | 0.978 | 0.977 | 0.981 |
| A1 | 0.983 | 0.985 | 0.981 | 0.987 | 0.981 | 0.982 | 0.980 | 0.984 |

## LOGO-macro TPR@1%FPR

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.517 | 0.546 | 0.510 | 0.622 | 0.449 | 0.478 | 0.547 | 0.624 |
| A1 | 0.715 | 0.736 | 0.710 | 0.764 | 0.678 | 0.700 | 0.739 | 0.762 |

## LOGO-macro AP

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.937 | 0.936 | 0.930 | 0.945 | 0.929 | 0.931 | 0.931 | 0.939 |
| A1 | 0.955 | 0.961 | 0.950 | 0.966 | 0.948 | 0.951 | 0.950 | 0.959 |

## Paired bootstrap delta vs A1 (A2_sec vs A1_sec), LOGO-macro AUROC, 95% CI (1000 reps, seed 0)

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0-A1 | -0.004 [-0.012,+0.003] | -0.005 [-0.015,+0.001] | -0.004 [-0.013,+0.003] | -0.004 [-0.013,+0.002] | -0.004 [-0.012,+0.002] | -0.004 [-0.013,+0.002] | -0.004 [-0.012,+0.003] | -0.004 [-0.012,+0.002] |

## Threshold drift (full-T model; threshold = 5% FPR quantile of clean calib natives; test-split clean reals)

| arm | FPR native | FPR WA | FPR FB-HQ | drift_WA | drift_FB | mean drift |
|---|---|---|---|---|---|---|
| A0 | 0.036 | 0.034 | 0.031 | 0.003 | 0.006 | 0.004 |
| A1 | 0.020 | 0.020 | 0.020 | 0.000 | 0.000 | 0.000 |

## Per-generator AUROC at jpeg75_resize05

| arm | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|
| A0 | 0.962 | 0.991 | 0.975 | 0.978 | 0.981 | 0.993 |
| A1 | 0.963 | 0.996 | 0.985 | 0.988 | 0.983 | 0.996 |

## Projector

- primary (1536-d): k*=91 (centered variant k*=93); energy at k=16/64/256: 0.296/0.690/0.962; mean-shift share of trace(M) 0.027

## LOGO-macro AUROC with 95% cluster-bootstrap CI (primary reals)

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.979 [0.961,0.992] | 0.980 [0.962,0.993] | 0.977 [0.959,0.991] | 0.983 [0.967,0.994] | 0.977 [0.958,0.991] | 0.978 [0.959,0.991] | 0.977 [0.959,0.990] | 0.981 [0.964,0.992] |
| A1 | 0.983 [0.972,0.992] | 0.985 [0.975,0.993] | 0.981 [0.969,0.991] | 0.987 [0.979,0.994] | 0.981 [0.967,0.991] | 0.982 [0.970,0.991] | 0.980 [0.968,0.990] | 0.984 [0.974,0.992] |

## LOGO-macro AUROC per real pool, condition clean (test reals of that pool only; fakes of g)

| arm | native (n=358) | WA (n=358) | FBH (n=358) |
|---|---|---|---|
| A0 | 0.979 [0.961,0.992] | 0.979 [0.961,0.992] | 0.980 [0.963,0.992] |
| A1 | 0.983 [0.972,0.992] | 0.983 [0.972,0.992] | 0.984 [0.974,0.993] |

## LOGO-macro AUROC per real pool, condition jpeg75_resize05 (test reals of that pool only; fakes of g)

| arm | native (n=358) | WA (n=358) | FBH (n=358) |
|---|---|---|---|
| A0 | 0.980 [0.962,0.993] | 0.980 [0.963,0.993] | 0.980 [0.963,0.993] |
| A1 | 0.985 [0.975,0.993] | 0.985 [0.975,0.993] | 0.985 [0.975,0.993] |

## Paired cluster-bootstrap delta per real pool (LOGO-macro AUROC)

| pool | pair | clean | jpeg75_resize05 |
|---|---|---|---|
| native | A0-A1 | -0.004 [-0.012,+0.003] | -0.005 [-0.015,+0.001] |
| WA | A0-A1 | -0.004 [-0.012,+0.003] | -0.005 [-0.014,+0.001] |
| FBH | A0-A1 | -0.004 [-0.012,+0.002] | -0.005 [-0.014,+0.001] |

## Verdict: PILOT_verdict = **NOT_EVALUATED (needs A1 and A2)**

- headroom rule: A1 LOGO-macro jpeg75_resize05 = 0.9852 (needs <= 0.96) -> NO HEADROOM: verdict UNINFORMATIVE_NO_HEADROOM
