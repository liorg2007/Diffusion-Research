# PILOT_V2: experiment 005 (H1 nuisance projection). Not a Phase 4 verdict.

Options: subdir=pilot, splits=experiments/004-phase4-features/pilot/splits_pilot.csv, grouped_cv=True, balanced=True, group_bootstrap=True, per_pool=True, headroom_rule=True, primary_pools=['vision_native'], generators=['dalle2', 'dalle3', 'firefly', 'midjourney-v5', 'stable-diffusion-2', 'stable-diffusion-xl'], heldout_generators=[], n_boot=1000
Bootstrap: cluster: reals by device (RAISE per image), fakes by image stratified per generator

## LOGO-macro AUROC (6 generators; test reals of pools vision_native; fakes of g)

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.992 | 0.985 | 0.992 | 0.994 | 0.991 | 0.991 | 0.985 | 0.987 |
| A1 | 0.989 | 0.987 | 0.990 | 0.992 | 0.989 | 0.988 | 0.985 | 0.987 |

## LOGO-macro TPR@1%FPR

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.879 | 0.782 | 0.869 | 0.908 | 0.876 | 0.850 | 0.754 | 0.835 |
| A1 | 0.846 | 0.811 | 0.864 | 0.868 | 0.879 | 0.832 | 0.807 | 0.840 |

## LOGO-macro AP

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.982 | 0.965 | 0.981 | 0.984 | 0.981 | 0.979 | 0.963 | 0.971 |
| A1 | 0.974 | 0.968 | 0.978 | 0.981 | 0.976 | 0.973 | 0.964 | 0.971 |

## Paired bootstrap delta vs A1 (A2_sec vs A1_sec), LOGO-macro AUROC, 95% CI (1000 reps, seed 0)

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0-A1 | +0.003 [+0.000,+0.007] | -0.002 [-0.005,+0.002] | +0.002 [-0.001,+0.004] | +0.001 [-0.001,+0.004] | +0.002 [+0.000,+0.005] | +0.003 [-0.000,+0.006] | -0.001 [-0.004,+0.003] | -0.000 [-0.003,+0.003] |

## Threshold drift (full-T model; threshold = 5% FPR quantile of clean calib natives; test-split clean reals)

| arm | FPR native | FPR WA | FPR FB-HQ | drift_WA | drift_FB | mean drift |
|---|---|---|---|---|---|---|
| A0 | 0.050 | 0.313 | 0.182 | 0.263 | 0.131 | 0.197 |
| A1 | 0.042 | 0.126 | 0.075 | 0.084 | 0.034 | 0.059 |

## Per-generator AUROC at jpeg75_resize05

| arm | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|
| A0 | 0.978 | 0.992 | 0.986 | 0.991 | 0.968 | 0.995 |
| A1 | 0.977 | 0.990 | 0.987 | 0.993 | 0.979 | 0.996 |

## Projector

- primary (1536-d): k*=90 (centered variant k*=92); energy at k=16/64/256: 0.328/0.707/0.957; mean-shift share of trace(M) 0.031

## LOGO-macro AUROC with 95% cluster-bootstrap CI (primary reals)

| arm | clean | jpeg75_resize05 | social_sim | resize(0.5) | jpeg(75) | jpeg(90) | mavali_medium | screenshot_sim |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.992 [0.987,0.997] | 0.985 [0.977,0.991] | 0.992 [0.986,0.996] | 0.994 [0.990,0.997] | 0.991 [0.986,0.996] | 0.991 [0.985,0.996] | 0.985 [0.977,0.991] | 0.987 [0.980,0.992] |
| A1 | 0.989 [0.981,0.995] | 0.987 [0.981,0.992] | 0.990 [0.985,0.995] | 0.992 [0.988,0.996] | 0.989 [0.984,0.994] | 0.988 [0.980,0.995] | 0.985 [0.978,0.992] | 0.987 [0.980,0.993] |

## LOGO-macro AUROC per real pool, condition clean (test reals of that pool only; fakes of g)

| arm | native (n=358) | WA (n=358) | FBH (n=358) |
|---|---|---|---|
| A0 | 0.992 [0.987,0.997] | 0.966 [0.956,0.975] | 0.982 [0.975,0.988] |
| A1 | 0.989 [0.981,0.995] | 0.972 [0.960,0.981] | 0.982 [0.974,0.989] |

## LOGO-macro AUROC per real pool, condition jpeg75_resize05 (test reals of that pool only; fakes of g)

| arm | native (n=358) | WA (n=358) | FBH (n=358) |
|---|---|---|---|
| A0 | 0.985 [0.977,0.991] | 0.950 [0.937,0.961] | 0.970 [0.960,0.979] |
| A1 | 0.987 [0.981,0.992] | 0.966 [0.954,0.977] | 0.977 [0.967,0.986] |

## Paired cluster-bootstrap delta per real pool (LOGO-macro AUROC)

| pool | pair | clean | jpeg75_resize05 |
|---|---|---|---|
| native | A0-A1 | +0.003 [+0.000,+0.007] | -0.002 [-0.005,+0.002] |
| WA | A0-A1 | -0.006 [-0.011,-0.000] | -0.017 [-0.023,-0.010] |
| FBH | A0-A1 | -0.000 [-0.004,+0.003] | -0.007 [-0.012,-0.001] |

## Verdict: PILOT_verdict = **NOT_EVALUATED (needs A1 and A2)**

- headroom rule: A1 LOGO-macro jpeg75_resize05 = 0.9868 (needs <= 0.96) -> NO HEADROOM: verdict UNINFORMATIVE_NO_HEADROOM
