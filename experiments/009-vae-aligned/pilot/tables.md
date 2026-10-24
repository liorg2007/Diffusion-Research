# E2 PILOT tables (experiments/009-vae-aligned). Source: pilot/results.json. PILOT: not a Phase 4 verdict, not counted toward the stop rule.

Timing gate: {"sd-vae-ft-mse": {"sec_per_pass": 0.3675383567810059, "batch": 4, "peak_mem_MiB": 3413.74169921875}, "sdxl-vae": {"sec_per_pass": 0.3544528841972351, "batch": 4, "peak_mem_MiB": 3413.74169921875}, "worst_sec_per_pass": 0.3675383567810059, "cut_applied": false}

## Labels (label arm = orig)

```
{
 "six_A": {
  "value": 0.7776691495965239,
  "ci": [
   0.7305516229281768,
   0.8169070427198802
  ],
  "arm": "orig",
  "cond": "jpeg75_resize05",
  "label": "PILOT INCONCLUSIVE (0.75 <= x < 0.85)"
 },
 "six_C_vs_B": {
  "B_logo_macro_jpeg75_resize05": 0.9845786778398512,
  "label": "no headroom / uninformative (B > 0.96)",
  "descriptive_C_minus_B": {
   "delta": -0.0435482619491,
   "ci_lo": -0.06258786347266013,
   "ci_hi": -0.02809892185927557
  }
 },
 "seven": {
  "label": "PILOT FAIL",
  "validity_gate": {
   "valid": true,
   "rule_A6_lower_bound_gt_0p5": true,
   "literal_ci_includes_0p5": false
  },
  "arm": "orig",
  "ratio_clean": 0.6007681564245809,
  "ratio_clean_ci": [
   0.539642847094332,
   0.6685706014015758
  ],
  "single_clean": 0.31855990068280576,
  "texture_clean": 0.7778708876474241,
  "ratio_laundered": 0.3177917442582247,
  "single_laundered": 0.3119258224705152,
  "texture_laundered": 0.6863516449410304,
  "crit_ratio_ge_single_plus_0p05": true,
  "crit_ratio_laundered_ge_0p65": false,
  "crit_beats_texture_both_conds": false
 }
}
```

## #6 macro AUROC over 6 generators vs 358 test natives, arm orig (95% cluster-bootstrap CI)

| cond | A (aligned only) | B (=A0 LOGO) | C (B+aligned) | C_all_reals | C-B | A SD-family | A other | B SD-family | B other |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.819 [0.769, 0.861] | 0.992 [0.987, 0.996] | 0.972 [0.956, 0.985] | 0.947 [0.922, 0.967] | -0.0201 [-0.0317, -0.0100] | 0.865 [0.817, 0.906] | 0.796 [0.743, 0.842] | 0.997 [0.994, 0.999] | 0.990 [0.983, 0.995] |
| jpeg75_resize05 | 0.778 [0.731, 0.817] | 0.985 [0.976, 0.991] | 0.941 [0.917, 0.959] | 0.932 [0.904, 0.954] | -0.0435 [-0.0626, -0.0281] | 0.814 [0.763, 0.855] | 0.759 [0.711, 0.802] | 0.981 [0.969, 0.990] | 0.986 [0.978, 0.993] |

Per generator AUROC of A:

| cond | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|
| clean | 0.798 | 0.754 | 0.913 | 0.719 | 0.870 | 0.860 |
| jpeg75_resize05 | 0.784 | 0.707 | 0.843 | 0.703 | 0.811 | 0.817 |

B vs 005 pilot_v2 A0 (reproduction): {"clean": {"ours": 0.9921477343265054, "e005": 0.9920352265673494, "abs_diff": 0.00011250775915594424}, "jpeg75_resize05": {"ours": 0.9845786778398512, "e005": 0.9849355990068281, "abs_diff": 0.0003569211669769423}}

### Texture check, arm orig

| cond | AUROC reals vs aligned (fake=lower Laplacian) | (fake=higher) | median lap real / aligned | A test overall | A within-quintile mean (n usable) | A matched on log-Lap (pairs) | texture baseline test (fake=lower) |
|---|---|---|---|---|---|---|---|
| clean | 0.550 [0.541, 0.566] | 0.450 [0.434, 0.459] | 0.0313 / 0.0272 | 0.819 | 0.836 (5) | 0.797 (271) | 0.228 |
| jpeg75_resize05 | 0.543 [0.536, 0.564] | 0.457 [0.436, 0.464] | 0.0401 / 0.0364 | 0.778 | 0.785 (5) | 0.760 (262) | 0.311 |

Quintile strata of A (clean): [{"bin": 0, "lap_lo": 0.0014013891341164708, "lap_hi": 0.02409118413925171, "n_real": 144, "n_fake": 72, "auroc": 0.7505787037037037, "usable": true}, {"bin": 1, "lap_lo": 0.02427319437265396, "lap_hi": 0.04669575020670891, "n_real": 101, "n_fake": 114, "auroc": 0.7676741358346361, "usable": true}, {"bin": 2, "lap_lo": 0.04688546806573868, "lap_hi": 0.073800228536129, "n_real": 60, "n_fake": 156, "auroc": 0.8368589743589744, "usable": true}, {"bin": 3, "lap_lo": 0.07388222962617874, "lap_hi": 0.11505933851003647, "n_real": 28, "n_fake": 187, "auroc": 0.9037433155080213, "usable": true}, {"bin": 4, "lap_lo": 0.1152341365814209, "lap_hi": 0.39026474952697754, "n_real": 25, "n_fake": 191, "auroc": 0.9231413612565446, "usable": true}]

## #7 macro AUROC, arm orig (fake-score fixed: -ratio / -single error; n = {'natives': 358, 'fakes': 360})

| cond | ratio (min over VAEs) | single-pass (min) | texture baseline | ratio-single | ratio-texture | ratio SD-fam | ratio other | ratio sd-vae-ft-mse |
|---|---|---|---|---|---|---|---|---|
| clean | 0.601 [0.540, 0.669] | 0.319 [0.240, 0.403] | 0.778 [0.723, 0.827] (sign +1) | +0.282 [+0.248, +0.317] | -0.177 [-0.283, -0.061] | 0.710 | 0.546 | 0.601 |
| jpeg75_resize05 | 0.318 [0.249, 0.395] | 0.312 [0.235, 0.399] | 0.686 [0.606, 0.757] (sign +1) | +0.006 [-0.031, +0.041] | -0.369 [-0.501, -0.217] | 0.248 | 0.353 | 0.318 |

## #6 macro AUROC over 6 generators vs 358 test natives, arm fov512 (95% cluster-bootstrap CI)

| cond | A (aligned only) | B (=A0 LOGO) | C (B+aligned) | C_all_reals | C-B | A SD-family | A other | B SD-family | B other |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.778 [0.745, 0.808] | 0.980 [0.961, 0.993] | 0.944 [0.908, 0.967] | 0.966 [0.940, 0.983] | -0.0365 [-0.0545, -0.0228] | 0.837 [0.802, 0.872] | 0.748 [0.711, 0.783] | 0.989 [0.976, 0.997] | 0.976 [0.953, 0.991] |
| jpeg75_resize05 | 0.681 [0.638, 0.720] | 0.980 [0.961, 0.992] | 0.926 [0.888, 0.952] | 0.959 [0.930, 0.977] | -0.0536 [-0.0737, -0.0376] | 0.747 [0.699, 0.791] | 0.648 [0.601, 0.692] | 0.987 [0.974, 0.996] | 0.976 [0.955, 0.991] |

Per generator AUROC of A:

| cond | dalle2 | dalle3 | firefly | midjourney-v5 | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|
| clean | 0.845 | 0.678 | 0.761 | 0.707 | 0.822 | 0.852 |
| jpeg75_resize05 | 0.688 | 0.610 | 0.615 | 0.680 | 0.753 | 0.741 |

B vs 005 pilot_v2 A0 (reproduction): {"clean": {"ours": 0.9800900062073247, "e005": 0.9792442582247052, "abs_diff": 0.0008457479826194936}, "jpeg75_resize05": {"ours": 0.9797059279950342, "e005": 0.9799270639354437, "abs_diff": 0.00022113594040951146}}

### Texture check, arm fov512

| cond | AUROC reals vs aligned (fake=lower Laplacian) | (fake=higher) | median lap real / aligned | A test overall | A within-quintile mean (n usable) | A matched on log-Lap (pairs) | texture baseline test (fake=lower) |
|---|---|---|---|---|---|---|---|
| clean | 0.556 [0.540, 0.591] | 0.444 [0.409, 0.460] | 0.0468 / 0.0442 | 0.778 | 0.783 (5) | 0.775 (316) | 0.394 |
| jpeg75_resize05 | 0.556 [0.548, 0.577] | 0.444 [0.423, 0.452] | 0.0605 / 0.0560 | 0.681 | 0.698 (5) | 0.701 (320) | 0.379 |

Quintile strata of A (clean): [{"bin": 0, "lap_lo": 0.0029527402948588133, "lap_hi": 0.03710419684648514, "n_real": 127, "n_fake": 89, "auroc": 0.6971600460054853, "usable": true}, {"bin": 1, "lap_lo": 0.03724026679992676, "lap_hi": 0.05791430175304413, "n_real": 66, "n_fake": 149, "auroc": 0.8201138905836892, "usable": true}, {"bin": 2, "lap_lo": 0.05804989114403725, "lap_hi": 0.08121149241924286, "n_real": 49, "n_fake": 167, "auroc": 0.806672369546621, "usable": true}, {"bin": 3, "lap_lo": 0.08125045895576477, "lap_hi": 0.11058180034160614, "n_real": 38, "n_fake": 177, "auroc": 0.7694023193577163, "usable": true}, {"bin": 4, "lap_lo": 0.11066427081823349, "lap_hi": 0.33373022079467773, "n_real": 78, "n_fake": 138, "auroc": 0.8219063545150501, "usable": true}]

## #7 macro AUROC, arm fov512 (fake-score fixed: -ratio / -single error; n = {'natives': 358, 'fakes': 360})

| cond | ratio (min over VAEs) | single-pass (min) | texture baseline | ratio-single | ratio-texture | ratio SD-fam | ratio other | ratio sd-vae-ft-mse |
|---|---|---|---|---|---|---|---|---|
| clean | 0.422 [0.350, 0.497] | 0.390 [0.304, 0.485] | 0.607 [0.526, 0.681] (sign +1) | +0.032 [-0.021, +0.081] | -0.185 [-0.324, -0.033] | 0.382 | 0.442 | 0.422 |
| jpeg75_resize05 | 0.335 [0.269, 0.407] | 0.374 [0.299, 0.454] | 0.619 [0.543, 0.688] (sign +1) | -0.039 [-0.081, +0.001] | -0.284 [-0.414, -0.140] | 0.278 | 0.363 | 0.335 |

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
   "ok": true
  },
  "sdxl-vae": {
   "n": 32,
   "max_abs_dpixel_x1": 0.005263566970825195,
   "max_abs_dpixel_x2": 0.0012486577033996582,
   "tol": 0.001,
   "uint8_mismatch_frac_x1": 0.0007102886835734049,
   "ok": false
  }
 },
 "downstream": {
  "sdxl-vae": {
   "n": 32,
   "pixel_guard": {
    "n": 32,
    "max_abs_dpixel_x1": 0.005263566970825195,
    "max_abs_dpixel_x2": 0.0012486577033996582,
    "tol": 0.001,
    "uint8_mismatch_frac_x1": 0.0007102886835734049,
    "ok": false
   },
   "max_abs_dpixel_uint8_x1": 2,
   "uint8_mismatch_frac_x1": 0.0007102886835734049,
   "uint8_mismatch_frac_x2": 0.04615561167399088,
   "max_rel_dratio": 0.005403158531397993,
   "spearman_ratio_gpu_vs_cpu": 0.9989002932551319,
   "features_device": "cuda (both sets)",
   "logit_tol": 0.001,
   "max_abs_dlogit_B0_clean": 0.014793410236667803,
   "max_scaled_dfeature_clean": 0.009565620683133602,
   "max_abs_dlogit_B0_jpeg75_resize05": 0.12160977677618945,
   "max_scaled_dfeature_jpeg75_resize05": 0.08621767163276672,
   "max_abs_dlogit_B0": 0.12160977677618945,
   "ok": false
  }
 },
 "bank": [
  "sd-vae-ft-mse"
 ],
 "probe": {
  "ok": true,
  "tol": 0.001,
  "items": {
   "orig/clean": {
    "max_abs_feature_diff": 8.106231689453125e-05,
    "max_scaled_feature_diff": 2.6613473892211914e-05,
    "max_abs_dlogit_A": 5.4272404361455084e-06,
    "max_abs_dlogit_C_full": 2.5786495563284717e-05,
    "ok": true
   },
   "orig/jpeg75_resize05": {
    "max_abs_feature_diff": 6.866455078125e-05,
    "max_scaled_feature_diff": 1.1771505342039745e-05,
    "max_abs_dlogit_A": 5.6281933261126404e-06,
    "max_abs_dlogit_C_full": 1.614715928921484e-05,
    "ok": true
   },
   "fov512/clean": {
    "max_abs_feature_diff": 6.532669067382812e-05,
    "max_scaled_feature_diff": 1.1104873010481242e-05,
    "max_abs_dlogit_A": 0.00015302865199195992,
    "max_abs_dlogit_C_full": 1.9946283702942935e-05,
    "ok": true
   },
   "fov512/jpeg75_resize05": {
    "max_abs_feature_diff": 9.822845458984375e-05,
    "max_scaled_feature_diff": 1.0043156180472579e-05,
    "max_abs_dlogit_A": 0.0001425213881400822,
    "max_abs_dlogit_C_full": 1.1704664690803668e-05,
    "ok": true
   }
  }
 }
}
```
