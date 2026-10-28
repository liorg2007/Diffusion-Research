# 024-attr-vae: H-A6 two-VAE reconstruction signature (non-counting diagnostic, docs/09 decision 1)

**Label: INCONCLUSIVE** (pre-declared rule in config.json `label_rule`; point estimates on the PASS side for clean, on the KILL side for clean-train transfer under laundering). Non-counting, so it does not enter the D-018 stop rule. Files: results/results.json, results/tables.md, results/selftest.txt, results/d008_guard.json, results/d008_pixel.json, results/run.log.

## What was run
- Main: Synthbuster 8 generators (1,020 fakes) -> 3 families (SD2 = kl-f8, SDXL = XL, other 6 = other). Native top-left 512 crop, conditioned with `experiments/lib/perturb.py` (same call for all classes), cut to a multiple of 8. 8 features = 4 per VAE (log e1, log e2, log AEDR, log hp8; see config.json). Caption-family grouped 70/30 split (seed 0), 10 further split seeds as sensitivity.
- Read-outs: logistic regression (fixed C = 1, balanced; clean-train and matched-train sources) and a no-training argmin rule (A2 SD2 vs SDXL; A3 3-way with tau = unlabeled median).
- **Deviation (cap, pre-declared fallback):** sdxl-vae failed the feature-level D-008 guard on the GPU (max |dlogit| 0.15-0.23 vs tol 1e-3; raw feature diff up to 1.7e-2), so it ran on CPU. As allowed by the task, the analysis set is a seeded subsample of **120 per family (360 images; test 108 = 34/33/41)** and **50 per SD1.x generator**. Cap sizes were fixed in config.json BEFORE any CPU extraction ran (first draft 40/class, raised to 120/class after the CPU speed was re-measured; no result had been read).
- **Deviation (sd-vae on CPU too):** sd-vae features computed on GPU on this subset differed from CPU by up to 1.1e-3 .. 3.0e-3 in logit (> 1e-3; the first analyze run, kept as results/results_before_cpu_sdvae_D008fail.json, was therefore INVALID by docs/09 decision 4). The remedy was to recompute the sd-vae features for the 360 + 100 subset on CPU; the reported analysis uses CPU features for both VAEs. The sd-vae-only 1,020-row table in tables.md uses GPU features (sd-vae pixel guard passed, 1.2e-4, but the feature-level logit guard is marginal): treat it as descriptive.
- **Bugs fixed after the first real analyze (logged):** (1) `write_tables` crashed on the sd-vae-only table (`KeyError: 'sd'`, since that table only has the `both` key). Before: crash after results.json was written. After: uses `e['both']` (sdxl columns zeroed). (2) `stage_analyze` now uses all-CPU features when they exist for the capped set. No change to features, splits, read-outs, or label rule after seeing numbers; the D-008 fix is a data-device change forced by the guard.
- Runtime (wall, shared machine): GPU extraction of both VAEs on all 1,620 images x 3 conditions about 1.6 h incl. GPU contention; CPU sdxl on 360+100 images 3.4 h; CPU sd-vae 3.7 h; guard about 1 h.

## Key numbers (results/tables.md; 3-way bacc, chance 0.333, 95% family-cluster bootstrap CI)
| condition (source) | 8-feature LR | M1 texture | argmin A3 |
|---|---|---|---|
| clean | 0.967 [0.933, 0.993] | 0.754 | 0.740 |
| jpeg75_resize05 (clean-train) | 0.333 [0.333, 0.333] | 0.422 | 0.206 |
| jpeg75_resize05 (matched-train) | 0.584 [0.487, 0.672] | 0.735 | |
| social_sim (clean-train) | 0.333 [0.333, 0.333] | 0.515 | 0.303 |
| social_sim (matched-train) | 0.591 [0.494, 0.678] | 0.701 | |
- Clean: SD2 and SDXL recalled 34/34 and 33/33 (A2 argmin SD2-vs-SDXL bacc 1.000, no training); beats M1 by +0.214 [+0.134, +0.299].
- Laundered, clean-trained: the model collapses to "other" for every image (bacc exactly 0.333); the VAE error scales shift under resize/JPEG. Matched-trained recovers 0.58-0.59 but does not beat M1 (-0.15 and -0.11 vs M1) and the argmin rules are at or below chance.
- Split-seed sensitivity: clean 0.950 +- 0.024; jpeg75_resize05 clean-train 0.333; matched 0.610 +- 0.039.
- Label: PASS needs clean >= 0.70 and jpeg75_resize05 >= 0.50 under clean-train (0.333 fails); KILL needs < 0.45 under both sources (matched 0.584 does not); so INCONCLUSIVE. With the D-018 reading ("INCONCLUSIVE with the point estimate on the KILL side") the realistic clean-train transfer is on the KILL side; the prediction (laundered KILL) is broadly confirmed for the realistic setting.
- Secondary SD1.3/SD1.4 (50 each, not in training): with the clean-trained model 100% and 98-100% are predicted "other", 0% kl-f8, while SD2 test images map to kl-f8 at 1.00 (other -> kl-f8 false-map 0.07). So SD1.x does NOT map to the kl-f8 family in this feature space on clean images. Under matched laundering SD1.3 maps to kl-f8 at 0.38 (jpeg) / 0.60 (social), SD1.4 0.28 / 0.46, against SD2 recall 0.68/0.62 and false-map of "other" 0.20/0.22: weakly in favour, CIs wide. Caveat: whether sd-vae-ft-mse equals the VAE of SD1.x or SD2 is [unverified], SD1.x files are exact 512 crops whereas SD2/SDXL in Synthbuster are 576-768 px, and these arms are descriptive.
- RAID cross-dataset arm: **not run**. The gen_k -> generator mapping is [unverified] (kb/datasets/raid.md, docs/08) and nothing in the repo verifies it; not guessed.

## Gate checklist (docs/08 section 3)
- [x] Identical preprocessing: native top-left 512 crop, no resize in clean, perturb.py path for all classes, format from content, EXIF unused (selftest: crop checks).
- [x] Baselines: M1 texture (same protocol; CI lower bound of diff vs M1 on clean = +0.134 >= 0.10, but not on laundered) and M0 on preprocessed tensors (conditioned crop shape) at chance (0.333-0.350).
- [x] Grouped splits: caption-family (015 RID_RE via 021 `load_caption_ids`), no family on both sides (asserted, planted-leak selftest); folds/seeds fixed in config.json before any feature was read.
- [x] Laundering: clean, jpeg75_resize05, social_sim; clean-train and matched-train both reported. screenshot_sim/mavali_medium not run (not required for the label).
- [x] Label-leak: no hyper-parameter chosen on test (fixed C); tau from the unlabeled train pool only; permutation control (200 shuffles) mean 0.32-0.35 in every cell, real clean bacc p = 0.005; sha256 and clean-crop-pixel dedupe across main + SD1.x pass.
- [x] Chance levels in tables (0.333, A2 0.5, majority rate 0.667 in results.json).
- [x] D-008: sd-vae pixel guard (CUDA vs CPU, 32 images) 1.2e-4 <= 1e-3; sdxl-vae failed at feature level, so sdxl is CPU; sd-vae logit guard marginal on GPU, so the analysis set is all CPU (no accelerator features in the reported models).
- [x] Selftest 36/36 before the real analysis (results/selftest.txt; re-run after the bug fixes, still 36/36).
- [ ] Opus/high critique (critique.md): not produced here (lead's step).
