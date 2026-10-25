# 016-loso (E7, Amendment 16) - leave-one-real-source-out test of the mid-level detector

Exploratory (D-016). Verdict: **KILL** (primary arm fov512, both FPR-threshold readings, all drop-one-generator subsets). OWNER photos: **PENDING** (`data/raw/owner_phone/` does not exist), so the label covers VISION, RAISE and OI only. A KILL cannot be reversed by adding a fourth source.

Run: `env/.venv/bin/python -I experiments/016-loso/run.py all --workers 14` (stages `splits`, `prep`, `feats`, `fit`, `eval`, resumable via `cache/`). Self-test: `selftest.py` (output `results/selftest.txt`, all PASS). Lib tests `experiments/lib/tests/run_all.py`: 62 passed. Every number below is in `results/results.json` / `results/tables.md`. Constants in `config.json` (frozen before any result). Runtime about 11 min (prep 20 s, features 3.5 min GPU, fits 5.2 min CPU with 14 workers, eval 13 s).

## Design (Amendment 16 as implemented; interpretations marked I)
- Sources: VISION (natives, groups = device), RAISE (B-Free PNGs, caption families), OI (Open Images V7 validation, 600 CC BY 2.0 photos from `openimages_manifest.csv`, unchanged; sha256 of every file rechecked; split by image, `default_rng(0).permutation` of sorted ids: 300 train / 100 calib / 200 test). OWNER pending.
- Roles for the 2,919 003/004 rows come from E6's `build_roles` (caption-family majority rule, imported from `015-midlayer-probe/run.py`): train 230 fakes + 99 natives + 84 RAISE reals, no caption id shared by train and calib/test (self-test). Fakes: 7 train generators (LOGO) + FLUX (test only).
- Arms: **fov512 = primary**, orig reported (OI orig = plain 512 centre crop of the CVDF file, no resize), fov224 secondary (below). Crops of the NEW images only were made with `004/fov_prep.py --no-clobber` (600 OI per arm, 0 existing files touched; reports `results/prep_report_*.json`).
- Features: 004 pipeline (DINOv2-base, blocks 4/8/12 CLS+mean, 224 centre crop, batch 20) for the new (image, condition) pairs only, into `cache/<arm>/`; the 004 caches are read, never written. F_b8 = columns 1536:3072, F_last (= B0) = 3072:4608. Conditions clean, jpeg75_resize05, social_sim, screenshot_sim, mavali_medium (the last two never used for fitting).
- Training (E6 recipe): `lib/probe.fit_probe`, balanced, grouped CV (device / family id / image), T-aug = clean + jpeg75_resize05 + social_sim copies sharing their source's group. Reals = `role == train` rows of the training sources (VISION: natives only, as in E6); fakes = `role == train` fakes minus the LOGO generator. No subsampling of sources (OI contributes 300 train reals, VISION 99, RAISE 84; 230 fakes).
- Per held-out source S: train on all other sources (and on each single other source for the learning curve), score S's TEST reals against each held-out generator's TEST fakes (LOGO model per generator, `full` model for FLUX), macro over the 8 generators. S's reals never enter training or calibration (self-test checks every fit's rows).
  - I - VISION test reals = natives + WA + FBH (1,074 rows = 358 images x 3 copies; Amendment: "natives + WA + FBH"); training uses natives only.
  - FPR: threshold = 1% FPR of the `full` model on the TRAINING sources' calib reals (VISION calib = natives + WA + FBH as in E6, RAISE 53, OI 100). I - the Amendment says "clean and laundered": **reading A** (headline) = threshold from clean calib scores, applied at jpeg75_resize05 (E6/D-009 convention); **reading B** = threshold from calib scores under the same condition. Both reported; labels agree.
  - Learning curve: mean over held-out sources of the mean over their k-source training subsets (k = 1: 2 subsets per source, k = 2: 1 subset).
  - Bootstrap (1,000 replicates, one global plan per arm, `Plan`): clusters drawn with replacement within kind: caption families (RAISE reals and ALL fakes with that id move together; D3 id likewise), devices (VISION, 25 test phones), images (OI). Same weights for every source, feature set and condition (paired differences). This differs from E6's plan (E6 drew fakes by image per generator). FPR CIs: Clopper-Pearson plus the cluster bootstrap with the threshold fixed.
  - Labels: PASS/KILL/INCONCLUSIVE exactly as in the Amendment, point estimates for AUROC and FPR thresholds (CIs reported), CI lower bound > 0 for the two differences. Drop-one-generator robustness: macros recomputed without generator g, and the FPR taken from the LOGO model that never saw g (FLUX / no-FLUX: `full`).
- D-008: 13 (arm, condition) pairs, 32 fixed new images each (CUDA vs CPU, TF32 off, batch 4): max |dlogit| 1.21e-04 (B0 probe and block-8 probe both tested), max scaled feature diff 2.0e-05, all <= 1e-3 (`results/d008_guard.json`).

## Results, primary arm fov512, jpeg75_resize05, F_b8 vs F_last (train on all other sources)
| held-out | AUROC F_b8 [CI] | AUROC F_last | FPR F_b8, reading A (k/n) [CP95] | reading B | FPR F_last (A) |
|---|---|---|---|---|---|
| VISION | 0.917 [0.893, 0.938] | 0.885 [0.855, 0.913] | 0.6% (6/1074) [0.2, 1.2] | 0.7% | 0.3% |
| RAISE | 0.689 [0.650, 0.726] | 0.600 [0.560, 0.636] | 19.1% (35/183) [13.7, 25.6] | 26.8% | 10.4% |
| OI | 0.809 [0.774, 0.843] | 0.758 [0.717, 0.795] | 16.0% (32/200) [11.2, 21.8] | 23.0% | 4.5% |

- **Label: KILL** (A and B). Kill triggers: RAISE AUROC 0.689 < 0.75 and FPR 19.1% > 10%; OI FPR 16.0% > 10% (OI AUROC 0.809 is above 0.75 but below the 0.85 pass bar). Only VISION meets AUROC >= 0.85 and FPR <= 5%. The RAISE fold is content-confounded (every fake is prompted from a RAISE caption, critic F1); the label does NOT depend on it, since OI alone already triggers the FPR kill.
- b8 - last, mean over held-out sources: **+0.057 [+0.035, +0.079]** (criterion met). Per source: VISION +0.032, RAISE +0.089, OI +0.051.
- Learning curve (mean held-out AUROC, F_b8): 1 source 0.773 [0.752, 0.793], 2 sources 0.805 [0.783, 0.826], difference **+0.032 [+0.025, +0.039]** (criterion met). Per source, 1 -> 2 sources: VISION 0.826 / 0.921 -> 0.917, RAISE 0.700 / 0.651 -> 0.689, OI 0.804 / 0.736 -> 0.809. FPR at the transferred threshold: VISION 29.1% (RAISE only) / 3.8% (OI only) -> 0.6%; RAISE 25.1% / 40.4% -> 19.1%; OI 10.5% / 44.0% -> 16.0% (more sources do not fix OI or RAISE).
- Held-out launderings (F_b8 macro AUROC; clean | social_sim | screenshot_sim | mavali_medium): VISION 0.920 | 0.911 | 0.930 | 0.894; RAISE 0.651 | 0.668 | 0.662 | 0.663; OI 0.845 | 0.807 | 0.847 | 0.794. FLUX at jpeg75_resize05: VISION 0.880, RAISE 0.627, OI 0.689 (F_last: 0.856 / 0.551 / 0.710, i.e. block 8 is not better on FLUX for OI). FPR on OI is 12.0% clean and 13.5-17.5% under the laundered conditions (reading A); RAISE 35.0% clean.
- Drop-one-generator: KILL in all 10 subsets (all, no FLUX, each of the 8 dropped) under A and B. Min held-out AUROC 0.667-0.716, max FPR 17.5-29.5%, b8 - last +0.052 to +0.062 (lower bound > 0 in all), learning curve +0.029 to +0.035 (lower bound > 0 in all). So the criteria on the layer and on the number of sources hold, but the per-source bar fails in every subset.
- Cross-check with E6: the fov512 "RAISE only -> VISION" fit here gives AUROC 0.826 [0.784, 0.863], the same point estimate as E6's fov512 cross-source number (0.826 [0.786, 0.865]; different bootstrap plan).

## Other arms (descriptive; same code)
- **orig:** KILL. AUROC / FPR (A): VISION 0.851 / 2.3%, RAISE 0.624 / 34.4%, OI 0.753 / 35.0%. b8 - last +0.041 [+0.017, +0.066]; learning curve 0.701 -> 0.743 (+0.041 [+0.031, +0.052]). Scale makes the source shift worse, as the E6 critic found.
- **fov224 (secondary, pilot-only D3 source):** KILL. Only JPEG-only conditions can go through the 224 centre-crop backbone (jpeg75_resize05, social_sim, screenshot_sim and mavali_medium all shrink a 224 px image below 224 px), so this arm uses clean, jpeg(75), jpeg(90), T-aug = those three, and the "laundered" label condition is jpeg(75). Resize launderings are NOT evaluated here. D3 sources: 60 of the 100 RAID/D3 reals have short side >= 224 (the 40 smaller ones are dropped, never upscaled; median short side 250 px, not 324), id split 60/20/20 gives D3 reals 39 train / 10 calib / 11 test, test fakes 20 per D3 generator (gen_0..gen_3 treated as LOGO generators, 12 generators in the macro). AUROC / FPR (A) at jpeg(75): VISION 0.891 / 0.7%, RAISE 0.774 / 12.6%, OI 0.795 / 16.5%, D3 0.800 [0.647, 0.931] / 18.2% (2/11, too few reals to be informative). b8 - last +0.129 [+0.095, +0.169]; learning curve k = 1, 2, 3: 0.742, 0.785, 0.815 (monotone, +0.043 and +0.030 steps).

## Deviations / interpretations to review
1. fov224 arm restricted to JPEG-only conditions (above); the amendment's five conditions are impossible at 224 px. Secondary, no effect on the label.
2. FPR threshold reading A is the headline (E6 convention); B is the sensitivity. Same label.
3. 40 of 100 D3 reals unusable (< 224 px), not 324 px as written in Amendment 16.
4. Bootstrap plan is family-joint (not E6's per-generator image plan); CIs are therefore not directly comparable with E6's.
5. VISION calib reals include WA/FBH copies (E6 convention), so VISION is up-weighted in the pooled threshold when it is a training source.
6. No equal-count subsampling of training sources (the critic's proposal; the Amendment does not require it). OI (300) dominates the reals when it trains.
7. PASS/KILL use point estimates of AUROC and FPR against the thresholds; CIs are reported next to them.

## Caveats
- Three held-out sources only (OWNER pending); the OI test set is 200 reals (FPR resolution 0.5%); all fakes carry RAISE (or LAION in the fov224 arm) content, so only the RAISE and D3 folds have content-matched reals.
- The cause of the failure is not tested here (source style vs content vs scale); E7 only shows that adding the second real source does not make block 8 transfer to an unseen one at the required FPR.
- Files: `config.json`, `run.py`, `selftest.py`, `splits_{orig,fov512,fov224}.csv` (003/004 rows + new rows, `path` = crop), `results/` (`results.json`, `tables.md`, `scores_long.csv.gz`, `d008_guard.json`, `run.log`, `prep_report_*.json`, `selftest.txt`), `cache/` (git-ignored: new-image features, CPU references, fitted scores), `_prep/` (git-ignored scratch).
