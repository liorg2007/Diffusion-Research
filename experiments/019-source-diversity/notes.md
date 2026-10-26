# 019-source-diversity (E10, Amendment 19) - does real-source diversity fix the unseen-pipeline problem?

STATUS: code written and self-tested; NOT run on real data (downloads in progress when written). Exploratory (D-016, D-017). Spec = `docs/04-phase4-plan.md` Amendment 19 (frozen); this file only documents the implementation.

Run: `env/.venv/bin/python -I experiments/019-source-diversity/run.py all --workers 14` (stages `splits`, `prep`, `feats`, `fit`, `eval`, each resumable via `cache/`; `splits --dry-run` checks the logic and writes nothing). Self-test: `selftest.py` (output `results/selftest.txt`). Constants: `config.json`.
Prerequisites: `unsplash_manifest.csv`, `commons_manifest.csv`, `hdrplus_manifest.csv` (columns image_id, path, sha256, width, height, license, group) and the raw files they point to (`fov_prep.py` re-checks every raw sha256). Prep and feature caching must run BEFORE the HDR+ originals are deleted.

## What it does (generalises 016-loso/run.py, fov512 arm only)
- Rows: 016's `splits_fov512.csv` (2,919 003/004 rows + 600 OI; unchanged, E7 roles/groups reproduced) + UNS / WMC / HDRP reals. Crops of the NEW rows only, via 004 `fov_prep.py --no-clobber`. Final file `splits_fov512.csv` (adds a `group` column).
- New-source split: whole groups, seeded permutation (seed 0) of sorted group ids, first-fit into train (300) then calib (100) then test (200); a group that fits nowhere goes to the bucket with the most room left. Exact counts are written to `results/split_counts.json` and `tables.md`. Images with short side < 512 in a manifest are dropped and counted.
- Features: 004 caches (old rows) + `016-loso/cache/fov512` (OI, read only) + `cache/feats` (new rows). 004 pipeline, D-008 guard on the new GPU features (`results/d008_guard.json`).
- Fits (`fit_scores` = E7's recipe: balanced, grouped CV, T-aug clean + jpeg75_resize05 + social_sim, LOGO over the 7 training generators + `full` model for FLUX). A fit depends on the training set T only, so one fit serves every held-out source outside T. Required fits: F_b8 on all 62 non-empty T with |T| <= 5 (covers the 6 x 31 = 186 (S, T) pairs); F_last on the 6 sets with |T| = 5; size-matched control: F_b8 on T = the 5 others, training reals subsampled to the mean real-train count of that S's 10 k=2 subsets (whole groups = device / caption family / image / group, seeded, 5 seeds), fakes unchanged = 30 sets. **110 combos x 8 models = 880 probes** (includes 12 extra F_last sets, see deviation 1). Each combo is cached as `cache/scores_<feat>__<T>[__ctl<seed>].npz`.
- Reuse of E7: the refit on {VISION, RAISE, OI} subsets reproduces E7's cached scores exactly (max |diff| 0.0 on RAISE+OI `full`, checked once; `eval` re-checks all of them and prints `parity_e7`). The bootstrap plan on the E7 rows is identical to E7's (draw order fam, dev, img first, same seed; checked in selftest), so the key secondary is paired on identical rows AND weights. Scores for the new rows need new fits, so E7's score files cannot be reused directly.
- Bootstrap: 1,000 replicates; clusters = caption family (RAISE reals + all fakes), device (VISION), image (OI), and one kind per new source with cluster = `group` (photographer / uploader / burst prefix). FPR CIs: Clopper-Pearson and the cluster bootstrap with the threshold fixed. Threshold readings A (clean calib) and B (same-condition calib).
- Labels (all reported, `decide()`): DIVERSITY HELPS, DIVERSITY NOT QUANTITY (needs the size-matched AUROC part), DISCOVERY, E7-rule label, KILL of the line, each for reading A and B, plus the drop-one-generator (10 subsets; FPR from the LOGO model that never saw the dropped generator) and drop-one-source (each of UNS / WMC / HDRP removed from the pool; k_max = 4) robustness. Key secondary: VISION / RAISE / OI held out, 5 sources vs E7's 2, per condition, with E7's published numbers and the parity check next to it.
- Leak check as asserts (selftest plants each leak and requires the assert to fire):
  - `audit_train_rows`: role = train only; reals only from the training sources' own train pools (so the held-out source contributes nothing, no train/calib/test row); no FLUX; the LOGO generator excluded. Called in every fit worker and, with S, for every evaluated fold.
  - `audit_calib`: threshold rows are calib-role reals of the training sources only.
  - `audit_eval`: evaluation reals are the held-out source's TEST reals only.
  - `audit_fake_rows`: the hash of the fake test rows (8 generators) is identical in every fold.
  - `audit_taug`: both classes get exactly the same T-aug conditions; `audit_same_prep`: every row (any class, any source) is a fov512 crop `data/processed/fov512/images/<sha256>.png`. No per-source normalisation or adaptation exists in the code.
- OWNER: pending (reported in results). Status read from `data/raw/owner_phone`.

## Expected cost (estimates; real-data stages not run)
- Fits: 110 combos, 880 probes. Timed here: 1.4 s per probe at 713 training rows (E7 {V,R,O}), so about 4-5 s at the 1,500+ rows of a k=5 fit; about 45 CPU-minutes, so 4-6 min with 14 workers (16 cores here).
- Prep about 1-2 min (1,800 images), features about 5-10 min on GPU (1,800 images x 5 conditions, E7 did 600 x 3 arms in 3.5 min), eval about 3-6 min (not timed on real data). Whole run about 20-25 min.
- Memory: features 5 conditions x about 5,300 rows x 4,608 float32 = about 0.5 GB (forked workers share it).

## Interpretations (Amendment 19 wording that needed a choice)
- E7-rule learning-curve criterion "k = 5 vs k = 1..4": implemented as AUROC(k=5) > AUROC(k) with CI lo > 0 for EACH k = 1..4 (strict). The weaker reading (vs k = 1 only, as in E7's 2 > 1) is reported next to it (`e7_learning_curve_ok_vs_k1_only`); it can only matter for a PASS, never for a KILL.
- "Mean FPR difference has CI hi < 0": mean over the 6 held-out sources of FPR(k=5) - mean over the 10 k=2 subsets of FPR, `full` model, the CI from the cluster bootstrap with thresholds fixed (as E7).
- "AUROC part" of DIVERSITY NOT QUANTITY: mean over held-out sources of (mean over the 5 seeds of the size-matched k=5 AUROC) - (mean k=2 AUROC), >= 0.03 with CI lo > 0.
- Per-source AUROC/FPR bars in DISCOVERY, the E7 rule and KILL use point estimates (CIs reported), as in E7. VISION test reals = natives + WA + FBH (1,074 rows), VISION calib = natives + WA + FBH, training = natives only (E6/E7 convention).
- For k < 5 the bootstrap metrics are computed at jpeg75_resize05 only (enough for every label and the learning curve); all five conditions are computed for k = 5, the size-matched control and the E7 pair of subsets (key secondary).

## Proposed deviations (NOT applied silently; the lead decides)
1. **12 extra F_last fits at |T| = 4** (96 probes, about 10 % more compute; `drop_source_flast` in config.json). Amendment 19 says F_last only at k = 5, but the drop-one-source robustness of the E7-rule label needs b8 - last on the 5-source pool, i.e. F_last at the 4-source sets. If set to false, the drop-one-source E7-rule label is reported as `PASS?-needs-F_last` or INCONCLUSIVE/KILL only.
2. **Drop-one-source does not re-run the size-matched control** (a new pool changes the matched count and needs 5 x 5 more fits per dropped source), so DIVERSITY NOT QUANTITY is not re-evaluated there (`use_ctl=False`); k_max = 4 and the k=2 baseline is taken from the remaining 4 other sources.
3. **OWNER source.** Amendment 19 says "used if >= 150 photos, otherwise pending" but defines no split. If >= 150 photos appear, the code does NOT use them (status `PRESENT-NOT-USED`). Options: test-only 7th held-out pipeline (cheap: scoring only, no new fits if it never trains) or a 7th training source (63 subsets per S, about 2x the fits). Needs a decision before the run.
4. **Split rule detail.** "Greedily filling train then calib then test" is implemented as first-fit over the permuted groups. With 600 images and groups <= 3 this normally gives exactly 300/100/200, but a last group can overflow a cap by up to 2 images (reported). Groups larger than 3 (HDR+ burst prefixes may be) can shift counts by a few images.
5. **Size-matched achieved count** is the largest whole-group subsample not exceeding the target (it can be a few reals below it); achieved counts are in `tables.md`.
6. A manifest row with short side < 512 is dropped and counted (E7 asserted instead); `fov_prep` would otherwise abort the run.

## Files
`config.json`, `run.py`, `selftest.py`, `notes.md`; written by the run: `splits_fov512.csv`, `results/{results.json, tables.md, scores_long.csv.gz, split_counts.json, d008_guard.json, prep_report_fov512.json, selftest.txt, run.log}`; git-ignored: `cache/`, `_prep/`.
