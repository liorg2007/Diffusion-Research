**Verdict: READY WITH FIXES.** No finding changes the Amendment 7 primary labels of CommFor, RINE, ClipDet or B0. Fix F1 before the run. F2–F4 are cheap hardening/reporting fixes.

Reviewer: independent pre-run code review, 2026-10-24. Inputs: `run.py`, `config.json`, `notes.md` and docs/04-phase4-plan.md Amendments 3 and 7. No GPU was used. CPU checks only: split-file alignment, device disjointness, and reading the scorers' transforms. `experiments/lib/tests/run_all.py` gives 58 passed, 0 failed.

## Findings, ranked by severity

### F1 (medium, secondary table only) — the B0 "cross" rows in the threshold-transfer table mix two different probes
- **Where:** `run.py:401-414`, `analyze()`, the loop `for d in INPUTS + XB0`.
- **What's wrong:** for `d == "b0"`, `S[src][c]["b0"]` is the probe trained in arm `src`, and `S[tgt][c]["b0"]` is a *different* probe, retrained in arm `tgt`. Rows keyed `thr:orig->test:fov512(cross)` and the reverse therefore apply one model's calib threshold to another model's scores. That number has no meaning.
- **Which rows are correct:** the real cross-zoom deployment of one fixed B0 is already given by the `b0_trained_orig` and `b0_trained_fov512` rows (same model, other arm's features). Those are correct.
- **Fix:** for `d == "b0"`, emit only the within-arm rows (`src == tgt`). Alternatively, rename the cross rows "not meaningful" and drop them from `tables.md`.
- **Can it change a label?** No. Amendment 7 secondary #1 is descriptive. It would mislead the reader of the transfer table.

### F2 (low) — the B0 fov512 reproduction check is logged but not enforced
- **Where:** `run.py:198-199`.
- **What's wrong:** `chk[arm]` (max |Δscore| against the cached 005/pilot_v2 A0 scores) is only logged. The orig arm gave 0.0 (`pilot/run.log`). The fov512 fit is not done yet. If the fresh fov512 fit differs from 005's fov512 A0, the run continues silently.
- **Fix:** `assert chk[arm] <= 1e-6`, or abort with a message. A difference would mean B0 is not 005's A0, as Amendment 7 requires.
- **Can it change a label?** Only if a hidden discrepancy exists. The check costs nothing.

### F3 (low) — the texture baseline's sign is chosen per arm, so its "delta" is not a fixed scorer's zoom effect
- **Where:** `run.py:356-367`, sign per `(input, arm, cond)`.
- **What's wrong:** if the train AUROC of the raw Laplacian falls on opposite sides of 0.5 in orig and fov512, the texture score is flipped between arms. Its ΔAUROC and its `scale_sensitive` flag then compare two different scorers. The pretrained detectors, by contrast, have a fixed direction.
- **Fix:** set one sign per `(input, cond)`, either from the orig-arm train data (the D-009 label arm) or from both arms pooled, and use it in both arms. Keep the per-arm sign table as a descriptive output.
- **Can it change a label?** No. Texture rows are descriptive, not Amendment 7 detectors.

### F4 (low, reporting) — non-pre-registered rows carry a "scale-sensitive YES/no" label in the primary table
- **Where:** `run.py:369` (`allnames`), and `tables()` at `run.py:451-458`.
- **What's wrong:** `texture_*` and `b0_trained_*` get the same YES/no column as the four Amendment 7 detectors.
- **Fix:** mark them "(extra, descriptive; not pre-registered)" or move them to a separate table. Only CommFor, RINE, ClipDet and B0 should carry the Amendment 7 label.
- **Can it change a label?** No, but it removes a risk of misreading.

### F5 (info) — B0's numbers will not exactly equal the 005 tables
- **Where:** `run.py:303`, the `Plan` fakes.
- **What's different:** E1 evaluates every detector on *test-split* fakes only (60 per generator). 005's LOGO tables use all 120 per generator.
- **Why it's acceptable:** this is the right choice for E1, because it gives the same fakes for every detector and a paired bootstrap. But E1's B0 AUROC will differ slightly from 005/pilot_v2.
- **Fix:** state this in the final notes so nobody "reconciles" the two.

### F6 (info) — the orig arm's D-008 guard is a GPU re-score, not CPU vs CUDA
- **Where:** `run.py:244-253`.
- **What it does:** it compares 32 GPU re-scores with 007's stored scores. 007's stored scores already passed CPU vs CUDA (`007/cache/d008_guard.json`).
- **Verdict:** acceptable as designed. The fov512 arm gets a proper CPU vs CUDA guard (`run.py:255-264`, tolerance 1e-3, abort).

## The agent's declared deviations (notes.md)

1. **Texture direction set on train natives vs train fakes, because calib has no fakes.** Harmless:
   - no test rows are used;
   - it is a 1-bit choice;
   - the train fakes come from the same 6 generators, so nothing leaks from test.
   - Keep it, with the per-arm caveat in F3.
2. **Extra threshold variant from all 129 calib reals.** Harmless and descriptive. Describe it accurately:
   - it is 43 photos × 3 platforms (WA and FBH are copies of the calib natives), so the effective n is about 43;
   - the threshold is the 2nd-highest of the 129 scores (floor(0.01·129) = 1).
   - The pre-registered threshold is `calib_natives`: 43 reals, so calib max + eps. That is implemented correctly (`metrics.threshold_for_fpr`).
3. **Fixed-probe cross-arm B0 rows (`b0_trained_*`).** Harmless and useful: they are the only valid cross-zoom B0 deployment. The standardization correctly uses the training arm's mu/sd (`run.py:203-210`). Keep them, labelled as extra (F4). This is also why F1 matters.

## Verified OK
- **Primary measure.**
  - ΔAUROC = AUROC(orig) − AUROC(fov512), macro over the 6 generators against test natives.
  - Rule: |Δ| ≥ 0.03 and the 95% CI excludes 0 (`label_for`, `run.py:328`).
  - 2,000 resamples, seed 0.
- **Paired cluster bootstrap.**
  - One `Plan` per real pool (reals by device, fakes by image stratified per generator), shared by both arms and every detector.
  - Row alignment holds: `splits_pilot.csv` and `splits_pilot_fov512.csv` have identical sha256 order (2,220 rows, checked).
  - The B0 scores come from `r.SPLITS = SPL["orig"]` in both arms, so they share the orig row order.
- **Split hygiene.**
  - Test devices are disjoint from train and calib devices (25 test devices; overlap 0, checked).
  - Thresholds use calib rows only. The texture sign uses train rows only.
  - B0 LOGO model `logo_g` never saw generator g. `b0_train` ⊆ the train split (459 = 99 natives + 360 fakes).
- **Identical processing.**
  - `perturb.apply_condition(img, cond, key=sha256, seed=0)` is applied after the arm's crop, for both classes.
  - The Laplacian transforms match each scorer's pre-normalisation transform:
    - CommFor: asserted;
    - RINE `CenterCrop(224)` and ClipDet `Resize(224, BICUBIC)+CenterCrop`: checked against `002-baselines/scorers`;
    - B0: `center_crop(224)`, as in `lib/features`.
- **Caches.**
  - fov512 scores are keyed `scores_{det}_fov512_{cond}`, and the CPU reference is fov512-only.
  - The dry and full runs use separate directories.
  - The B0 cache deletes `scores_A0.npz` before a fresh fit, so it cannot pair stale scores with new models.
  - `lap_*.npy` is per arm, with 2,022 rows: 1,563 scored + 99 train natives + 360 train fakes.
- **D-008.**
  - TF32 is off.
  - The fov512 arm is checked CUDA vs CPU (batch 4) on 007's fixed 32 rows, tolerance 1e-3, with abort.
