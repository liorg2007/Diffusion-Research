# 003-baselines-eval — notes

Run: `run.py prep → score → b0 → report` (seed 0). Every number below is from `tables.md` (generated from `results.json`).
Test set: 358 VISION-native + 199 RAISE real crops (512 px); 6 Synthbuster generators + SD3.5-large + FLUX (FLUX test-only). WA/FB-HQ = platform copies of the VISION test natives, used for FPR only.
Conditions: `clean`; `jpeg75_resize05` = 0.5× resize then JPEG q75 (the "laundering" condition); `social_sim` = random resize in [0.7, 1.0] then JPEG q in [70, 100], seeded per image.

## Findings
1. **Clean:** Community Forensics is best (AUROC 0.992), followed by B-Free 0.970, B0 0.932, RINE 0.844 and ClipDet 0.780.
2. **Laundering (jpeg75_resize05) hurts everything; no method reaches the ≥0.95 target.**
   - B0 0.875 is the most robust, then RINE 0.807, CommFor 0.800, B-Free 0.714 and ClipDet 0.595.
   - The newest generators suffer most. CommFor: SD3.5 0.656, FLUX 0.593. B-Free: SD3.5 0.511, FLUX 0.469 (chance level).
   - TPR@1%FPR under laundering is ≤0.321 for every method (best is CommFor).
3. **social_sim is much milder than jpeg75_resize05.** For example, CommFor scores 0.977 vs 0.800. So most of the damage comes from the 0.5× downscale + q75 combination. Which factor dominates is not yet separated (resize(0.5) vs jpeg(75) ablation pending).
4. **False positives on real phone photos are far above the ≤1% target.**
   - At the native threshold (logit 0), clean, VISION-native: B-Free 17.3%, ClipDet 10.1%, CommFor 4.2%, RINE 2.2%.
   - At the clean-calibrated threshold, B-Free's FPR on VISION-native rises from 23.5% (clean) to 33.0% (laundered).
   - Caveat: VISION devices are 2015–2017 phones, before neural ISPs (kb/datasets/vision-phone-photos.md). This is NOT the neural-ISP audit.
5. **B0 shows a format shortcut** (supports hypothesis drift-format-shortcut).
   - Its AUROC depends on which real pool it is compared against: 0.962 vs VISION-native (JPEG) but 0.876 vs RAISE (uncompressed PNG).
   - Its FPR on RAISE at the calibrated threshold is 9–12%, against 1–4% on VISION.
   - Reading: the probe partly learned "has JPEG traces ⇒ real". RAISE reals and the PNG fakes both lack these traces.
   - The metadata-only format probe separates VISION-native from fakes at 0.989. It uses original-file features, so this is an upper bound on what a crop model sees.
   - CommFor and B-Free are much less affected on clean images (both pools ≥0.957).
6. **B0 leave-one-generator-out (LOGO):** macro AUROC 0.916 clean / 0.861 laundered / 0.903 social_sim.
   - Weakest held-out generators: dalle2 and midjourney-v5 (0.867 clean); SD3.5 laundered 0.797.
   - FLUX, never trained on: 0.886 / 0.813 / 0.862.
7. **Anomaly to check:** RINE scores DALL·E 3 at 0.215 clean (inverted) but 0.849 laundered. This may be a preprocessing interaction (crop224 of a 512 crop). It is not investigated.

## Caveats
- **"calib5" thresholds are re-fit per condition, which is optimistic.** The "clean_calib5" table is the deployment view.
- **The B0 D-008 check reads 0.00 trivially.** Since D-010, B0 runs entirely on CPU, so the check compares CPU with CPU. The 12 pretrained-scorer checks are genuine MPS-vs-CPU comparisons (max 8.7e-5).
- **ClipDet is CLIP-only.** It is not the repo's CLIP + Corvi-R50 fusion, so it underestimates the published method.
- **B-Free is non-commercial licensed.**
- **The git HEAD in `tables.md` (dff2011) predates the D-010 edits used by this run.** Those edits are committed in the Phase 3 commit.
- **Scale:** n per generator is small (Synthbuster 60 test images each), so per-generator CIs are wide. See `results.json`.

## Implications for Phase 4
- The gap is laundering robustness plus real-photo FPR, not clean accuracy.
- B0 (frozen DINOv2 + linear probe) is the right base for hypothesis 1, laundering-nuisance-projection: it is the most laundering-robust model and the cheapest to modify.
- Its format shortcut (finding 5) must be removed first, or measured alongside. Hypothesis 4 is now evidence-backed.
- The Phase 2 plan assumed DINOv3 and 9 Synthbuster generators. Actual setup: DINOv2-base and 6 Synthbuster + SD3.5 + FLUX. Reconcile before Phase 4.
