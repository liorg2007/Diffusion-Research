# 012-self-ref: H-D1 PILOT (self-referenced decoder-trace contrast)

Source of criteria: `kb/ideas/self-ref-trace-contrast.md` (ca58d61) + `docs/04-phase4-plan.md` Amendment 12. Pilot under D-014; does not count toward the stop rule.
Numbers are not repeated in the design part below; see `pilot/tables.md` and `pilot/results.json` (results section at the end).

## Run
    CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/012-self-ref/selftest.py
    env/.venv/bin/python -I experiments/012-self-ref/run.py --stage all --dry    # tiny subset -> dry/ (plumbing only)
    env/.venv/bin/python -I experiments/012-self-ref/run.py --stage all          # feat -> vv -> probes -> blur -> guard -> analyze (resumable from cache/)

## What is computed
For an OBSERVED image t (already in its test condition, center-cropped to a multiple of 8 = t'):
- A  = -(d_p(phi(VAE t')) - d_p(phi(t'))), d_p = E4's primary F2 probe (clean pairs); no L0.
- B  = -(d_m(phi(L0 VAE t')) - d_m(phi(L0 t'))), d_m = E4's matched F2 probe (jpeg75_resize05 pairs); same canonical L0 on both sides. LABEL SCORE.
- A2 = -decision of a generator-free logistic probe on difference vectors (Delta1 = phi(VAE x)-phi(x) vs Delta2 = phi(VAE VAE x)-phi(VAE x), 297 train reals, clean); A2L = A2 with L0.
- Mandatory matched-blur control: VAE(t') replaced by GaussianBlur(t', sigma); sigma per (arm, condition, A or B) chosen on the 297 train reals so that the mean shift equals the VAE's.
- Baselines: pooled F2 single-image probe, Laplacian texture (sign from calib natives vs train fakes), pixel L2 round-trip error (fake = low error, fixed a priori), B0 LOGO quoted from 005.
- The refit of E4's F2 probes is checked against `011/pilot/results.json` (abs diff 0 required <1e-6): `pilot/check_probes_vs_011.json`.

## Interpretation choices and deviations (all fixed before the full run; flagged for the lead)
1. **L0 on an already-laundered observation.** jpeg75_resize05 of a 256^2 image is 128^2, smaller than the 224 DINOv2 crop. Primary rule "min": if the L0 output is shorter than 224 it is bicubic-resized back to the pre-L0 size (Amendment 9 precedent), otherwise left at half size (= the literal 011 pipeline for clean 512 inputs; this matches how the matched probe was trained). Sensitivity "uniform" (always back up) is reported as Bu. This is an adaptation, not in the idea file.
2. **VAE(t') is the raw uint8 output for both classes in both arms** (no JPEG-at-source-QF re-save as in E4's orig arm: fakes have no QF, and t is arbitrary at test time). w for A in the orig arm was trained on E4's JPEG@QF round-trips (unchanged E4 probe).
3. **t' = center crop to a multiple of 8** (identity for 512 and 256; <= 3 px for social_sim).
4. **FPR rule.** "1% threshold calibrated on held-out-device natives, evaluated on the other devices' natives, WA and FBH copies": primary = 2-fold cross-fit over the 25 test devices (calibrate on one half's natives, evaluate on the other half), per condition. The 43 calib-split natives share their devices with train, so the alternative (calib-split threshold) is reported as a second row, not as the decision. Criterion applied: max FPR over conditions and the three pools <= 2.0%.
5. **Gate 2** (Spearman >= 0.8) is not tied to a condition in the idea file; it is evaluated for B in all three conditions and must hold in all (for A also reported).
6. **Gate 5** (Amendments 6/8 validity): B clean CI lo > 0.5 and B clean > texture clean.
7. **A2 variants.** The idea file does not say whether A2 uses L0; A2 (no L0) is the arm, A2L is a descriptive extension.
8. **pixel-L2 orientation** fixed a priori (fake = low error, AEROBLADE premise); the flipped orientation is printed (descriptive). E4's critic found it inverted on this data.
9. Texture direction uses the train-split fakes (360) and the calib natives (009 convention); all 720 fakes are scored.
10. GPU was shared with experiment 013 during the dry run (first orig chunk took ~9 min under contention); no effect on numbers.

## D-008
VAE pixel guard reused from 009 (same weights revision, same recon function). New GPU features: 32 fixed images per arm (16 test natives + 16 fakes) x condition, stored VAE outputs and blur images re-embedded on CPU, max |dscore| over all scores <= 1e-3 (`pilot/d008_guard.json`); same for the second round-trip features (16 natives).
