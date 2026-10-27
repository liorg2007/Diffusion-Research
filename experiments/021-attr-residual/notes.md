# 021-attr-residual: H-A2, training-free residual fingerprints (D-018 cycle)

Frozen design: docs/08 H-A2 and §3, docs/09 (config.json restates it). Code: run.py (`gen`, `dev`, `report`), selftest.py. Heavy intermediates (fingerprint sums, residual cache) live in the session scratch dir, not in the repo; `env`-pinned Python only, CPU only.
All numbers below are from `results/results.json` / `results/tables.md` (full tables there).

## Labels (pre-registered rules, point estimates)
| part | label | basis |
|---|---|---|
| (i) generators | **INCONCLUSIVE-i** | clean bacc 0.522 [0.478, 0.567] (cluster bootstrap over 163 test caption families): above KILL-i (< 0.30), below PASS-i (>= 0.60) |
| (ii) devices | **PASS-ii** | native 14-way top-1 0.985 [0.969, 1.000] (>= 0.70) and within-model 1.000 [1.000, 1.000] (>= 0.70) |
| H-A2 overall | **PASS** (via ii) | docs/08 verdict rule "PASS if (i) or (ii) passes"; not a D-018 counting failure |
Annotations (not part of the label): in (i) the M1 margin of gate 2 is NOT met (clean bacc minus M1 = -0.008 [-0.072, 0.057], need CI lower bound >= +0.10), so (i) would not have counted even at >= 0.60. No M1-type baseline gates (ii); a metadata-only M0 for devices (m0_dev.py: dims, bytes/pixel, JPEG quantisation table; 7 distinct tables) gives top-1 0.498 (chance 0.071), far below 0.985.

## (i) Generators (K = 8, 1,020 fakes, 717 train / 303 test, split by caption family 70/30, seed 0)
- Clean: bacc 0.522; per-generator recall dalle2 0.175, dalle3 0.367, firefly 1.000, midjourney-v5 0.472, SD2 0.419, SDXL 0.345, sd3_large 0.400, flux 1.000. Without Firefly (refit, K = 7, chance 0.143): 0.479. Pairwise on matched families 0.840 [0.776, 0.894] (chance 0.5). Mean over 10 other family splits: 0.506 +- 0.018.
- M1 texture baseline on the same crops: clean 0.530. M0 raw-file metadata (random forest): 0.674 [0.628, 0.716], so file metadata alone beats the residual fingerprint; M0 on the preprocessed tensors is exactly chance (0.125, constant features).
- Laundered (fingerprints from CLEAN train images = realistic): jpeg75_resize05 0.114 [0.080, 0.151], social_sim 0.164 [0.125, 0.204]; both at or just above chance (0.125) and below M1 (0.278, 0.305). The predicted collapse is confirmed.
- Laundered (fingerprints from matched-condition train images): jpeg75_resize05 0.523 [0.476, 0.567] (M1 0.454; bacc - M1 = +0.069 [-0.006, 0.139]); social_sim 0.211 [0.169, 0.257] (M1 0.426). **This does not match the prediction "jpeg75_resize05 near chance"**: when the fingerprint is rebuilt under the same laundering, the residual still carries class information. Not judged by the pre-registered label; unexplained here. A possible reading (untested): after resize, JPEG and top-left crop the residual keeps generator-specific texture/spectrum that M1 (0.454) also sees, i.e. not a clean "fingerprint" effect.
- Spectral variant (reported, not judged): 0.254 clean, 0.193 / 0.290 / 0.205 / 0.234 in the laundered cells (clean / matched fingerprints).
- Firefly (JPEG q~95 under .png names) and flux are separated perfectly in clean; the Firefly case is the JPEG grid, the flux case is not analysed here.

## (ii) VISION devices (14 devices, 197 photos, full resolution, PRNU MLE fingerprint, PCE, leave-one-photo-out)
Devices verified from data/manifest_raw.csv: D02, D04, D05, D06, D10, D11, D14, D15, D18, D19, D21, D29, D33, D34 (all long side 3264, short 2448; D33 holds 9 portrait natives). 14 photos each, D10 has 15 (197 triplets, nat/WA/FBH sharing the photo index).
| arm | top-1 (chance 0.071) | within-model (chance 0.445) | TPR @ PCE 60 | FPR @ PCE 60 (all / same-model / other-model) | true-device PCE peak at zero shift (PCE >= 60) |
|---|---|---|---|---|---|
| native -> native | 0.985 [0.969, 1.000] | 1.000 [1.000, 1.000] | 0.964 [0.939, 0.985] | 0 / 0 / 0 | 0.989 (190 cases) |
| WA -> WA | 0.930 [0.895, 0.964] | 0.977 [0.947, 1.000] | 0.371 [0.320, 0.421] | 0 / 0 / 0 | 1.000 (73) |
| FBH -> FBH | 0.871 [0.829, 0.911] | 0.977 [0.946, 1.000] | 0.381 [0.330, 0.431] | 0 / 0 / 0 | 1.000 (75) |
| native fingerprint resized -> WA | 0.915 [0.880, 0.951] | 0.969 [0.938, 0.993] | 0.452 [0.401, 0.503] | 0 / 0 / 0 | 1.000 (89) |
- Median PCE true device 1091 vs 1.7 for wrong devices (native). Native misses: 3 of 197 photos (D05 2, D33 1). Same-model sets: iPhone 4s (D02/D10), 5c (D05/D14/D18), 6 (D06/D15), 5 (D29/D34) all 1.000. CIs: bootstrap of photos within device (stratified), 1,000 replicates.
- Scene content is not the signal: no wrong-device fingerprint ever reached PCE 60 (0 false hits in 197 x 13 pairs x 4 arms), and the true-device peak sits at zero shift. Photos of all devices come from one VISION acquisition protocol, so a content leak would lift the wrong devices too; it did not.
- Caveats: 2010s devices only (no modern neural-ISP phones); fingerprints come from 13-14 natural photos of the same collection (a platform copy sees the same camera session); VISION's own flat-field images were not used (fingerprint from natural photos only).

## Gate checklist (docs/08 §3)
- [x] 1 Identical preprocessing: decode with PIL, EXIF never propagated, RGB / gray L; format read from content (Firefly JPEG); generators: condition via `experiments/lib/perturb.py apply_condition(img, cond, key=sha256, seed=0)` on the FULL decoded image, identical code path for all 8 classes, then a top-left 256x256 crop (clean: native pixel grid, no resize; jpeg75_resize05 and social_sim resize before the crop, so the crop covers the top-left 512 / 256-366 px of the original); devices: no resize or crop at full resolution, resize only in the "native fingerprint -> WA" arm.
- [x] 2 Baselines shown: M1 (11 low-level statistics: Laplacian variance, 8x8 blockiness, residual std, saturation mean and clipped fraction, RGB mean/std; logistic regression, C = 1) in every row; M0 raw (0.674) and M0 on tensors (0.125, chance); device M0 0.498. M1 margin NOT met in (i) (see above).
- [x] 3 Grouped splits: caption family from `experiments/015-midlayer-probe/run.py caption_ids` (E6 code; `check_grouped` asserts no family on both sides, and the same family contains the same RAISE caption across generators); devices: leave-one-photo-out with the photo's nat/WA/FBH copies excluded (each arm builds its fingerprint only from other photo indices; for "nat -> WA" the native copy of the test photo is excluded); `verify_lopo_subtraction` re-sums D02 without photo 19 and matches the subtraction (max |dD| = 0, max |dN| = 1.4e-3, inside the float32 rounding of the sums; acceptance threshold 1e-4 relative). Pairwise scoring on matched families reported for (i). Split seeds committed in config.json before any result.
- [x] 4 Laundering: clean, jpeg75_resize05, social_sim; fingerprints from clean train images (realistic) and from matched-condition images; screenshot_sim and mavali_medium not run (H-A2 defines only the three D-009 conditions).
- [x] 5 Label-leak checklist: no test family/photo in training; no hyper-parameter tuned (nothing was chosen on test folds; C of M1 fixed at 1); permutation control (200 shuffles of train labels): generators, clean: mean 0.140 (chance 0.125), 97.5% bound well below the observed 0.522, max 0.317; laundered cells 0.125-0.135, max 0.256; devices (cheap zero-shift NCC statistic on a 1024^2 centre crop, true labels 0.974): mean 0.074 (chance 0.071), 95% interval [0.015, 0.161], max 0.179, p(perm >= real) = 0.005. sha256 dedupe: 1,020 fakes unique; 14-device VISION files (591) unique; decoded-crop hash unique per condition. D-008 guard: not applicable (no accelerator used). PCE peak-at-zero-shift: see device table (98.9% of true PCE >= 60 cases in the native arm; the remaining 2 have the global maximum elsewhere while the zero-shift PCE is >= 60).
- [x] 6 Chance levels in every table (0.125, 0.071, within-model 0.445 pooled = mean of 1/|set| over the 9 devices in same-model sets, pairwise 0.5, majority-class rate = 0.165 (sd3_large, 50 of 303 test images)).
- Validity: no gate violated; label is not INVALID. Selftest: results/selftest.txt PASSED (27 checks: planted fingerprint recovered, null at chance, planted ungrouped-split leak caught, planted test-in-fingerprint leak inflates and LOPO gate raises, duplicate hash caught, PCE equals brute-force correlation, shift flagged, 180-degree rotation recovered, resized-fingerprint arm, permutation controls).

## Implementation choices and deviations (documented, none changes the pre-registered rule)
1. Wiener 3x3 uses reflect borders (scipy.signal.wiener zero-pads, which makes the first row/column of every residual an artefact and the crop starts at the border). The interior formula is otherwise scipy's (checked equal in selftest with zero padding, max |d| 6.5e-13).
2. Orientation (devices): fingerprints are built in canonical landscape sensor orientation. Natives are used as stored; WA/FBH copies get the inverse of the native copy's EXIF orientation (the platform applied it); the 9 portrait D33 natives (stored portrait, EXIF 1) are rotated 90 degrees with the direction (k = 1 or 3) resolved per photo by zero-shift NCC against a fingerprint of the device's landscape natives (all nine chose k = 1; NCC 0.002-0.013 vs 0.000-0.002). That uses the labelled device for orientation only (< 1 bit per photo). Tests follow the spec: portrait rotated to landscape, then 0 and 180 degrees tried and the max PCE kept; for WA/FBH copies all images are tried at 0 and 180 (some iPhone copies are 180-rotated). A canonical thumbnail check confirmed every WA and FBH copy aligns with its native (min correlation 1.000).
3. PCE: signed zero-shift value rho0^2 / mean(rho^2 outside an 11x11 window), circular cross-correlation via FFT of the full image; the decision is argmax of the zero-shift PCE (peak elsewhere is reported, not used). Fingerprint eps = 1.0 in K = N / (D + eps).
4. Device permutation control uses the cheap zero-shift NCC (not PCE) on a 1024^2 centre crop, because PCE for 200 shuffles x 197 photos x 14 candidates at full resolution is not affordable; the same statistic with true labels (0.974) is the reference.
5. Generator split: own 70/30 split by caption family (seed 0) over the 8 generators, not the E6 train/eval roles; the E6 `caption_ids` function supplies the family. LDM (256 px) is not in the main K = 8 and was not run in a side arm.
6. Spectral variant: mean |FFT2| of the residual centred by the all-train mean spectrum, NCC scoring.
7. Bootstrap for (ii) is over photos within device (stratified), not over devices, because 14 devices are too few clusters; the within-model CI of [1.000, 1.000] is therefore a statement about the 127 same-model photos only.

## Bug log (code frozen after selftest.txt; one bug fix, logged)
- Before: `prnu_perm_control` computed the row norms of W as `np.sqrt((W.astype(np.float64) ** 2).sum(1))`; with 12 worker processes each holding a 1.6 GB float64 copy the permutation stage ran out of memory (BrokenProcessPool; log kept in `results/run_dev_crash1.log`). Per-photo scores (`results/prnu_scores.csv`) were already saved and were not touched.
- After: row-wise norms (`np.asarray(W[i], float64)`), plus a resume branch that reuses a saved `prnu_scores.csv`. The selftest was re-run (PASSED) and the permutation stage completed (`results/run_dev2.log`). No change to any scored quantity.

## Runtime (shared 16-core CPU, load average 20-30 from other jobs)
Generators ~8.5 min wall (28 s feature extraction, rest = evaluation with 200 permutations x 5 cells); devices ~26 min scoring + fingerprint sums + the first permutation attempt, then 7 min for the permutation rerun; total about 45 min.

## Files
results/selftest.txt, results/results.json, results/tables.md, results/gen_results.json, results/prnu_scores.csv (11,032 rows: arm, device, photo, candidate, PCE, rho0, peak-at-zero, shift, orientation variant), results/dev_perm_control.json, results/dev_meta.json, results/vision_photos.csv, results/m0_dev.json, logs.
