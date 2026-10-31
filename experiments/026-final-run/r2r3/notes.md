# 026-final-run/r2r3: R2 (mechanism) and R3 (laundering stress map)

Pre-registration: docs/11-final-run-plan.md (D-020). Code: run.py (stages guards, extract, fit, report), selftest.py (62/62, results/selftest.txt, run BEFORE the real run), config.json (all constants, fixed before any real result). Results: results/results.json, results/tables.md, results/guards.json, logs results/run_*.log. Features cache (git-ignored): cache/.
Data, folds, probe, bootstrap are imported from experiments/020-attr-foundation/run.py (K=8 main rows, folds_ha1_main.json, T-mix = clean + jpeg75_resize05 + social_sim, standardised LR with C in {1e-4..1e-1} by inner grouped 3-fold CV, 1,000 caption-family cluster-bootstrap replicates). Nothing outside r2r3/ was modified.

## Headline (primary condition jpeg75_resize05, 8-way bacc, chance 0.125)
- Unmanipulated nested best-mid-block probe: 0.857 [0.835, 0.878] (020 nested: 0.855). Block 12: 0.561. Mid minus last +0.297 [0.264, 0.332].
- **R2 (a) tile shuffle keeps >= 70%: CONFIRMED**, retained 0.944 [0.918, 0.970] (bacc 0.816). CI-robust.
- **R2 (b) blur sigma 1.5 keeps < 50%: REFUTED**, retained 0.972 [0.952, 0.991] (bacc 0.837). CI-robust. Fixed-block (B* = 6) variant agrees.
- **R2 mid > last persists under (a): CONFIRMED**, delta +0.241 [0.211, 0.271] (unmanipulated +0.297).
- Descriptive: high-pass keeps 0.854 [0.818, 0.886]; grayscale keeps 0.917 [0.887, 0.948]. The signal needs neither global layout, nor colour, nor the low frequencies, nor the finest detail left after laundering. Every manipulation lowers bacc a little (shuffle -0.041 [-0.061, -0.022], blur -0.021, gray -0.061, high-pass -0.107).
- **R3 smooth decay: REFUTED by the declared rule, but borderline.** Max single-step drop along an axis slice is 0.204 [0.189, 0.234] vs the 0.20 limit, always at the resize step 0.35 -> 0.25 (frozen probe, e.g. q75 0.764 -> 0.560). The JPEG axis is smooth (largest step 0.08). No non-monotone rise > 0.05. The 0.25 column is also the most upsampled cell (see deviation 2), so cliff vs. interpolation artefact cannot be separated.
- **R3 extended T-mix shifts the breaking point by >= 1 step: CONFIRMED, but on very thin evidence.** The frozen probe goes below 0.5 in only 1 of 30 cells [1, 3] ((q30, f0.25) = 0.458; (q45, f0.25) = 0.500); the extended probe in none. Only 2 of 11 axis slices contain a break, both shift 1. Descriptive (post hoc, not a verdict): cells below 0.60 bacc: frozen 5 (the whole f0.25 column except q95), extended 0. Extended gain grows with severity: +0.18 at f0.25, +0.12 at (q30, f0.35); within +-0.03 at mild cells (slight loss -0.026 at (q95, f1.0)).
- The probes are far from breaking over most of the grid: frozen bacc >= 0.62 everywhere except the f0.25 column (0.46-0.62).

## Gates (all passed; tables.md "Gates")
- **Parity** with the 004 cache, 16 images (2 per class) x 5 conditions, blocks 4/8/12: max scaled diff 8.5e-6 (tol 1e-3, 004 definition), before any extraction (results/guards.json).
- **D-008** GPU vs CPU, 32 images (4 per class), 6 variants (5 manipulations at jpeg75_resize05, plus jr(30,0.25)): max |dlogit| 3.0e-5, max scaled feature diff 1.6e-5; CPU run-to-run spread about 3e-6, so the tolerance was max(1e-3, 2 x spread) = 1e-3. Logit probe: LR (C 0.01) on the cached 004 block-8 features, jpeg75_resize05, folds != 0.
- Permutation control (100 label shuffles per manipulation, block 8, C 0.01, T-mix, primary): means 0.1232-0.1252 (chance 0.125), p95 <= 0.1444 (tol 0.01 / +0.03).
- sha256 dedupe (0 duplicates, 0 cross-class), committed folds reproduced and checked (sha256 bc2894a4...), no group in both train and test (assert in every task), no held-out laundering (screenshot_sim, mavali_medium) in any training set, frozen R3 probe reproduces R2 (0.8575 both), identical processing for all classes (one code path; manipulation depends on image and sha256 only; selftest audit).
- No OOM occurred (0 retries).

## Decisions and interpretations (fixed in config.json before any real result)
1. **Best mid block** = blocks 3-9 (relative depth 0.25-0.75 of 12). Primary R2 numerator and denominator = nested choice per outer fold by inner grouped CV on train folds only, under the same manipulation (same rule in both). Sensitivities in tables.md: fixed oracle block B* (= 6, best unmanipulated mid block) and per-manipulation oracle best mid block.
2. **Small-image rule (deviation from "centre crop only"):** resize factors 0.35 and 0.25 of the 512 crop give 179 and 128 px, below the 224 input. These images are bicubically upsampled so the short side is 224, then centre-cropped; identical for all classes and for the extended-T-mix training cell. Factors >= 0.5 use the plain centre crop exactly as in 004/020.
3. **Extended T-mix** = T-mix + jr(45,0.5) + jr(75,0.35): "q45 and resize 0.35" read as one new cell on each axis through the training point (q75, f0.5). Both are also grid cells, so the extended probe has seen those two cells (with other images); block and C were re-selected by inner CV under the extended training set. Frozen probe = the unmanipulated R2 nested-mid probe.
4. **Thresholds not in docs/11, declared in config.json:** cliff = max single-step bacc drop > 0.20; break = point bacc < 0.5 (docs/11); breaking index per axis slice (5 resize rows scanning quality, 6 quality columns scanning resize), shift = extended - frozen, headline = median over slices where either probe breaks; verdict by point estimate, "CI-robust" if the 95% CI lies on one side of the threshold. "mid > last persists" = CI lower bound of [nested mid - block 12] under (a) > 0.
5. Tile-shuffle permutation seeded from sha256 (same permutation for every condition of an image); blur and high-pass use scipy gaussian_filter (sigma 1.5, reflect, truncate 4) with uint8 rounding; high-pass has no contrast stretch; grayscale = PIL luma, 3 channels.
6. Permutation control uses 100 shuffles (020: 200) because it runs for 5 manipulations; tolerances as 020.

## Caveats
- One dataset, one backbone, n = 1,020 (about 24 test images per generator per fold). The retained-fraction CIs capture caption-family resampling only.
- The manipulation is applied after laundering, so "blur keeps the signal" means the signal does not live in the finest scale that survives jpeg75_resize05; it does not say it survives blur before laundering.
- Tile shuffle at 32 px does not align with DINOv2's 14 px patch grid (32/14 is not an integer), so patches on tile seams see mixed content; this adds seam artefacts on top of removing global layout. The retained 94% therefore holds despite these seams.
- The extended probe's large gain at f0.25 may partly be the extra upsampled training cell (f0.35) rather than "harsher compression" in general.
- High-pass loses most (-0.107), so some information is in low/mid frequencies or colour; the claim "local texture only" is not established, only "not layout, not fine detail".

## Gate checklist / bug log
- Selftest 62/62 before the real run (synthetic manipulations, crop rule, embedder batching and OOM retry, caching/resume, surface metrics with planted cliff/shift, end-to-end synthetic R2/R3 with planted signal, planted leaks: group split across folds, duplicates across the split memorise, cross-class duplicate, permutation gate, held-out laundering in training).
- Selftest iterations before the real run (synthetic design only): planted signal amplitudes, permutation tolerance for the 320-row synthetic set, and dimensionality of the duplicate-memorisation demo were adjusted; no run.py logic changed. run.py: a permutation-gate summary was added to the gates table before the real run.
- After the real results (cosmetic, no number changed): the tables.md line "fixed-block variant agrees: {...}" was reworded to state what the dict means (prediction confirmed under the fixed-block sensitivity). Before: `fixed-block (B* = block 6) variant agrees: {'a': True, 'b': False}`. After: `sensitivity, fixed block B* = block 6 ... prediction (a) confirmed = True, prediction (b) confirmed = False`. results.json compared before and after the report re-run (config removed): identical.
