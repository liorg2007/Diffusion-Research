# 026-final-run/r1r4: R1 (generality across backbones) and R4 (scaling with data)

Plan: docs/11 (D-020) + addendum (on-disk backbones only, no download). Code: run.py, selftest.py (76/76 pass, results/selftest.txt), config.json (frozen before any real result). Results: results/results.json, results/tables.md (full per-block profiles for 5 backbones), results/profiles.png, results/r4.json, results/d008_guard.json, results/parity.json. Logs: results/*.log. Feature cache: cache/ (git-ignored; regenerate with `run.py extract_all`, about 35 min on the shared GPU/CPU).

## What was run
- Rows/folds: the 1,020 fakes of 020 ha1_main; the committed `folds_ha1_main.json` (sha256 bc2894a4...) equals `make_folds` and the row order (asserted). Probe = 020 `closed_task` imported unchanged (LR, standardised, C by inner 3-fold grouped CV scored at jpeg75_resize05, T-mix training), one probe per block (84 blocks x 5 folds), cluster bootstrap = 020 `cluster_weights` (1,000 replicates, same weights for every block/backbone, so all differences are paired).
- Features: per block CLS and mean-patch (registers skipped), pre-LayerNorm residual stream, float32, TF32 off, from forward hooks (DINOv2: the 004 `Backbone` with layers 1..12). Images: 004 fov512 crop -> `perturb.apply_condition(key=sha, seed=0)` -> 224 centre crop + the model's normalisation.
- R1 selection rule (fixed in config.json before results): the verdict uses the NESTED choice (per outer fold, the mid block 0.25-0.75 relative depth with the best inner-CV bacc on the train folds) compared with the LAST block, paired CI. The "oracle" best-mid (chosen on test points) is a sensitivity only. Headline per backbone = nested over all blocks and nested over mid blocks.
- R4: n in {15,30,60,90,all} per generator in each outer fold's training rows, 5 seeded draws (rng [0,7001,s,n,fold], identical for every model; each caption family holds at most one image of a given generator, so this is a draw of caption families); C re-chosen by inner CV on the subsample; error = 1-bacc at jpeg75_resize05; power law by `curve_fit`; bootstrap = family clusters x resampling of the 5 subsamples. Models: DINOv2-base block 6 (mode of inner-CV choices among mid blocks) and CLIP ViT-L/14 OpenAI block 13 (best non-reference by mean inner-CV bacc over train folds; block = mode of nested-all choices). Dict keys in the "why" text of r4.json/tables.md are 0-based block indices.

## Headline (jpeg75_resize05, chance 0.125; full numbers in tables.md)
- DINOv2-base nested 0.856 [0.834, 0.877] (020: 0.855), last block 0.562, gap +0.295 [0.262, 0.330].
- Gap nested-mid minus last (all satisfy >= 0.10 with CI lo > 0): CLIP-OpenAI +0.249 [0.219, 0.282], CLIP-LAION +0.196 [0.167, 0.226], B-Free +0.167 [0.136, 0.200], CommFor +0.148 [0.118, 0.179].
- Nested headline vs DINOv2-base nested-all (paired): CLIP-OpenAI 0.896 (+0.040 [0.020, 0.062]), CLIP-LAION 0.886 (+0.029 [0.010, 0.050]), CommFor 0.883 (+0.026 [0.006, 0.047]), B-Free 0.839 (-0.017 [-0.036, 0.003]).
- **R1 verdict: GENERAL (4 of 4). Frozen CLIP pair: general across pretrained backbones (2 of 2).** Fine-tuned pair (B-Free, CommFor): both satisfied, reported separately. Oracle sensitivity gives the same verdicts. At the two never-trained launderings the gap is large for DINOv2/CLIPs (0.20-0.30) but small for CommFor (screenshot_sim +0.059 [0.034, 0.085], mavali_medium +0.078 [0.049, 0.106]).
- The inner-CV choice over ALL blocks never left the mid range for any backbone (blocks chosen: DINOv2 5-6, CLIP-OpenAI 12-14, CLIP-LAION 11-14, B-Free 6, CommFor 6), so nested_all = nested_mid.
- **R4:** DINOv2 block 6 bacc 0.691 / 0.778 / 0.832 / 0.853 / 0.861 for n = 15 / 30 / 60 / 90 / all (102); fit b = 0.661 [0.394, 1.081], c = 0.074 [0.000, 0.128]; plateau (CI of b includes 0): NO. CLIP-OpenAI block 13: 0.744 / 0.829 / 0.872 / 0.895 / 0.898; b = 0.808 [0.483, 1.284], c = 0.061 [0.000, 0.100]; plateau: NO. EXTRAPOLATED bacc (not measured; 10x and 100x beyond the data; the CI covers sampling noise only, not misspecification of the power law): DINOv2 n=1,000 0.911 [0.868, 0.954], n=10,000 0.923 [0.872, 0.984]; CLIP-OpenAI n=1,000 0.932 [0.898, 0.972], n=10,000 0.938 [0.900, 0.991]. The data still rise between n = 60 and 90 (+0.02 to +0.03) and flatten between 90 and all (+0.003 to +0.008).

## Gates (all passed; tables.md "Validity gates")
- Row/fold identity with 020; sha256 dedupe (0 duplicates); no held-out laundering in training; `assert_disjoint_groups` inside every task (planted-group-leak selftest proves it trips).
- D-008 per backbone, tolerance max(1e-3, 2 x CPU run-to-run spread): scaled GPU-vs-CPU max diff DINOv2 4.9e-5, CLIP-OpenAI 9.4e-5, CLIP-LAION 8.5e-5, B-Free 4.6e-5, CommFor 4.0e-5; CPU spread (two runs, 8 vs 3 threads, batch 4 vs 5) 4e-6 to 2.3e-5, so the tolerance stayed 1e-3. Metric: |dF| / max(1, column max over the 32 images) over all conditions, blocks, CLS/mean parts (the frozen backbones have no logit).
- Hook placement: the model's own output equals the output rebuilt from the hooked last block (diff 0.0 for all five).
- Permutation control (200 shuffles, middle block, fixed C = 0.01): means 0.124-0.127, p95 <= 0.146 for all five.
- DINOv2 parity: the 020 probe harness on the CACHED blocks 4/8/12 reproduces 020 results.json exactly; fresh GPU features differ from the cache by <= 7.9e-6 (scaled); the probe on fresh features differs from 020 by <= 0.0023 bacc.

## Deviations and interpretations (all fixed before results)
1. B-Free at a 224 crop: its own 5-window wrapper replicate-wraps the 16x16 token grid up to its 36x36 window (the five windows coincide), so the 224 crop is tiled about 2.25 times. Mean-patch features average over those tiled tokens. A native-resolution B-Free pass (as in 023) was not run. B-Free's 0.839 should therefore be read with this caveat.
2. CommFor uses its native Resize(440) -> CenterCrop(384) (position embeddings are 384 px), so jpeg75_resize05 images (256 px) are upsampled to 440 first; this is not a 224 crop.
3. Gate metric for D-008 is a scaled feature difference (no logits for frozen backbones); the tolerance rule is the one in docs/11.
4. Plateau test: the 2.5th percentile of b is <= 0.05 (b is bounded below by 0, so a flat curve gives a percentile near, not exactly at, 0). The first draft of the rule was <= 0.001; the selftest flat-curve case showed it was too strict, so it was changed before any real result (config.json and run.py).
5. Test-side fixes during the selftest (not result-dependent): the planted-group-leak test needed a memorisable fingerprint (one-hot group id, 4 rows per group); the subsample-subset test had a loop-variable bug; `Extractor.close()` added so GPU memory is freed.
6. R1 "best block" = nested inner-CV choice among mid blocks (not the test maximum); the oracle version is reported next to it.
7. The profile at relative depth uses block index / L (1-based), so the DINOv2 last block is depth 1.0 and the first block 0.08.

## Post-result code changes
None. (No logic change after the real run started; the first and only real run is the one reported.)

## Caveats
- n = 1,020 images, 542 caption families; the gap CIs are family-bootstrap CIs on a single 8-generator set (Synthbuster + sd3 + flux), a single laundering family for training; R1 shows the mid-over-last pattern is shared by five ViTs on this data, not that it holds on other generators or data.
- Fine-tuned detectors (B-Free, CommFor) were trained on real/fake detection, possibly on related generators (023 overlap flags); the claim "pretrained backbones" rests on the two frozen CLIPs.
- R5 (literature) and R2/R3 are not part of this run. The opus critic for the whole run is the caller's step.
