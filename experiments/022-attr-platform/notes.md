# 022-attr-platform: H-A3, platform history on content-identical VISION triplets

Frozen by docs/08 (H-A3, sec. 3) + docs/09. Code: run.py, selftest.py (42/42 pass, results/selftest.txt), config.json. Results: results/results.json, results/tables.md, results/run.log. Runtime: 5098 s (CPU, 14 workers). Not committed (lead commits).

## Result
- **Label: INCONCLUSIVE** (primary = T-mix training, per docs/08 "Common definitions"; the clean-train sensitivity label is also INCONCLUSIVE). `counts_as_failure` = false under the rule fixed in config.json before the run (INCONCLUSIVE with max(A_clean,B_clean) < 0.80 would count; A clean is 0.929). The lead may judge otherwise: social_sim point estimate 0.523 is below the 0.60 bar.
- Method A (native-grid DCT, 3-way bacc, device-cluster 95% CI), T-mix: clean 0.929 [0.900, 0.955]; social_sim 0.523 [0.490, 0.554]; jpeg75_resize05 0.584 [0.554, 0.612]. Clean-train: 0.995 / 0.335 / 0.333 (no transfer to laundering).
- Method B (DINOv2 fov512, block 8 chosen in 5/5 folds), T-mix: 0.610 [0.573, 0.645] / 0.425 [0.402, 0.448] / 0.399 [0.381, 0.418]. Clean-train: 0.621 / 0.385 / 0.351. B is KILL-side on clean (< 0.50 not reached, but < 0.80).
- M0 (file size + quantisation table, raw files, clean): 0.985 [0.964, 0.999] (not exactly 1.0: some native files carry the same table as FB copies). M0 on preprocessed tensors: 0.333.
- Per-device and confusion: results/tables.md. A's clean errors are native<->FB; WhatsApp is almost perfect (484/485). After social_sim all three classes blur (recall 0.61 / 0.49 / 0.47 for native / WA / FB).
- Exploratory camera-model arms (no label): VISION same-model pair transfer, mean AUROC 0.747 [0.679, 0.810] over 14 ordered pairs (CI over pairs only), control 0.501; weak for iPhone 5 (0.45, 0.54), good for 5c and S3 Mini (0.77 to 0.89). HDR+ 6-model ID (43 burst groups): b4 bacc 0.200 [0.176, 0.335], chance 0.167, group-shuffle permutation mean 0.133 (2.5-97.5%: 0.097 to 0.170, max 0.188); b8 0.173, b12 0.114. Weak signal at best; rare models (Nexus 5X, one burst group) are never predicted.

## Gate checklist (docs/08 sec. 3)
1. Identical preprocessing: [x] decode raw JPEG, no EXIF transpose (the as-stored grid is the object; no metadata read for A), RGB; laundering via experiments/lib/perturb.apply_condition, same call for all classes, key = raw-file sha256, seed 0 (same as 004 load_cond). B reads the 004 fov512 caches (short side 512, centre crop, same for all classes). M0 is the only method that reads headers.
2. Baselines: [x] M0 shown. [x] M0 on preprocessed tensors at chance (0.333). [ ] M1 texture NOT computed here (task named M0 only; M1 lives in 020), so the "CI lower bound beats M1 by 0.10" test is not applied; A's clean value is far above any plausible texture baseline but this is unverified.
3. Grouped splits: [x] 5-fold GroupKFold by device (folds in results.json); all copies of a photo and all conditions of a photo share a fold; assert_no_group_overlap in code; real-data check in selftest. D13 excluded (485 triplets, 1455 images, 34 devices).
4. Laundering: [x] clean, social_sim, jpeg75_resize05; both clean-train transfer and T-mix reported. screenshot_sim and mavali_medium were not requested here (not run).
5. Label-leak checklist: [x] no device in train and test; block (B) and C chosen by inner device-grouped CV on training folds only; [x] permutation control, 200 within-triplet shuffles, mean 0.332 to 0.334 for A and B in all 3 conditions (gate |mean-1/3| <= 0.02 passed; real p = 0.005, the minimum for 200 shuffles); C fixed to the real run's median per method for speed (a deviation from a full-pipeline shuffle); [x] sha256 unique over 1455 raw files, recomputed from disk; [x] D-008: no new GPU pass (B uses only the existing 004 caches, whose guard is 004 cache/fov512/d008_guard.json, max |dlogit| about 1e-5); A is CPU numpy.
6. Chance levels: [x] 0.333 (3-way; majority rate 0.333 since classes are balanced), 0.5 AUROC, 1/6 and majority rate 0.377 for HDR+.

## Deviations and caveats (none changed after the real results)
- Laundering differs by method by construction: A launders the native-resolution decoded image; B uses the cached features where laundering was applied to the 512 px crop (so B's laundering is milder in absolute pixel terms but harsher relative to the grid; the two columns are not directly comparable).
- Primary training mode = T-mix (my reading of docs/08 "Training = T-mix"); fixed in config.json before running.
- Same-model devices can fall into different folds (e.g. iPhone 5c D05/D14/D18), so "unseen device" does not mean "unseen model". Platform identification from A may read output scale (WA 1280 px, FB 2048 px, natives 2560 to 3968 px) through block statistics; no size-controlled variant was run.
- The A periodicity estimator recovers the true luminance steps on 36 to 53% of decoded-JPEG coefficients (selftest, informational); the histograms carry most of the signal.
- VISION model-transfer CI covers pair variation only; HDR+ permutation shuffles whole burst-group labels.
- Selftest history: the first two selftest runs failed one test (planted-leak design: a within-triplet-shuffled device fingerprint cannot leak; then a 72-photo chance fluctuation). Only selftest.py was changed (photo-level random labels, averaged over 6 draws) before the real run; run.py was not changed. No logic change was made after the real results.
- The agent session hit an API rate limit while the run was in progress; the run process itself finished (log "done in 5098s"); results.json holds all sections.
