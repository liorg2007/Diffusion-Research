# 020-attr-foundation: M0, M1, H-A1, H-A4 (D-018 cycle)

Code: run.py, selftest.py (59/59 checks, results/selftest.txt), tables.py, config.json (frozen before any attribution score was read). Results: results/results.json, results/tables.md. No new GPU features (cached DINOv2-base, 004/016/019 caches). Runtime: prep 3 min, closed 3 min, report 2.5 min; logo/ha4/perm stages ran under the shared CPU (about 1 h wall). CPU only, 13 workers.

## Labels (pre-registered criteria, docs/08 + docs/09)
- **H-A1: PASS.** jpeg75_resize05, T-mix, inner-CV nested feature set: bacc 0.855 [0.834, 0.876] (chance 0.125; CI lower bound is 0.43 above M1 0.425); pairwise 1.000 [1.000, 1.000] (macro over generator pairs); RAID 4-way bacc 0.688 [0.650, 0.725] at jpeg(75) (chance 0.25; M1 0.477, M0 0.478).
- **Open-set usable: NO.** -max softmax mean LOGO AUROC 0.768, min 0.703 (mean < 0.80); Mahalanobis mean 0.546, min 0.464.
- **H-A4: KILL.** Mahalanobis b8 T-mix mean novelty AUROC 0.531 (min 0.386, HDRP). **DISCOVERY: NO** (min kind accuracy over the 8 real sources 0.733, UNS).
- All validity gates passed (tables.md); the labels are therefore not INVALID.

## Caveats the reader needs
- M0 (raw metadata) already reaches bacc 0.688 [0.665, 0.710] on the 8 generators (RAID 0.478): much of "attribution" is free from file metadata. Under preprocessing M0 is at chance (gate).
- Clean-train transfer collapses: nested bacc 0.542 at jpeg75_resize05 (T-mix 0.855); layer contrast b8 - b12 +0.254 [+0.221, +0.287] (T-mix). Held-out launderings (T-mix): clean 0.913, social_sim 0.872, screenshot_sim 0.866, mavali_medium 0.813. Without firefly: 0.855.
- Weakest classes: midjourney-v5 recall 0.667, SDXL 0.783. Pairwise 1.0 means caption-matched pairs are separable, the 2-AFC metric is easy (strict "both" reading: 0.988).
- H-A4 novelty is at chance because known real sources cover the space (VISION nat/WA/FBH are near each other; kind accuracy for VISION 0.998 is trivially due to the sibling pools). Closed-set 16-way is high (b8 T-mix 0.695, kind accuracy 0.95) but unseen sources are not flagged.
- n is small (about 24 test images per generator per fold); RAID gen_k mapping is [unverified].

## Interpretations / deviations (all fixed before results; none changes docs/08 criteria)
1. Pairwise = 2-AFC on log-probabilities, macro over generator pairs (strict reading also reported).
2. RAID primary condition = jpeg(75) (the only JPEG-only conditions exist at fov224).
3. "CI lower bound >= M1 + 0.10" read literally (lo of method vs M1 point estimate); paired difference also reported.
4. H-A4 primary: Mahalanobis, b8 fixed a priori; b4/b12 with Holm; concat omitted (cost). kNN-10 reported. Own grouped 5-fold cross-fitting (flux has no train-role rows, so E6 roles are not used).
5. Open-set usable = either score meets the rule (both flags in results).
6. Permutation control for H-A4 uses a 128-d random projection of b8 (cost); H-A1 uses b8 with C=0.01, T-mix.
7. M1 label model = logistic; gradient boosting (HGB) is sensitivity only. M0 = random forest on raw metadata; HDR+ raw originals are deleted, so M0 uses manifest size/bytes and no quantisation table for 600 HDR+ rows.
8. **Dedupe finding (gate 5):** 6 D13 (iPad 2) WhatsApp copies have pixel-identical processed crops to the native copy (the 960 px images are never resized). The later duplicate was dropped (6 of 5319 rows) before folds were committed; kept set re-checked (0 duplicates). Affects H-A4 only.
9. Folds are committed in results/folds_*.json (sha256 in results/folds_commit.json) before any feature is read.

## Gate checklist (docs/08 section 3)
- [x] 1 Identical preprocessing (same fov crop dir, no EXIF/ICC, RGB, one perturb.py code path, true format from content, cache meta equal)
- [x] 2 M0 and M1 shown; M0 on preprocessed tensors at chance (<= chance + 0.04/0.06); method beats M1 by the margin (H-A1 criterion)
- [x] 3 Grouped splits (caption family, RAID id, device/image/group), pairwise on matched families
- [x] 4 Laundering: clean, jpeg75_resize05, social_sim in training; screenshot_sim and mavali_medium never trained on; clean-train and T-mix both reported (no thresholds used)
- [x] 5 Leak checklist: assert_disjoint_groups in every task; inner-CV layer choice + Holm; 200-shuffle permutation controls (H-A1 main 0.125, RAID 0.250, H-A4 0.500); sha256 dedupe; D-008 verified on existing guards (max |dlogit| 1.4e-4, no new GPU pass)
- [x] 6 Chance levels in every table

## Bug log (code changes after the first full run; no criteria or logic changes)
- selftest: boot_conf used float32, point estimate differed from sklearn by 1e-7. Changed to float64. Selftest synthetic separation raised (0.08 -> 0.2). Re-run: 59/59.
- build_analyses: read-only mask array (`.copy()` added). m1 npz shas saved as object dtype (now `dtype=str`).
- report: results dict lacked `drop_macro` (tables.py KeyError after results.json was written); added the key and re-ran the report stage on the cached task outputs. Numbers unchanged.
