# Critique: 007 PILOT (H2 worst-env conformal). Critic: opus/high, 2026-10-23

**Verdict: the pilot result is uninformative for H2.** Code and FPR numbers are valid, but with 42–45 calibration reals per device, M3 is exactly the pooled calibration maximum, so the worst-group mechanism was never exercised. The TPR ratio (A0: 0.59) only compares a 1% operating point with one at ≈0.1%.

## 1. Code audit (analyze.py): no verdict-changing bug
- **Groups:** `device`, pooled over native/WA/FBH; 25 test and 10 disjoint calib devices asserted (:83-85); 42–45 reals per device (critic's check).
- **LODO:** no leak (`cal = T[T.device != d]`, :105). M1 uses only the calib devices (:98).
- **LOPO leak:** real but small.
  - Native/WA/FBH are calibrated on other versions of the same photos (:126).
  - The synthetic platforms are calibrated on the very natives that are then transformed (:131-133).
  - Critic's LODO×LOPO re-run (A0 `full`): only jpeg75_resize05 M3 changes (0.00% → 0.28%); WA stays M2 3.07%, M3 1.96%.
- **Conformal index:** correct. It takes the ⌈(n+1)(1−α)⌉-th order statistic, returns +inf if that exceeds n, and flags with strict `>` (:52-57).
- **M3 degeneracy:** `np.quantile(..., 0.99, method="higher")` (:64) returns the maximum for every n ≤ 100.
  - Critic's check: A0 `full` M3 = 8.8725 = pooled max = `thr_conformal(s, 0.001)`.
  - A true per-group (Mondrian) conformal threshold is +inf for n < 99. So H2 at α = 1% is not computable on VISION device groups.
- **Strict vs non-strict:** immaterial.
  - RINE has a point mass at 0.567941: 12 images, including 2 test reals and 1 calib real. This explains M1 = M2.
  - `>=` would move FPR by ≤2/1074.
- **TPR protocol:** valid.
  - The `logo_<g>` real scores set the thresholds (:218), and that model never saw g (005 run.py:150).
  - The test devices are disjoint from `b0_train`.
- **Verdict logic:** matches plan:93-98. Harmless deviations: "native" added as a 5th LOPO platform; ratio of macro means (:224-225; mean of per-generator ratios 0.577 / 0.587).
- notes.md is stale (says A0 not analysed).

## 2. Is H2 testable on VISION? Not as specified.
- **What the test measures.** It reduces to "pooled max vs pooled 1% quantile", i.e. the TPR cost of moving from 1% to ≈0.1% FPR. For A0 that cost is 41% (77.9 → 46.0%, tables_scores_long.md).
- **The evidence says the hedge only buys conservatism.**
  - Pooled calibration already transfers across devices: M2 LODO 1.02% [0.51, 1.83]; M1 0.37%.
  - M1 is partly device-draw luck: in LOPO-WA, M1 = 6.23 > M3 = 5.74 (`results_scores_long.json` A0.FPR.LOPO.WA.thresholds).
  - The only M1/M2 cell above 2% is WA: M2 3.07% (11/358) vs M3 1.96% (7/358), a difference of 4 images.
  - M4 (pooled conformal at α/2) reaches LODO 0.47% and WA 2.79% while keeping TPR ratios of 0.88 and 0.90.
- **Fix, pre-registered in spirit:** a pilot-informed amendment, logged before the official run.
  1. Use groups with n ≥ 100, with per-group split-conformal thresholds:
     - primary grouping: platform/channel (358+ reals each);
     - secondary grouping: brand/model, merging groups up to at least 100.
  2. Make the decisive test a frontier comparison: held-out pooled and worst-device FPR vs TPR, against pooled conformal over an α sweep. Worst-group must beat that frontier; otherwise it is just a smaller α.

## 3. The FPR ≤1% target
- **The default threshold is broken, but the pilot exaggerates this.**
  - B0 is trained on 99 reals vs 360 fakes with no class weighting (probe.py:21). This offsets logit 0 by about ln(360/99) ≈ 1.3 (critic's estimate).
  - The 29.6% WA FPR also carries the field-of-view (FOV) confound: crops are taken without resizing (005 critique §2).
  - Still broken for pretrained detectors: B-Free 17.3%, ClipDet 10.1% (003 notes.md:15).
- **Real-only calibration fixes cross-device FPR** on 2015–17 VISION phones: M2 is 1.02–1.12% for all four detectors.
- **What remains open:** real shift beyond VISION.
  - Neural-ISP phones (H3).
  - Distribution channels such as WhatsApp.
  - Other sources: Phase 3 B0 had 9–12% FPR on RAISE at a calibrated threshold (003 notes.md:20).

## 4. Laundering mostly moves the reals
- **Critic's medians (A0 `logo_<g>`):** under jpeg75_resize05, real logits rise by +2.5 to +5.0 (e.g. −8.02 → −2.99). Fake medians move by −0.8 to +1.2.
- **The TPR "gain" is bought with FPR.** At the clean threshold, TPR goes from 77.9% to 82.5% under laundering (M2). Recalibrated at 1% FPR on laundered natives, macro TPR is 0.740, below the clean 0.779 (critic's computation).
- Laundering erases a "real" cue (format/FOV), consistent with a shortcut.
- **Deployment implications:**
  - Calibrate on reals passed through the expected channel, or on the worst channel.
  - Always report TPR at matched FPR for each condition.

## 5. Recommendations
1. **Amend H2 before the official run.**
   - Use n ≥ 100 groups (channel first, then brand/model), Mondrian split-conformal thresholds, and the α-sweep frontier test.
   - The "problem is real" clause should require M1 or M2 > 2%, not M0.
   - Evaluate with LODO×LOPO and a device-cluster bootstrap.
2. **Reprioritise Phase 4.** The H1 and H2 pilots both show that, within VISION, neither ranking nor cross-device thresholds is the bottleneck.
   - First: H4, the FOV fix (resize before crop) and a class-balanced probe.
   - Then re-run 007 (seconds).
3. **Get modern-phone data now.** Ask the owner for the ~250 modern-phone photos (H3), and keep RAISE in the real pool; that is the only real test of FPR ≤1%.
   - Reframe H2 as protocol plus recipe, not a method.
