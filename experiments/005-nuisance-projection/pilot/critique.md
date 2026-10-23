# Critique: 005 PILOT (H1 nuisance projection).

**Verdict: Pilot FAIL is uninformative.** The code is correct, but a PASS was arithmetically impossible: A1 = 0.988 at `jpeg75_resize05` (tables.md), so the best possible gain is +0.012, below the +0.03 bar. The negative direction (−0.0072, CI [−0.0104, −0.0044], `results.json:primary_endpoint`) is real but small and confounded.

## 1. Code audit (run.py): no verdict-changing bug
- **Leakage: none.**
  - Δ uses only the 99 train triplets, asserted equal to the `b0_train` reals (run.py:60-64, 71).
  - Grid features exist only for train rows (p4features.py:55-56).
  - Each LOGO model drops g's fakes from its clean and augmented rows (run.py:151-158). It is tested on all 120 g fakes vs the 358 test natives (run.py:177-181).
  - Calib natives set only the drift threshold (run.py:216).
- **Pairing is correct:** keyed by (device, vidx), triplets asserted complete (run.py:57-62).
- **Bootstrap is paired:** one `ridx`/`fidx` set for every arm (run.py:295-307). It resamples images, not the 25 devices, so the CIs are too narrow.
- **Projection then StandardScaler: harmless.** The effective z-space weight is P S w, which stays orthogonal to U_k. Only the L2 geometry changes.
- **Deviation:** M is uncentered (run.py:247) while the plan says "covariance" (plan:40). The centered A2c also fails (−0.009), so this does not matter.
- **Conditions, drift and verdict logic** match the plan (run.py:214-226, 313-326; plan:47-52). Drift is a mean over WA and FB. A per-component reading fails too: FB drift goes 0.022 → 0.025.
- **Minor issue (both arms):** `fit_probe` CV puts augmented copies of one image into different folds (probe.py:21). This biases C: A1/A2 chose C = 0.01, A0 chose C = 10 (run.log). Use GroupKFold by source image.

## 2. Sensitivity: the pilot is near ceiling
- Every arm except A4 scores 0.970–0.999. Phase 3 B0 LOGO laundered was 0.861 (003 notes.md:24).
- **Cause 1: no RAISE.** Phase 3 B0 scored 0.962 vs VISION but 0.876 vs RAISE (003 tables.md:51).
- **Cause 2: a field-of-view confound.** Measured by the critic with PIL on 40 test rows per pool; not in a results file.
  - Median raw short side: native 2448 px, WA 960, FB-HQ 1536, fakes 800–1024.
  - Preprocessing is a center crop with no resize (preprocess.py:52-54), then a 224 crop (004 run.py:63). Native crops are zoomed in about 2.5–3× compared with fakes.
  - It also explains A0 WA FPR 0.307.
- **Format alone is not the driver.**
  - With `jpeg(90)` on both classes, A0 moves only 0.992 → 0.991.
  - Firefly (a JPEG fake) scores 0.978 clean for A0, close to the others.
  - Scene and content differences are the likely residual cause `[unverified]`.
- **It can only show "hurts" vs "≈0"; it shows "hurts slightly".**
  - This holds across all 8 conditions and both feature sets.
  - TPR@1%FPR falls 0.794 → 0.722, but 1% FPR here is about 4 reals, so this metric is noisy.

## 3. Do the laundering and fake subspaces overlap?
- **Evidence that they do:**
  - Monotone k-sweep at `jpeg75_resize05`: k16 −0.002, k64 −0.005, k90 −0.007, k256 −0.018.
  - A3 < A0 with no augmentation: 0.975 vs 0.985 laundered, 0.982 vs 0.992 clean.
  - So the real-only Δ span holds directions the probe uses.
- **What it does not show:** whether those directions carry generator traces or the pilot's FOV/scene shortcut.
  - Platform Δ rows, which are dominated by FOV change, carry the most energy: 1224 vs 747 per row for the grid (`info.energy_pairs_check`).
  - On test WA reals, whose FOV is close to the fakes', the cost disappears: A2 0.972 vs A1 0.971 laundered. Critic's computation from `cache/*.npz`.
- **Projection does cut drift:** A3 0.010 vs A0 0.187; A2_sec 0.014 vs A1_sec 0.148.
  - But A1 vs A2 (0.038 vs 0.022) is within binomial noise (SE ≈ 0.012 at n = 358, my calculation). The 41%-vs-50% gap is not meaningful.

## 4. A1_sec at 0.999 / 0.982: a shortcut, but probably not a format one
- **Against a format shortcut:**
  - `jpeg(90)` on both classes gives 0.998.
  - Firefly reaches 0.994 clean and 0.998 laundered.
  - WA reals vs fakes still score 0.983 clean.
- **Likely cause (unproven):** a content shortcut, exploited more by multi-layer features.
- **Cheap tests with the existing cache:**
  - (a) Real-pool swap: AUROC with natives vs WA vs FB-HQ as the real pool.
  - (b) Format direction: fit a probe on train fakes, `clean` vs `jr(90,1.0)` (same images with and without JPEG). Report its test AUROC and its cosine with A1_sec's weights.
  - (c) Mixed FOV: reals at `resize(0.5)` vs fakes at `clean`. Quick run: A1_sec 0.993 (critic).
- **The decisive test needs RAISE** (H4 settings ii–iv).

## 5. Recommendations
1. **Do not retire H1, and do not count this pilot** (Amendment 2). Run the official version.
   - Before looking, pre-register:
     - a headroom check: the +0.03 test is valid only if A1 ≤ 0.96;
     - RAISE in the real pool, with a per-pool breakdown;
     - GroupKFold, and device-cluster bootstrap CIs.
   - My expectation is FAIL or INCONCLUSIVE `[unverified]`.
2. **Move H4 earlier**, and add a FOV-matched preprocessing check: downscale reals to about 1024 px short side before the 512 crop. This confound affects all Phase 4 numbers, including H2's WA FPR.
3. **Reframe H1's value as FPR stability, not AUROC.** Feed A1_sec and A2_sec into H2.
   - Promote A1_sec (multi-layer + augmentation) to baseline only after it is confirmed against RAISE.
   - Its +0.010 over A1 is measured at ceiling, and its WA drift (0.279) is worse than A1's.
