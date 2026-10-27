# H-A3 critique (opus/high critic, 2026-10-27)

Scope: docs/08 H-A3 + §3 gates, docs/09 (frozen), config.json, run.py, notes.md, results/{tables.md, results.json, selftest.txt}.
New critic checks: critic/feats.py and critic/probe.py, with output in critic/probe_results.json and critic/probe.log.
They reuse run.py's table, folds, probe (T-mix, inner device-CV for C) and device-cluster bootstrap. The probes ran single-threaded BLAS with 4 workers.

## 1. Label verdict: INCONCLUSIVE is correct, but the A-clean number behind it fails gate 2
- Applied as written, the docs/08 rule gives INCONCLUSIVE:
  - PASS needs A or B to reach clean ≥ 0.80 and social_sim ≥ 0.60. A has 0.929 / 0.523; B has 0.610 / 0.425.
  - KILL needs A < 0.60 and B < 0.50 on clean, and neither is true.
- **Gate 2 (M0 on preprocessed tensors at chance) was checked only on B's input.**
  - The notes tick "M0 on preprocessed tensors 0.333", but that used the fov512 constants (`m0_tensor`).
  - Method A's preprocessed tensor is the native-grid decoded image. Its size alone gives the platform:
    - WhatsApp: long side always 1280.
    - FB-HQ: always 2048.
    - Native: 2048 to 5312, median 3264. Only D08 has native = 2048.
  - A depth-6 tree on (w, h, aspect, long side), with the same folds, gives **bacc 0.990 [0.971, 1.000]** (critic/probe_results.json `dims_decoded_clean`).
  - Under docs/09 decision 4, a strict reading makes the A arm INVALID.
- This does not change the label:
  - B alone gives INCONCLUSIVE (clean 0.61, between the 0.50 KILL bar and the 0.80 PASS bar).
  - A gate-compliant variant of A also gives INCONCLUSIVE (next row).
- **A at a common scale** (critic arm, not pre-registered, descriptive): every copy is resized so its long side is 1024 (perturb.resize, bicubic, antialiased), then A runs unchanged.
  - Train mode is T-mix. Columns are clean / social_sim / jpeg75_resize05:
    - **0.690 [0.646, 0.728] / 0.365 [0.341, 0.390] / 0.366 [0.344, 0.388]**.
  - With clean-train: 0.720 / 0.335 / 0.335.
- No route reaches PASS: native-grid A, B, or common-scale A. The social_sim bar fails in every arm, and every CI upper bound is ≤ 0.554, below 0.60.

## 2. M1 (docs/08 §3) is now computed
M1 = the 020 `texture_stats` (12 stats, 224 centre crop), copied verbatim, run on the same input A sees, laundered the same way.

| M1 input | train | clean | social_sim | jpeg75_resize05 |
|---|---|---|---|---|
| native grid (A's input) | T-mix | 0.619 [0.584, 0.655] | 0.439 [0.423, 0.456] | 0.444 [0.422, 0.468] |
| native grid | clean | 0.718 [0.673, 0.757] | 0.412 | 0.416 |
| common 1024 | T-mix | 0.378 | 0.363 | 0.357 |

Gate "A's CI lower bound ≥ M1 + 0.10" (T-mix, native grid):

| condition | A lower bound | M1 + 0.10 | result |
|---|---|---|---|
| clean | 0.900 | 0.719 | passes |
| social_sim | 0.490 | 0.539 | **fails** |
| jpeg75_resize05 | 0.554 | 0.544 | passes by 0.01 |

- So A's laundered numbers do not clear the M1 bar in a useful way.
- M1 on the native grid sits well above chance (0.62 clean). M1 at common scale is near chance (0.38). That gap shows M1 itself reads scale.
- B's laundered numbers (lower bound ≤ 0.402) are below "M1 on B's input + 0.10" for any plausible M1. M1 on the fov512 crop was not computed. The common-1024 M1 of about 0.36 would give a bar of about 0.46.

## 3. Failure count (docs/09 decision 1): count it as a failure (recommended)
- docs/09 counts INCONCLUSIVE as a failure when "the point estimate [is] on the KILL side".
- The run fixed this before seeing results (config.json `label_rule`): failure = INCONCLUSIVE with max(A_clean, B_clean) < 0.80.
- It reported "not a failure" only because native-grid A reached 0.929 on clean. That number comes from an input whose dimensions alone classify at 0.99, which fails gate 2.
- On gate-valid inputs, max clean is 0.690 (common-scale A) or 0.610 (B):
  - The run's own pre-fixed rule then gives **counts_as_failure = true**.
  - B is also on the KILL side of the 0.50–0.80 midpoint (0.65).
- The distinctive claim of H-A3, "partly survives one more laundering", fails on every arm. All CIs are below 0.60, and common-scale A is at chance after laundering.
- If the lead instead accepts native-grid A as specified in docs/08, the literal result is "not a failure". The lead should log that choice in DECISIONS.
- My recommendation is to count it, since both halves of the hypothesis fail once the scale shortcut is removed.

## 4. What method A actually reads
- **Clean, native grid (0.929): the last JPEG quantisation table, read through the pixel grid.** In effect this is M0 recovered from pixels; M0 itself is 0.985 from headers.
  - WhatsApp has a single quantisation table (485/485, results.json `qt_hash_by_platform`) and A's WA recall is 484/485.
  - Native-vs-FB confusions match tables shared between some natives and FB copies.
  - It is not only image size: D08 natives are 2048 px, the same size as FB, and A still scores 0.98 on D08.
  - When the grid is broken by the common-scale resample, WA folds into native: confusion [[317, 146, 22], [158, 308, 19], [50, 56, 379]]. Only FB stays partly separable (recall 0.78), plausibly from FB's own downscale or sharpening [unverified].
- **Laundered, native grid (0.52 / 0.58): output scale, not compression history.** Three lines of evidence:
  1. At common scale, laundered A drops to chance (0.365 / 0.366).
  2. jpeg75_resize05 (0.584) beats social_sim (0.523) even though social_sim uses milder JPEG on average (q 70 to 100). jpeg75_resize05 keeps each platform's relative size exactly. social_sim adds a random ×0.7 to 1.0 resize per copy, which blurs the size cue.
  3. M1 on the native grid shows the same pattern (0.44 laundered vs 0.36 at common scale).
- This is the failure mode docs/08 pre-declared ("A's block statistics also read the output scale"). It is now measured: the scale cue accounts for the whole margin above chance after laundering.

## 5. Bugs, leaks and deviations
- **Wrong gate tick (major for reporting):** gate 2 M0-tensor is ticked for both methods but is valid for B only (§1). The notes.md checklist should say so.
- **M1 missing:** now computed (§2). A's social_sim fails the M1 + 0.10 bar.
- **A and B were laundered on different inputs** (native vs the 512 crop), as the notes disclose, so the A and B laundered columns are not comparable.
  - The common-1024 arm is closer to B's regime: clean 0.69 vs B 0.61; laundered 0.37 vs B 0.40 to 0.43. Both are near chance after laundering.
- **Gate 4:** screenshot_sim and mavali_medium were not run. §3 lists them as mandatory. They do not affect the label, but this is a deviation to log.
- **Permutation control:** it fixes C at the real run's median instead of shuffling the full pipeline. That is acceptable: the gap between permuted max (≤ 0.379) and real values is large, and it is disclosed.
- **No leak found:**
  - Device-grouped folds are asserted in code.
  - sha256 values are unique.
  - The laundering key is the raw-file sha (not label-dependent).
  - Selftest is 42/42, and the selftest edit was made before the real run.
- Minor: the "unseen device" splits do not hold out the model (iPhone 5c devices span folds). This is irrelevant to platform ID.
- Minor: the HDR+ arm, b4 0.200 [0.176, 0.335] against a permutation max of 0.188, is weak and correctly left without a label.

## 6. Interpretation: how much of the "real pipeline" in E7–E10 is platform processing?
- The DINOv2 fov512 features that E7–E10 use (B, block 8) carry weak platform information on clean images: 0.61 vs chance 0.33.
- Most of B's clean errors involve WA (WA→FB 115 and WA→native 116 of 485; FB→WA 137 and native→WA 106), so WA is the least distinct copy in DINOv2 space.
- After one more laundering, the platform signal is almost gone (0.40 to 0.43). A scale-equalised compression-history method is also at chance there (0.37).
- This agrees with 019/results/tables.md line 73: E10 FPR at jpeg75_resize05 per VISION pool is native 0.011, WA 0.011, FBH 0.008, with no platform dependence.
- **Conclusion:** at the operating conditions that matter (laundered), the "real" class the detectors learned is essentially platform-invariant. WhatsApp/Facebook history is neither a cue the detectors rely on nor a recoverable attribute after re-compression and resize.
- Platform history is recoverable only from first-generation files on their stored grid. That is the header and quantisation-table information M0 already gives, and the same holds from pixels while the grid is intact.
- Caveats:
  - These are 2010s VISION phones with 2017-era WA/FB pipelines; modern pipelines are untested.
  - The probes are linear on cached features, so a non-linear head could find more [unverified].
