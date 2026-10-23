# 005 nuisance projection (H1): PILOT notes

Status: PILOT (Amendment 2 of docs/04-phase4-plan.md). 6 Synthbuster generators, VISION-only reals. Not a Phase 4 verdict; does not count toward the 3-failed-hypotheses rule. Pass/fail rule is the pre-registered one, unchanged.

Run: `env/.venv/bin/python -I experiments/005-nuisance-projection/run.py` (resumable; per-arm score caches in `pilot/cache/`, git head and timings in `pilot/results.json`, ~32 min total). Outputs: `pilot/{results.json,tables.md,scores_long.csv.gz,run.log}`.

## Design
- Features: 004 pilot cache, primary 1536-d (= B0) and secondary 4608-d. z = (f - mu)/max(sigma, 1e-6), mu/sigma from clean features of the 99 `b0_train` native reals.
- Delta rows (n = 1089): z(WA) - z(native) and z(FB-HQ) - z(native) for the 99 train-device triplets matched by (device, vidx) (all 99 triplets complete, asserted), plus z(jr cell) - z(clean) for the 99 train natives x 9 cells. Reals only.
- M = (1/n) D^T D (uncentered). Centered variant (A2c) reported as sensitivity. k* = smallest k with >= 80% of trace(M).
- Arms A0..A3, A2_k{16,64,256}, A2c, A4 (DRIFT-lite), A1_sec, A2_sec as specified. Probe = `probe.fit_probe` (seed 0, C by 5-fold CV).
- LOGO: per g, train on `b0_train` rows minus g's fakes (plus the 9 grid copies of the same rows for augmented arms); test = 358 test-split VISION natives vs all 120 fakes of g. Metrics via `metrics.compute_metrics`, macro = mean over 6 g.
- Primary bootstrap: 1000 reps, seed 0; one set of resampled real indices (shared by all g) and one per-g fake index matrix, reused for every arm and condition (paired, stratified).
- Drift: "full" model per arm, threshold = `threshold_for_fpr(clean calib natives (43), 0.05)`, FPR on clean test natives/WA/FB-HQ (358 each).

## Judgment calls
1. mu/sigma from the 99 train natives only (not the WA/FB train reals), because "clean" reals = natives.
2. Probe training reals = 99 natives only (b0_train), as in B0; WA/FB train reals are used only for the projector's Delta, not in the probe.
3. Augmented arms augment only `b0_train` rows (99 reals + 360 fakes): 459 x 10 = 4590 probe rows.
4. A4: only the stated architecture/loss/300 epochs/Adam 1e-3 were used; unstated details were fixed before any detection metric was seen: pair mini-batch 256 shuffled each epoch, variance/covariance terms computed on all 99 unique clean train natives each step (not on the repeated a-side), std = sqrt(var + 1e-4), cov = off-diagonal squared sum / d. Single run, no tuning. NOTE: the covariance term stayed at ~1.7 and the invariance term ~2e-4 at the end (`pilot/cache/a4_hist.json`): with only 99 unique clean reals the 256-d covariance has rank <= 98, so the off-diagonal penalty cannot be satisfied and the embedding is dominated by decorrelation. A4's poor result (LOGO-macro 0.83 clean) therefore reflects this pilot's tiny real set plus untuned weights, NOT a fair test of DRIFT. Not tuned afterwards, per the pre-registration; comparison only.
5. Secondary-feature arms (A1_sec, A2_sec) recompute mu/sigma, the eigendecomposition and k* on the 4608-d features. Their bootstrap is A2_sec - A1_sec; A1_sec - A1 in the bootstrap table is a feature-set comparison.
6. Verdict logic: PASS if all four criteria hold; FAIL if gain < +0.01 or clean loss > 0.02; else INCONCLUSIVE.
7. Test-time scores of train-split rows are in scores_long.csv.gz too (they were fit on); use the sha256 + split from splits_pilot.csv when joining.

## Results (all from pilot/results.json, pilot/tables.md)
- k* = 90 (primary; centered 92); 80% energy at k=90, 50% at 32, 90% at 152, 99% at 474 of 1536 dims. Secondary: k* = 80 (centered 90). The mean shift is 3.1% of trace(M) (primary), 13.7% (secondary).
- Primary endpoint A2 - A1 at jpeg75_resize05: -0.0072, 95% CI [-0.0104, -0.0044] (A1 0.988, A2 0.981). Negative, CI excludes 0.
- Clean A2 - A1 = -0.0085 (holds, but the CI excludes 0); screenshot_sim A2 - A1 = -0.0061 (fails); mean threshold drift A1 0.038 -> A2 0.022 (41% reduction, below 50%).
- PILOT_verdict = FAIL (gain < +0.01).
- Projection costs TPR@1%FPR a lot (clean A1 0.829, A2 0.761) and AUROC falls monotonically with k (k16 -0.003, k64 -0.006, k256 -0.019 vs A1). The nuisance subspace overlaps the fake-signal subspace, the falsification risk named in the KB idea.
- A3 (projection, clean only) is worse than A0 on every AUROC condition except drift (mean drift 0.010 vs A0 0.187); A1 alone already cuts drift from 0.187 to 0.038.
- Secondary features: A1_sec is far stronger (0.999 at jpeg75_resize05); A2_sec - A1_sec = -0.003 at jpeg75_resize05, -0.011 clean. Drift A1_sec 0.148 (WA FPR 0.31) vs A2_sec 0.014: the projection helps drift there, at an AUROC cost.
- Caveat: AUROC ceiling. All arms are 0.97-1.0, leaving little room for a +0.03 gain; a possible partial explanation [unverified] is the format shortcut (JPEG VISION reals vs lossless PNG fakes, H4), which makes these pilot numbers optimistic.

## Sanity check
A0 LOGO-macro clean AUROC = 0.9917 vs 003 B0 LOGO clean 0.916. Higher than 003 as expected: reals are VISION natives only (no RAISE), 6 generators only (no SD3.5), and the JPEG-vs-PNG format shortcut is uncontrolled. Plausible for this pilot but not comparable; the official run on full data must be rechecked.


## Amendment-3 options and the pilot_v2 runs (2026-10-23; PILOT, no verdict)
New `run.py` options (defaults reproduce the original pilot byte for byte: checked results.json / tables.md / scores_long.csv.gz against `pilot/`, and a fresh A0 refit equals `pilot/cache/scores_A0.npz`): `--subdir`, `--splits`, `--out-dir`, `--label`, `--arms`, `--generators`, `--heldout-generators`, `--grouped-cv` (StratifiedGroupKFold, group = source image), `--balanced`, `--group-bootstrap` (reals by device, RAISE per image; fakes by image stratified per generator, one plan shared by all arms), `--per-pool`, `--headroom-rule` (+0.03 test only if A1 LOGO-macro jpeg75_resize05 <= 0.96, else `UNINFORMATIVE_NO_HEADROOM`), `--primary-pools`. With held-out generators (FLUX) the gain criterion uses the minimum over endpoints (LOGO-macro and each held-out generator); this is my reading of "both FLUX and LOGO-macro", please confirm. RAISE train reals (official run) enter the standardization and the grid deltas only; platform pairs stay native-only. A score cache is tied to its options (`cache/opts.json`).
pilot_v2 = A0/A1 only, `--grouped-cv --balanced --group-bootstrap --per-pool --headroom-rule`, on the original pilot features (`pilot_v2/orig`) and the fov512 features (`pilot_v2/fov512`). Numbers: `pilot_v2/*/results.json` and `tables.md` (key `per_pool`).
