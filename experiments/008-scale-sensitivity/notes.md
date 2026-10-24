# 008-scale-sensitivity: E1 zoom sensitivity of published detectors (Amendment 7). PILOT, exploratory.

STATUS (2026-10-24): code complete and dry-run OK (20 images, GPU, D-008 and orig re-score guards passed in the dry run; dry outputs in cache/dry_out/, git-ignored).
The full run was stopped at the owner's request (no GPU work). NO RESULT EXISTS YET.

Cached so far (cache/full/, git-ignored): Laplacian of every detector input (lap_orig.npy, lap_fov512.npy, 2022 rows each, both conditions);
B0 (005 A0, grouped CV + balanced) for the orig arm, fresh fit with pickled models: max|d| vs 005/pilot_v2 cache = 0.0.
Not yet done: B0 fov512 fit (CPU, ~3 min), pretrained scoring of fov512 (GPU, ~1563 rows x 2 conds x 3 detectors), CPU reference, D-008 guards, analysis.

Resume (resumable, same command): env/.venv/bin/python -I experiments/008-scale-sensitivity/run.py > experiments/008-scale-sensitivity/pilot/run.log 2>&1

Design (all per Amendment 7; criteria unchanged):
- orig-arm pretrained scores come from 007/pilot/scores_pretrained.csv.gz, after re-scoring 007's 32-image check subset on the GPU (abort if |d| > 1e-3).
- fov512 arm scored with 007's score_rows/get_fn (own preprocessing per detector), D-008: CUDA vs CPU (batch 4) on the same 32 rows, abort > 1e-3.
- B0 via the code of 005/run.py (load_data, run_arm; --grouped-cv --balanced). Besides the retrained-per-arm B0 (spec), two extra rows b0_trained_orig / b0_trained_fov512 score ONE fixed probe on the other arm's features (a real cross-zoom deployment).
- Texture baseline: mean |Laplacian| of each detector's actual input (pre-normalisation tensor transform reproduced).

Deviations / judgment calls (to be restated in the final report):
1. Texture direction: one sign per (detector input, condition), set on the orig arm's TRAIN natives vs train fakes (calib has reals only), used in both arms (Amendment 8 clarifications).
2. 1%-FPR thresholds on 43 calib natives are the calib max + eps (floor(0.01*43)=0); a sensitivity variant uses all 129 calib reals.
3. Clopper-Pearson CIs treat images as independent (reals cluster by device); descriptive. B0 transfer CIs are approximate (mean over 6 LOGO models).
4. Pretrained detectors are scored only on calib+test rows.
5. Per code_review.md / Amendment 7-8 clarifications: b0 transfer has within-arm rows only; the B0 fov512 reproduction vs 005/pilot_v2 is asserted (<= 1e-6); the 129-calib variant and b0_trained_* rows are flagged 'not pre-registered' in tables.md/results.json.
6. E1 evaluates all detectors on the 60 test fakes per generator, so B0's AUROC will differ slightly from 005's tables (which use 120 fakes per generator).

## E1b (Amendment 10): du2 arm, zoom vs resampling. PILOT, exploratory. Outputs: pilot_du2/
- Crops: `fov_prep.py --mode du2` (new; default modes verified byte-identical to the existing fov512/crop512 PNGs on 20 rows; `lib/preprocess.process_image(mode="du2")`, test `lib/tests/test_du2.py`). Splits `004/pilot/splits_pilot_du2.csv`, report `du2_prep_report.json`. Features: 004 `run.py --splits pilot/splits_pilot_du2.csv --cache-subdir pilot_du2 --conditions clean jpeg75_resize05` (guard in cache/pilot_du2/d008_guard.json, log pilot/extract_du2.log).
- Script `run_du2.py` (reuses run.py/E1 caches: orig = 007 scores, fov512 = E1 scores, B0 models from cache/full). D-008: du2 CUDA vs CPU on 007's 32-row subset for every detector x condition, abort > 1e-3; the orig/fov512 scores were guarded in E1 and not re-scored.
- Run: `env/.venv/bin/python -I experiments/008-scale-sensitivity/run_du2.py > experiments/008-scale-sensitivity/pilot_du2/run.log 2>&1` (resumable).
- Labels follow Amendment 10 (only rows E1 flagged scale-sensitive in that condition). Rows beyond it: b0_trained_du2, texture rows (descriptive).
- Cross-scale FPR CIs: joint bootstrap over calib-native devices (10), test devices (25) and test fake images; rules `max` (calib max + eps) and `second` (second-highest + eps). No Clopper-Pearson here.
