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
1. Texture direction is set on TRAIN natives vs train fakes, because the calib split contains reals only.
2. 1%-FPR thresholds on 43 calib natives are the calib max + eps (floor(0.01*43)=0); a sensitivity variant uses all 129 calib reals.
3. Clopper-Pearson CIs treat images as independent (reals cluster by device); descriptive. B0 transfer CIs are approximate (mean over 6 LOGO models).
4. Pretrained detectors are scored only on calib+test rows.
