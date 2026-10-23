# 006 format shortcut: H4 PILOT (docs/04-phase4-plan.md, Amendment 5)

Status: PILOT, not a Phase 4 verdict, does not count toward the stop rule. Labels below are the ones computed by `run.py`; no interpretation is added here.

## What was run
```
env/.venv/bin/python experiments/006-format-shortcut/run.py --dry-run        # 16 test images + 5 calib natives, n_boot=50, output in cache/dryrun_out (git-ignored)
env/.venv/bin/python experiments/006-format-shortcut/run.py --subdir pilot   # full pilot -> pilot/
```
All parameters: `config.json`. Outputs in `pilot/`: `results.json`, `tables.md` (all readable tables), `scores_long.csv.gz` (setting, arm, feat, sigma, sha256, label, generator, pool, device, split, score), `d008_guard.json`, `run.log`. Score caches: `cache/pilot/` (git-ignored).

## How
- Data: `004/pilot/splits_pilot.csv` (arm `orig`) and `splits_pilot_fov512.csv` (arm `fov512`), same rows/order (asserted). Test split: 358 native reals (25 devices; WA and FBH copies only in the descriptive per-pool table) and 360 fakes (6 generators x 60).
- RIGID-style score: 004's `load_cond` applies the condition (`clean`, `jpeg(90)`, `jpeg75_resize05`) to the 512 crop, then 224 center crop, [0,1], Gaussian noise (sigma in {0.01, 0.02, 0.05}, 4 draws, CPU generator seeded by sha256(image sha, draw, sigma)), clip, ImageNet mean/std, DINOv2-base. Score = mean over draws of 1 - cos(f(x), f(x+n)); higher = fake, never flipped.
- Two CLS variants come from the same forward: `post` (last_hidden_state, after the final LayerNorm) is the PRIMARY one, fixed in config.json before any result; `pre` (hidden_states[-1], the 004/B0 convention) is a sensitivity analysis. Labels for both are in `tables.md`.
- AUROC = LOGO-macro over generators (each generator's test fakes vs the same native reals), for RIGID and B0 alike (pooled AUROC is identical because every generator has 60 fakes). CIs: `metrics.bootstrap_weights` (reals by device, fakes by image per generator), 2000 resamples, seed 0; gaps and orig-fov512 differences are paired on the same resamples.
- B0 = 005's A0 on the cached 004 primary features: standardised with b0_train reals, `probe.fit_probe(seed 0, groups=sha256, class_weight="balanced")`, one `logo_<g>` model per generator, scored on clean / jpeg(90) / jpeg75_resize05 cached features. Primary B0 numbers use the TEST-split fakes of g (same images as RIGID). The 005-style variant (all 120 fakes of g, train+test) is reported too; its orig / clean / with-Firefly value 0.99204 equals 005 pilot_v2's A0 clean macro exactly (`results.json: meta.check_005`).
- Labels follow Amendment 5 on point estimates: best sigma = argmax AUROC(i) per (arm, Firefly variant); SUPPORTED if gap(i)-(iii-p) >= 0.10 and B0 gap < 0.05; REFUTED if the gap < 0.03 for every sigma; else INCONCLUSIVE. Primary cell = orig, with Firefly, `post`. The (iv-p) gap is reported, not used for the label.
- 1%-FPR threshold: `metrics.threshold_for_fpr` on the 43 clean calib native reals (so it is the calib maximum plus epsilon), applied to test rows.

## Checks
- D-008 (`pilot/d008_guard.json`): 32 fixed images (16 native test reals + 16 test fakes), CUDA vs CPU, clean + noised (draw 0, all sigmas); TF32 off. Scaled |dfeat| 1.2e-05 (post) / 1.1e-05 (pre), tolerance 1e-3; max |dscore| 1.6e-06. Passed.
- Pipeline check: our unnoised pre-LN CLS equals the 004 cache primary[:, :768] on the same 32 images (max abs diff 0).
- `experiments/lib/tests/run_all.py`: 51 passed, 0 failed. `pytest` is not installed in the venv (and may not be installed), so the project runner was used.

## Caveats and deviations
- Batch size is 32 (config.json), not 64: another process was using about 5.7 GB of the 8 GB GPU during the run. The `orig` arm score caches were first computed with batch 64 in an interrupted first launch (killed because the GPU was contended and slow; resumed from the caches, which store per-condition scores keyed by the image list), and the `fov512` arm with batch 32. Batch size changes results only at float rounding level (D-008 scale). `pilot/run.log` is appended across the launches. `results.meta.timings` therefore lists only the fov512 conditions that were computed in the final launch; total wall time of the final launch was 1074 s, plus about 6 min for the orig arm in the first launch.
- Judgment calls not fixed by the spec: macro-over-generators aggregation for RIGID, test-split fakes for B0's primary numbers, noise added on the 224 crop, post-LayerNorm CLS as primary.
- RIGID's AUROC(i) is far below 0.5 on this data (reals score higher than fakes), and the direction was fixed in advance, so the absolute RIGID AUROCs are not those of a working detector; the gap criterion is applied as specified.
- In (iii-p) the reals have been compressed twice, so only their final format matches the fakes (Amendment 5).
- Firefly files are already JPEG q~95; every number is given with and without Firefly.
