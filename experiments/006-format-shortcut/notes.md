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


# Official run (code ready 2026-10-24; NOT yet run: RAISE PNG reals and SD3.5/FLUX fakes are on grip.unina.it, currently down)

Spec: `docs/04-phase4-plan.md`, H4 section + Amendments 3, 5, 6. The pilot (`run.py`, `pilot/`) is frozen and unchanged; the official run is `run_official.py` (imports the pilot module for the RIGID score, the noise generator and the D-008 guard). Parameters: `config_official.json` (sigma/draws/batch come from `config.json`). New pure functions: `experiments/lib/h4.py` (+ `lib/tests/test_h4.py`).

## What `run_official.py` does (Amendment 6 mapping)
- **Rows:** official splits = `experiments/003-baselines-eval/splits.csv` (arm `orig`), `004-phase4-features/splits_fov512.csv` (`fov512`), `splits_fov224.csv` (`fov224`); same rows in the same order (asserted). Aborts if any crop, arm, feature cache, RAISE test row, `sd3_large` or `flux` is missing (`--allow-missing` only for dry runs). FLUX is test-only (asserted: no train/calib/b0_train row) and never enters B0 training, texture-direction calibration or residualisation.
- **Settings** (test split, LOGO-macro AUROC over the 7 train-having generators dalle2, dalle3, firefly, midjourney-v5, stable-diffusion-2, stable-diffusion-xl, sd3_large; with and without Firefly; plus a `with_firefly_plus_flux` macro and per-generator tables that include FLUX): (i) VISION native clean vs clean fakes; (ii) RAISE clean vs clean; **M** RAISE `jpeg(90)` vs clean fakes; (iii) both `jpeg(90)`; (iv) both `jpeg75_resize05`; continuity (iii-p)/(iv-p) = the pilot's native-based versions.
- **Primary contrast** AUROC(M) - AUROC(ii) (same RAISE reals, same fakes: paired CI). Secondary: (M)-(iii), (i)-(iii). Also (i)-(iii-p), (i)-(iv-p), (ii)-(iii), (ii)-(iv).
- **Best sigma** = highest AUROC(ii). **Validity gate** (at that sigma): CI lower bound of AUROC(ii) > 0.5 AND RIGID AUROC(ii) > Laplacian-texture AUROC(ii) (direction set once on `clean`: calib reals of the pool vs TRAIN fakes, because the calib split holds no fakes), else the label is `UNINFORMATIVE`; labels `SUPPORTED` / `REFUTED` / `INCONCLUSIVE` use the unchanged thresholds (gap >= 0.10 with B0 gap < 0.05; gap < 0.03 for every sigma), `NOT_EVALUABLE` if (ii)/(M) are missing.
- **Label arm** = `label_arm` in the config (`orig` until D-013 decides). `orig`, `fov512` and `fov224` are all reported; `fov224` has only the `clean` and `jpeg(90)` conditions (resize(0.5) would leave 112 px < 224), so settings (iv)/(iv-p) are skipped there.
- **Manipulation check** (per contrast): mean Laplacian of the test reals and of the test fakes under the two settings (difference and ratio), next to B0's gap and the texture baseline's gap.
- **Texture-residualized RIGID:** OLS of the score on log-Laplacian fitted on the calib reals of the setting's pool and real condition; AUROC of the residual on test (descriptive).
- **Cross-scale B0:** probe fitted on arm A (b0_train, clean, standardised with A's reals) applied to arm B's cached features, all ordered arm pairs and all conditions the two arms share, next to the within-arm B0.
- Also: orig - fov512 / orig - fov224 differences (setting i, ii, M), per real pool (native, RAISE, WA, FBH) clean tables, per-generator tables, 1%-FPR operating points from calib reals, `scores_long.csv.gz` (with the Laplacian), D-008 guard (CUDA vs CPU, abort > 1e-3) + pipeline-vs-004-cache check.

## Exact command sequence (run from the repo root)
```
# 0. data (network; owner/lead; records SOURCES): re-fetch RAISE, SD3.5, FLUX, weights when grip.unina.it is back
env/.venv/bin/python -I env/bootstrap_data.py
# 1. crops for ALL rows, CPU, ~3-4 min each. --no-clobber NEVER overwrites data/processed/* (outside experiments/, owner's rule): files that
#    already exist (the pilot crops) are re-created beside themselves as <sha>.chk.png, compared pixel-wise, deleted, and the run ABORTS if any
#    differs (report: no_clobber.n_existing_equal / n_existing_differ / n_newly_written); only the missing files (RAISE, SD3.5, FLUX, ...) are
#    written. Do NOT use 003 `prep` (no sidecar -> it would redo everything and overwrite splits.csv).
env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py --no-clobber --mode crop --out-splits experiments/004-phase4-features/official/splits_crop512_check.csv --report experiments/004-phase4-features/official/crop512_prep_report.json
#    then: the written splits must equal 003 splits.csv (diff -q experiments/004-phase4-features/official/splits_crop512_check.csv experiments/003-baselines-eval/splits.csv)
env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py --no-clobber --out-splits experiments/004-phase4-features/splits_fov512.csv --report experiments/004-phase4-features/fov512_prep_report.json
env/.venv/bin/python -I experiments/004-phase4-features/fov_prep.py --no-clobber --size 224 --out-splits experiments/004-phase4-features/splits_fov224.csv --report experiments/004-phase4-features/fov224_prep_report.json
# 2. 004 features (GPU). B0 needs clean, jpeg(90), jpeg75_resize05 (fov224: clean, jpeg(90)) and, for B0-JM, jm(85,95) = the train FAKES
#    JPEG-compressed with prob 0.5 at q~U[85,95], seeded per image (435 train fakes per arm). The full 17-condition run is what H1/H2 need.
env/.venv/bin/python -I experiments/004-phase4-features/run.py --check-sizes                       # orig -> cache/feats, 17 conditions (~15-25 min [estimate]); H4-only: --conditions clean "jpeg(90)" jpeg75_resize05
env/.venv/bin/python -I experiments/004-phase4-features/run.py --conditions "jm(85,95)"            # orig, B0-JM training fakes
env/.venv/bin/python -I experiments/004-phase4-features/run.py --check-sizes --splits experiments/004-phase4-features/splits_fov512.csv --cache-subdir fov512
env/.venv/bin/python -I experiments/004-phase4-features/run.py --splits experiments/004-phase4-features/splits_fov512.csv --cache-subdir fov512 --conditions "jm(85,95)"
env/.venv/bin/python -I experiments/004-phase4-features/run.py --splits experiments/004-phase4-features/splits_fov224.csv --cache-subdir fov224 --conditions clean "jpeg(90)" "jm(85,95)"   # no --check-sizes: that asserts 512x512 crops
# 3. tests, then H4 official (GPU, batch 32 or --batch 16 if the GPU is shared; ~40-100 min depending on contention [estimate from the pilot: 0.5 s/image contended])
env/.venv/bin/python -I experiments/lib/tests/run_all.py
env/.venv/bin/python -I experiments/006-format-shortcut/run_official.py
```
Official mode is strict: it ABORTS (never skips or drops) on a missing arm, crop, RAISE/SD3.5/FLUX data, a missing B0 feature cache or condition, any scored row or b0_train row without features, missing `jm(85,95)` features, a non-CUDA device, or a stale RIGID cache (cache files carry sigmas, draws, seed scheme, feature variants, config hash; a mismatch is an error, delete the cache). The D-008 guard runs for every arm and every condition of the arm (CUDA vs CPU, 32 images incl. 8 RAISE reals, plus the pipeline-vs-004-cache identity) and aborts above 1e-3.

Outputs: `experiments/006-format-shortcut/official/{results.json, tables.md, scores_long.csv.gz, d008_guard.json, run.log}`; score caches `cache/official_scores/` (git-ignored, resumable: re-running reuses them). Not for the label: the dry-run directories under `cache/`.

## Dry-run / code-path verification (not results)
```
env/.venv/bin/python -I experiments/006-format-shortcut/run_official.py --dry-run [--device cpu --skip-guard]            # pilot data, RAISE settings skipped -> cache/official_dryrun/
env/.venv/bin/python -I experiments/006-format-shortcut/make_standin.py                                                  # 20 VISION natives re-saved as PNG -> cache/standin/ (pool raise_standin), prints the 004 extraction commands
env/.venv/bin/python -I experiments/006-format-shortcut/run_official.py --dry-run --standin-raise [--device cpu --skip-guard]   # -> cache/official_standin_dryrun/
```
The stand-in is a fixture for the (ii)/(M)/(iii)/(iv) code and the gate; every number it produces is meaningless and its outputs live only under `cache/` (git-ignored).

## 004 changes made for this (defaults unchanged, byte-identical outputs checked on 20 rows)
`fov_prep.py`: `--size N` (fov224 arm; default 512), `--mode crop` (re-creates 003's plain centre crops from the raw files at the splits' own `path`, raw sha256 checked, no resize), `--no-clobber` (never overwrite; verify existing files pixel-wise, abort on any difference), `--out-dir` is resolved to an absolute path. `lib/perturb.py` + `p4features.py`: new condition `jm(lo,hi)` (B0-JM; `JM_CONDITIONS`, extracted only for split == train fakes, not part of the default 17). Report key names such as `n_short_side_lt_512` keep their old names for any size.

## Pilot addendum (`pilot_addendum/`, CPU only, descriptive, post-hoc to the pilot, pre-registered in Amendment 6.5-6.6)
`pilot_addendum.py` -> `pilot_addendum/{results.json,tables.md}`: cross-scale B0 (orig <-> fov512) and the Laplacian texture baseline for setting (i), on the pilot test split.

## Dry-run status (2026-10-24, CPU only: the owner paused GPU work; `--device cpu --skip-guard`, so the D-008 guard and GPU forwards were NOT exercised)
- `--dry-run` (pilot data, no RAISE): ran end to end; settings (ii)/(M)/(iii)/(iv) skipped, labels `NOT_EVALUABLE`; (i), (iii-p), (iv-p), texture, B0, cross-scale, per-pool, operating points produced for orig, fov512 and fov224. Output (not a result): `cache/official_dryrun/out/`.
- `--dry-run --standin-raise` (20 VISION natives as `raise_standin`): ran end to end incl. (ii)/(M)/(iii)/(iv), the gate, the labels (all `UNINFORMATIVE`, as expected for an inverted score on 14 test reals), residualised RIGID, manipulation check, cross-scale B0. Output (not a result): `cache/official_standin_dryrun/out/`.
- The pilot-data fov224 arm was prepared for this (`004/pilot/splits_pilot_fov224.csv`, features `004/cache/pilot_fov224`, extracted on CPU in 591 s, log `004/pilot/extract_fov224_cpu.log`).
- After the code-review fixes (JM arm, strict mode, cache meta, guard over every arm/condition, per-pool bootstrap streams) the stand-in dry run was repeated on CUDA with the guard: `run_official.py --dry-run --standin-raise --batch 16` -> OK; D-008 guard over 3 arms x 2-3 conditions (32 images each, 8 RAISE-stand-in), max scaled feature diff 1.8e-05, max |dscore| 2.5e-06, pipeline-vs-004-cache 9.9e-06 (tol 1e-3). The B0-JM table is exercised on the pilot `jm(85,95)` features (extracted for pilot, pilot_fov512, pilot_fov224); on the stand-in it is meaningless. `python run_official.py` (official, no flags) aborts immediately on the missing fov512 splits, as intended.
