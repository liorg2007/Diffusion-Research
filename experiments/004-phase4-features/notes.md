# 004-phase4-features: shared DINOv2-base feature cache for Phase 4 (005-007)

## What it does
One frozen-DINOv2-base pass (`hf:dinov2:facebook/dinov2-base`, `HF_HOME=data/hf_cache`, offline, 224 center crop of the
512 crop, batch 20, seed 0, fp32, TF32 off) per image and condition, with `layers=(4,8,12)`, CLS + mean-patch each
(6 x 768 = 4608-d, float32). Two feature views are served from that single cache:
- **primary** (1536-d) = the last 1536 columns = block 12 = `hidden_states[-1]`: exactly B0's feature (003: `layers=(-1,)`,
  `use_cls`, `use_mean`, `input_size=224`, crop mode "crop", same preprocessing via `features.Backbone.to_tensor`).
  By design: `check_design_vs_003` asserts the config equals 003/config.json `b0`; `selfcheck_primary` asserts a separate
  `layers=(-1,)` forward equals the slice (max diff 0 on CUDA and CPU in the smoke test); `run.py --verify003` compares our clean
  primary features with 003's cache file (`003-baselines-eval/cache/b0/hf-dinov2-facebook_dinov2-base_clean.npz`) when present
  (scaled diff must be <= 1e-3; the small residual comes from batch 16 vs 20).
- **secondary** (4608-d) = blocks 4, 8, 12 (`hidden_states[4,8,12]`, index 0 = embeddings), CLS + mean each.

## Rows and conditions (rows = `experiments/003-baselines-eval/splits.csv`, image paths from its `path` column)
- 8 eval conditions on all rows with `in_scoring_set or split in (train, calib)` (= all 2,919 rows):
  `clean, jpeg75_resize05, social_sim, resize(0.5), jpeg(75), jpeg(90), mavali_medium, screenshot_sim`.
  `mavali_medium`, `screenshot_sim` are held out from all fitting.
- 9 grid conditions `jr(q,f)`, q in {60,75,90} x f in {0.5,0.75,1.0}, on `split == 'train'` rows only (872 rows).
  `jr(75,0.5)`, `jr(75,1.0)`, `jr(90,1.0)` are byte-identical to `jpeg75_resize05`, `jpeg(75)`, `jpeg(90)` but are recomputed for simplicity (cheap).
- Conditions are seeded per image (`seed_from_key(sha256, 0)`), applied identically to both classes; the images are the EXIF-free PNG crops.

## Condition definitions (experiments/lib/perturb.py; existing conditions unchanged and tested)
- `mavali_medium` = `jpeg(resize(img, 0.75), 50)` (bicubic, libjpeg 4:2:0).
- `screenshot_sim`: factor drawn from {0.5, 0.67, 1.25}, PIL BILINEAR (antialiased when shrinking), 8-bit RGB, no JPEG, then ONE seeded fraction
  in [0.02, 0.05] removed from each of the four sides (symmetric).
- `jr(q,f)` = `jpeg(resize(img, f), q)`; f = 1.0 skips the resize.

## Caveats
- **Image smaller than 224**: `Backbone.to_tensor` raises `ValueError` (never silently resizes). `run.py` checks every conditioned image and aborts with the path
  and size. With 512 crops the smallest outputs are 256 (f=0.5), 230 (screenshot_sim 0.5 + 5% crop each side) and 358 (social_sim), so all are >= 224.
- **screenshot_sim with factor 1.25** gives 640 px (up to 576 after the crop): the 224 center crop then sees only 179 source pixels (a zoomed centre);
  with 0.5 the crop spans 448 source pixels (almost the whole width: a full-image view). Factors therefore change the field of view as well as the
  resampling; this is part of the "screenshot" condition and is not corrected.
- Every 224 center crop is taken AFTER the condition, as in 003 (`jpeg75_resize05` shows the central 448x448 of the original).
- Intermediate-layer features are pre-final-LayerNorm hidden states (the primary one is too, as in B0).

## D-008 guard
Per condition, a fixed 32-image subset (16 real + 16 fake, seed 0, sorted shas; eval conditions use 003's `check_subset` rule, so it is the same
32 images as 003; grid conditions use `split == 'train'` rows): accelerator features (from the cache) vs CPU recomputation (batch 4, TF32 off).
- Feature test: `max_j |dF_j| / max(1, max_i |F_cpu_ij|) <= 1e-3` (relative for large-magnitude dims, absolute below 1; DINOv2 has a few
  massive-activation dimensions, so a raw absolute tolerance would be unfair). Raw max |dF| is logged too.
- Logit test (same as 003): max |dlogit| <= 1e-3 of the B0 probe (`probe.fit_probe` on `b0_train` clean primary features, seed 0), applied to
  the primary slice. Skipped (logged) if clean features are not cached yet.
- Any failure aborts. Results: `cache/d008_guard.json`; CPU references are cached in `cache/cpuref/`.
- Smoke result with real weights on random synthetic images (CUDA, 8 images): scaled diff <= 6e-5, |dlogit| <= 5e-6.

## Run
```
env/.venv/bin/python -I experiments/004-phase4-features/run.py                 # all 17 conditions, resumable (shards of 240 images)
env/.venv/bin/python -I experiments/004-phase4-features/run.py --conditions clean "jr(60,0.5)"
env/.venv/bin/python -I experiments/004-phase4-features/run.py --verify003     # primary clean vs 003 B0 cache
env/.venv/bin/python -I experiments/004-phase4-features/run.py --dry-run       # synthetic smoke test (tiny random model, deletes its scratch dir)
```
Exits 2 with a message if the weights or any image are missing. Output: `cache/feats/<condition>/part_*.npz` (keys = sha256, feats float32),
`cache/timings.json`, `cache/d008_guard.json` (all git-ignored); `config.json` holds all parameters.

## Using the features (005-007)
```python
sys.path.insert(0, ROOT / "experiments/004-phase4-features")
from p4features import load_features, load_features_aligned, GRID_CONDITIONS
shas, X = load_features("primary", "jpeg75_resize05")        # (sha256 array, float32 [n, 1536]); 'secondary' -> 4608-d
X = load_features_aligned(splits_df, "primary", "clean")      # aligned to df rows, NaN where not extracted
```
Column order of the 4608-d matrix: `[b4 CLS | b4 mean | b8 CLS | b8 mean | b12 CLS | b12 mean]`.

## Runtime [estimate]
31,200 conditioned images (8 x 2,919 + 9 x 872). GPU forward is cheap (ViT-B/14 at 224, fp32, a few hundred img/s); the CPU side (PNG decode + resize + JPEG,
4 threads) is the bottleneck at roughly 25-40 ms/img, so about 15-25 min, plus about 5 min for the CPU guard (17 x 32 images). Cache size about 0.7 GB.


## fov512 arm (D-013 / Amendment 3; PILOT run 2026-10-23, official run pending)
- `fov_prep.py`: raw file -> EXIF-orient -> resize short side to 512 (PIL BICUBIC, antialiased; `lib/preprocess.process_image(mode="fov")`) -> center crop 512 -> `data/processed/fov512/images/<sha>.png`. Writes a splits file with the same rows/columns as its input (only `path` changes) and a report. Short side < 512 is never upscaled: the script lists such images and ABORTS unless `--drop-small`.
- Pilot: `pilot/splits_pilot_fov512.csv`, `pilot/fov512_prep_report.json`: 2,220/2,220 ok, 512x512 RGB, raw sha256 match 2,220/2,220, 0 images with short side < 512 (manifest census over all 2,920 non-latent-diffusion rows incl. RAISE and the SD3.5/FLUX rows not yet re-fetched: min short side 576 px, 0 below 512).
- Features: `run.py --check-sizes --splits pilot/splits_pilot_fov512.csv --cache-subdir pilot_fov512` (same 8 eval + 9 grid conditions; `--check-sizes` asserts every condition leaves >= 224 px on a 512 crop and every crop is 512x512: min sizes resize(0.5) 256, screenshot_sim 230). Log `pilot/extract_fov512.log`, guard `cache/pilot_fov512/d008_guard.json`: all 17 conditions ok, max |dlogit| 2.78e-05 (jr(90,1.0)), max scaled feature diff 4.81e-05, tol 1e-3. Official run: same command with the 003 splits file replaced by `splits_fov512.csv` (from `fov_prep.py` with no `--splits`) and `--cache-subdir fov512`.
