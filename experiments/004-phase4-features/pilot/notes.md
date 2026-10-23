# 004 pilot feature extraction (PILOT, Amendment 2 of docs/04-phase4-plan.md)

Status: PILOT. grip.unina.it was down, so only VISION and Synthbuster raw files exist. Nothing here is a Phase 4 verdict.

## What was run
1. `pilot_prep.py`: kept the rows of 003 `splits.csv` whose `src_path` exists (2,220 of 2,919; 699 dropped because the raw files are missing). It regenerated each 512 crop at the row's `path` with 003's own call, `preprocess.process_image(src, dst, 512)` (same lib function and arguments as `stage_prep._proc_one`). Output: `pilot/splits_pilot.csv` (same columns and values as 003) and `pilot/prep_report.json`. Prep took the time in `prep_report.json["seconds"]`.
2. `run.py --splits ... --cache-subdir pilot`: the full 17-condition extraction (8 eval conditions on all 2,220 rows, 9 `jr(q,f)` grid conditions on the 657 `split == train` rows). CUDA, RTX 4060 Ti, batch 20, DINOv2-base blocks 4/8/12 CLS+mean. Cache: `cache/pilot/<condition>/part_*.npz` (git-ignored). Log: `pilot/extract.log`; config: `pilot/config_pilot.json` (the main `config.json` is not touched in pilot mode).

## Commands
```
env/.venv/bin/python -I experiments/004-phase4-features/pilot_prep.py
env/.venv/bin/python -I experiments/004-phase4-features/run.py --splits experiments/004-phase4-features/pilot/splits_pilot.csv --cache-subdir pilot
env/.venv/bin/python -I experiments/lib/tests/run_all.py      # 30 passed, 0 failed
```
Next experiments load with: `from p4features import load_features, load_features_aligned`;
`load_features("primary", cond, subdir="pilot")` (or `load_features_aligned(pilot_df, "secondary", cond, subdir="pilot")`), where `pilot_df = run.load_splits(".../pilot/splits_pilot.csv")` or `pd.read_csv(..., keep_default_na=False, dtype={"vidx": str})`. Default (no `subdir`) behaviour is unchanged.

## Counts (source: pilot/prep_report.json)
- Reals: train 297 (99 native + 99 WhatsApp + 99 FB-HQ; `b0_train` reals = the 99 natives), calib 129 (43 x 3), test 1,074 (358 x 3).
- Fakes: 60 train + 60 test for each of dalle2, dalle3, firefly, midjourney-v5, stable-diffusion-2, stable-diffusion-xl (720 total). No sd3_large, flux, RAISE.
- All 2,220 crops: status ok, 512x512 RGB PNG.

## sha256 portability check
- IMPORTANT: in 003 the `sha256` column is the hash of the ORIGINAL raw file (it only names the crop, `<sha>.png`, and keys the sidecar). It is not the hash of the crop, so "sha256(crop) == row sha256" cannot hold (measured: 0/2,220 equal, as expected). The plan's wording in Amendment 2 should be read as below.
- raw file sha256 == row sha256: **2,220 / 2,220** match, no mismatches (the raw data is bit-identical to what the Mac hashed).
- PIL-detected original format (this machine, Linux Pillow) == row `orig_format`: **2,220 / 2,220**.
- Crop pixel-level portability (Pillow/libjpeg Mac vs Linux) is NOT verified: no reference crop hash or Mac-side crop exists in the repo (`data/processed` and 003's cache are absent). The Linux crop sha256s are stored in `prep_report.json["crop_sha256"]` so a later Mac or GPU-PC rebuild can be compared against them. The `verify003` check against the 003 B0 cache also SKIPs here (cache absent).

## D-008 guard (accelerator vs CPU, 32 images = 16 real + 16 fake, seed 0, drawn from the pilot df; source: cache/pilot/d008_guard.json)
All 17 conditions ok. Maxima over conditions: max |dlogit| (B0 probe) = 1.63e-04 (jr(60,1.0)), tol 1e-3; max scaled feature diff = 5.19e-05 (jr(90,1.0)), tol 1e-3; max raw |dF| = 2.67e-04. Per-condition values are in the json. Also the dry run passes (tiny random model, guard max 2.1e-07), and the primary-slice self-check is exactly 0.
Note: the B0 probe used for the logit guard is fit on the pilot `b0_train` rows (99 natives + 360 fakes), not 003's probe.

## Runtime
Total 276 s wall for run.py (17:25:19 to 17:29:59, includes CPU guard recomputation); 197 s in the extraction loops (sum of `cache/pilot/timings.json`).

## Code changes
`run.py`: `--splits PATH` (default = 003 splits.csv), `--cache-subdir NAME` (cache goes to `cache/<NAME>`; pilot/custom runs write `config_pilot.json` rather than `config.json`). The 003-cache check (`verify003`) was already optional (SKIP when absent). `p4features.py`: `cache_root(subdir)`, and `load_features` / `load_features_aligned` take `subdir=None`.
