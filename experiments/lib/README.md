# experiments/lib: shared evaluation harness

Flat modules; add this dir to `sys.path` (`sys.path.insert(0, ".../experiments/lib")`). Python: `env/.venv/bin/python`.
No downloads happen anywhere except `features.load_backbone()` (pretrained weights; never called by tests).

| module | purpose |
|---|---|
| `manifest.py` | CSV schema `path,label,source_dataset,generator,camera_or_platform,license,split,sha256`; `assign_splits(df, holdout_generator, seed)`; `check_splits` |
| `preprocess.py` | RGB, EXIF-orient, re-encode (strips EXIF/ICC), center crop (default 512, no resize), PNG; `sidecar.csv` with original format / JPEG quality / size |
| `perturb.py` | `apply_condition(img, cond, key, seed)`: `clean`, `jpeg75_resize05`, `resize(f)`, `jpeg(q)`, `social_sim` |
| `metrics.py` | AUROC, AP, bal-acc (fixed thr and FPR=5%-calibrated), FPR@95TPR, TPR@1%FPR, seeded bootstrap CIs, per-generator |
| `features.py` | `Backbone` adapters (HF CLIP/DINOv2/DINOv3, timm, torchvision), cached `extract_features` |
| `probe.py` | standardized logistic probe (C by CV on train), `standard_eval`, `logo_eval` |
| `format_bias_probe.py` | dumb file-size/quant-table/dims/HF-energy baseline (leakage detector) |
| `run_eval.py` | CLI producing `results.json` |

Usage:
```
# 1. manifest (make_row per file) -> preprocess -> splits
pm, side, rep = preprocess_manifest(df, "data/proc", size=512)     # writes manifest_processed.csv + sidecar.csv
pm = assign_splits(pm, holdout_generator="sdxl", seed=0); write_manifest(pm, "data/proc/manifest_processed.csv")
# 2a. black-box scorer
python run_eval.py --manifest data/proc/manifest_processed.csv --out exp/results --conditions clean jpeg75_resize05 social_sim --scorer mypkg.mod:score
# 2b. frozen features + linear probe (+ leave-one-generator-out)
python run_eval.py --manifest ... --out ... --backbone hf:dinov3:<repo> --layers -1 -4 --use-mean --logo --exp-dir exp
```
Tests: `env/.venv/bin/python -I experiments/lib/tests/run_all.py` (pytest is not installed; tests are pytest-compatible).

## Eval protocol
1. Same preprocessing for both classes; "clean" = center crop only, never a resize. Images smaller than the crop are skipped and counted.
2. One generator is held out ENTIRELY (all in `test`); other fakes are split train/val/test per generator.
3. Reals are split by `camera_or_platform` group: test cameras never appear in train/val/calib. Calib reals (train-side cameras) set the FPR=5% threshold.
4. Scorers see only PIL images; higher score = more fake. Conditions are applied to test AND calib images identically, deterministically (seeded per image via sha256).
5. Probes are trained on `train`-split features of `--train-condition` (default clean) and tested under every condition: this measures robustness, not re-fit-per-condition.
6. LOGO: for each generator g, train on reals + other generators' train fakes; test g (all its images) vs test-split reals.
7. Per-generator metrics = that generator's fakes vs ALL test reals; headline numbers are the held-out generator and LOGO macro-mean, not pooled seen-generator AUROC.
8. Report AUROC, AP, balanced accuracy at thr 0.5 and at the calibrated FPR=5% threshold (plus the FPR actually reached on test reals), FPR@95TPR, TPR@1%FPR, all with 95% stratified bootstrap CIs (1000 resamples, seeded, calib set resampled too).
9. Check `format_bias` in every `results.json` (logistic regression on bytes/pixel, quant-table presence + quality, dims, HF energy of the ORIGINAL files via the sidecar). If its AUROC CI lower bound > 0.6 the dataset leaks format, and detector results are not trustworthy.
10. Tiny sets: CIs are NaN if a class has <2 samples; `warnings` flags <100 reals (TPR@1%FPR then means zero false positives) and <20 calib reals.
11. Acceptance targets live in the project README (FPR <= 1% on fresh phone photos; AUROC >= 0.95 after laundering).

Design notes: feature crops are center crops to `input_size` (default 224), never resizes, so `jpeg75_resize05` needs a source crop >= 2x input_size (512 -> 256 is fine). Intermediate-layer features use HF `hidden_states` (pre final LayerNorm). MPS is used when available; any MPS RuntimeError falls back to CPU.
Feature cache: `<exp_dir>/cache/<backbone.name>_<condition>.npz`, keyed by sha256 of the processed file; a different backbone config invalidates it.
