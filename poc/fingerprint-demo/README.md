# Fingerprint Lab (local proof-of-concept)

A local web demo of the Diffusion-Research findings (`docs/10-attribution-report.md`, `docs/03-final-report.md`): upload or pick an image, launder it, and see (1) an "is it AI?" score and (2) which of 8 known generators the image resembles. Further tabs show the findings as charts and a 14x14 phone-fingerprint (PRNU) heatmap. **Research demo, not a product.**

## Run (from the repo root; Python = `env/.venv/bin/python -I`, Node 12 built-ins only, no npm)
```
env/.venv/bin/python -I poc/fingerprint-demo/train_models.py     # ~15 s, CPU: models/*.npz, meta.json, eligible.json, parity_expected.json
env/.venv/bin/python -I poc/fingerprint-demo/build_findings.py   # findings.json from the result files
env/.venv/bin/python -I poc/fingerprint-demo/make_samples.py     # samples/ (34 thumbnails + samples.json)
env/.venv/bin/python -I poc/fingerprint-demo/worker.py --selfcheck  # parity of live extraction vs cached features
node poc/fingerprint-demo/server.js                              # http://127.0.0.1:5173  (PORT=... to change)
node poc/fingerprint-demo/e2e_test.js                            # optional end-to-end check against the running server
```
Needs the cached features (`experiments/004|016|019/cache`), `data/processed/fov512`, `data/raw` and the DINOv2-base weights in `data/hf_cache` (loaded offline, no downloads). The worker uses CUDA if available, else CPU. `models/`, `samples/`, `tmp/` are git-ignored (non-commercial / no-redistribution licences: Synthbuster CC BY-NC-SA 4.0, B-Free GRIP NC, RAISE, Unsplash terms).

## What is in it
| file | role |
|---|---|
| `train_models.py` | trains the two probes from cached features (no pickles: npz with standardiser mean/scale, logistic coef/intercept, class names) |
| `worker.py` | persistent JSON-lines process: EXIF-orient + RGB -> fov512 (`preprocess.fov_resize` + centre crop) -> `perturb.apply_condition` -> DINOv2-base (`features.load_backbone`, blocks 4/8/12 CLS+mean, 224 centre crop) -> both probes |
| `server.js` | built-in `http` server; `/api/analyze` (multipart, JSON base64 or `sample_id`; <=15 MB; temp file deleted after), `/api/samples`, `/api/findings`, `/api/meta`, `/api/status`; supervises and restarts the worker; request queue |
| `build_findings.py` | parses the project's markdown tables and `prnu_scores.csv` into `findings.json` (each block carries its source path) |
| `make_samples.py` | seeded-random test-fold samples (3 per generator, 2 per real source; HDR+ skipped) as <=768 px JPEG q95 |
| `public/` | single-page UI (vanilla JS, inline SVG, dark/light) |

## The two models and where the numbers come from
**Attribution ("which generator?").** 8-way logistic probe on DINOv2 fov512 `cat` features (blocks 4/8/12, CLS+mean, 4608-d), T-mix training (clean + jpeg75_resize05 + social_sim), exactly 020's recipe (`experiments/020-attr-foundation/run.py`: `inner_table` picks C, `fit_model` fits). It is trained on the 4 non-test folds of `results/folds_ha1_main.json` (fold 0 = test, 204 images, C = 0.1 as 020 chose for that fold), so test-fold samples are honest. Held-out balanced accuracy on that fold is stored in `models/meta.json` and shown in the UI next to the project-wide 5-fold values from `020/results/tables.md` (clean 0.913, jpeg75_resize05 0.855, social_sim 0.872, screenshot_sim 0.866, mavali_medium 0.813). The "uncertain" flag fires when the top probability is below tau = 0.707, the 10th percentile of out-of-fold max-softmax among correct predictions (3-fold grouped inner CV on the train rows). **Open-set caveat:** the max-softmax score separates unknown generators at only 0.768 AUROC (`020/results/tables.md`; docs/10), so the flag is not an unknown-generator detector.

**Detection ("is it AI?").** E10's `F_b8` (block-8 CLS+mean, 1536-d) real-vs-fake probe with T-aug, via 019's own `Data`, `train_rows`, `probe.fit_probe` (balanced, grouped CV), trained on the train-role reals of all 6 sources (VISION native, RAISE, Open Images, Unsplash, Commons, HDR+) and the 7 train generators (FLUX is test-only). Threshold = 1% FPR on the clean scores of the 581 calib reals (reading A), applied unchanged at every laundering. The UI states the honest scope: FPR on *unseen* real sources was 0.5% to 17% (`019/results/tables.md`, "Held-out source at k = 5", F_b8, reading A: HDR+ 0.005 ... Unsplash 0.170); the test-role FPR on the *seen* sources (about 1.5% to 2.5%) and the per-generator TPR (FLUX the lowest) come from `models/meta.json` and are not unseen-source numbers.

**Findings tab** charts: hypothesis scorecard (docs/10 section 4; detection labels docs/03 section 3.3), laundering bars and layer contrast (`020/results/tables.md`), detector-embedding comparison (`023/results/tables.md`), real-source learning curve and unseen-source FPR (`019/results/tables.md`), open-set AUROC (`020/results/tables.md`), SD1.3 vs SD1.4 (`020/secondary/tables.md`), plus the demo model vs its 5-fold reference. **Phone tab:** `experiments/021-attr-residual/results/prnu_scores.csv`; cell = median PCE over the photos of the row device against the column device's fingerprint (log colour); the top-1 accuracies recomputed from the CSV (native 0.985, WhatsApp 0.929, Facebook 0.868, native-fp on WhatsApp 0.914) match `021/results/tables.md` (0.985, 0.930, 0.871, 0.915) to 0.003 (tie handling in my argmax).

## Verification (this machine, CUDA)
* Parity (`worker.py --selfcheck`, 20 cached test rows x 4 conditions through live extraction vs cached features): max |dprob| = 1.9e-6 (attribution), max |dlogit| = 1.0e-5 (detector); `models/parity_result.json`.
* Held-out bacc (fold 0, n = 204): clean 0.918, jpeg75_resize05 0.853, social_sim 0.864, screenshot_sim 0.836, mavali_medium 0.802.
* `e2e_test.js` (34 samples): generator named right 21/24, 20/24, 21/24, 22/24 at clean, jpeg75_resize05, social_sim, screenshot_sim; fakes flagged 14, 15, 14, 18 of 24; real photos left unflagged 10/10 at each. With 3 samples per generator the rates are noisy (about +/-10 points).

## Deviations and notes
* Samples are 768 px JPEG q95 re-encodes of the raw originals (not the cached PNG crops), so they differ slightly from the images behind the cached numbers. Uploads with a short side below 512 px are upscaled (the validated pipeline skips them); the UI warns.
* The seeded launderings (`social_sim`, `screenshot_sim`) are keyed by the sha256 of the uploaded bytes, as the cache keys them by the original file hash.
* `mavali_medium` is evaluated in `meta.json` but not offered in the UI.
* Models inherit the non-commercial licences of the training data. Every displayed number is either parsed from a result file (cited in the UI) or computed by `train_models.py` from cached features (`models/meta.json`).
