**Verdict: READY WITH FIXES.** The core recipe matches Amendment 8: VAE I/O, A sees no generator image, no train→test leakage, the #7 formula and direction, paired bootstrap. Before any GPU run, the lead must:
- settle F1 in writing, because it can change the #7 label;
- fix F2 and F3 (fov512 format confound; photo-level CV groups);
- add the F4 guard enforcement.

Reviewer: independent pre-run code review, 2026-10-24. Inputs: `run.py`, `config.json`, `notes.md`, and docs/04-phase4-plan.md Amendments 3, 6 and 8. No GPU was used. CPU checks I ran:
- **Tiny VAE round-trip.** Scratch script `/tmp/claude-1000/-home-zera-project-Diffusion-Research/8e17439b-23a0-499c-acb6-22c219043654/scratchpad/review2/vae_tiny.py`, run on a 128² patch on the CPU.
  - Both pinned VAEs load offline in fp32.
  - `latent_dist.mode()` equals the posterior mean.
  - The latent has shape /8.
  - The [-1,1] in/out convention gives a sane reconstruction (per-pixel RMSE ≈ 0.006).
  - The `accelerate` warning in notes.md only means a slow load; it is not a failure.
- **Source JPEG check on all 297 train reals.** The quality factor (QF) is estimable for every file:
  - native: QF 88–100, median 96;
  - WhatsApp: QF 92;
  - FB-HQ: QF 71–91, median 73;
  - all files are 4:2:0, which matches PIL's default when re-saving.
- **Unit tests.** `experiments/lib/tests/run_all.py`: 58 passed, 0 failed.

## Findings, ranked by severity

### F1 (HIGH: can change the #7 label) — the #7 validity gate does not match Amendment 6 when the AUROC is below chance
- **Where:** `run.py:692-693`, `ci_has_half = ci_lo <= 0.5 <= ci_hi`.
- **What's wrong:** Amendment 8 says "Validity gate as in Amendment 6", then restates it as "CI includes 0.5". Amendment 6's gate is "lower bound of the 95% CI > 0.5". The two differ when the ratio is *below* chance (the CI entirely < 0.5):
  - the code labels that **PILOT FAIL**;
  - Amendment 6, and the H4 precedent (below-chance RIGID → UNINFORMATIVE), would label it **UNINFORMATIVE**.
- **Why it is likely to matter:** this case is plausible here. VAE reconstruction error tracks texture. Smooth, zoomed-in phone crops reconstruct well, so they "look fake" under −e1 and −ratio. This is the same mechanism that inverted RIGID in the H4 pilot.
- **Fix:** before the run, the lead records which reading applies. My recommendation:
  - `UNINFORMATIVE if ci_lo <= 0.5` (Amendment 6, and consistent with H4);
  - store both readings in `results.json` (`gate_A6`, `gate_literal`) for transparency.
- **Can it change a label?** Yes: FAIL ↔ UNINFORMATIVE for #7.

### F2 (medium; the label arm orig is not affected) — in the fov512 arm, the JPEG re-save adds a format difference instead of removing one
- **Where:** `run.py:249`. Every round-trip is saved as `JPEG(quality=qf[sha])` in BOTH arms.
- **Why orig is fine:** in orig, the real crop's last operation is a decode of the source JPEG at that QF, so real and aligned fake share their format. This is the intent of Amendment 8.
- **What's wrong in fov512:** the real crop was bicubic-downscaled after decoding (about 4.8× for natives, 1.9× for WA, 3× for FBH) and saved as PNG, so its JPEG blocking is largely gone. The aligned fakes, by contrast, get a fresh, grid-aligned JPEG at QF 71–100; FB-HQ fakes at about QF 73 are clearly blocky.
- **Consequence:** probe A and probes C/C_all in fov512 can learn "fresh JPEG = fake". That is a format shortcut, the exact confound H4 studies. Synthbuster test fakes do not carry it, so the fov512 A/C numbers would be confounded in either direction.
- **Fix:** in the fov512 arm, save the round-trips losslessly (PNG). That matches the real crop's last operation (resample → PNG). Record this as an Amendment 8 clarification: "format matched to the real crop's last operation".
- **Can it change a label?** Not the label (orig decides). It changes the fov512 column that is reported alongside.

### F3 (medium) — grouped CV groups by sha256, but WA and FB-HQ are copies of the same source photo
- **Where:** `run.py:521-522` (A), `528-532` (C, C_all_reals), `537-538` and `543-547` (guard refits). Groups are `sha256` of the real or of its source real.
- **What's wrong:**
  - The native, WA and FBH versions of one photo (same `device`, `vidx`) have three different shas. So the photo's three reals and its six round-trips are spread over up to 3 CV folds.
  - Example: RT(native P) is labelled fake in a training fold, while WA-copy P is labelled real in the validation fold.
  - Amendment 3 defines a group as "the source image, so augmented copies stay in one fold". For VISION, the source image is the photo.
- **Scope:** this affects only the selection of the regularization strength C, not the test set (test devices are disjoint).
- **Fix:** set `group = f"{device}:{vidx}"` for VISION reals and for their round-trips (carry the source's key). Fakes keep their sha256.
- **Can it change a label?** Possibly, through a different C choice, but only near the 0.75/0.85 cut-offs. It is cheap to fix.

### F4 (medium-low, process) — the D-008 guards are not enforced before results exist
- **Where:**
  - `run.py:800`: the order is timing → **vae** → guard. All GPU round-trips and #7 errors are computed before the VAE guard.
  - `stage_analyze`: never checks `d008_guard.json["vae"]["*"]["ok"]`. If the stages are run one by one (as notes.md instructs), a failed VAE guard does not stop the analysis.
  - `run.py:662-669`: the probe guard runs *after* `results.json`, `tables.md` and `scores_long` are written. Its abort comes after the labels already exist.
- **Fix:**
  1. Order the stages timing → guard → vae → feat → analyze.
  2. At the start of `stage_analyze`, assert that the VAE guard file exists and that every entry is ok.
  3. Compute `probe_guard` before `jdump(results.json)`. On failure, either write nothing or write the labels prefixed "D-008 FAILED".
- **Can it change a label?** No, but it prevents a guard-failed result from being published.

### F5 (low) — B's reproduction of 005's A0 is only reported, not checked
- **Where:** `run.py:576-583`.
- **What's wrong:** B should equal 005 pilot_v2 A0. 005's outer z-scoring is absorbed by the pipeline's StandardScaler, so the fits should agree up to floating-point rounding. Nothing stops the run if they differ.
- **Fix:** `assert abs_diff <= 1e-3` per condition, or abort with a message. The C−B comparison and the headroom rule depend on B being B0.

### F6 (low) — round-trip files are not written atomically, and the cache counts any existing file as done
- **Where:** `run.py:241` and `249`. The same applies to `r7/*.npz` and `feats/*.npz`, which are written via `np.savez` straight to the final name.
- **What's wrong:** a run killed mid-write leaves a truncated `.jpg` that is silently reused. Nothing exists yet, so there is no current harm.
- **Fix:** write to `*.tmp`, then `os.replace` (as `score_pretrained.save_scores` already does).

### F7 (low, optional) — the re-save uses IJG tables at the estimated QF, not the source's actual tables
- **What's different:** camera quantization tables differ from IJG's. PIL can reproduce them exactly with `qtables=src.quantization` (subsampling is already 4:2:0, as checked).
- **Recommendation:** the pre-registration says "estimated QF", so keep it. Optionally add a note. I see no expected effect on DINOv2 features.

### F8 (info, disclosed judgment calls; acceptable if recorded before the run, which they are in config.json and notes.md)
- **"Beats the texture baseline"** is read as: point estimate, at BOTH conditions (`run.py:690-691`). This is the stricter reading.
- **Texture sign** is set on calib natives vs train fakes (`run.py:630-631`). E1 used train natives vs train fakes. Both are harmless (no test rows); the inconsistency is only cosmetic.
- **The timing-gate cut** applies to the #7 test set only. #6 uses cached test features and has no VAE cost.
- **Timing method:** the time is amortized over batch ≤4 on a shared GPU. Contention can only trigger the cut spuriously, which lowers #7's power but adds no bias.
- **C_all_reals** is a sensible extra sensitivity arm, because otherwise the WA/FBH-style aligned fakes have no matching reals in C.

## Verified OK (the checks the lead asked for)
- **VAE preprocessing:**
  - uint8 → [-1,1] fp32;
  - `latent_dist.mode()`;
  - decode;
  - (y+1)/2 clamped to [0,1];
  - `np.rint` → uint8 before saving and before the second pass;
  - TF32 off (`features.get_device`);
  - pinned revisions with `local_files_only`, recorded with SHA256 in `env/SOURCES.md`.
- **Aligned fakes:** every train real (99 + 99 + 99) is round-tripped through each VAE at its own arm's crop, giving 594 (asserted). Each is re-saved at the source's estimated QF, taken from `src_path`, the raw file (correct; the crops themselves are PNG).
- **Conditions after the round-trip:** `p4run.load_cond(rt_path, sha, cond)`, the same function and key the 004 cache used for the reals. The DINOv2 config is checked by `check_design_vs_003`. #7 applies the condition before the VAE, identically for natives and fakes.
- **No train→test leakage:**
  - round-trips come only from `split == train` reals;
  - test natives come from 25 devices disjoint from train and calib;
  - the texture sign uses calib and train rows only;
  - the probe guard samples train round-trips only.
- **A never sees a generator image.** Its training rows are `label == 0 & split == train` plus round-trips. No outer standardization is fitted on generator data. Its evaluation uses test natives against all 120 fakes per generator; all of them are unseen by A.
- **B and C:**
  - B is the A0 recipe: `b0_train` LOGO, balanced weights, grouped CV.
  - C = B's LOGO rows + the 594 round-trips.
  - Each LOGO model g is scored only on g's fakes.
- **#7:**
  - ratio = ‖x−x1‖₂ / ‖x1−x2‖₂ in [0,1] float space, with x1 and x2 quantized;
  - min over the 2 VAEs;
  - fake score = −ratio, fixed;
  - single-pass = −min ‖x−x1‖₂, fixed.
- **Labels:**
  - A: ≥ 0.85 PASS, < 0.75 FAIL, otherwise INCONCLUSIVE, at jpeg75_resize05 in the orig arm;
  - C vs B: +0.02 only if B ≤ 0.96 at jpeg75_resize05;
  - #7: c1, c2 and c3 as pre-registered (apart from F1).
- **Bootstrap:**
  - reals by device, fakes by image stratified per generator;
  - one plan shared by A, B and C, and by every #7 method, so the differences (C−B, ratio−single, ratio−texture) are paired;
  - 2,000 resamples; row 0 is the point estimate.
  - The texture check (reals vs aligned fakes) moves a real and its round-trips together by device.
- **Caches are keyed by arm** (`rt/<arm>`, `feats/<arm>`, `r7/<arm>__<cond>__<vae>`, `scores6_<arm>`, `lap_<arm>`). `qf.json` is keyed by sha and is arm-independent by design. No mixing of arms or conditions is possible.
- **D-008 tolerances** match Amendment 8:
  - VAE: max |Δpixel| ≤ 1e-3 on the float output, x1 and x2 (x2 from the same quantized x1);
  - probes: scaled |ΔF| ≤ 1e-3 and |Δlogit| ≤ 1e-3 for A and C_full.
