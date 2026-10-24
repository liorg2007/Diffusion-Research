# 010-raid-transfer: E3 PILOT (idea #5, canonicalization vs RAID transfer attacks)

Pre-registration: `docs/04-phase4-plan.md` Amendment 9. Data: `kb/datasets/raid.md`, `data/raw/raid/` (1,500 files listed in `raid_manifest.csv`, sha256 == HF LFS oid; the run verifies all of them and never edits the manifest). PILOT: does not count toward the stop rule. Status: see LOG.md; results in `pilot/`.

## Design (as pre-registered)
- 100 ids x 5 classes (real, gen_0..gen_3); the SAME 100 ids appear in all 5 classes (asserted). Per item: original JPEG, eps8 PNG, eps16 PNG.
- 6 input sets: `clean`, `eps8`, `eps16`, and each with `_canon`.
- Detectors: `b0` (A0 recipe: balanced logistic probe on the 1536-d primary DINOv2-base features, trained on ALL 459 b0_train rows of `004/pilot/splits_pilot.csv` = 6 Synthbuster train generators + 99 VISION train natives, 'orig' arm, 004 cache subdir `pilot`, group-CV by sha256 as in 005 pilot_v2 `--grouped-cv --balanced`); RAID images are embedded with the 004 backbone config (`load_backbone` with 004 `config.json` primary block: 224 center crop of the 256 input). The refit reproduces 005 `pilot_v2/orig/cache/scores_A0.npz['full']` on clean features to 8.3e-4 (logged; `cache/b0_probe_info.json`). References: CommFor, RINE (LDM checkpoint; in the attack ensemble), ClipDet, via `experiments/002-baselines/scorers` with their own preprocessing, same call pattern as `007/score_pretrained.py`.
- Metrics (primary set; the sensitivity set gets the same tables without a label): pooled and per-generator AUROC, bootstrap CIs; drop, recovery, canon clean cost (eps8 primary, eps16 descriptive); balanced accuracy at logit 0 (descriptive anchor to RAID Table 4 accuracy); B0 label with the validity gate, exactly per Amendment 9 (`b0_label`, unit-tested in `--stage selftest`). Also reported: the share of bootstrap replicates giving each label.
- Bootstrap: unit = id, within class (reals and each generator's fakes independently; `lib/metrics.bootstrap_weights`, 2000 resamples, seed 0), one plan shared by all input sets and detectors so drop / gain / cost are paired. Recovery CI uses only replicates with drop > 0.02 (count reported); point recovery is None if drop <= 0.
- D-008: for every detector x input set, the fixed 32-item subset (16 real + 16 fake, seed 0) is scored on CUDA and compared with a CPU reference (chunk of 4, subprocess with CUDA hidden, ALWAYS computed fresh, never reused; the GPU subset scores are rescored fresh too); abort if max |dlogit| > 1e-3. For `--device cpu` runs the guard is recorded as not applicable (the reference is CPU).

## Deviations and caveats (the lead must see these)
- **D1, canonicalization size (not covered by Amendment 9).** `jpeg75_resize05` on a 256 image gives 128 px, which B0's 224 center crop (and RINE's, which would zero-pad) cannot take. Implemented: resize 0.5 (bicubic) -> JPEG q75 -> bicubic upsample back to 256, applied identically to both classes and to clean and attacked inputs. CommFor and ClipDet resize internally, so for them the upsample is nearly neutral. This is a deployable "purify then restore the nominal size" step, but the upsample is part of the canonicalizer and may blur residual attack energy. Alternative if the owner prefers: feed the 128 px image directly to the resizing detectors only.
- **D2, clean control (replaced 2026-10-24 per Amendment 9 clarification 2 and `code_review.md`).** RAID attacks the PLAIN center crop of the original JPEG to 256x256, no resize, zero-padded when a side is under 256 (torchvision `CenterCrop`). The clean control is exactly that crop. `verify` asserts the pairing for every id: mean |attacked - clean| <= 10 grey levels (eps 8) and <= 18 (eps 16), else abort. (The first version resized the short side to 256; it showed a different field of view for 4 of 5 classes and was dropped before any result.) Original sizes in the subset are NOT mostly 324: real short side median 250 (70 to 1444 px), gen_0 exactly 256, gen_1..3 about 360-512.
- **D3, padding shortcut.** 52 of the 100 reals are under 256 px, so after RAID's zero-pad crop they carry black borders that no fake has (clean and attacked alike). The PRIMARY analysis, and the only one that gets the B0 label, therefore uses unpadded reals (both sides >= 256) vs unpadded fakes; all 100 reals is a SENSITIVITY analysis that is reported but never labelled. Counts are reported. Bootstrap: reals and each generator's fakes resampled independently by id within class.
- The originals are already JPEG ~q75, so `clean_canon` cost is probably understated (as pre-stated).
- B0 sees 256 -> 224 crops but was trained on 512 -> 224 crops (domain shift, stated in Amendment 9). A clean AUROC < 0.80 makes the B0 label UNINFORMATIVE by the validity gate.
- Generator mapping gen_0..gen_3, caption pairing between real and gen of the same id, and upstream image licences are [unverified] (raid.md). Treat as research use only.
- The smoke numbers (25 items) are meaningless; they only prove the code path.

## Files
- `config.json`, `run.py`, `raid_manifest.csv` (input, untouched).
- `pilot/` (full run): `results.json`, `tables.md`, `scores_long.csv.gz`, `d008_guard.json`, `run.log`.
- `cache/` (git-ignored via `cache/.gitignore`): `b0_probe.pkl`, `scores_<det>_<set>_<prep version>_<device>.npz` (resume granularity: 100 items; the key includes input set, preprocessing version hash and device, so stale or other-device caches are never read), `cpuref_<det>_<ver>.json`, `d008_guard_<ver>_<device>.json`, `check_subset.json`. Smoke runs live ONLY in `cache/smoke/` (outputs in `cache/smoke/out/`).

## Commands (always `env/.venv/bin/python -I`; run from the repo root)
Synthetic self-test (no data, no models): `CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/010-raid-transfer/run.py --stage selftest`

CPU smoke (one detector at a time; --smoke N takes N ids per class, half of them with unpadded reals):
`CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/010-raid-transfer/run.py --stage all --device cpu --smoke 6 --detectors b0`
then `--stage score ... --detectors rine` (etc.) and `--stage export --device cpu --smoke 6`.

Full run, GPU (only with the owner's go-ahead; resumable, re-run the same command after any interruption):
`env/.venv/bin/python -I experiments/010-raid-transfer/run.py --stage all --device cuda`
Stage by stage: `--stage verify`, `--stage b0fit`, `--stage score --detectors rine` (one or several of b0 communityforensics rine clipdet), `--stage export`.

Full run, CPU (no D-008 guard; ~2 h at 4 threads): `CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/010-raid-transfer/run.py --stage all --device cpu --threads 4`

## Cost estimates (from the first CPU smoke, 4 threads; 3,000 forwards per detector for 500 items x 6 sets)
- CPU: about 0.4 s/img B0, 0.2 s/img CommFor, 1 s/img RINE and ClipDet, so about 2 h in total [estimate from 150-image timings including model load].
- GPU: a few minutes per detector for scoring; plus the CPU references of the D-008 guard (32 items x 6 sets x 4 detectors = 768 forwards, about 10 min).
