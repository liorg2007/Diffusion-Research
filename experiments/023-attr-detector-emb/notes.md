# 023-attr-detector-emb (H-A5) notes

**Label: KILL** (docs/08 H-A5: best detector embedding <= H-A1 best layer). Source: results/results.json `labels.H-A5`, results/tables.md.

## What was run
- Rows: the 1,020 fakes of the main K=8 set (020 ha1_main rows, same order, same caption-family folds: `fold_identity_with_020` = identical).
- Pipeline: data/processed/fov512 crop -> lib/perturb.apply_condition (key = sha256, seed 0; clean, jpeg75_resize05, social_sim, screenshot_sim, mavali_medium) -> each detector's own test transform (taken unmodified from experiments/002-baselines/scorers) -> forward hook on the layer feeding the head (commfor 384-d, bfree 768-d, rine 1024-d `proj2` output, clipdet 1024-d L2-normalised).
- Probe: the functions of experiments/020-attr-foundation/run.py imported unchanged (folds, inner-CV C on the primary condition, LR, T-mix training, clean-train arm, pairwise, LOGO, permutation task, cluster bootstrap). DINOv2 b4/b8/b12/cat were re-run through the same code on the same folds: they reproduce 020 results.json exactly (max |bacc diff| = 0.0).
- 020 `results/tables.md` did not exist when this finished (020's own report stage crashed in tables.py with KeyError 'drop_macro' after writing results.json). I used 020's code and results.json. Re-run `run.py report` once 020 is final; provenance sha256 of 020 run.py/config.json are stored in results.json.

## Headline (jpeg75_resize05, T-mix, bacc, chance 0.125; 95% family-bootstrap CI)
H-A1 best fixed layer cat 0.858 [0.838, 0.878] (nested 0.855); commfor 0.740 [0.713, 0.766]; clipdet 0.709 [0.681, 0.738]; bfree 0.660 [0.630, 0.688]; rine 0.157 [0.136, 0.176]. Paired diff best detector vs cat: -0.118 [-0.149, -0.087]. No detector beats DINOv2 b4/b8/cat at any condition except b12 (which is weak); every detector is below cat at all 5 conditions.
- RINE: z is collapsed (933/1024 dims always 0, effective rank 2 at 95% variance) because the binary-trained head squeezes it through ReLUs; hooks verified (head(z) == logit). Not a bug, but it means RINE's penultimate vector carries ~no generator identity. Its per-block CLS stack (not used) might; out of the pre-registered scope.
- B-Free and CommFor (fine-tuned ViTs) lose most under jpeg75_resize05 (clean 0.86-0.88 -> 0.66-0.74); ClipDet is flatter but lower.
- Open set: no embedding is "open-set usable" (best mean LOGO AUROC: cat msp 0.770; detectors <= 0.656 msp, <= 0.595 maha).
- Overlap flags: tables.md, from config.json `overlap_flags`. Overlapped classes (SD2 for B-Free; SD2/SDXL family for CommFor/RINE/ClipDet) do not stand out in recall: B-Free SD2 0.683 vs dalle3 0.825 (no overlap); so leakage is not what limits these embeddings.

## Gate checklist (docs/08 §3; docs/09 gate 4 -> INVALID if failed)
- [x] 1 Identical preprocessing: one code path (`load_cond` + the detector transform) for all rows; 1,020 crops are 512x512 RGB with no EXIF/ICC (gate `identical_preprocessing`).
- [x] 2 Baselines shown: M0 raw 0.688 [0.665, 0.710], M1 (T-mix) 0.425 [0.397, 0.452] from 020 (same rows/folds); M0 on tensors at chance (<= 0.137). Not part of H-A5's own rule.
- [x] 3 Grouped splits: caption-family folds identical to 020 (sha256 bc2894a4...), assert_disjoint_groups in every task, pairwise 2-AFC computed on 695 same-family pairs.
- [x] 4 Laundering: all 5 conditions; screenshot_sim and mavali_medium never used for training; clean-train transfer reported.
- [x] 5 Label-leak checklist: C by inner CV on train folds only; 200-shuffle permutation per detector at chance (means 0.1251-0.1268, p95 <= 0.1454); sha256 dedupe ok; D-008 ok (below). Best-of-4 detector choice is on test results: reported with Holm p (all 1.0000, irrelevant for KILL).
- [x] 6 Chance levels in every table.
- [x] D-008: 32 images x 5 conditions, GPU vs CPU (TF32 off): worst scaled embedding diff 5.3e-05, worst |dlogit| 1.6e-04 (tol 1e-3), all four detectors ok (results/d008_guard.json).
- [x] Selftest (results/selftest.txt): 0 failed (hook placement, official-scorer logit match, batch invariance, 5-window B-Free path, planted leaks, label rule).

## Deviations / interpretation choices
- "H-A1 best layer" = max point bacc over fixed b4/b8/b12/cat (conservative); nested value reported next to it.
- Detector embeddings use the DINOv2 C grid (same probe). RINE embedding = proj2 output z (head input); per-block CLS stack not probed.
- Permutation control uses 020's `perm_closed_task` (fixed C = 0.01, T-mix, primary condition) per detector.
- Rate-limit interruption: the stages had all finished (caches/guard/probe pickles complete); only `report` was (re)run.

## Post-result changes (bugs only)
- run.py `stage_report`: `labels.all_gates_ok` was never set, so tables.md printed "Gates all ok: None". Added the assignment (display only). Before: results/results.json copy in the session scratchpad (label KILL, same numbers). After: label KILL, all_gates_ok True. No numerical change.
