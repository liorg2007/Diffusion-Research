# Critique of 023-attr-detector-emb (H-A5): opus/high critic, 2026-10-27

**Verdict: the KILL label is correct under the docs/08 rule. It counts as a failure for D-018, with one deviation from pre-registration (RINE) that must be logged.**

## 1. Label check (docs/08 H-A5)
- The rule says KILL if the best detector embedding's bacc at jpeg75_resize05 (T-mix) is ≤ the H-A1 best layer.
- The best detector is CommFor at 0.740 [0.713, 0.766]. The H-A1 best fixed set is cat at 0.858 [0.838, 0.878]; the nested value is 0.855. The paired diff is -0.118 [-0.149, -0.087] (results/tables.md). So 0.740 ≤ 0.858, and the label is KILL.
- The KILL holds by a wide margin:
  - Every detector is also below the single layers b4 (0.841) and b8 (0.816).
  - Every detector is below cat at all 5 conditions (tables.md, "Same diff" table).
- Two choices of what to compare both favour the detectors, so they cannot cause a false KILL:
  - The best of the 4 detectors is picked on test results.
  - Using "max fixed DINOv2 set on test" (0.858) instead of nested (0.855) inflates DINOv2 by only 0.003.

## 2. Fairness
**C grid (checked).** The 020 DINOv2 grid tops out at C = 0.1, and detectors often pick that boundary: CommFor in 3/5 folds for T-mix and 5/5 for clean (results.json `fs/*/chosen`). I ran an oracle sweep, which picks C on test and is therefore optimistic: critic/csweep.py, output in critic/csweep_commfor_bfree_clipdet.json. Bacc is flat or falling above 0.1:

| detector | C = 0.1 | C = 1 | C = 10 |
|---|---|---|---|
| CommFor | 0.739 | 0.734 | 0.736 |
| B-Free | 0.664 | 0.654 | 0.644 |
| ClipDet | 0.703 | 0.691 | 0.677 |

The grid does not limit the detectors.

**Dimensionality and layer.**
- DINOv2 cat is 4,608-d: three layers, each CLS+mean. The detectors give 384–1,024-d, one pooled vector from the last layer.
- Dimension alone does not explain the gap: single-layer b4 and b8 (1,536-d each) beat CommFor too.
- There is a structural asymmetry. Each detector is probed only at its last layer, where binary training compresses the most. DINOv2 gets its best layer. The DINOv2 analogue of a "penultimate layer" is b12 (0.562), and CommFor, B-Free and ClipDet all beat it (+0.10 to +0.18).
- This follows the pre-registration ("penultimate embeddings"), so the label stands. But the result does NOT show that detector backbones lack generator identity: their intermediate blocks were never probed.

**Input resolution.** Each method gets its own published test transform. This is identical for all classes, so there is no leak.
- DINOv2 sees a native 224 centre crop (004 `PRIMARY_CFG` crop_mode=crop).
- RINE sees a native 224 centre crop.
- CommFor applies Resize(440) then CenterCrop(384). At jpeg75_resize05 the image is 256 px, so it gets upsampled ×1.7.
- B-Free sees the full image.
- ClipDet applies a bicubic Resize(224): on clean images this is a ×0.44 downscale, which removes high-frequency traces.
- This is fair to the detectors as published. It does mean ClipDet's lower number is partly a resolution effect.

**RINE: a deviation from pre-registration that is mis-described in notes.md.**
- docs/08 H-A5 names the RINE embedding explicitly as the "CLIP-L/14 per-block CLS stack". The run probed the `proj2` output instead. That output has 933/1,024 dims always 0 and effective rank 2; I reproduced both numbers (critic/csweep_*.json `rine_diag`).
- notes.md calls the CLS stack "out of the pre-registered scope". That is wrong: the CLS stack IS the pre-registered embedding.
- Could the CLS stack change the label? Strictly by the letter, it is untested. In substance, no:
  - RINE's CLIP backbone is frozen OpenAI CLIP (scorers/rine.py docstring). Its per-block CLS stack is a generic foundation-model feature, not something detector training shaped.
  - A high number there would measure CLIP-L against DINOv2-B (an H-A1-style question). It would not support H-A5's mechanism ("trained with augmentation on many generators").
- The same applies to ClipDet. Its embedding is a frozen CommonPool CLIP feature, and only the linear head was trained (config.json `overlap_flags.clipdet.note`).
- So only CommFor and B-Free test H-A5's mechanism, and both fail it cleanly.
- Recommendation:
  - Log the deviation in DECISIONS/LOG.
  - Optionally close the letter with one cheap GPU pass: RINE CLS stack on clean, jpeg75_resize05 and social_sim, about 3k CLIP-L forwards. Label it as a descriptive CLIP-vs-DINOv2 comparison, not a rescue of H-A5.

## 3. Leaks
None found. The checks:
- Folds are identical to 020 (sha256 bc2894a4…), and DINOv2 reproduces 020 exactly (max |diff| 0.0).
- Every detector's permutation test is at chance (means 0.1251–0.1268).
- Dedupe passes. D-008 passes: worst |dlogit| 1.6e-4, against a 1e-3 tolerance.
- Training-set overlap would favour the detectors, so it cannot produce a false KILL. The overlapped classes do not stand out anyway (B-Free SD2 recall 0.683).
- One residual I did not byte-verify: DINOv2 features come from the 004 cache. 023 re-implements the same `apply_condition(im, cond, key=sha, seed=0)` call (run.py `load_cond` vs 004 run.py:101). Same call, so the risk is low.

## 4. Interpretation: why detectors carry less generator identity here
- **Much of the loss comes from laundering, not from collapse.**
  - On clean images CommFor ≈ b4 (-0.016 [-0.040, 0.008]), and its same-family pairwise score stays at 0.985.
  - The gap opens at jpeg75_resize05 (-0.100 vs b4). CommFor and B-Free lose 0.14 and 0.20 from clean, while cat loses 0.06.
  - Fine-tuned real/fake detectors rely on low-level traces, and JPEG plus a ×0.5 downscale destroys exactly those.
  - DINOv2's mid layers carry mid-level texture and style cues that survive laundering. The generator "look" (MJ, DALL-E 3) is likely part of what is being attributed. Pairwise scoring removes content, but not style.
- **"Collapse into a fake direction" shows up in pure form only in RINE's `proj2`** (rank 2, bacc 0.157 ≈ chance 0.125). For CommFor and B-Free the collapse is partial.
- **The detectors add a little information that DINOv2 lacks.** In an oracle-C fusion (critic/csweep_cat_cat+commfor.json, no CI, descriptive), cat+CommFor scores 0.880 against cat alone at 0.861, about +0.02.
- **Open set:** the detectors are also worse (best mean LOGO AUROC 0.656 msp, against cat's 0.770).

## 5. Is the comparison meaningful while H-A1 is uncritiqued?
- Yes, as a relative claim. H-A5 is a paired comparison: same 1,020 rows, same caption-family folds, same imported probe, and DINOv2 reproduced bit-for-bit.
- The H-A1 risks (style attribution, aspect-ratio crops, Firefly's JPEG history) apply to both arms of the comparison, so the paired diff stays valid.
- Only a DINOv2-specific defect, for example a different pipeline in the 004 cache, would undermine the KILL. Nothing points to one.
- Housekeeping: 020 results/tables.md now exists, so rerun `run.py report` as notes.md says.

## 6. Counts as a D-018 failure?
**Yes.** The pre-registered KILL criterion is met by a wide margin. It is robust to the choice of C, and every possible selection bias favoured the detectors.

Caveat to record: the RINE embedding deviated from pre-registration. Because RINE's backbone is frozen CLIP, the deviation does not touch the hypothesis's mechanism.
