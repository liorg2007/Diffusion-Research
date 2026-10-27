# Critique of 020-attr-foundation (H-A1, H-A4): opus/high critic, 2026-10-27

**Verdict.**
- **H-A1 = PASS** follows docs/08 + docs/09 exactly. The 8-way 0.855 at jpeg75_resize05 is **not** a geometry or metadata shortcut: it holds between generators of identical native size (see §1).
- **Open-set usable = NO** is correct (-max softmax mean 0.768 < 0.80).
- **H-A4 = KILL** is correct and robust. It is counting failure **2 of 3** (after H-A5).
- I found no leak and no bug that changes a scored number. Two weaknesses:
  - the pairwise criterion is saturated and non-diagnostic (§2);
  - the M0-on-tensors gate cannot detect the cue it was meant to catch (§1).

Critic checks (CPU, judged pipeline untouched; committed folds/OOF + 004 fov512 cache; fixed C = 0.01, no test tuning): `critic/c1_sizes.py` (size census, size-stratified OOF), `critic/c2_probes.py` (fresh size-matched probes, scale decodability, unmatched 2-AFC), `critic/c3_raid.py` (RAID without gen_0); outputs `critic/*.txt`.

## 1. Geometry / resample-scale shortcut (fov512 = short side → 512, bicubic AA, centre crop)
**Native sizes (c1, from cache/m0_fov512.json):**
- **Short side 1024 (scale 0.5):** dalle2, sd3_large and flux are all 1024² only. dalle3's 3 sizes all have short side 1024 (100 square, 20 at 1792×1024).
- **Unique scales:** MJ has short side 896 only (scale 0.571). Firefly has short sides 1536/1792/2048 (scale 0.25–0.33).
- **SD2 and SDXL share the same 26 sizes** (short sides 576–1024), so their size distributions match.
- So a scale cue could at most separate {MJ}, {firefly}, {SD2/SDXL}, {dalle2, dalle3, sd3, flux}. It cannot separate within those blocks.

**Attribution within blocks of identical scale (jpeg75_resize05, T-mix):**

| test (chance) | source | bacc |
|---|---|---|
| dalle2 / sd3 / flux, all 1024², fresh 3-way probe (0.333) | c2 A | cat 0.964 [0.943, 0.981]; b4 0.949; b8 0.925 |
| dalle2 / sd3 / flux, argmax restricted from the judged 8-way OOF | c1 | 0.971 |
| 4-way native-1024² only, incl. square dalle3 (0.25) | c2 B | 0.969 [0.952, 0.983] |
| 1024² rows in the judged 8-way OOF, 8-way argmax | c1 | 0.927 |
| SD2 vs SDXL, identical 26-size sets, fresh probe (0.5) | c2 C | b4 0.933 [0.901, 0.963]; cat 0.921 |
| SD2 vs SDXL, restricted from the judged OOF | c1 | 0.942 |

**Other size evidence:** within SD2/SDXL, recall shows no trend with short side (c1: SD2 0.67–0.94, SDXL 0.58–1.00 across buckets); only 19% of the 142 T-mix errors land on a class with an image of the same native size; the unique-scale classes are not the easy ones (MJ is the *weakest*, recall 0.667; firefly 0.858).

**Is scale decodable at all?** (c2 D/D2, short side ≤704 vs ≥896 inside SD2+SDXL)
- On clean crops, yes: a probe trained on SD2 transfers to SDXL with AUROC 0.861 (b4) / 0.855 (cat).
- At jpeg75_resize05 it is near chance: bacc 0.540 (b4) and 0.596 (cat) [0.516, 0.674]; transfer AUROC 0.61–0.72.
- So scale is a real *clean-condition* confound, but it carries little at the primary condition. It may inflate the clean column (0.913–0.942) a little; it does not explain the 0.855.

**Gate weakness (no label change):** "M0 on preprocessed tensors at chance" uses only tensor shape, always 512² (224² for RAID), so it is chance by construction and cannot detect resample-history leakage. The checks above are the real control; future runs should add a native-scale baseline or a size-matched arm to gate 2.

**RAID (c3).**
- gen_0 is the only exactly-256-px class (scale 0.875 vs about 0.45–0.6 for the others). Its recall (0.90) matches the cue that M0 (1.00) and M1 (0.71) also read.
- Without gen_0, the 3-way among gen_1–3 (sizes not separable: M0 recall about 0.3) is 0.677 restricted (chance 0.333). That is the same relative lift, so the replication survives.
- gen_1↔gen_2 are confused (pairwise 0.690); the gen_k mapping is still [unverified].

## 2. Pairwise 1.000: computed correctly, but saturated
**Correct (code read, run.py `build_pairs` / `pair_scores` / `pair_boot`):**
- Same-family, different-class pairs (695, c2 E, as in docs/08 §1); 2-AFC s_A(a) − s_B(a) > s_A(b) − s_B(b). Families are the CV groups, so both images of a pair are always in the same test fold (asserted `inv[I] == inv[J]`). No fold leak.

**Why it is perfect.**
- The 2-AFC is a binary pairwise ranking on the A-vs-B logit, and any per-class bias cancels.
- On **all unmatched cross-family pairs** the same statistic is already 0.991 (min 0.958). Clean-train gives 0.983 even though its 8-way bacc is 0.542.
- So content matching adds about 0.01, and the 0.75 threshold could not fail. The CI [1.000, 1.000] is a bootstrap of all-correct outcomes, not precision: 24 of 28 pair types have only 10–23 pairs. For example, n = 10 all correct still allows a true rate down to about 0.74 (one-sided 95%).
- Useful reading: matched (1.000) ≥ unmatched (0.991), so the probe does **not** lean on caption content. The metric is weak evidence, not a flaw that changes the label.
- Future pre-registrations should use the strict "both" reading (0.988 here, 0.882 RAID) or matched-minus-unmatched.

## 3. Content, prompt and "style"
**Not content semantics:**
- b12 (most semantic) is the worst layer: 0.562 vs b4 0.841. Layer contrast b8 − b12 +0.254 [+0.221, +0.287].
- Same-caption pairs show no content reliance (§2). M1 colour/texture moments reach only 0.425.

**Probably "rendering style" (mid-frequency texture and tone), not a pixel fingerprint.**
- With clean-train, the high-frequency trace dies at jpeg75_resize05: 0.942 → 0.542, mavali 0.507.
- The laundered errors pile into the "smooth" classes: 183 non-sd3 images are predicted sd3, 118 predicted MJ and 96 predicted dalle3 (tables.md confusion).
- dalle3 (0.992) and sd3 (0.967) survive laundering even under clean-train. These are global rendering looks.
- T-mix learns cues that survive a 4× pixel loss. That they also hold on never-trained launderings (screenshot_sim 0.866, mavali_medium 0.813) is the strongest part of the result.
- This is legitimate attribution of a model *as deployed*. It is also the kind of cue that fine-tunes, LoRAs, style presets (MJ `--style`/version) and prompt rewriting change. Expect it to be version-specific.

**Prompt confound, not testable here:**
- KB says Synthbuster prompts were "edited to be photorealistic" (kb/datasets/synthbuster.md). Whether they were edited *per generator* is [unverified]: prompts.csv was not downloaded.
- DALL-E 3 rewrites prompts server-side [unverified for Synthbuster]. So for dalle3, "same caption" ≠ same content.
- The same-caption control is therefore weaker for dalle3 (and possibly MJ) than for the SD/FLUX classes. The 1024² and SD2/SDXL results above do not depend on that.

## 4. Labels against docs/08 §2 + docs/09 (checked against results.json and tables.md)
**H-A1 PASS.** All four primary criteria hold at jpeg75_resize05 with T-mix (the docs/08 "Training" definition):
- bacc 0.855 [0.834, 0.876] ≥ 0.60;
- CI lower bound 0.834 ≥ M1 0.425 + 0.10 (paired difference +0.430 [+0.398, +0.464], so either reading passes);
- pairwise 1.000 ≥ 0.75;
- RAID 0.688 [0.650, 0.725] ≥ 0.55.

- Nested layer choice uses inner folds only; permutation 0.125; sha256 dedupe clean. Clean-train 0.542 is descriptive (it would fail 0.60 if it were the criterion); report it next to the headline every time.

**Open-set usable: NO.**
- At the primary condition: msp mean 0.768 (min 0.703), Mahalanobis 0.546.
- At the clean condition the T-mix msp mean is 0.828, but the label is fixed at the primary condition, so it stays NO.

**H-A4 KILL** (mean novelty AUROC 0.531 < 0.70). Robust to every choice:
- b4/b8/b12 × {T-mix, clean} all fall in 0.51–0.53; kNN-10 is ≈0.55.
- Removing the three VISION platform sources (held-out siblings of the same photos, a design flaw in docs/08) does not rescue it: the 8 fakes average 0.504.
- Many sources score below 0.5 (HDRP 0.386, MJ 0.405). Tight held-out sources sit nearer the class means than the spread of known test images. The direction is right where there is signal (SD2 0.721, UNS 0.763), so this is not a sign bug.
- DISCOVERY NO (UNS kind accuracy 0.733).

**Stop-rule count (docs/09 decision 1):** H-A5 KILL = failure 1; H-A4 KILL = failure 2; H-A1 PASS, H-A2 PASS, H-A3 INCONCLUSIVE with clean 0.93 (not on the KILL side) = 0. Total **2 of 3**; all five counting hypotheses are now run; H-A6 does not count.

## 5. Novelty against the literature (kb/papers)
**Anchors:**
- rpa-arxiv2026: 98.9% caption-matched 25-way clean; open-set AUROC 0.896.
- wild-ijcnn2025: ViT 95.8 → 88.67 bacc after 3 random processing steps, with clean training and 1,000 images per class; CLIP+MLP 72.4; ViT open-set AUROC 0.857 → 0.801 under 1–3 steps.
- defake-ccs2023: 0.864 image-only / 0.936 with prompt, clean, no laundering.
- pose-cvpr2023: open-set reference.

**Assessment:**
- 0.855 at 8-way, laundered, with a frozen DINOv2 and a linear probe on about 100 images per class is a **solid, reproducible baseline, not a state-of-the-art claim**.
- It is below WILD's ViT under heavier processing. Our *clean-train* transfer (0.542) is far worse than WILD's clean-trained ViT. Neither comparison is like-for-like (faces vs RAISE scenes, different operators, n).
- What is new or useful here:
  - mid layers ≫ last layer for laundered attribution (+0.254), echoing RINE-style intermediate features;
  - transfer to never-seen launderings;
  - explicit size-matched controls (≥0.92 within identical-size blocks), which most attribution papers lack (rpa itself flags the size/format risk);
  - a negative open-set result (0.768 msp, Mahalanobis about chance), consistent with POSE/WILD showing open-set as the hard part.

## 6. Requests to the lead (shared files are not edited by me)
- **README / LOG headline:** quote "0.855 (T-mix, seen laundering); 0.866 / 0.813 never-seen launderings; clean-train 0.542". State that size-matched blocks hold (3-way 1024² 0.964; SD2 vs SDXL 0.93).
- **notes.md:** say the pairwise criterion is saturated (unmatched 0.991). Say the M0-on-tensors gate is vacuous for resample history.
- **Next cycle:** pre-register a native-scale baseline and a version-shift test (a new MJ/DALL-E version as a "same generator, new version" probe) before claiming deployable attribution.
