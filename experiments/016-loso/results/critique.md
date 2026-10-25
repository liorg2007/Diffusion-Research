# E7 (016-loso) critique: independent critic, opus/high, 2026-10-25

**Rating: no bug or leakage found, and no deviation can flip the label. KILL is the correct reading of Amendment 16, so the mid-level line should close as pre-registered. The OI kill is a real-source domain effect. The OI photos the detector flags are polished "enthusiast" photographs: telephoto wildlife, shallow depth of field, saturated colour, bokeh and tone-mapped HDR. It is not caused by grayscale images, JPEG history, EXIF or image scale.**

Scope: CPU only, no downloads. `run.py`, the config, the caches and the results were not modified. The checks recompute everything from `results/scores_long.csv.gz`, the splits files and the raw OI files with my own code, using plain sklearn AUROC and my own threshold code following the rule documented in `metrics.threshold_for_fpr`. Files are in `results/critic_checks/`.

| Check | What it does | Files |
|---|---|---|
| c1 | Independent recomputation of the FPRs, calibration-set composition, FLUX and the OI split | `c1_recompute.{py,out,json}` |
| c2 | Anatomy of the OI false positives (mode, JPEG quality, bpp, size, EXIF), alternative thresholds, montages | `c2_oi_fp_anatomy.{py,out,json}`, `c2_oi_top32.csv`, `c2_oi_top18_fp.jpg`, `c2_oi_lowest12.jpg` |
| c3 | fov512 scale factor per pool; score shift of the held-out reals vs the fakes | `c3_scale_shift.{py,out}` |

Unless stated otherwise, every number is for the fov512 arm at jpeg75_resize05 with F_b8, from the `full` model trained on all other sources.

## Q1: bugs, leakage, deviations
**F1 (verified): the headline reproduces exactly from the raw scores.**
- FPR reading A: VISION 6/1074, RAISE 35/183, OI 32/200. Reading B: 0.7% / 26.8% / 23.0%. The orig arm matches too (c1).
- FLUX AUROC recomputed: 0.880 / 0.627 / 0.689 (c2).
- **Calibration sets contain only the training sources** (asserted in c1). n_calib is 153 / 229 / 182 = RAISE 53 + OI 100, VISION 129 + OI 100, VISION 129 + RAISE 53.
- **OI split** is 300 / 100 / 200, by image, with unique sha256 values.
  - 8 Flickr authors appear in both OI train and OI test. This is irrelevant to the OI fold, because OI is held out whole there and its train rows are unused (the self-test checks this).
- My plain macro AUROC from `scores_long` is lower (OI 0.772 vs 0.809). This is expected and is **not** a discrepancy. `scores_long` stores the reals' `full`-model scores, whereas run.py correctly scores both classes with the same per-generator LOGO model. The FLUX cells, where both classes come from `full`, match exactly.

**F2: no deviation can flip KILL.** OI alone triggers the kill on FPR, and that does not depend on any interpretation:
- **Clopper-Pearson CI** [11.2%, 21.8%]: its lower bound is above the 10% kill line. The cluster-bootstrap CI {11.0%, 20.5%} is also entirely above it.
- **Other thresholds** (c2). The pre-registered pooled threshold equals the VISION-only threshold (3.35). A RAISE-only threshold gives **28.5%**, and the same-condition threshold (reading B) gives 23.0%. Every threshold available without knowing the image's source gives at least 16%.
  - Reaching 10% FPR on OI would need a threshold of 5.44. At that threshold the TPR on test fakes is **0.29**, so the AUROC of 0.81 is weak at any useful operating point.
- **Deviations.** Each one leaves the label unchanged:
  - fov224 conditions: secondary arm.
  - D3 60/100: pilot-only source.
  - Family-joint bootstrap: CIs only. The labels use point estimates.
  - Clean vs same-condition calibration: both give KILL.
  - Missing equal-count subsampling: in the OI fold the training reals are VISION 99 + RAISE 84, so OI's 300 never trains there.
  - Class balance: `class_weight="balanced"`.
- **OI re-encoding.** The CVDF copies are re-encoded and resized: 0/600 match OriginalMD5, 594/600 have a 1024 px long side, none has EXIF. They then go through the same fov_prep (resize, centre crop, PNG) and the same laundering code as every other pool. Nothing is processed differently for one class.

**F3 (minor, disclosure): the RAISE fold fails partly by construction.** All fakes are prompted from RAISE captions. When RAISE is held out, the probe can learn "RAISE content = fake" (E6 critic F1).
- Amendment 16 did not exempt this fold, so the pre-registered label counts it.
- The notes correctly say that the label does not depend on it.

## Q2: is it a real-source domain effect? Yes
**F4: the whole OI distribution shifts, not just a tail** (c2, c3).
- Median score: OI test reals −0.72, VISION calib −6.06, RAISE calib −4.33, test fakes +3.61.
- 15.5% of OI reals score above the median fake.

**F5: what the flagged images look like.**
- **Montage** (`c2_oi_top18_fp.jpg`): the top-scoring OI reals are polished, enthusiast or "stock-like" photographs: a cockatoo on blue, a robin, a seagull close-up, vintage cars, a lion, a parasail against the sky, a light-painting orb, bokeh lights, a tone-mapped pinball machine.
- The lowest-scoring OI reals (`c2_oi_lowest12.jpg`) are casual snapshots: children, a passport on a table, a whiteboard, a wrestling event under flat light.
- So the probe's "real" direction is largely "casual snapshot". With OI held out, the only real training data is VISION phone snapshots plus RAISE, and RAISE content is shared with the fakes.
- **Not grayscale.** There are only 2 'L'-mode images in OI test (1 flagged); 8 of the 11 L-mode images are in train. FPR without them is 15.7%.
- **Not EXIF.** No OI file has any.
- **Not size.**
  - Medians are equal in the FP and non-FP groups (MWU p > 0.3).
  - Spearman correlation of score with bpp / JPEG quality: 0.03 / 0.06.
  - FPR by JPEG-quality bin is not monotone: 17% (q ≤ 85), 13%, 3%, 30% (q ≥ 98). The q ≈ 99 exports (bpp ≈ 0.8, typical of a camera or Lightroom export) are over-represented. That fits the enthusiast-photography reading, not a JPEG-artefact reading.
- **Not scale.** fov512 down-scale factors: OI 0.67, RAISE 0.61, fakes 0.29–0.67, VISION natives 0.21, WA 0.53 (c3). OI is scale-matched to the fakes and to RAISE.
  - In the VISION fold, the WA copies (0.53) have the same 0.6% FPR as the natives (0.21).
- **Caveat.** Style vs content was judged by eye on 30 images. That is suggestive, not a controlled test. E7 cannot separate "aesthetic style" from "Flickr pipeline" formally.

## Q3: is KILL the right reading, and should the line close? Yes
**F6: KILL is unambiguous.**
- Amendment 16's kill rule is "any held-out source with AUROC < 0.75 or FPR > 10%".
- **RAISE** trips both parts; **OI** trips the FPR part with its CI lower bound above 10%. OI is the content-fair fold, because its content differs from the fakes, which should have made it *easier*.
- The label holds in both readings, all 10 drop-one subsets and all three arms.
- **Mixed evidence:** criteria (b) b8 − last +0.057 and (c) learning curve +0.032 are met. Real, but secondary:
  - Each added source buys about 0.03 AUROC.
  - Adding RAISE *raised* OI's FPR, from 10.5% (VISION only) to 16.0%.
  - At that rate the 0.85 / 5% bar needs many more real pipelines, with no guarantee that FPR follows.
- **Closing is justified.** D-016 granted exactly one experiment with a pre-committed close-on-KILL, and the result is not borderline. Record it in `kb/` as a negative result with the positive sub-findings (block 8 > block 12 across sources; more source diversity helps AUROC) and with F5's mechanism.

## Q4: plain-language summary for the owner
- **What we established.**
  - A frozen DINOv2 feature probe detects laundered fakes (JPEG, resize, social-media simulation) well *on the kinds of real photos it was trained with*: about 0.95 AUROC (E6). Laundering alone does not erase the difference between real and generated images (E5).
  - Every time we test on a **new kind of real photo**, many genuine photos get called fake: 16–19% here, against a 1% target. The cause is not compression or resolution. The detector has partly learned "what our real collections look like", for example casual phone snapshots, rather than "what a diffusion image looks like". So beautifully composed, vivid, professional-looking real photos look "AI" to it.
  - That is exactly the risk for modern phone photos with computational bokeh and HDR.
  - Our hand-built "diffusion-specific" hypotheses (H1, H2, H4, VAE traces, reconstruction scores) all failed or were uninformative under the pre-registered rules.
- **The single most valuable next step:** write the **Phase-5 report of findings now**, as a negative-results-plus-lessons report. Its core is the source-shift false-positive mechanism above, together with the evaluation lessons (twin leakage, zoom/scale shortcut, pooled-threshold pitfalls, RAISE content confound).
  - Secondary: ask the owner for ≥ 150–250 of their own phone photos (H3). They cost no download and are the Phase-5 acceptance test. Any future detector, including the E7 probe as a reported datapoint, should be scored on them before any method claim.
  - Do not reopen the mid-level line.
