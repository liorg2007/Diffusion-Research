# E6 (015-midlayer-probe) critique: independent critic, opus/high, 2026-10-25

**Rating: the code is sound and I found no leakage bug, so all three labels stand. P1's gain is real but local to the two known real sources. It sits mostly in the RAISE pool, and its FLUX part depends on the training set. P2's failure is a genuine false-positive blow-up on an unseen real pipeline, and much of it is driven by image scale (fov512 cross-source 0.826 vs orig 0.644). Raising the limit is worth exactly one bounded, pre-registered test (E7, leave-one-real-source-out). It is not worth a method claim.**

Scope: CPU only, no downloads. `run.py`, the config and the caches were not modified. My checks import `run.py`'s own `Arm`, `Plan`, `TwinPlan` and `diff_summary` (the same bootstrap plan, seed 0, 1,000 replicates). Files in `results/critic_checks/`:

| Check | What it does | Files |
|---|---|---|
| c1 | Split facts; cross-source diagnosis (FPR vs missed fakes); per-source thresholds | `c1_data_crosssource.{py,out,json}` |
| c2 | Paired decomposition of the gain: layer vs laundering augmentation, per real pool, plus twin-paired | `c2_layer_vs_aug.{py,out,json}` |
| c3 | Refit on the larger 003 `b0_train` rows (training-size control); RAISE-only learning curve | `c3_trainsize.{py,out,json}` |
| c4 | P3 tail: false positives per device, calib-vs-test quantiles | `c4_fpr_tail.{py,out,json}` |
| c5 | Cross-source transfer, block 8 vs block 12, paired | `c5_crosssource_layer.{py,out,json}` |

All numbers below are at `jpeg75_resize05`, arm `orig`, as LOGO-macro over 8 generators with 95% CI, unless stated otherwise.

## Findings, ranked

### F1 (HIGH): P2 fails because unseen real photos are flagged as fake, not because fakes are missed
The fakes are identical in both comparisons, so the whole drop from in-source to cross-source comes from where the new reals land (c1, `full` model, F_b8 T-aug):

| Probe trained on | AUROC fakes vs RAISE test reals | AUROC fakes vs VISION test reals | Unseen reals vs seen reals | FPR on the unseen source at a 1% threshold calibrated on the training source |
|---|---|---|---|---|
| RAISE only (orig) | 0.917 | 0.674 | VISION scores above RAISE: AUROC 0.855 | VISION natives **41.9%** (clean 75.7%); RAISE test 2.2%; TPR 0.57 |
| VISION only (orig) | 0.656 | 0.997 | RAISE scores above VISION: 0.992 | RAISE **72.7%** (clean 96.7%) |
| RAISE only (fov512) | 0.944 | 0.848 | 0.780 | VISION natives 12.8%; TPR 0.70 |

- **The failure is spread across phones.** With the RAISE-only probe at a 5% RAISE threshold, the per-device FPR on VISION natives runs from 33% to 93% across all 25 test phones (c1). It is not the one-phone D17 tail seen with B0.
- **Block 8 is the more source-sensitive layer.** Trained on RAISE only, block 8 separates VISION reals from RAISE reals at 0.855, against 0.665 for block 12. Its in-source advantage over block 12 is 0.917 vs 0.805. That advantage disappears on the new pipeline in `orig`:
  - cross-source block 8 minus block 12 is −0.021 [−0.058, +0.018] at jpeg75_resize05, −0.138 clean, −0.101 screenshot_sim;
  - on FLUX it is −0.163 [−0.236, −0.092] (c5).
- **Scale explains a large part.** With fov512 (every image resized to a 512 short side), RAISE→VISION rises to 0.826. There, block 8 does beat block 12 cross-source: +0.066 [+0.032, +0.102]. The FLUX difference is −0.000, and the clean condition is −0.018 (not significant).
  - In orig, a RAISE image is a 4×-downscaled DSLR picture (B-Free), while a VISION crop is at native phone resolution.
  - So in orig the probe's "real" direction is largely "looks like a downscaled RAISE image".
- **The failure is not a sample-size artefact.** RAISE-only cross-source with 21, 42 and 84 RAISE train reals gives 0.60–0.75, 0.64–0.68 and 0.674 (c3 B, flat). The larger 003 rows (140 RAISE reals plus 2× fakes, with twin leakage) give 0.777, still below 0.85. More images from the same source do not fix it; it needs more *kinds* of real pipeline.
- **The VISION→RAISE direction is not a fair test.** All 1,020 fakes are prompted from RAISE-1k captions (`fakes_with_caption_id_frac` = 1.0, c1). A probe trained on VISION reals can therefore separate the classes by content, and RAISE reals share the fakes' content. That direction's failure (0.604) is mostly a content confound and should not be weighted equally with RAISE→VISION. RAISE→VISION is the informative direction: training there is content-matched, and the VISION reals were flagged *despite* having different content from the fakes, which should have made them easier to call real.

### F2 (HIGH): P1 is genuine, but the gain sits where the probe can exploit a known real source
Paired decomposition (c2), point estimate with 95% CI:

| Contrast | Macro over 8 | FLUX | RAISE pool | Native pool | Twin-paired |
|---|---|---|---|---|---|
| Layer at T-aug (F_b8 − F_last, both T-aug) | +0.064 [+0.044, +0.087] | +0.026 [−0.015, +0.071] | **+0.133** | +0.028 | +0.148 |
| Augmentation, last block | +0.028 [+0.019, +0.038] | +0.050 [+0.033, +0.069] | +0.036 | +0.025 | +0.040 |
| Augmentation, block 8 | +0.046 | +0.097 | +0.078 | +0.030 | +0.091 |
| P1 total vs matched B0 | +0.092 | +0.077 | +0.169 | +0.053 | +0.188 |

- **Layer vs augmentation.** About two thirds of the macro gain comes from the layer and one third from laundering augmentation, with an interaction of about +0.018. The FLUX gain comes from augmentation and the interaction; the layer alone adds nothing significant on FLUX.
- **The gain holds on launderings never trained on.** Layer at T-aug is +0.059 macro on screenshot_sim and +0.052 on mavali_medium. On FLUX it is +0.056 for screenshot_sim and +0.014 [−0.021, +0.053] for mavali_medium.
  - social_sim is used in training **and** is one of the three D-009 conditions. Only screenshot_sim and mavali_medium are truly held out.
- **The gain is concentrated on RAISE**, the in-source, content-controlled pool. That is where F1 shows block 8 is most source-specific. The data cannot tell "block 8 sees diffusion artefacts better" apart from "block 8 recognises the RAISE pipeline better". Pre-selection of block 8 (from E5, same images) is not a worry, because F_cat (no layer chosen) gives the same 0.945.

### F3 (MEDIUM): the shrunken training set does not inflate the macro gain, but P1's FLUX clause depends on which training set is used
c3 refits on the 003 `b0_train` rows (413 shared + 261 more rows, about 2× fakes; twin leakage is symmetric for both methods). F_b8 T-aug vs F_last T-clean:
- macro **+0.082 [+0.057, +0.112]** (E6: +0.092);
- FLUX +0.040 **[−0.007, +0.089]**: the CI lower bound is below 0, so the FLUX criterion would *fail*.

F_last gains more from extra data on FLUX (0.824 vs 0.797). Layer at T-aug on FLUX: +0.010 [−0.033, +0.052]. The label "P1 PASS" should therefore carry this caveat: robust on macro, fragile on FLUX.

### F4 (MEDIUM): P3 has two causes, a pooled threshold over two real populations and a block-8 sensitivity to JPEG history
- **Different score levels.** In the joint model, RAISE reals score higher (more "fake") than VISION reals: VISION-vs-RAISE AUROC 0.30 (c1).
- **The pooled calib set is 71% VISION copies.** It holds 129 VISION rows (43 images × 3 copies) and 53 RAISE rows, so the effective n is about 96 images and the "1%" cut-off is roughly the second-highest score.
- **A RAISE-only calib threshold would give RAISE 2.7% and natives 0.0%.** That needs the source to be known, which is impossible in the wild, so P3 FAIL stands.
- **False positives spread across phones (c4).** With F_b8 T-aug, they are spread over about 15 devices, mostly WA/FBH copies (WA 3.1%). This is not B0's D17 tail.
- **Block 8 reacts to platform recompression.** Trained clean, block 8 scores WA copies 0.891 vs natives 0.980 at clean, and the WA FPR is 6.4% at jpeg75_resize05. T-aug largely repairs this, and fov512 removes it: per-pool AUROC is equal there, and FPR is 0.6–0.8% on the VISION pools and 2.7% on RAISE.
- **Wording fix for notes.md.** The calib phones are also training phones, so calib reals score slightly *lower* than test reals (99th percentile 0.63 vs 1.08 at clean). The threshold therefore comes out too low, and the measured test FPR (unseen devices) is honest or slightly pessimistic. The *nominal* 1% is what is optimistic.

### F5 (LOW): no leakage found in the questions asked
- **Twin families.** No caption id is shared between train and calib/test (self-test, re-derived in c1). Calib and test may share ids, which only affects the threshold.
- **Leave-one-generator-out (LOGO).** Each held-out generator's test fakes belong to eval-side families, so even their *captions* are unseen.
- **Augmented copies** carry their source's group into the StratifiedGroupKFold. This only affects the choice of C.
- **Device split.** Test VISION devices are disjoint from train devices: 10 train/calib phones, 25 test phones.
- **FLUX** has no train rows; 19 FLUX rows were dropped because their families were assigned to the train side.
- **Comparators.**
  - The matched B0 uses the same rows, groups and CV, so only the features differ.
  - The 003-literal B0 trains on rows of which 261 share caption ids with E6 eval rows. That favours B0, so it is the conservative comparator, as the notes say.
  - All methods are scored on identical test rows with one shared bootstrap plan.
- **Minor.** C hit the grid edge in several fits. The macro CIs do not resample twin families jointly, so they are slightly narrow.

## Recommendations
1. **Keep the labels** (P1 PASS, P2 FAIL, P3 FAIL), with two edits:
   - P1's FLUX clause fails on the 003 training rows (F3);
   - most of the layer gain is on the in-source RAISE pool (F2).
   Record in the KB, as a negative result: "block 8 adds in-source power that does not transfer to an unseen real pipeline in orig; partial transfer in fov512".
2. **Make fov512 the primary arm for any follow-up.** Scale accounts for about +0.18 of the cross-source AUROC. fov512 is the only setting where block 8 beats block 12 across sources, and it removes the WA/FBH recompression effect.
3. **Do not treat VISION→RAISE as a symmetric falsifier** while every fake carries a RAISE caption (F1). The fair falsifiers are training that is content-matched or content-diverse, and FPR on a held-out real source.

## Plain-language verdict for the owner
The new detector does catch more laundered fakes than our baseline: about 0.95 vs 0.86. But it learned partly "what our two real photo collections look like" rather than "what a generated image looks like". Trained on one collection, it labels 40–75% of photos from the other collection as fake (13% once all images are resized to the same scale). An unseen phone or camera is exactly the in-the-wild situation we care about, so this is not yet a method. There is one honest reason for hope: with scale normalised (fov512), the mid-layer transfers better than the baseline layer (+0.07) and comes close to the bar (0.83 vs 0.85). That justifies **one** more bounded experiment, and closing the line if it fails, rather than an open-ended extension.

## Proposed pre-registered E7: leave-one-real-source-out (LORSO), fov512 primary
- **Real sources (at least 3, ideally 4; equal training counts, 150 each, subsampled; at least 150 test images per held-out source):**
  - S1 RAISE (B-Free, on disk; non-commercial);
  - S2 VISION natives (on disk; CC BY-SA 4.0);
  - S3 Open Images V7 photos, filtered to licence == CC BY via the official ID lists and downloader. Use original files, never thumbnails (see `kb/datasets/openimages-v7-ccby.md`; record per-image licences in `env/SOURCES.md`; well under 1 GB);
  - S4 the owner's own phone photos: at least 150 unedited originals from at least 2 recent phones with neural ISPs and varied scenes. The licence is clean, it matches the Phase 5 target, and no download is needed.
  - Optional pilot only: the 100 D3/LAION reals already on disk under `data/raw/raid/original/real`. They are tiny web JPEGs and the upstream licence is [unverified].
- **Fakes:** the same 8 generators, twin-disjoint as in E6, with FLUX test-only. Add the on-disk D3 fakes (`raid/original/gen_0..3`, LAION captions) to the training fake pool, so the fakes do not all share RAISE content.
  - Report the RAISE-held-out fold separately as content-confounded.
  - If owner-approved generator weights become available, caption-twins for S3/S4 would remove the confound (over 5 GB, so this needs approval first).
- **Protocol:** F_b8 vs F_last (same rows), T-aug (clean + jpeg75_resize05 + social_sim). For each held-out source h:
  - train on the other sources plus the fakes;
  - set the threshold at 1% FPR on the training sources' calib reals (clean, pooled);
  - report the FPR on h (all 5 conditions, Clopper-Pearson CIs);
  - report LOGO-macro AUROC and FLUX for h-reals vs test fakes;
  - report twin-paired AUROC where twins exist;
  - add a k-sources curve (train on 1, 2, 3 sources).
- **Pass (all must hold, fov512, jpeg75_resize05):**
  - (a) for every non-confounded held-out source: AUROC ≥ 0.85 **and** FPR ≤ 5% at the transferred 1% threshold;
  - (b) mean LORSO AUROC, F_b8 − F_last ≥ +0.03 with CI lower bound > 0, and FLUX not worse;
  - (c) held-out AUROC rises from k=1 to k=3 training sources.
- **Kill rule:** any held-out source with AUROC < 0.75 or FPR > 10%, or (b) fails. Then close the mid-layer line and record it in the KB as a negative result.
- **Power:** 150 reals per source detects FPR > 5% (Clopper-Pearson). Certifying ≤ 1% for Phase 5 would need at least 300 reals with 0–1 errors per source, so E7 is a go/no-go test, not the Phase 5 certification.
