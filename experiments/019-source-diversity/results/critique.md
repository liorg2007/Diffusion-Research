# E10 (019-source-diversity) critique: independent critic, opus/high, 2026-10-27

**Rating.** No bug or leak found. All five labels follow Amendment 19 and its addendum exactly. **KILL OF LINE is correct and robust**: it holds in readings A and B, in all 10 drop-one-generator subsets and in all 3 drop-one-source pools. The lead's reading, "diversity mainly recalibrates; separability is capped by a linear probe", is **half right**:
- The FPR cut is mostly a move along the ROC.
- The cap is **not** the linear head. It is the frozen F_b8 features together with the unseen-generator problem. **Even in-distribution, this probe tops out at a mean AUROC of 0.855** (LOGO, jpeg75_resize05).

Everything in Q3 and Q4 is **post-hoc**. CPU only. Scripts and outputs are in `critic/` (`c1`–`c7` `.py`/`.out`/`.json`). Nothing outside `critic/` and this file was written.

## Q1. Bugs and leaks: none found
- **Held-out source excluded.** I read `run.py`:
  - `audit_train_rows` (with S) rejects any pool of S in the training rows.
  - `audit_calib` restricts threshold rows to calib reals of the sources in T only.
  - `audit_eval` restricts the evaluation reals to S's test reals.
  - These run for every fold (216 audited, `tables.md`). The k=2 and k=5 FPRs recomputed from the cached scores match `tables.md` to 3 decimals (`c1_thresholds.out`).
- **Transforms are identical across classes.** Every row is a `fov512/<sha256>.png` crop (`audit_same_prep`), and the T-aug conditions are the same for both classes (`audit_taug`). The new-source features pass D-008, max |Δlogit| 1.4e-4 (`d008_guard.json`).
- **Near-duplicates.** No test real has a block-12 CLS cosine > 0.95 to any non-test real of another source or to any training fake. The maximum is ≤ 0.91 (`c5_semantic.out`).
- **Size-matched control.** It is fair for the AUROC part, which is the only part the label uses:
  - training reals are subsampled by whole groups, achieving exactly the target count (514/520/433);
  - the fakes are unchanged.
  - Caveat: the **calibration set is not size-matched**. The control keeps all 5 sources' calib reals, so its FPR part (−0.108, against −0.095 for full k=5) is not a clean "diversity vs quantity" test. This does not affect any label.
- **Key secondary is paired correctly.**
  - The refits reproduce E7's cached scores exactly (parity 0.0 on all 6 sets).
  - E7's published AUROCs *and* CIs reappear: VISION 0.917 [0.893, 0.938], RAISE 0.689 [0.650, 0.726], OI 0.809 [0.774, 0.843] (`tables.md`). So the Plan weights on E7 rows are identical.
- **HDRP split (22/9/12 groups).** No leak: HDRP's train rows are never used when HDRP is held out. But the test set is effectively a handful of shooters: the 3 largest groups hold 134/201 test images (48, 45, 41), and test is 99 Nexus 6, 70 Pixel XL and 22 Nexus 6P images. A bootstrap over 12 clusters is anti-conservative, and Clopper-Pearson (which assumes independence) understates the FPR uncertainty. It cannot change a label, since HDRP is not the binding source.
- **Housekeeping (not bugs).** `splits_fov512.csv` is an untracked run output that the tables depend on; the lead decides whether to commit it. `data/raw/hdrplus` (2.6 GB, 600 originals) is still on disk, although Amendment 19 says it is deleted after prep and caching.

## Q2. Labels: correct per Amendment 19

| Label | Rule | Value (reading A, `tables.md`) | Verdict |
|---|---|---|---|
| DIVERSITY HELPS | ΔAUROC ≥ 0.03, CI lo > 0, and ΔFPR CI hi < 0 | +0.040 [+0.032, +0.049]; −0.095 [−0.107, −0.084] | **True** ✔ |
| NOT QUANTITY | control ΔAUROC ≥ 0.03, CI lo > 0 | +0.023 [+0.016, +0.030] | **False** ✔ |
| DISCOVERY | every S ≥ 0.90 and ≤ 5% | UNS 0.660, RAISE 0.670 | **False** ✔ |
| E7 rule | any S < 0.75 or > 10% → KILL | RAISE and UNS trip both | **KILL** ✔ |
| KILL OF LINE | CI lo ≤ 0, or any S < 0.75 or > 10% | the per-source part trips | **True** ✔ |

- **The strict learning-curve gate is met.** k=5 − k=4 is +0.009 [+0.007, +0.012]. The b8 − last gate fails: +0.003 [−0.019, +0.027].
- **Drop UNS.** With UNS removed from the pool, DIVERSITY HELPS turns False (+0.023). With either other new source removed, it stays True.
- **No borderline call.** UNS's FPR CI {12.3%, 22.5%} is wholly above 10% (RAISE's {9.4%, 19.3%} straddles it), and the UNS and RAISE AUROC CIs (hi 0.703, 0.707) are wholly below 0.75.

## Q3. Interpretation

**(a) Most of the FPR gain is a move along the ROC** (`c1`, `c2`; mean over the 6 held-out sources; LOGO-consistent).

| | FPR (A) | TPR at that threshold |
|---|---|---|
| k=1 | 0.403 | 0.647 |
| k=2 | 0.164 | 0.437 |
| k=5 | 0.069 | 0.308 |

- **Matched-FPR comparison.** At k=5's own held-out FPR, the k=2 probes would have TPR 0.234, against 0.308 for k=5.
- **Decomposition.** Going from k=2 to k=5 costs 0.20 TPR in operating-point shift and buys back +0.074 TPR in real ROC improvement. The improvement is largest on HDRP (+0.16) and smallest on VISION (+0.02).
- **Oracle separability stays low.** The macro TPR at 5% held-out FPR is 0.283 → 0.348, and at 1% FPR it is 0.138 → 0.194.
- So E10 gives a threshold that transfers better (diverse calib reals) plus a small genuine gain. At the operating point it is still a detector that catches about a third of fakes.

**(b) The cap is not the linear head.** A 1-hidden-layer MLP on the same k=5 F_b8 rows changes the full-model macro AUROC by −0.006 to +0.027 per source, about +0.009 on average (`c3_capacity.out`).

**(c) A 2×2 decomposition shows where the AUROC goes** (`c7_ceiling.out`; F_b8 linear, jpeg75_resize05, mean of 6 sources).

| | seen generators (full model) | unseen generator (LOGO) |
|---|---|---|
| real source in training (k=6) | 0.926 | 0.855 |
| real source held out (k=5) | 0.858 | **0.767** |

- The unseen real source costs about 0.07, and so does the unseen generator.
- **E10 can only attack the first.** Even if diversity closed the source gap completely, the LOGO ceiling is **0.855**: RAISE 0.772, WMC 0.814, UNS 0.844. That is below the 0.90 DISCOVERY bar on 3 of 6 sources.
- Per-generator AUROC at k=5 confirms it. dalle2, SD2, SD3 and FLUX are weak even on the easiest source, HDRP: 0.758 / 0.763 / 0.782 / 0.855. On UNS, dalle2 is at chance: 0.491 (`results.json` `per_gen`).

**(d) "More diversity" is not monotone. It mostly insures against a bad pair** (`c6_subsets.out`; oracle selection, descriptive only).
- **k=5 is below the best 2-source pair for every held-out source** (k=5 vs best pair): VISION 0.861 vs 0.917 (RAISE+OI), RAISE 0.670 vs 0.689 (VISION+OI), OI 0.811 vs 0.840 (VISION+WMC), UNS 0.660 vs 0.738 (VISION+OI), WMC 0.738 vs 0.806 (VISION+OI), HDRP 0.863 vs 0.891 (VISION+WMC).
- **Adding UNS to the other 4 lowers AUROC** on VISION (−0.034), OI (−0.028) and WMC (−0.053). It raises RAISE (+0.034).
- The pre-registered comparator is the *mean* over k=2 pairs, which is dragged down by the pathological pairs (worst pair 0.487–0.740). So "+0.040" mostly measures robustness to the choice of pair, not a higher ceiling.
- VISION's 0.917 → 0.861 drop (key secondary, −0.056 [−0.078, −0.037]) is the same effect. UNS-like "fake-looking" reals pull the linear boundary.

**(e) Alternative explanations for UNS and RAISE** (`c4_attributes.out`, `c5_semantic.out`).
- **Content confound.** It is real but partial.
  - RAISE reals share captions with the fakes, and RAISE has the lowest ceiling even in-distribution (0.772 LOGO, 0.876 seen). This suggests part of the other sources' separability is content, not generator trace. The ceiling may also partly reflect that only 84 RAISE reals are used for training.
  - The top tertile of semantic closeness to training fakes (block-12 kNN) has higher FPR than the other two tertiles: UNS 24% vs 12–15%, WMC 16% vs 2–3%, RAISE 21% vs 10–12%.
  - But the correlation with score is weak (ρ = 0.07–0.21). UNS's FPR is ≥ 12% even in its least fake-like tertile. So UNS's problem is a **source-wide shift**: median real score −0.82, against the median test fake +0.47 and the threshold 1.91. Content does not explain it.
- **Imgix re-encode or bitrate.** Not supported.
  - Within UNS, ρ(score, bpp) is +0.03, and FPR by bpp tertile is 19% / 18% / 13%.
  - WMC is also a server re-encode (thumbnails) and has 7% FPR.
  - It cannot be excluded without the originals.
- **Simple image attributes.** Six attributes (saturation, colourfulness, brightness, contrast, sharpness, flat area) predict UNS false positives at CV AUROC **0.535**, i.e. chance. The largest correlate is contrast, ρ = +0.35.
- **Orientation or resolution.** A weak signal.
  - UNS portrait photos have FPR 23.5% (n = 85) against 12.2% for landscape (n = 115), Fisher p = 0.038; WMC shows the same direction (11.8% vs 5.4%).
  - Portrait crops are both more downscaled (scale ≈ 0.32 vs 0.48) and more subject-centred, so scale and content are confounded. One uncorrected post-hoc test: suggestive only.
- **Probe capacity.** Rejected in (b).
- **Most consistent reading.** Unsplash's curated "polished" look sits in the same F_b8 region as the generators. This repeats E7's OI mechanism with more force, and it is a property of the features, not of one attribute.

## Q4. Recommendation: close the line. One zero-cost follow-up is worth keeping
- **Close the real-source diversity line as pre-registered** (KILL OF LINE). The 2×2 in (c) shows that no amount of real-source diversity can reach the DISCOVERY bar with this probe, because the in-distribution LOGO ceiling is 0.855. A further real-source step (more sources, reweighting or source selection) has at most about 0.09 mean AUROC of headroom, and picking the "best pair" would be post-hoc tuning.
- **Keep, at no cost: the OWNER confirmatory run (Amendment 19 addendum item 3).** It is test-only and frozen at k=5, F_b8, threshold A.
  - Phone-like sources already have low FPR at k=5: HDRP 0.5% (1/201), VISION 1.0% (11/1074).
  - But TPR at that threshold is only 0.24–0.36 (c1), so report it as an FPR datapoint, not as a detector.
- **Record as positive sub-findings in `kb/`.**
  1. A diverse calibration set makes a 1% threshold transfer far better: mean FPR 40% → 7% from k=1 to k=5.
  2. Real-source diversity is not monotone; one "fake-looking" source (UNS) can cost 0.03–0.05 AUROC on others.
  3. The 2×2 loss budget (source ≈ generator ≈ 0.07).
- **If the owner wants to keep exploring**, the evidence points to a *different* line, which needs a new pre-registration and, for data or GPU, the owner's approval:
  - the generator axis: more or more varied training fakes (only 230 train-role fakes now, `tables.md` n_train) [effect unverified];
  - or a non-frozen or different backbone. A different head on frozen F_b8 does not help (b).
  - Do not raise the limit again for real-source diversity.

## Plain-language summary for the owner
- **What E10 did.** Training on 5 kinds of real photos instead of 2 makes the detector's alarm level much more trustworthy on a new kind of photo: false alarms fall from about 16% to about 7%.
- **Why it is not a fix.**
  - Most of that comes from the detector becoming more cautious. At the same false-alarm rate, the 5-source detector catches only about 7 more fakes in every 100 than the 2-source one (31 vs 23).
  - It never gets good at telling Unsplash-style polished photos from AI images.
  - Even on photo types it was *trained* on, this frozen-feature detector only reaches about 0.86 AUROC against an AI generator it has not seen. More kinds of real photos cannot lift it past that ceiling.
- **Verdict.** The pre-registered verdict, close this line, is right.
