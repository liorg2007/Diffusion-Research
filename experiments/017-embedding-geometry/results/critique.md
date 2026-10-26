# E8 (017-embedding-geometry) critique: independent critic, opus/high, 2026-10-26

**Rating.** I found no bug or leakage. No deviation flips a pre-registered label: R1 holds but is uninformative, R2 FAILS and R3 is a KILL. Two of the interpretations in `notes.md`, however, need correcting before they go into the report.

1. **The R2 cosine comparison is framed wrongly.**
   - The notes say the cross-source cosines are "at the noise ceiling". They are not.
   - On matched, disjoint halves, cross-source cosines are about half the within-source ones. So the probe directions really do differ between sources.
2. **The R3 KILL means "the 2-source subspace does not contain the new source's shift". It does not mean "removal does not help".**
   - An oracle check adds one direction: the held-out source's own real-mean offset, estimated from its train-role reals, which are disjoint from the test images.
   - That single direction lifts RAISE from 0.764 to 0.936 and OI from 0.856 to 0.933 (full-model macro). FPR falls from 19.1% to 3.8% (RAISE) and from 16.0% to 6.5% (OI).

Scope: CPU only, no downloads, nothing outside `results/critic_checks/` and this file was modified. Scripts, outputs and JSON are in `results/critic_checks/`:

| File | What it checks |
|---|---|
| `c1_geometry.{py,out,json}` | c1–c4 below |
| `c5_fewshot.{py,out,json}` | The oracle with only n target reals |
| `c5b_baseline_insource.*` | The no-projection reference for c5 |

All numbers are fov512, F_b8, jpeg75_resize05.

**Sanity anchor.** My independent no-projection refit reproduces E7's FLUX AUROCs exactly: VISION 0.880, RAISE 0.627, OI 0.689 (`c1_geometry.out`; compare `016-loso/results/critique.md` F1).

## Q1: bugs, leakage, deviations

**Verified clean**
- U is estimated from the training sources' train-role **reals** only (`m3u_task`: `rows[D.lab[rows]==0]`). Its scaler comes from those reals.
- Twin pairs are train-role only.
- The d_twin scaler uses RAISE train rows only.
- The LOGO normals are used per generator.
- M3 is a KILL for every k and every uniform j (`tables.md`). So neither the subspace-construction deviation nor the calib-reals-only k rule can flip R3.

**F1: R2's cosine leg is mis-framed and was nearly impossible to pass. R2 still fails.**
- The notes compare full-data cross-source cosines (0.29–0.39) with a **half-data** within-source reference (0.34 / 0.50 / 0.39). That is not a like-for-like comparison.
- **Matched check (c1).** One global group split into disjoint halves, with each source's full-fit C, 6 replicates:

  | Pair type | Cosine (mean ± sd) |
  |---|---|
  | Within-source: VISION | 0.33 ± 0.03 |
  | Within-source: RAISE | 0.49 ± 0.02 |
  | Within-source: OI | 0.36 ± 0.02 |
  | Cross-source: VISION–RAISE | 0.15 ± 0.06 |
  | Cross-source: VISION–OI | 0.18 ± 0.04 |
  | Cross-source: RAISE–OI | 0.15 ± 0.03 |

  Cross-source is clearly below within-source, so the directions differ beyond noise.
- **Noise-corrected shared fraction:**
  - matched halves (cross / √(within·within)): 0.37 / 0.53 / 0.36;
  - Spearman-Brown on the full-data cosines: 0.61 / 0.73 / 0.48.

  Either way the direction is partly shared, not invariant.
- **The ≥ 0.5 threshold was nearly unattainable.** Even an identical true direction would give a raw full-data cosine of only about √(rel_a·rel_b) ≈ 0.53–0.61. This is a pre-registration weakness.
- R2 fails on its other leg anyway: the d_twin AUROC on OI is 0.554 [0.507, 0.597].
- d_twin is itself a crude, unwhitened mean of 99 pairs, with only 0.83 AUROC in-source. That leg is a weak test, not proof of absence.

**F2: functional transfer shows a shared component plus a source-specific part (c2).** A single-source probe normal is applied to another source's test reals vs test fakes (LOGO macro):

| Probe trained on → evaluated on | VISION | RAISE | OI |
|---|---|---|---|
| VISION | 0.994 (in-source) | 0.700 | 0.804 |
| RAISE | 0.826 | 0.935 (in-source) | 0.736 |
| OI | 0.921 | 0.651 | 0.962 (in-source) |

Transfer is well above chance but costs 0.07–0.29 AUROC. Cosines alone cannot show this.

**F3: M3's k-selection target is near-tautological.**
- Projecting out the between-source mean difference makes a *linear* source classifier chance-level by construction: the projected class means coincide (the LEACE / linear-erasure property).
- So "calib source-ID 0.97 → 0.47–0.52" proves only that the projection is correct. It says nothing about removing the source information that matters for a *new* source.

**F4: the 2-source subspace is too low-rank for the held-out shift (c3).**

| Held-out source | Share of its real-mean offset inside U_k at chosen k | At k = 10 | Random baseline |
|---|---|---|---|
| OI | 2.5% (k=2) | 45% | k/768 ≤ 0.7% |
| VISION | 6.3% (k=4) | 20% | k/768 ≤ 0.7% |
| RAISE | 14.5% (k=3) | 48% | k/768 ≤ 0.7% |

With 2 training sources the between-source span is rank 1. A third source's offset is mostly a new direction.

How the held-out offset aligns with the fake offset:

| Held-out source | Cosine with the fake offset |
|---|---|
| VISION | 0.08 |
| OI | 0.30 |
| RAISE | 0.74 (content-confounded) |

**F5: the oracle shows that removal works when the subspace contains the new source's direction (c4, c5).** One extra direction (the held-out source's train-role real mean, k_chosen + 1) gives:

| Source | Macro AUROC | FLUX AUROC | FPR (reading A) | Real shift / real-fake margin |
|---|---|---|---|---|
| RAISE | 0.764 → **0.936** | 0.627 → 0.836 | 19.1% → **3.8%** | 0.68 → 0.06 |
| OI | 0.856 → **0.933** | 0.689 → 0.845 | 16.0% → **6.5%** | 0.43 → 0.07 |
| VISION | 0.949 → 0.960 | (not reported) | 0.6% → 0.6% | (not reported) |

- **With only 25–50 target reals:**
  - OI: 0.895–0.929, FPR 7.5–11.5%;
  - RAISE: 0.885–0.933, FPR 2.7–9.3%.
- **Cost on the sources already covered** (vs c5b): in-source AUROC drops 0.01–0.025, and FPR rises from 0.3–0.7% to 0.6–3%.
- **Caveats:**
  - not pre-registered;
  - full model (7 generators seen in training; FLUX is the unseen-generator column);
  - one split, no CIs;
  - 8 OI Flickr authors overlap train and test (E7 critique).
- Even the oracle leaves OI FPR at 6.5%, far from the 1% target.

## Q2: is "no source-invariant linear direction" justified?
Only in a weaker form. The justified statement is:

> With DINOv2 block-8 features and three real sources, the real-vs-fake direction is **partly shared**. Noise-corrected cosine is about 0.4–0.6, and single-source probes transfer at 0.65–0.92. But each real pipeline adds its own offset, partly toward "fake" (OI). A training set with 2 real sources cannot estimate that offset for a third.

What E8 does **not** show:
- that an invariant direction is absent in general;
- that nuisance removal is useless. F5 points the other way, if the new pipeline's offset is known.

It is a "not findable with 2–3 sources and this backbone" result.

## Q3: what the 2-D map shows (`fig1_map_pca.png`, `fig2_map_tsne.png`)
- **PCA.** PC1 and PC2 explain only 6.1% and 4.7% of the variance.
- **PC1 is a content/source axis.** RAISE and the fakes, which share RAISE captions, sit to the right; VISION and OI sit to the left. Source accounts for 30–32% of PC1 variance among reals (η², `results.json` map).
- **Real and fake are not separable in 2-D.** The fakes overlap the RAISE region.
- **t-SNE.** VISION and OI form tight islands (devices or scenes), while the fakes form one diffuse central cloud mixed with RAISE.
- **Reading.** The dominant variance is content and source. The detection signal lives in low-variance directions. This is descriptive only, but consistent with F2 and F4.

## Q4: plain-language verdict and next steps
**Verdict for the owner.**
- We looked for one "AI-made" direction in DINOv2's features that works the same whatever camera or photo site the real photos come from. We did not find one.
- The direction is about half shared. Every new photo pipeline (a phone, Flickr, a camera archive) also pushes its real photos part of the way toward "looks AI-made".
- Removing the differences we can learn from two known pipelines does not help with a third: that third pipeline's push points somewhere new.
- If you know a few dozen real photos from the new pipeline, removing *its* push fixes most of the problem in this check (AUROC about 0.93). That turns the tool into a per-pipeline-calibrated detector, not a universal one. Even then, the false-alarm rate (about 4–11%) stays far above the 1% target.

**Ranked by expected information per cost:**
1. **Default: stop and finalize.** Add E8 to `docs/03-final-report.md` as a pre-registered negative (R2 FAIL, R3 KILL), with F1, F4 and F5 as interpretation. The universal-detector line is closed.
2. **Only if the owner wants one more bounded step** (this needs another D-016 raise): a pre-registered E9 on "real-anchored pipeline adaptation".
   - It is CPU-only, uses the existing caches and takes minutes.
   - Design: n ∈ {10, 25, 50, 100} target reals, LOGO + FLUX, CIs, all folds, with the in-source cost reported.
   - The decisive test is the owner's phone photos (half used as the anchor, half for test).
   - It changes the claim to a calibrated, per-pipeline detector, and on present evidence it would not meet the ≤1% FPR target.
3. **More real sources** (download, needs owner approval). This tests whether a non-oracle subspace built from many sources generalizes. Medium cost, medium information.
4. **Other backbones (CLIP, larger DINO).** Downloads plus feature extraction. Expected information is low: E5–E8 point to a real-pipeline shift in semantic features, not a backbone quirk.
5. **Non-linear or ensembled heads.** Lowest value. C2ST is already at ceiling, and a more flexible head learns source shortcuts more easily. It does nothing about the offset.
