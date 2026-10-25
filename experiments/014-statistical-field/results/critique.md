# E5 (014-statistical-field) critique: independent critic, opus/high, 2026-10-25

**Rating: the code is sound and I found no bug. P1, P3 and P4 stand, and the P2 refutation is robust. But the headline laundered C2ST (DINOv2-B0 0.940, block-8 0.970) is not a "diffusion vs real" effect size. About half of the gap to 003's 0.875 comes from training on laundered data. The other half comes from a real pool that is 79% VISION photos, which DINOv2 separates mostly by content (scale-matched WhatsApp copies alone give 0.98). With content, scale and format controlled, the defensible laundered AUROC is 0.92–0.95 (block 8), 0.82–0.87 (B0) and 0.69–0.72 (hand-built battery). All of these rest on a single real source: B-Free-resized RAISE.**

Scope: CPU only (≤4 processes); no downloads; run.py and its caches were not modified. The scratch checks reuse run.py's own functions (`load_df`, `get_space`, `fit_score`, `deal`, `twin_pairs`). Files in `results/critic_checks/`:

| Check | What it does | Files |
|---|---|---|
| c1 | C2ST controls in the E5 fold scheme | `c1_c2st_controls.{py,out,json}` |
| c2 | Bridge to the 003 B0 protocol | `c2_003_bridge.{py,out,json}` |
| c3 | Cluster-level MMD permutations | `c3_mmd_cluster_perm.{py,out,json}` |
| c4 | Twin-paired C2ST with clean training | `c4_twin_crosscond.{py,out,json}` |

Sanity checks:
- c1 reproduces E5's repeat-0 value exactly (0.93807).
- c4's matched rows reproduce E5's twin table exactly (0.837 / 0.869 / 0.927 / 0.953).
- c2 reproduces 003's B0 laundered AUROC exactly (0.875) from the 004 block-12 features.

## Findings, ranked

### F1 (HIGH, interpretation): 0.940 vs 0.875 is explained by the training condition and the real-pool mix, not by leakage
Bridge, step by step, at `jpeg75_resize05`, `orig`, DINOv2-B0 (c2, c1):

| Setup | AUROC |
|---|---|
| 003 recipe (train `b0_train` clean; test natives + RAISE test reals) | **0.875** (exact `fit_probe` recipe: 0.875) |
| Same split, train on laundered features (matched) | **0.917** (+0.042; `fit_probe` 0.913) |
| Same split, matched, twin-free training | 0.907 |
| E5 folds (device- and generator-held-out), reals = native + RAISE, matched | **0.906** (clean-trained 0.882; strict id-pruned 0.895) |
| E5 folds, reals = all, which adds 1,000 WA/FBH copies (79% of reals are VISION), matched | **0.940** (+0.034; clean-trained 0.927; strict 0.932) |

- **Training condition.** The C2ST correctly trains on the condition it tests, which is the right design for a two-sample test. 003 trains clean. Matched training adds +0.02 to +0.04 for B0 and +0.04 to +0.09 for block 8 (block 8 RAISE-only, 003 split: clean 0.848 → matched 0.936, c2).
- **Real mix.** Pooled C2ST is roughly the share-weighted mix of the per-pool values:
  - per pool: native 0.987, WA 0.981, FBH 0.981, RAISE 0.840 (tables.md);
  - VISION is 1,500 of the 1,899 reals;
  - 003's pool is 64% natives and has no WA/FBH.
- **Generator held out vs seen.** E5 holds out two generators per fold, while 003's 0.875 test mostly contains generators the probe has seen. The held-out case is no easier (0.906 in E5 folds vs 0.917 on the 003 split, both matched), so this does not inflate E5.

### F2 (HIGH): the VISION pools are content-confounded even when scale-matched, so "WA+RAISE fov512" is not a clean control for DINOv2
- **WA alone** (scale-matched per 008, about 960 px short side) is the easiest pool in every arm and space at `jpeg75_resize05`:
  - B0: 0.981 (orig) and 0.986 (fov512);
  - block 8: 0.988 and 0.990.
  - Strict id pruning leaves these unchanged (c1).
- **RAISE alone** is much harder for B0 (0.840 / 0.854), although RAISE is also near the fakes' scale (1258×833, 4× downscaled by B-Free; header check).
- **Reading.** VISION phone snapshots differ in *content and composition* from fakes prompted with RAISE captions. Zoom does not explain the WA result. So any DINOv2 number that includes VISION reals (0.940, 0.970, and the 0.918 for WA+RAISE fov512) overstates the content-controlled difference.
- **The content-controlled estimates** are RAISE-only and twin-paired (see F3). P4 is unaffected because it concerns the battery.

### F3 (HIGH, the owner's question): after content, scale and format control, a laundered difference remains
Twin-paired rows: RAISE real vs its own caption twins, id- and generator-held-out. RAISE-only rows: E5 folds plus strict id pruning (c1). E5 rows are in tables.md; c1/c4 rows are mine.

| Space | Arm | Twin-paired, matched (E5) | Twin-paired, clean-trained (c4) | RAISE-only, strict, matched (c1) | RAISE-only, clean-trained (c1) |
|---|---|---|---|---|---|
| block 8 | orig | 0.927 [0.913, 0.942] | 0.862 | 0.923 | 0.869 |
| block 8 | fov512 | **0.953** [0.943, 0.963] | 0.924 | 0.940 | 0.929 |
| B0 | orig | 0.837 [0.818, 0.856] | 0.804 | 0.823 | 0.808 |
| B0 | fov512 | 0.869 [0.854, 0.886] | 0.830 | 0.839 | 0.831 |
| battery | orig | 0.687 [0.660, 0.715] | 0.629 | – | – |
| battery | fov512 | 0.722 [0.695, 0.747] | 0.663 | 0.711 | – |

- **Question (e) is answered.** Laundered twin-paired DINOv2 was already computed in E5 and stays high: B0 0.84–0.87, block 8 0.93–0.95. The matched-pair win rate is 0.88–0.96.
- **Remaining confound: one real pipeline.** All real images here come from three Nikon DSLRs → raw conversion → one B-Free resize kernel (4×). Part of the residual difference may be "RAISE look", such as the antialiasing and noise suppression from the 4× downscale, which survives JPEG-75 + 0.5× resize. The data cannot separate "diffusion signature" from "this particular real pipeline". There is also no camera id for RAISE, so cameras cannot be held out.
- **Small sample:** 361 ids and 796 pairs.
- **Mild selection:** block 8 is the best of three DINOv2 depths that were inspected.

### F4 (MEDIUM): fold scheme: no leakage that matters
- **Device grouping.** Each of the 35 VISION devices keeps its native, WA and FBH copies in one fold, because `rgroup` uses the `device` column and that column is identical across the copies.
- **Generators.** All 8, including FLUX, are dealt by name, two per fold, and never seen in training.
- **Twin pruning.** It works as described.
- **One unpruned route.** Training fakes from *other* generators that share a caption id with a test fake. Pruning those as well ("strict") lowers C2ST by only 0.003–0.017 across all cells (c1), so content memorisation across generators is a small inflation.
- **Pooling.** Pooled out-of-fold AUROC matches the mean within-fold AUROC to within 0.01 in every cell, so there is no pooling artefact.
- **Class balance.** 1,899 reals vs 1,020 fakes, with `class_weight=balanced`. This is irrelevant for AUROC. The real *mix* is what matters (F1).

### F5 (LOW): MMD p-values: the image-level permutation is anticonservative in principle, but the verdict does not change
- With cluster-level permutations (each VISION device = one cluster, each RAISE image and each fake = its own cluster, 999 permutations), every generator stays at the permutation floor: p = 0.001, Holm 0.008.
- This holds in battery, B0 and block 8, for orig native+RAISE and for fov512 WA+RAISE, at `jpeg75_resize05` (c3).
- MMD² is reported but not interpretable as an effect size across spaces, because the bandwidth is set per space.

### F6 (LOW–MEDIUM): implementations of the statistics and of P3
- **S3** matches the config: 4-neighbour residual, MAD/√1.25 correction (correct for i.i.d. noise), 8 equal-count bins.
  - The self-test recovers the slope but not the intercept (2.09 vs 4, inside a loose window), and textured bins inflate MAD.
  - S3 effects are small (mean |δ| 0.08–0.24), so S3 carries little weight in P3 either way.
- **S4** is implemented as described. The 1-px offset from the valid-region high-pass is harmless because the phase statistics are symmetric.
  - (a) After a 0.5× resize, the 2×2 phase statistics are meaningless by construction, since period-2 structure aliases to DC. Part of S4's 0.52 ratio under `jpeg75_resize05` is arithmetic, not "laundering removes colour traces".
  - (b) The S4 correlation effects are not a JPEG chroma-subsampling artefact of VISION: they are as large against PNG RAISE reals (s4_corr_RG clean |δ| 0.52 RAISE vs 0.42 native; atlas_long).
  - (c) The signs flip between generators (s4_corr_dRG_dBG: SDXL +1.00, SD3.5 −0.69). This is why large per-generator |δ| does not translate into a high generator-held-out linear battery C2ST.
- **S2:** db2 was pre-registered as "Haar/db2" but only Haar was computed. This is disclosed in notes.md and is minor.
- **P3 operationalisation.** After a 0.5× resize, "S2 level 1" measures the band that was level 2 before. So clean/laundered ratios compare different physical frequencies. P3b also takes ratios of small means: S3's clean mean is 0.16.
  - REFUTED is a fair label. The informative content is the descriptive pattern: S4 is suppressed, while S2L2-4 and S5 are unchanged.
- **P2 operationalisation.** The strict reading is fine. Even the most conservative cell (B0, twin-paired, clean-trained, 0.80–0.83) is far above 0.70.
  - The battery component alone is borderline: 0.710 [0.676, 0.739] orig, 0.699 fov512, 0.687 twin.
  - The fair summary is "hand-built statistics nearly close; learned features do not".
- **P4** is confirmed as pre-registered. The single-factor table already shows that most of the battery's 0.972 against natives is scale/format.
- **notes.md wording.** "Leaves the DINOv2 separation essentially intact (0.948 → 0.940)" is a *matched-training* statement. A clean-trained classifier loses 0.03–0.11 on the content-controlled pools (block 8 RAISE orig: 0.978 clean → 0.869 laundered when trained clean). The difference persists, but its direction moves.

## Plain answer for the owner
**Yes, a measurable difference remains.** Hold content fixed (each fake vs the very photo whose caption made it), roughly equalise scale, wash both classes through JPEG-75 + half-size resize, and never let the classifier see the test photo's caption or the test generator. A simple linear read-out of DINOv2 mid-layer features still tells real from diffusion images apart with AUROC about **0.93–0.95** (block 8). The final-layer features (B0) reach about **0.84–0.87**, and our 38 hand-built signal statistics reach about **0.69–0.72**.

That is "different but overlapping": B0 at 0.85 means a random fake outranks a random real 85% of the time.

**Best defensible number:** about 0.92–0.93 (block 8, twin-paired, orig, matched training). It falls to about 0.86 if the classifier never saw laundered training data.

**Limitation:** these numbers are for *one* real camera pipeline (resized RAISE DSLR photos) and 8 generators. They do not yet show that the difference is a property of diffusion rather than of RAISE's processing. The much higher 0.94–0.97 figures include VISION phone photos, which DINOv2 recognises largely by content.

## Recommendations (pre-registered follow-up for a detector built on this)
1. **Correct the headline wording** in notes.md / README. Report 0.940 / 0.970 as "all-reals, matched-training, content-mismatched". Report the content-controlled numbers (F3 table) as the estimate of the distribution difference.
2. **Freeze the detector before seeing new data.** Block-8 CLS+mean, z-score + logistic regression C=0.01, trained on laundered + clean augmentations. Fix the training data, the threshold rule (calibrated on held-out reals) and the seeds in the pre-registration.
3. **Primary test = a new real pipeline with matched content.** Real photos the model has never seen (fresh phone photos with neural ISP, and a second DSLR set not resized by B-Free), paired with fakes from held-out generators prompted by captions of those exact photos. Primary metrics:
   - AUROC and TPR@1%FPR at `jpeg75_resize05` and `social_sim`;
   - FPR on the fresh phone photos, with a target of ≤1% (the Phase 5 target).
4. **Pre-register these controls:**
   - (a) clean-trained vs matched-trained;
   - (b) strict id-disjoint folds;
   - (c) cluster-level CIs (device/id);
   - (d) a format/scale probe: a classifier on file size, dimensions and blur that must fail (AUROC < 0.65) on the new pairs;
   - (e) a RAISE-pipeline control: resize the new phone photos with the B-Free kernel and check that the score does not track that operation.
5. **Pre-registered falsifier:** block-8 twin-paired laundered AUROC < 0.85 on the new real pipeline means the E5 residual was RAISE-specific.
6. **Do not use WA/FBH/native VISION copies as evidence for distribution separation in DINOv2 spaces** unless the fakes are content-matched to them.
