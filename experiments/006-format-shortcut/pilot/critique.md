# Critique: H4 PILOT (experiment 006, docs/04-phase4-plan.md Amendment 5)

**Critic:** independent critic (opus/high), 2026-10-23. I edited no files outside this one and `pilot/critic_checks/`.
**Overall rating: no bug found. The pilot is UNINFORMATIVE for the H4 question, but it is INFORMATIVE as a negative result about the score.** The inversion is real and not a coding error. On these crops the RIGID-style drift mostly measures how little high-frequency texture the 224 input has. Read REFUTED (orig) as "uninformative: the score does not detect on this data", not as "the format shortcut does not drive RIGID". Treat INCONCLUSIVE (fov512) the same way.

Scripts I ran (all read-only on the data; outputs sit next to them):
- `critic_checks/c1_scores.py`: label alignment, AUROCs recomputed with sklearn, per-generator values, medians.
- `critic_checks/c2_texture.py` → `c2_stats.csv.gz`: texture statistics of the exact 224 input, setting (i).
- `critic_checks/c3_texture_settings.py` → `c3_lap.csv.gz`: texture in every setting and arm, texture-residualised AUROCs, per-device table.

"lap" = mean |discrete Laplacian| of the grey 224 center crop that the model sees. It is a simple proxy for high-frequency (HF) energy.

---

## Q1. Is the below-chance AUROC a bug or real? **Real. No bug found.**

### Checks that rule out a bug
| Check | Evidence | Result |
|---|---|---|
| Direction convention | RIGID (arXiv 2405.20112, Sec. 1 and Sec. 3.1, Eq. 1) flags an image as generated when cos(f(x), f(x+λδ)) ≤ ε, i.e. fakes have *lower* similarity. `run.py:118` uses score = 1 − cos with higher = fake. | Same direction as RIGID. |
| Label/score alignment | `c1`: all 34,956 rows of `scores_long` merged with `splits_pilot.csv` on sha256 give 0 label, 0 pool and 0 split mismatches. | Aligned. |
| AUROC recomputation | `c1`: sklearn LOGO-macro reproduces every value in `tables.md`. Examples: orig (i) σ=.01 0.292 / 0.237 (with / without Firefly); fov512 (iii-p) σ=.01 0.338. | Reproduced. |
| Identical processing | `run.py:132-146`: both classes go through the same `load_cond` → 224 center crop → noise → clip → normalise, with noise seeded per (sha, draw, σ). Pipeline check: the unnoised pre-LN CLS equals the 004 cache exactly (`d008_guard.json`, max diff 0). | Identical. |
| Post-LN vs pre-LN CLS | Both variants give the same picture (`tables.md`: post 0.292 vs pre 0.298 at orig σ=.01) and the same labels in every cell (`run.log`). | Not a factor. |
| Noise scale | RIGID uses λ = 0.05 (App. A). The paper does not say whether noise is added in pixel or normalised space, and it gives no code URL (fetched arXiv HTML) **[unverified which]**. In ImageNet-normalised space (std ≈ 0.225), λ = 0.05 ≈ σ_pixel 0.011. In [0,1] pixel space it is σ = 0.05. The grid {0.01, 0.02, 0.05} brackets both readings. | The inversion holds at every σ (orig 0.27–0.29, fov512 0.39–0.41), so noise scale cannot explain it. |

### What drives the inversion: high-frequency texture of the 224 input (zoom and smoothness)
1. **Pool texture.** At orig, VISION native crops are much smoother than fakes (`c2`): median lap 0.031 vs 0.080, pixel std 0.123 vs 0.172. A fixed-σ perturbation is relatively larger on a smooth, low-content patch. In the orig arm, the 224 input covers only about 224/2448 ≈ 9% of a phone photo's short side (Amendment 3 confound).
2. **Texture predicts drift within each class.** Spearman(score, lap) at orig σ=.01 is −0.73 for reals and −0.69 for fakes (`c2`). Across the 25 devices, median drift vs median lap gives Spearman −0.73. Only 3 of 25 devices have AUROC > 0.5, and they are the three most textured: D22, D26, D24 (`c3`).
3. **Generators order the same way (`c1` + `c2`).**

   | Generator | Median lap (orig) | AUROC |
   |---|---|---|
   | DALL-E 3 | 0.123 (most textured) | 0.131 (most inverted) |
   | Firefly | 0.035 (closest to natives; 1792 px, the most zoomed fake) | 0.571 (the only generator above chance) |

4. **Real pools order the same way (`tables.md` per-pool, `c2`).**

   | Pool | lap | AUROC (σ=.01) |
   |---|---|---|
   | WA | 0.055 | 0.408 |
   | native | 0.031 | 0.292 |
   | FBH | 0.029 | 0.229 |

   This is texture order, not zoom order: FBH is *less* zoomed than native but more inverted.
5. **The trivial texture score beats RIGID.** The score "−lap" alone has AUROC 0.222 at orig (`c2`), more inverted than RIGID.
6. **Residualising removes the signal.** I regressed log-score on log-lap, fit on reals only, and scored the residual (`c3`).

   | Arm | σ=.01 | σ=.02 | σ=.05 |
   |---|---|---|---|
   | orig (raw 0.29) | 0.515 | 0.428 | 0.385 |
   | fov512 | 0.471 | 0.452 | 0.427 |

   At the RIGID-equivalent σ ≈ 0.01, almost nothing is left beyond texture. At larger σ a modest residual inversion remains, which I cannot explain (possibly sensor-noise structure) **[unexplained]**.

### Corroboration from the literature (KB)
Below-chance RIGID is not unique to us. kb/papers/manifold-bias-iclr2025.md (Table 1) reports RIGID zero-shot AUC 0.439 averaged over ~20 generators.

### Remaining fidelity gaps to RIGID (not bugs, but they limit generalisation)
- ViT-B/14 instead of ViT-L/14.
- A 224 crop of a 512 crop, instead of (very likely) a whole-scene resize to 224 on ImageNet/LSUN-scale inputs. RIGID's preprocessing is not stated in what I could read **[unverified]**.

Our orig arm feeds DINOv2 near-content-free patches, which is far from RIGID's regime.

## Q2. How should the label be read? **"Uninformative: the score does not work on this data."**

- **The criterion assumes AUROC > 0.5.** The gap AUROC(i) − AUROC(iii-p) is a proxy for "how much separability disappears when formats match". Below 0.5, the sign means the opposite.
  - In **fov512**, AUROC goes 0.405 → 0.338 at σ=.01. The separation |AUROC − 0.5| therefore *grows* from 0.095 to 0.162 when formats are matched. That is the opposite of a format shortcut, yet it counts toward SUPPORTED. INCONCLUSIVE there is a misfire of the criterion, not weak support.
  - Mechanistically, fov512 natives drift more after jpeg(90) (median 0.0130 → 0.0179) while fakes do not (0.0094 → 0.0088) (`c1`). This persists after texture residualisation (gap +0.080, `c3`) **[unexplained]**.
- **orig: "REFUTED" is literally true and substantively empty.** Re-encoding changes nothing (gap −0.003 [−0.023, +0.017]; texture-residualised +0.003). But there is no detection performance whose origin could be tested.
  - The KB idea's pre-registered consequence of refutation, "drift scores are a legitimate training-free component for fusion" (kb/ideas/drift-format-shortcut.md), must **not** be drawn. The pilot shows the opposite: on camera photos the score is a texture meter that ranks reals as more fake (1%-FPR TPR ≤ 0.017 at orig, `tables.md`).
- **The pilot manipulation is half the format question.** In (i) → (iii-p) the reals were already JPEG and stay JPEG (now double-compressed), so only the fakes change format.
  - jpeg(90) also barely changes texture: native lap 0.0307 → 0.0301, fakes 0.080 → 0.083 (`c3`).
  - B0's gap is also only +0.002. The pilot cannot show a format shortcut for B0 either.
  - Setting (ii) → (iii) in the official run, where the reals go PNG → JPEG, is the real test.
- **Stop rule.** The pilot does not count toward the stop rule (Amendment 5). The lead should log the label as "uninformative (score inverted)" next to the literal label.

## Q3. Is the secondary zoom finding sound? **Descriptively yes. The interpretation needs two caveats.**

1. **It is mediated by texture, not "scale" as such.** fov512 resampling raises native lap 0.031 → 0.051 (4.8× downscale concentrates detail) and lowers fakes' lap 0.080 → 0.071 (1.5–2× downscale with antialiasing) (`c2`). The texture gap shrinks, so AUROC moves toward 0.5.
   - After texture residualisation, the orig − fov512 difference falls from −0.112 / −0.127 / −0.104 to +0.044 / −0.025 / −0.042 (σ = .01 / .02 / .05, `c3`).
   - So: "RIGID is strongly sensitive to the HF content of its input, which zoom and resampling change". Do not phrase it as "RIGID detects zoom".
   - Because each class is downscaled by a different factor (reals ≈ 4.8×, fakes ≈ 1.5–2×, Firefly ≈ 3.5×), fov512 does not equalise processing history. Zoom and resampling cannot be separated in this design.
2. **The B0 comparison is not like-for-like.** `b0_scores` (run.py:267-285) **retrains** the probe on each arm's own features. "B0 barely moves (+0.010 / +0.015)" therefore means "a probe retrained at each scale reaches ~0.98–0.99 at both". It does not mean "a fixed B0 is scale-invariant".
   - A training-free score cannot re-adapt, so the asymmetry is partly by construction.
   - The fair comparison is B0 trained on orig and evaluated on fov512 features. It is not in the pilot.
3. Fine as reported: noise is added after the resampling, so the noise statistics themselves are identical across arms. Only the content changes.

## Q4. Other verdict-changing bugs? **None found.**

- **Bootstrap** (run.py:199-229, metrics.py:102-137):
  - reals are clustered by device (25 clusters) and fakes are resampled per generator;
  - the same (Wr, Wf) is reused across σ, settings and arms (row order asserted identical, run.py:505), so gaps and arm differences are properly paired;
  - ties count 0.5.
  - Limitation: B0's CIs ignore probe-fit variance. This is minor and cannot change any label.
- **B0 reproduction.** The 005-style macro is 0.99204, which equals 005 pilot_v2 A0 exactly (`run.log`, `results.json meta.check_005`). The primary B0 uses the test-split fakes, the same images as RIGID. Correct.
- **Firefly.** Every cell is reported with and without Firefly, and the labels agree.
  - Firefly is the only generator above chance (0.571). This is texture-consistent and does not change the picture.
- **D-008.** Passed (scaled |Δfeat| 1.2e-5, |Δscore| 1.6e-6 against scores of ~1e-2). Limitations:
  - The guard covers the orig arm, the clean condition and draw 0 only. It is the same code path, so this is low risk.
  - The orig caches were computed at batch 64 and the fov512 caches at batch 32 (notes.md). This is float-rounding level given TF32 is off.
- **Process note (transparency, not verdict-changing).**
  - The 16-image dry run (`cache/dryrun_out/run.log`, 23:06:41) already showed AUROC(i) of 0.06–0.20.
  - `config.json` was last modified at 23:17, after that dry run. notes.md says the edit was the batch-size change. I cannot verify the earlier content because the file is untracked.
  - Post and pre give identical labels in every cell, and the direction was not flipped, so a post-hoc choice of feature variant could not have changed anything.
- **Operating point.** The threshold at 1% FPR is the calibration maximum plus epsilon, from 43 reals, as documented. It is indicative only.

## Q5. Recommended pre-registerable changes for the OFFICIAL H4 run (RAISE, setting ii)

Each item is justified by this pilot and can be written down before RAISE data exists. Ordered by priority.

1. **Score-validity gate (no direction flip).** Before applying the gap criterion, require AUROC(ii) of the RIGID score > 0.5 with the bootstrap CI lower bound > 0.5, in the arm used for the label.
   - If the gate fails, the label is **"UNINFORMATIVE (score not a detector on this data)"**, never REFUTED or SUPPORTED.
   - Report the signed AUROC and the separation |AUROC − 0.5| side by side, so a "gap" that increases separation is never read as support.
2. **Texture baseline and covariate.**
   - Report AUROC of the trivial score −lap (mean |Laplacian| of the exact 224 input) in every setting.
   - Report the RIGID AUROC after residualising log-score on log-lap, fit on reals only (the c3 recipe).
   - Pre-register that RIGID counts as "working" only if it beats the −lap baseline. The gap criterion is also reported on the residualised score, as a sensitivity analysis.
3. **Use (ii) → (iii) as the label contrast (already the official plan) and state why.** In the pilot only the fakes changed format. In the official run both classes go PNG → single JPEG(90).
   - Add a "how strong is the manipulation" check: the change in lap and in B0 between (ii) and (iii) per class.
   - The pilot showed that jpeg(90) barely moves either.
4. **Fix the primary arm and add a RIGID-faithful input arm.**
   - RAISE's median short side is 833 (Amendment 3), close to the fakes', so orig is a much cleaner arm for (ii) than for VISION.
   - Pre-register which arm (orig or fov512) carries the label.
   - Add a whole-image arm as a sensitivity: resize the short side to 224, then center-crop 224 (`fov224`). This is closer to RIGID's likely preprocessing **[unverified]** and to the semantic-content regime the method was designed for.
5. **Like-for-like B0 for the scale comparison.** Add B0 trained on orig and evaluated on fov512 (and the reverse) for the secondary orig − fov512 comparison. Retrained-per-arm B0 cannot show scale sensitivity.
6. **Optional, needs the owner/lead because of the D-007 download budget: the RIGID backbone, DINOv2 ViT-L/14.** A ViT-B-only result can always be dismissed as "not RIGID".
   - Keep σ = 0.01 as the pre-declared RIGID-equivalent (λ = 0.05 in normalised space ≈ 0.011 pixel) **[unverified convention]**, while still reporting all three σ.
7. **Keep unchanged:** the fixed direction, best σ = argmax AUROC(ii), the with/without-Firefly reporting, the cluster bootstrap and the D-008 guard. Extend the guard to every arm and condition, which is cheap.

Not recommended: changing the 0.10 / 0.03 thresholds. The problem is validity (item 1), not the size of the thresholds.
