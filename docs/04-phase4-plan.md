# Phase 4 plan (pre-registered before any Phase 4 result exists)

**Written:** 2026-10-23 by the lead, after the owner's "continue the research".
**Gate:** no Phase 4 experiment runs until the SETUP_WSL step-6 reproduction passes. That means byte-identical splits and Phase 3 AUROCs within about ±0.005.
**Inputs:** docs/02-hypotheses.md, kb/ideas/*.md, experiments/003-baselines-eval/{notes.md,tables.md}.

## Reconciliation with the Phase 2 plan (what changed and why)
| Phase 2 assumed | Actual (Phase 3) | Consequence |
|---|---|---|
| Frozen DINOv3 ViT-L/16, multi-layer | DINOv2-base (DINOv3 gated; a larger model would push past the 9 GB cap, D-007) | Primary features = Phase 3 B0 features, so every comparison is with the published baseline. A multi-layer variant is secondary. |
| 9 Synthbuster generators + SD3.5 + FLUX | 6 Synthbuster + SD3.5 (train/test) + FLUX (test only) | LOGO runs over 7 generators. FLUX stays the untouched held-out generator. |
| VISION native + WA + FBH + FBL, 20 photos/device | native + WA + FBH. Splits: 10 train devices / 10 calib / 25 test (99/43/358 natives) | No FBL. Real platform pairs for H1 come from the 10 train devices only (2 × 99 pairs). |
| Conditions C0–C5 | D-009 has clean, jpeg75_resize05, social_sim | Added below: the factor ablation, a format-match condition and two held-out laundering types. |
| Mac MPS | CUDA (RTX 4060 Ti 8 GB) | Feature passes take minutes. The D-008 guard is now CUDA vs CPU. |

## Shared infrastructure: experiments/004-phase4-features
- **Backbone:** DINOv2-base, 224 center crop of the 512 crop (as in B0).
  - Primary feature: last layer CLS + mean patch (1536-d), identical to B0.
  - Secondary feature: blocks {4, 8, 12} CLS + mean (4608-d).
- **Evaluation conditions** (both classes, seeded per image):
  - D-009 conditions: `clean`, `jpeg75_resize05`, `social_sim`.
  - Factor ablation: `resize(0.5)`, `jpeg(75)`.
  - Format match: `jpeg(90)`.
  - **Held out from all fitting:**
    - `mavali_medium` = resize 0.75 then JPEG q50;
    - `screenshot_sim` = resample by a factor drawn from {0.5, 0.67, 1.25} (bilinear), 8-bit RGB, lossless PNG, then a 2–5% border crop.
- **Training-augmentation grid (train split only):** JPEG q ∈ {60, 75, 90} × resize ∈ {0.5, 0.75, 1.0}, which gives 9 conditions.
- **D-008 guard:** CUDA vs CPU on a fixed 32-image subset for every new condition. Abort if > 1e-3.

## H1 — laundering-nuisance-projection (experiments/005-nuisance-projection)
- **Arms:** all use the standardized logistic probe from `probe.fit_probe`, seed 0, with standardization fitted on clean train reals.

  | Arm | Training data | Projection |
  |---|---|---|
  | A0 | clean train | none (= B0) |
  | A1 | clean + 9-grid laundered copies of all train images, both classes | none |
  | **A2 (H1)** | same as A1 | projection P_k applied first |
  | A3 (ablation) | clean only | projection P_k applied first |

- **How P_k is built:** P_k = I − U_k U_kᵀ, where U_k holds the top-k eigenvectors of the covariance of the differences f(L(x)) − f(x).
  - The differences use **reals only**:
    - real platform pairs (train devices, native↔WA and native↔FBH);
    - the 9-grid on train reals (VISION train natives + RAISE train).
  - No fake image, and no calib or test image, enters the estimate.
- **Choice of k:** pre-registered as the smallest k that captures 80% of the difference energy. k ∈ {16, 64, 256} are also reported.
- **Primary endpoints:** AUROC on FLUX and LOGO-macro AUROC (7 generators) under `jpeg75_resize05`, A2 vs A1. Each comes with a paired-bootstrap 95% CI on the difference.
- **Pass** requires all of:
  - A2 − A1 ≥ +0.03 on both FLUX and LOGO-macro at `jpeg75_resize05`;
  - clean AUROC A2 ≥ A1 − 0.01;
  - `screenshot_sim` AUROC A2 ≥ A1;
  - threshold drift reduced by ≥ 50%. Drift = |FPR(WA) − FPR(native)| and |FPR(FBH) − FPR(native)| on test devices, at the 5%-FPR threshold set on calib-device natives.
- **Fail:** the gain is < +0.01, or the clean loss is > 0.02. Anything in between is inconclusive.

- **Amendment 1 (2026-10-23, before any result; from kb/novelty-check-phase4.md):**
  - The nearest prior work is DRIFT (arXiv 2606.06918). It learns real-only robust and fragile heads on a frozen DINOv2 ViT-B/14, using synthetic transforms only.
  - New secondary arm **A4, "DRIFT-lite":** a 2-layer MLP projection head trained on the same real-only differences (the same pairs as P_k). Its training objective makes the projected features invariant between f(x) and f(L(x)) while preserving the variance of the clean features. The head is followed by the same probe as A1.
  - A4 is a comparison only and has no pass/fail role. H1 is not claimed as "the first real-only nuisance subspace"; the claim is closed-form + real platform pairs + held-out laundering types + FPR drift.

## H4 — drift-format-shortcut, plus a B0 format fix (experiments/006-format-shortcut)
- **Settings** (test split only):
  - (i) VISION-native JPEG reals vs clean PNG fakes;
  - (ii) RAISE PNG reals vs PNG fakes;
  - (iii) RAISE and fakes both `jpeg(90)`;
  - (iv) RAISE and fakes both `jpeg75_resize05`.
  - Caveat: Firefly files are already JPEG q≈95, so results are reported both with and without Firefly.
- **RIGID-style score:** 1 − cos(f(x), f(x + n)), using the DINOv2-base last-layer CLS.
  - n is Gaussian noise added in [0,1] pixel space before normalization, with σ ∈ {0.01, 0.02, 0.05}, averaged over 4 seeded draws.
  - The direction is fixed in advance (higher = fake) and is never flipped after looking.
- Precedent: RIGID's Appendix D re-tested DIRE with both classes converted to JPEG, but never ran that test on RIGID itself (kb/novelty-check-phase4.md).
- **Supported if:** for the best σ, AUROC(i) − AUROC(iii) ≥ 0.10, while B0's gap is < 0.05.
- **Refuted if:** the gap is < 0.03 for every σ.
- **B0-JM (fix):** B0 trained with each train fake JPEG-compressed at q ~ U[85, 95] with probability 0.5.
  - Success: the per-real-pool gap AUROC(VISION) − AUROC(RAISE) falls from 0.086 (Phase 3) to ≤ 0.03, with a LOGO-macro clean loss ≤ 0.01.

## H2 — worst-env-conformal-fpr (experiments/007-worst-env-conformal)
- Framing: an evaluation protocol plus a recipe, not new conformal math. Related work: 2505.05084, 2606.09881, 2602.01973, 2512.15182.
- **Scores:**
  - Primary: B0. Secondary: the 005 arms A1/A2 (if they exist), CommFor, B-Free, RINE and ClipDet.
- **Reals:** the 25 test devices, with natives + WA + FBH, plus `jpeg75_resize05` and `screenshot_sim` versions of the natives as synthetic "platforms".
- **Thresholds:**

  | Id | Rule |
  |---|---|
  | M0 | native logit 0 |
  | M1 | 1% FPR quantile on the 10 calib devices (a different set of phones) |
  | M2 | pooled split-conformal, α = 1% |
  | **M3 (H2)** | maximum over calibration devices of each device's 99th percentile |
  | M4 | pooled conformal at α/2 |

- **Splits:**
  - Leave-one-device-out: calibrate on 24 test devices, test on the 25th, and pool the decisions.
  - Leave-one-platform-out: hold out WA, FBH, `jpeg75_resize05` and `screenshot_sim` in turn.
- **Pass** requires all of:
  - M3 leave-device-out FPR ≤ 1.5%, with a Clopper-Pearson upper bound ≤ 2.0%;
  - leave-platform-out FPR ≤ 3% on every held-out platform;
  - FLUX TPR at M3 ≥ 0.7 × the TPR at M2 (clean and `jpeg75_resize05`);
  - at least one of M0/M1/M2 exceeds 2% somewhere.
- **Fail:** M3 leave-device-out FPR > 2%, or a TPR ratio < 0.5.

## Not run now
- **H3 neural-isp-fpr-audit:** needs about 250 photos from the owner's 2021+ phone.
- **#5–#7:** deferred, as in Phase 2.

## Process
- Each experiment gets one sonnet/high code agent and an opus/high critic on its result (CLAUDE.md).
- A sonnet/medium agent re-reads the full text of DRIFT, RA-Det and RIGID, because Phase 2 checked them at abstract level only.
- Stop rule: 3 failed hypotheses → stop and report.

## Amendment 2 (2026-10-23 17:15, before any Phase 4 result): PILOT while grip.unina.it is down
**Situation.** The server has been unreachable for 6+ hours, from both WSL and Windows. RAISE, SD3.5, FLUX and the B-Free weights are unavailable. The owner chose "pilot now, final later".

**Pilot data.** Rows of 003 `splits.csv` whose raw files exist: VISION native/WA/FBH and the 6 Synthbuster generators. Splits are unchanged; missing rows are dropped.
- The crops are regenerated with 003's preprocessing. Each crop's sha256 must equal the `splits.csv` sha256. This is the partial portability check that is possible now.

**Pilot endpoint changes.**

| Hypothesis | Change |
|---|---|
| H1 | FLUX is replaced by LOGO-macro over the 6 Synthbuster generators. LOGO-macro becomes the sole primary endpoint. The pass/fail thresholds are unchanged. Train reals are 99 VISION natives (no RAISE). |
| H2 | Unchanged, except TPR is LOGO TPR (each generator scored by the probe that never saw it) instead of FLUX TPR. Detectors: B0 plus A1/A2, CommFor, RINE and ClipDet. B-Free is unavailable. |
| H4 | Deferred to the final run. Settings (ii)–(iv) need the RAISE PNG reals. |

**Status.** All pilot outputs are labelled PILOT and live in `<exp>/pilot/`. A pilot pass or fail is not a Phase 4 verdict and does not count toward the 3-failed-hypotheses stop rule. The official run repeats everything on the full data, together with the step-6 reproduction.

## Amendment 3 (2026-10-23, after both PILOT critiques, before any official result)
**Pilot outcome.** Both critics rate the pilots *uninformative*, with no verdict-changing bugs:
- `experiments/005-nuisance-projection/pilot/critique.md`
- `experiments/007-worst-env-conformal/pilot/critique.md`

H1 had no headroom: A1 was 0.988, so the largest possible gain was +0.012. H2's M3 degenerates to the calibration max, because there are only about 43 reals per device. Neither counts toward the stop rule.

**New confound (lead-verified, manifest_raw.csv).** The median short side is:

| Pool | Median short side (px) |
|---|---|
| VISION native | 2448 |
| Fakes | 768–1024 (Firefly 1792) |
| WhatsApp | 960 |
| FB-HQ | 1536 |
| RAISE | 833 |

Because D-009 takes a 512 crop with no resize, real crops are about 2.5–3× more zoomed in than fake crops.
- Lead check (`005/pilot/lead_pool_check.txt`): A0 LOGO-macro clean AUROC is 0.992 against natives, 0.982 against FB-HQ and 0.965 against WA copies of the same test photos. So the confound explains part of the ceiling, not all of it.

**Changes for the OFFICIAL run.** These are additive; D-009 stays the primary protocol unless the owner decides otherwise (D-013).
1. **Order:** H4 runs first. It is extended with a scale arm, `fov512`: resize the short side to 512 (bicubic, antialiased), then center-crop 512. Both classes get this, so the field of view is roughly matched and every image is resampled. All 005/007 headline tables add a `fov512` column and per-real-pool rows (native / WA / FBH / RAISE).
2. **Statistics, for all arms equally:**
   - grouped CV in `fit_probe`, where a group = the source image, so augmented copies stay in one fold;
   - a class-balanced logistic probe;
   - a bootstrap over devices (reals) and images (fakes).
3. **H1 headroom rule:** the +0.03 criterion is applied only if A1's LOGO-macro AUROC at jpeg75_resize05 is ≤ 0.96. Otherwise the result is reported as "no headroom / uninformative".
   - The primary endpoint is unchanged.
   - Drift (FPR stability) stays secondary, flagged as pilot-motivated.
   - The A1_sec and A2_sec arms are also passed to 007.
4. **H2 reformulated as protocol + recipe:**
   - Groups have ≥ 100 reals (platform first, then brand), each with a proper per-group conformal threshold.
   - The decisive comparison is worst-group vs pooled conformal at matched TPR, sweeping α.
   - "Problem is real" now requires M1 or M2 > 2% (not M0).
   - Evaluation is LODO × LOPO.
5. **H3 (owner's modern-phone photos)** becomes the main FPR ≤ 1% test. The owner was asked.

## Amendment 4 (2026-10-23, before the official run): settling the implementation's judgment calls
1. **H2-v2 groups:** platform × brand, with groups of ≥ 100 reals.
   - On the VISION test devices this degenerates to platform-only (Apple has 99 reals per platform). The official run therefore uses platform-only groups at min-group 100.
   - min-group 50 (platform × brand) is a reported sensitivity analysis.
2. **H2-v2 verdict rule:** adopted as implemented in `007/h2v2.py:32`.
   - SUPPORTED if the CI upper bound of (worst-group FPR − pooled FPR at matched clean TPR) is < 0 at ≥ 3 of the 4 α values.
   - NOT_SUPPORTED if the point estimate is ≥ 0 at ≥ 3 of the 4.
   - INCONCLUSIVE otherwise.
   - **Transparency:** the code author proposed this rule, and the lead adopts it after seeing the pilot_v2 labels (NOT_SUPPORTED everywhere, pilot only).
   - **Clipping:** groups too small for a finite conformal threshold are clipped to the group max, and the clipped share is reported.
3. **H1 "both":** the +0.03 gain must hold for FLUX AND for LOGO-macro separately. The code takes the minimum of the two, which matches the original pre-registration.

**Pilot_v2 note** (`005/pilot_v2/{orig,fov512}`, grouped CV + balanced probe + cluster bootstrap):
- With fov512, the real-pool gaps vanish: A0 clean is native 0.979 / WA 0.979 / FBH 0.980, against orig 0.992 / 0.966 / 0.982.
- The ceiling stays about 0.98, so there is still no headroom on the pilot data.
- WA pooled-conformal FPR at 1%: 3.31% orig vs 1.58% fov512 (`007/pilot_v2`, indicative).

## Amendment 5 (2026-10-23 22:58, before any H4 result): H4 PILOT on the data we have
**Why now.** grip.unina.it is still down, and H4 runs first in the official order (Amendment 3), because it decides how H1 and H2 are read. Two of its comparisons need no RAISE, so they are piloted now. Like every pilot, this does not count toward the stop rule and is not a Phase 4 verdict. Outputs go to `experiments/006-format-shortcut/pilot/`.

**Data.** The pilot test split from `004-phase4-features/pilot/splits_pilot.csv`: 1,074 VISION reals (native, WA, FBH; 25 devices) and 360 fakes (6 Synthbuster generators × 60). Each result is reported both with and without Firefly, whose files are already JPEG q≈95.

**Settings.** Every image passes through the same processing:

| Setting | Reals | Fakes | Replaces |
|---|---|---|---|
| (i) | VISION native, as is (camera JPEG) | clean PNG | (i) unchanged |
| (iii-p) | VISION native re-encoded `jpeg(90)` | `jpeg(90)` | (iii); RIGID App. D protocol, both classes converted to JPEG |
| (iv-p) | VISION native `jpeg75_resize05` | `jpeg75_resize05` | (iv) |

Setting (ii) (RAISE PNG vs PNG) has no pilot version. Caveat: in (iii-p) the reals are compressed twice, so their *final* format matches the fakes, but not their whole history.

**Scale arm.** Each setting is run twice: on the 003 crops (`orig`) and on `fov512` crops (short side resized to 512, then center crop), using the existing 004 preprocessing. The WA and FBH copies appear only in a descriptive per-pool table for setting (i).

**RIGID-style score.** As pre-registered in the H4 section:
- score = 1 − cos(f(x), f(x + n)), with f = the DINOv2-base last-layer CLS on the 224 input used by 004;
- n is Gaussian, added in [0,1] pixel space before normalization and clipped to [0,1];
- σ ∈ {0.01, 0.02, 0.05}, averaged over 4 seeded draws;
- the noise is generated on the CPU from a fixed seed per (image, draw), so it is identical across devices;
- the direction is fixed in advance (higher = fake) and is never flipped.

**Supervised comparator.** B0 = the A0 probe of 005 (primary 1536-d features, grouped CV, balanced). Its LOGO-macro AUROC comes from the cached 004 features for `clean` and `jpeg(90)` and `jpeg75_resize05`, in both `orig` and `fov512`.

**Pilot criteria.** These are the original H4 thresholds, applied with (iii-p) in place of (iii):

| Pilot result | Condition |
|---|---|
| SUPPORTED | For the best σ, AUROC(i) − AUROC(iii-p) ≥ 0.10, while B0's gap is < 0.05 |
| REFUTED | The gap is < 0.03 for every σ |
| INCONCLUSIVE | Otherwise |

- "Best σ" is the σ with the highest AUROC(i).
- Uncertainty comes from a cluster bootstrap: devices for reals, images for fakes, 2,000 resamples.

**Secondary (descriptive, no verdict).**
- The `orig` − `fov512` difference in AUROC(i), for the RIGID score and for B0. This is the first direct measurement of the zoom shortcut for a training-free score.
- The 1%-FPR operating point for the RIGID score, from the calib-split reals.

**D-008 guard.** CUDA vs CPU on 32 noised images: scaled |Δfeature| ≤ 1e-3, with max |Δscore| reported. Abort if the guard fails.

## Amendment 6 (2026-10-24, after the H4 pilot critique, before any official H4 data)
**Pilot outcome** (`experiments/006-format-shortcut/pilot/critique.md`, no bug found):
- The RIGID-style score is below chance on VISION + Synthbuster: AUROC(i) is 0.292 orig and 0.405 fov512.
- The cause is input texture. Zoomed-in phone crops are smooth (median Laplacian 0.031 vs 0.080 for fakes), and smooth inputs drift more under noise. Spearman(score, Laplacian) is −0.73 within reals; the lead re-checked this from the critic's `c2_stats.csv.gz`.
- Texture alone scores 0.222, and the texture-residualized score is about 0.51.
- Both pilot labels are therefore read as **UNINFORMATIVE**: the score does not work on this data. "REFUTED" must NOT be taken to mean that drift scores are a valid fusion component.

**Changes for the official H4 run.** Thresholds (0.10 / 0.03, B0 < 0.05) are unchanged; the direction is never flipped.
1. **Validity gate.** A label other than UNINFORMATIVE is allowed only if, for the best σ:
   - the lower bound of AUROC(ii)'s 95% CI is > 0.5; and
   - RIGID's AUROC(ii) beats the Laplacian texture baseline's AUROC(ii). The baseline is mean |Laplacian| of the 224 input, with its direction set on calib-split reals and fakes.
   - "Best σ" is now the σ with the highest AUROC(ii).
2. **Primary contrast isolates format.** New setting **M** = RAISE reals `jpeg(90)` vs clean PNG fakes. The primary gap is AUROC(M) − AUROC(ii). Only the reals' encoding differs: same reals, same fakes, and a JPEG-real/PNG-fake shortcut is available in M only.
   - The original (i) − (iii) and also (M) − (iii) are reported as secondary.
   - This replaces the critic's suggested (ii) → (iii), because in (ii) → (iii) neither side offers a format shortcut.
3. **Manipulation check.** For every contrast, report the mean change in Laplacian texture and B0's gap, so we can see how much the format change alters the inputs.
4. **Which arm decides.** The label comes from the arm that D-013 makes primary; until the owner decides, that is `orig` (D-009). `fov512` and a new whole-image arm `fov224` are reported alongside. `fov224` resizes the short side to 224 and then center-crops, which may be closer to RIGID's preprocessing [unverified].
5. **Texture-residualized RIGID** (descriptive). Regress the score on log-Laplacian using calib-split reals only, then report the AUROC of the residual on test.
6. **Cross-scale B0** (descriptive). Train on orig and test on fov512, and the reverse. Retraining B0 inside each arm hides how scale-dependent the probe itself is.
7. **Not adopted without the owner:** a DINOv2 ViT-L/14 backbone. It falls under the larger-backbone decision and the 9 GB cap (D-007).

**Disclosure.** The pilot agent edited config.json (adding the pre-LN variant and making post-LN primary) after the dry run had already shown the inversion. Post-LN and pre-LN give identical labels in every cell, so this changes nothing.

## Amendment 7 (2026-10-24, before any result): E1 zoom sensitivity of published detectors (experiments/008-scale-sensitivity)
**Why.** The H1 and H4 pilots found a field-of-view (zoom) confound. Phone crops are 2.5–3× more zoomed in than fake crops. That moved the RIGID-style score by 0.10–0.16 AUROC and B0 by only 0.01–0.015. Do the released detectors we use (CommFor, RINE, ClipDet) also depend on zoom? Their published phone-photo FPRs, and our Phase 3 numbers, would be affected.

**Status.** E1 is an exploratory diagnostic, not a hypothesis. It does not count toward the stop rule. It is PILOT data (VISION + 6 Synthbuster) and is repeated on the official data later.

**Data.**
- Pilot test split: VISION native / WA / FBH reals and 6 Synthbuster generators.
- Calib-split VISION reals are used for thresholds.
- Two arms with identical processing for both classes:
  - `orig`: 003 crops;
  - `fov512`: short side resized to 512, then center crop, using the existing 004 crops.
- Conditions: `clean` and `jpeg75_resize05`.

**Detectors.**
- CommFor, RINE and ClipDet, each at its own preprocessing, scored with `007/score_pretrained.py` on both arms. B-Free is unavailable.
- B0 = 005's A0 (grouped CV, balanced), from cached 004 features.
- Texture baseline: mean |Laplacian| of the detector input, with its direction set on calib.

**Primary measure (per detector).** ΔAUROC = AUROC(orig) − AUROC(fov512):
- AUROC is the macro over the 6 generators, against test natives;
- 95% CI from a paired cluster bootstrap (reals by device, fakes by image), 2,000 resamples.
- A detector is labelled **scale-sensitive** if |ΔAUROC| ≥ 0.03 and the CI excludes 0.

**Secondary (descriptive).**
1. **Cross-zoom threshold transfer.**
   - Fix a 1%-FPR threshold on calib natives in one arm. Report the test FPR on natives, WA and FBH in the *other* arm, and TPR.
   - Do this in both directions, against a within-arm threshold.
   - This mimics deployment, where the same photo can arrive at different resolutions.
2. **Per-pool AUROC** (native / WA / FBH) in each arm.
3. **Correlation** between each detector's real-photo score and the Laplacian texture.

**Caveat (stated in advance).** `fov512` downsamples phone photos by about 4.8× but fakes by only about 2×. So it changes resampling history differently for the two classes. E1 measures sensitivity to zoom *and* resampling together, and cannot separate them.

**D-008.** The guard runs for every new scoring run (|Δlogit| ≤ 1e-3).

## Amendment 8 (2026-10-24, before any result): E2 PILOT of ideas #6 (aligned-VAE probe) and #7 (VAE double-reconstruction ratio) (experiments/009-vae-aligned)
**Why now.** Phase 2 planned "a 20-image pilot when idle" for #6 and #7. The GPU is idle while grip.unina.it is down. Both ideas use only reals plus public VAEs, so they can be piloted now. Like every pilot, this does not count toward the stop rule.

**Shared setup.**
- VAEs: `stabilityai/sd-vae-ft-mse` (SD1.x family) and `stabilityai/sdxl-vae`.
- `diffusers` 0.35.2 has been added to the lock.
- The FLUX VAE is not used: FLUX is the held-out generator.
- Pilot splits come from `004/pilot`, in two arms: `orig` (003 512 crops) and `fov512`.
- Features are B0's DINOv2-base primary (1536-d) via the 004 pipeline.
- Conditions: `clean` and `jpeg75_resize05`, applied identically to both classes.
- **Timing gate.** First time 20 images. If one VAE pass at 512² takes more than 1 s, cut the test set to 30 fakes per generator and 150 natives, and say so.

**#6 aligned-VAE probe.**
- Aligned fakes = every train-split real (99 native + 99 WA + 99 FBH) round-tripped (encode → decode) through each VAE at its own crop. That gives 594 fakes.
- Each round-trip is re-saved as JPEG at the source's estimated quality factor (`lib/preprocess.jpeg_quality_estimate`), so real and aligned fake share format and zoom.
- **Arms:**
  - A = probe on reals vs aligned fakes only. It never sees a generator image, so all 6 Synthbuster generators are unseen.
  - B = B0 (005's A0, LOGO).
  - C = B's LOGO training data plus the aligned fakes.
- **Metrics:** macro AUROC over the 6 generators vs test natives, with cluster-bootstrap CIs. Generators are also grouped into SD-VAE family (SD2, SDXL) and other (DALL-E 2, DALL-E 3, Firefly, Midjourney v5).
- **Pilot criteria:**
  - A PASSES if macro AUROC at `jpeg75_resize05` is ≥ 0.85 in the label arm (`orig`).
  - A FAILS if it is < 0.75.
  - C vs B uses the +0.02 criterion only if B's LOGO-macro is ≤ 0.96 (headroom rule as in Amendment 3). Otherwise it is "no headroom / uninformative".
  - The UnivFD comparison in the idea file is dropped: no UnivFD head is cached.
- **Texture check** (lesson from H4): report the Laplacian-texture AUROC of reals vs aligned fakes, and A's AUROC within texture-matched strata. This shows whether A only learns "the VAE blurs".

**#7 double-reconstruction ratio.**
- x1 = VAE(x) and x2 = VAE(x1).
- Score = ‖x − x1‖₂ / ‖x1 − x2‖₂, using the minimum over the two VAEs. Higher = more real, so the score for "fake" is the negative.
- The direction is fixed by AEDR's premise and never flipped.
- Data: test natives and test fakes, both arms, both conditions.
- **Baselines:**
  - single-pass reconstruction error ‖x − x1‖₂, min over the bank. This is an AEROBLADE-style baseline with L2 in place of LPIPS. LPIPS would need a further VGG download; this deviation is disclosed.
  - the Laplacian texture baseline.
- **Pilot criteria:**
  - PASS if the ratio's macro AUROC (clean) is ≥ the single-pass baseline + 0.05, AND ≥ 0.65 at `jpeg75_resize05`, AND it beats the texture baseline.
  - FAIL if any of these is missed.
  - Validity gate as in Amendment 6: if the ratio's clean AUROC CI includes 0.5, the label is UNINFORMATIVE.
  - The fusion criterion (+0.02 over B0 on FLUX) is not testable in the pilot: B0 is at ~0.98 with no FLUX.

**D-008.** CUDA vs CPU VAE outputs on 32 images: max |Δpixel| ≤ 1e-3 (in [0,1]); max |Δlogit| ≤ 1e-3 for probes. Abort on failure. TF32 is disabled.

## Amendment 9 (2026-10-24, before any data is downloaded): E3 PILOT of idea #5 (canonicalization vs RAID transfer attacks) (experiments/010-raid-transfer)
**Source.** `kb/datasets/raid.md`:
- RAID is MIT-licensed, at aimagelab/RAID on HF. The upstream D3/LAION image licences are [unverified].
- The attack ensemble is Wang20, Corvi23, Ojha23 (UnivFD), Cavia24, Koutlis24 (RINE) and DRCT. DINOv2 is a held-out baseline, so our DINOv2-base probe is outside the ensemble.

**Subset.**
- 100 ids per class × 5 classes (real, gen_0..gen_3), chosen with seed 0 from ids present in both `original/` and `epsilon8/`.
- For each id: the eps-8 attacked PNG, the eps-16 attacked PNG (an anchor against the paper's Table 4) and the original JPEG.
- About 140 MB. Each file's sha256 is checked against the HF LFS oid and logged in a committed manifest.
- This is a pilot. It does not count toward the stop rule.

**Inputs.**
- Attacked images: 256² PNG, as released.
- Clean control: the original JPEG (324²), resized with the short side to 256 and center-cropped to 256. Bicubic, antialiased; RAID's exact preprocessing is [unverified].
- Every detector then applies its own preprocessing. For B0 that is the 224 center crop of the 256 image, which differs from the 512→224 crop it was trained on. This domain shift is stated up front.

**Detectors.**
- B0 = the A0 probe trained on ALL pilot train data (6 Synthbuster generators plus VISION train reals, `orig` arm). No LOGO is needed, since RAID's generators are unseen.
- References: CommFor and ClipDet (not in the ensemble) and RINE (in the ensemble, so a near-white-box reference).

**Test-time canonicalization.** `jpeg75_resize05`, applied to both classes, on clean and on attacked inputs.

**Metrics.** AUROC of real vs fake (all 4 generators pooled, plus per generator), with 95% CIs from a bootstrap over ids. Then:
- drop = AUROC(clean) − AUROC(attacked, eps 8);
- recovery = [AUROC(attacked + canon) − AUROC(attacked)] / drop;
- canonicalization clean cost = AUROC(clean) − AUROC(clean + canon). The originals are already JPEG at about q75, so this cost is probably understated. That caveat is stated in advance.

**Pilot criteria for B0.**

| Label | Condition |
|---|---|
| UNINFORMATIVE (validity gate) | AUROC(clean) < 0.80 |
| PASS | drop ≤ 0.10, and either drop < 0.02 or recovery ≥ 50% |
| FAIL | drop > 0.10 and recovery < 50% |
| INCONCLUSIVE | Otherwise |

Eps-16 numbers are descriptive only. They are compared with RAID Table 4 (DINOv2 accuracy 83.1 → 54.6).

**D-008.** The guard runs for every scorer (|Δlogit| ≤ 1e-3).

**Amendment 9 clarification** (2026-10-24, before any E3 result; only a 25-item CPU smoke test exists, numbers unread):
- On RAID's 256² inputs, `jpeg75_resize05` gives 128², which is smaller than B0's 224 crop.
- Canonicalization is therefore: resize ×0.5 → JPEG q75 → bicubic back up to 256.
- This applies to both classes, and to clean and attacked inputs, for every detector.
- The extra up-sampling is part of the canonicalization, so it counts toward the clean cost.

**Amendment 7–8 clarifications** (2026-10-24, from the pre-run code reviews `008/code_review.md` and `009/code_review.md`; no E1 or E2 result exists yet)
- **E2 #7 validity gate:**
  - Amendment 8 said "as in Amendment 6", but its parenthetical ("CI includes 0.5") misstated that gate.
  - The rule is Amendment 6's: a label other than UNINFORMATIVE requires the lower bound of the 95% CI of the ratio's clean AUROC to be > 0.5.
  - So a ratio that is significantly *below* chance is UNINFORMATIVE, not FAIL.
- **E2 aligned fakes in the `fov512` arm** are saved as PNG, not JPEG.
  - fov512 reals are resampled PNGs with no fresh JPEG pass, so a JPEG re-save would give only the fakes a format cue.
  - The `orig` arm (the label arm) keeps the JPEG re-save at the source quality factor.
- **E2 CV groups** = source photo (`device:vidx`). The native, WA and FBH copies of one photo, and all of their round-trips, then share a fold.
- **E2 guard order:**
  - the VAE D-008 guard runs before the bulk VAE stage;
  - `analyze` refuses to write results unless both guards passed.
- **E1 transfer table:**
  - For B0, only within-arm rows plus the fixed-probe rows `b0_trained_*` are reported. "Cross" rows that pair one arm's threshold with the other arm's retrained probe are dropped as meaningless.
  - Rows beyond Amendment 7 (the 129-calib-reals variant and the fixed-probe B0 rows) are marked "not pre-registered".
  - The texture sign is set once per (detector input, condition), on train data.

**Amendment 9 clarification 2** (2026-10-24, from `010/code_review.md`; no E3 result exists, only a 25-item CPU smoke test whose numbers were not read)
1. **Clean control = RAID's own preprocessing.** That is a plain center crop to 256², zero-padded when a side is under 256, with no resize. It replaces the earlier resize-then-crop.
   - The code asserts the pairing: for every id, the mean |attacked − clean| must be ≤ 10/255 at eps 8 and ≤ 18/255 at eps 16, or the run aborts.
2. **Padding shortcut.** 52 of the 100 selected reals are smaller than 256 px, so the crop gives them black borders that the fakes lack.
   - The **primary** label uses only reals with no padding (both sides ≥ 256) against all fakes.
   - **Sensitivity:** all 100 reals, reported but never labelled.
   - Any fake that needs padding is excluded from the primary analysis in the same way. The counts are reported.
   - The bootstrap resamples reals and fakes independently, by id, within each class.
3. **Canonicalization** (clarification 1) still applies on top of this 256² input.
4. **Caches** are keyed by (input set, preprocessing version, device). Score caches made before this clarification are invalid and must be deleted or ignored.

**Amendment 8 clarification 2** (2026-10-24 04:45; no E2 result exists, only guard numbers)
- **What happened.**
  - The pixel-level VAE guard (max |Δpixel| ≤ 1e-3, an addition beyond CLAUDE.md's logit-level D-008 rule) passed for sd-vae-ft-mse (9.2e-5).
  - It FAILED for sdxl-vae: max 5.26e-3 (x1) and 1.25e-3 (x2), with 0.07% of uint8 pixels differing.
  - A lead diagnostic on 8 of the guard images gives the same result with deterministic cuDNN: max 2.5e-3, p99.99 = 2.9e-4. TF32 was already off. So the error is inherent fp32 GPU-vs-CPU rounding in SDXL's VAE (which has large activations), not a silent failure like D-010's.
  - SDXL on CPU takes about 11 s per pass, roughly 10 h for the pilot.
- **Rule** (fixed before any result):
  - sdxl-vae is accepted on CUDA if the D-008 criterion holds downstream on the same 32 guard images.
  - The criterion: B0 (005 A0) logits on 004 features of the GPU-made vs CPU-made round-trips (features computed on the same device for both) differ by ≤ 1e-3 at max |Δlogit|.
  - Also reported: max relative |Δ| of the #7 ratio, and Spearman ρ between GPU and CPU ratios.
  - If the criterion fails, E2 runs with sd-vae-ft-mse ONLY. The "min over bank" becomes that single VAE, the change is reported, and SD-family results are then based on SD1.x only.
- The pixel guard for sd-vae-ft-mse stays as pre-registered.

## Amendment 10 (2026-10-24, before any result): E1b, separating zoom from resampling (experiments/008-scale-sensitivity, subdir pilot_du2)
**Why.** In E1, `fov512` changes zoom and resampling by the same factor, so the two cannot be separated (`008/pilot/critique.md`). This exploratory pilot does not count toward the stop rule.

**New arm `du2`.**
1. Resize the image so its short side is 512, exactly as `fov512` does (bicubic, antialiased).
2. Resize it back up to its original size (bicubic).
3. Take the 003-style 512 center crop with no further resize.

So `du2` has the zoom (field of view) of `orig` and the downsampling history of `fov512`. The same processing is applied to both classes, under the same conditions as E1 (`clean`, `jpeg75_resize05`).

**Decomposition** (per detector, macro AUROC over the 6 generators vs test natives, paired cluster bootstrap as in E1):
- Δ_resample = AUROC(orig) − AUROC(du2): the resampling effect at fixed zoom.
- Δ_zoom = AUROC(du2) − AUROC(fov512): the zoom effect at fixed resampling.
- Δ_resample + Δ_zoom equals E1's Δ exactly.

**Labels** (descriptive): "resampling-dominated" or "zoom-dominated" if |Δ_resample| − |Δ_zoom| has a 95% CI that excludes 0 in that direction; "mixed" otherwise. Only detectors E1 flagged as scale-sensitive in a condition get a label for that condition.

**Also.**
- A fixed-probe B0 (trained on `orig`, and on `fov512`) scored on `du2`.
- Cross-scale 1%-FPR with device-bootstrap CIs and a second-highest-calibration-score threshold variant, as the critic recommended.
- WA and RAISE-like scale-matched pools are reported per pool.

**D-008.** The guard runs for every new scoring run.

## Amendment 11 (2026-10-24 12:15, before any result): E4, how detectable is the diffusion VAE decoder trace? (experiments/011-vae-trace)
**Why** (D-014). Diffusion-specific methods (AEROBLADE, FIRE, LaRE², aligned training) assume the latent-diffusion decoder leaves a detectable trace. In E2, our B0 features separated real photos from their own SD-VAE round-trips at only 0.66 clean / 0.56 laundered (`009/pilot/critique.md`). E4 measures directly which features see the trace and whether it survives laundering. It is a diagnostic pilot that does not count toward the stop rule.

**Data.**
- Pilot splits (`004/pilot`, `orig` arm as the label arm, `fov512` reported).
- Pairs: each real (native, WA, FBH) and its own sd-vae-ft-mse round-trip, made as in E2 (latent mode, uint8). The orig arm re-saves at the source JPEG quality factor; fov512 saves PNG (Amendment 7–8 clarifications).
- Train on train-split pairs (297 + 297); test on test-split pairs (1,074 + 1,074).
- Groups and clusters are by source photo (`device:vidx`) and device.
- Conditions: `clean` and `jpeg75_resize05`, applied identically to both members of a pair.

**Feature families** (logistic probe, balanced, grouped CV for C):
- F1: B0 primary (DINOv2-base last-layer CLS + mean).
- F2: DINOv2-base blocks 4/8/12 CLS + mean (the 004 secondary features).
- F3: a fixed low-level residual spectrum. The image minus a 3×3 median filter, then the azimuthally averaged log power spectrum in 32 radial bins per channel (96-d), plus the 8×8-grid periodicity peaks at the latent stride (f = 1/8, 2/8, 3/8, 4/8 cycles/px in x and y, 8 values per channel). These are standard choices fixed here, before results.
- F4: F2 + F3 concatenated.
- Baseline: the Laplacian texture score (fixed direction set on train).

**Measures.**
1. **Pair AUROC** (real vs own round-trip) on test, per feature family and condition, with cluster-bootstrap CIs.
2. **Transfer:** each pair-trained probe scored on test natives vs Synthbuster fakes, macro AUROC split into SD-VAE family (SD2, SDXL) and other generators.

**Decision rules** (for the direction; this is not a hypothesis verdict):
- "VAE trace detectable" if the best family's pair AUROC is ≥ 0.90 clean, with CI lower bound > 0.85.
- "Survives laundering" if it is ≥ 0.75 at `jpeg75_resize05`, with CI lower bound > 0.70.
- "Trace-specific" if the best family beats the texture baseline by ≥ 0.10 on pairs.
- If the trace is not detectable or does not survive laundering, VAE-trace methods are deprioritized in the Phase 5 design.

**D-008.** Guards apply to VAE outputs (pixel ≤ 1e-3 for sd-vae-ft-mse, which already passed) and to new features (|Δlogit| ≤ 1e-3).

## Amendment 12 (2026-10-24, before any H-D1 or H-D2 result): pilots of H-D1 self-ref-trace-contrast and H-D2 latent-grid-periodicity-rescale (experiments/012-self-ref, experiments/013-latent-grid)
**Source of criteria.** The pass/fail rules, gates and predictions are those in `kb/ideas/self-ref-trace-contrast.md` and `kb/ideas/latent-grid-periodicity-rescale.md`, as committed in ca58d61, with the pilot adaptations below. Both are pilots under D-014 and do not count toward the stop rule.

**Disclosure (written before running).** The E4 critic (`011/pilot/critique.md`, post hoc, on a 60-images-per-generator sample) already checked a related reference score s(x) − s(VAE(x)). It reached 0.850 macro at clean native scale but fell below the single-image probe once laundered (0.56 orig, 0.45 fov512). It also found that a plain Gaussian blur reproduces the 0.97–0.99 paired win-rate, and that the 0.5× resize, more than JPEG, erases the trace. H-D1 is still run, because its deployable score (L0 applied to both t and VAE(t) of the OBSERVED image) differs from that check and costs about an hour. Its criteria are not changed in light of the preview.

**Pilot adaptations (both).**
- Pilot data: VISION + 6 Synthbuster. There is no FLUX; the SD-family = SD2, SDXL.
- E4's laundered win-rate came from pairs laundered after the VAE, so it is NOT a ceiling for H-D1's deployable score.
- The label arm is the one each idea file names (fov512 for H-D1); orig is reported.

**H-D1 additions.**
1. **Matched-blur control (mandatory).** The same score with VAE(t) replaced by a Gaussian blur of t, with σ chosen on train reals to match the mean F2 shift of the VAE round-trip. H-D1 counts as diffusion-specific only if its laundered AUROC beats the blur version by ≥ 0.05 (CI lower bound > 0). Otherwise the label is "not diffusion-specific", whatever the AUROC.
2. **Texture baseline and validity gate** as in Amendment 6.

**H-D2 additions** (from the E4 critic).
1. Extra conditions: JPEG-only (q75), resize-only (0.5× and 0.75×), and `jr(75, 0.75)`, so each component can be read separately.
2. A "folded-grid energy" feature: the image folded on the 8×8 phase grid (mean over each phase position of the high-pass residual), giving one energy number. It is reported beside the idea file's spectrum statistics.
3. JPEG-history-matched control as in the idea file.

**D-008.** Guards apply to any new GPU features (|Δlogit| ≤ 1e-3) and to VAE outputs (sd-vae-ft-mse pixel guard, already passed).
