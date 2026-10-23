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
