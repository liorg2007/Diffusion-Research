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
