# Worst-environment conformal threshold for a guaranteed FPR on unseen phones and platforms
- **id:** worst-env-conformal-fpr
- **Gap addressed:** G1 (thresholds do not transfer; FPR on fresh real photos is almost never reported), G3 (phone photos)
- **Scores (1-5):** novelty 2 · expected gain on unseen/in-the-wild 4 (directly targets the FPR <= 1% criterion) · Mac feasibility 5 · **product 40** (rank 2)

- **Hypothesis:** A threshold set as the maximum over calibration device-groups of each group's empirical 99th-percentile real score (Mondrian / worst-group split-conformal on real photos only, with each group's photos in native + WhatsApp + Facebook versions) keeps the pooled false-positive rate on held-out phone devices <= 1.5% (95% Clopper-Pearson upper bound <= 2.0%) and on a held-out platform transform <= 3%, while the standard thresholds (0.5 / validation-EER / pooled conformal) exceed 2% on at least one of these splits - at a TPR cost on the held-out generator of no more than 30% relative to the pooled-conformal threshold.

- **Rationale:**
  - The KB shows the ranking of real vs fake survives shift much better than the threshold: Corvi's Grag2021 keeps AUC 90.8 but accuracy 61.5 after resize+JPEG, and Platt calibration with 2 real + 2 fake per generator lifts fused accuracy 64.5 -> 87.8 (corvi-icassp2023 Tables 1-2); UnivFD 93.38 mAP vs 81.38 accuracy at a ProGAN-validated threshold (univfd-cvpr2023 Tables 2-3); on Chameleon most detectors get 95-100% real accuracy and 0-27% fake accuracy (aide-chameleon-iclr2025 Table 5); on real phone photos FPR goes from 0.2% to 39.1% for UnivFD depending only on which synthetic set the threshold came from (laion-mobile-arxiv2026).
  - For a deployment that must promise FPR <= 1%, the only side that can be calibrated without knowing future generators is the REAL side. Split-conformal gives a finite-sample guarantee only under exchangeability; phones and platforms break it. Taking the worst group's quantile is a cheap hedge whose cost (TPR loss) and benefit (FPR on held-out groups) are directly measurable with VISION's 35 devices x 4 versions.

- **Closest prior work:**
  - KB: corvi-icassp2023 (Platt with labeled fakes), clipdet-cvprw2024 (LLR output, fixed 0.5 threshold), bfree-cvpr2025 (reports ECE/NLL at fixed 0.5 threshold, on fixed test sets), manifold-bias-iclr2025 (threshold from 1,000 real images; FPR at that threshold not extracted), laion-mobile-arxiv2026 (measures FPR but does not propose a calibration).
  - Web check: (1) "Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated" (arXiv 2602.01973, AAAI 2026; https://arxiv.org/abs/2602.01973v1) fits a scalar logit correction on a target-distribution validation set; abstract targets fakes being missed and does not mention FPR control. (2) Conformal FPR bounding exists for machine-generated TEXT (arXiv 2505.05084, https://arxiv.org/pdf/2505.05084), which also reports that plain conformal costs detection power. (3) A non-peer-reviewed HF model card (https://huggingface.co/GPTchatly/slope-scope) calibrates fixed "budget" cutoffs on held-out VISION devices: its held-out VISION phones are 98-99% decided with 0% wrong, but on a private real set 16.9% of reals are decided wrong - a concrete example of device-held-out calibration NOT transferring to another real distribution. No image-detection paper found that evaluates group-conditional/worst-group conformal thresholds with leave-device-out and leave-platform-out splits.
  - Difference: real-only, label-free with respect to fakes, explicit worst-group construction, and evaluated on held-out devices, held-out platform transforms (incl. screenshot simulation), and a fresh modern-phone set (neural-isp-fpr-audit).

- **Evidence for / against:**
  - For: real distributions "remain relatively stable" relative to fake-side shift (secondary summary of 2602.01973; not verified in the PDF); VISION gives a clean grouping structure.
  - Against: the slope-scope card result above shows a large real-side shift beyond VISION; LAION-Mobile shows threshold choice dominates FPR; with only ~80-100 calibration images per device-group, the 1% quantile is noisy and the max over 25 groups may be dominated by one outlier device (TPR collapse). The pass criterion includes the TPR cost to catch this.

- **Test plan (Mac-feasible):**
  - Data: VISION eval devices D_eval (25 devices, 20 native photos each + WA/FBH/FBL copies = ~2,000 real images) + screenshot-sim and q75+0.5x versions generated locally; fakes = shared eval set (held-out FLUX for TPR).
  - Scores: B0 probe logit (frozen DINOv3 multi-layer + logistic regression, see laundering-nuisance-projection); repeat with UnivFD and CommFor ViT-S to show the method is detector-agnostic.
  - Thresholds: M0 = 0.5 / default; M1 = EER on the probe's generator validation split; M2 = pooled split-conformal at alpha = 1%; M3 (proposed) = max over calibration device-groups of per-group (1-alpha) quantile; M4 = pooled conformal at alpha/2 (simple alternative).
  - Splits: leave-one-device-out over 25 devices (calibrate on 24, test on 1, pool test decisions over folds -> ~2,000 test reals, 95% CI half-width ~ +/-0.5% at 1%); leave-one-platform-out (calibrate without WA, test on WA; same for FBL and screenshot-sim).
  - **Pass:** M3 pooled leave-device-out FPR <= 1.5% with CP upper bound <= 2.0%; leave-platform-out FPR <= 3% on every held-out platform; FLUX TPR at M3 >= 0.7 x TPR at M2 (clean and q75+0.5x). At least one of M0/M1/M2 exceeds 2% on some split (otherwise the problem is not real on this data and the idea is moot). **Fail:** M3 leave-device-out FPR > 2%, or TPR ratio < 0.5.
  - Cost: seconds on CPU once scores exist.

- **Status:** proposed
- **Official result:** OFFICIAL NOT_SUPPORTED (2026-10-25; experiments/007-worst-env-conformal/official/, critique.md): the worst-platform threshold does not beat pooled conformal for A0/A1/A2/B0/CommFor/RINE/B-Free (ClipDet SUPPORTED). B0 calibrated FPR 1.28% pooled, 2.51% worst platform. The tail is ONE phone (Microsoft Lumia 640), not a platform. alpha=0.5% is a built-in tie (disclosed).
- **Status (final):** official-not-supported
