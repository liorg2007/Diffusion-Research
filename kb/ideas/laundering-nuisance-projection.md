# Laundering-nuisance projection: remove real-photo laundering directions from frozen features before the probe
- **id:** laundering-nuisance-projection
- **Gap addressed:** G2 (laundering robustness is only bought with augmentation), G4 (frozen modern SSL backbones)
- **Scores (1-5):** novelty 3 · expected gain on unseen/in-the-wild 3 · Mac feasibility 5 · **product 45** (rank 1)

- **Hypothesis:** Projecting out the top-k directions of the covariance of feature *differences* f(L(x)) - f(x), where x are REAL photos and L(x) is the same photo after real platform sharing (VISION WhatsApp/Facebook copies) or simulated JPEG+resize, makes a frozen-DINOv3 multi-layer logistic probe lose less AUROC under laundering than the same probe trained with laundering augmentation alone: on the held-out generator (FLUX) after JPEG q75 + 0.5x resize, AUROC improves by >= +0.03 with <= 0.01 clean-AUROC loss.

- **Rationale:**
  - Augmentation teaches a probe to *tolerate* laundering only to the extent the head can; a linear head on a 10k-dim frozen feature has many directions that move with JPEG/resize and with nothing else. Estimating those directions in closed form from real-only paired data (no fakes, so no generator leakage) and deleting them is a nuisance-attribute projection (the idea used in speaker verification and LEACE-style concept erasure), not something the KB papers do.
  - VISION ships the same photo in native, WhatsApp, Facebook-high and Facebook-low versions (kb/datasets/vision-phone-photos.md) - real platform pairs for free, licence CC BY-SA 4.0.
  - The same projection should also shrink *threshold drift* (FPR on shared reals vs native reals), the failure mode TrueFake documents (MISLNet TNR on FFHQ via Facebook 0.41, truefake-ijcnn2025 Table IV).
  - Falsification risk is real and informative: laundering destroys low-level fake traces (corvi-icassp2023 Table 1; bfree-cvpr2025 Table 11), so the laundering subspace may overlap the fake-signal subspace; then the projection costs clean AUROC and the hypothesis fails.

- **Closest prior work:**
  - KB: rine-eccv2024 (multi-layer CLIP CLS, blur/JPEG aug; combination of perturbations "considerably reduces performance", Table 5 text), clipdet-cvprw2024 ("+" reference set with resized/compressed images: post-processed AUC 78.1 -> 85.2, Table 5), effort-icml2025 (SVD split of CLIP weights - orthogonal subspace, but of *weights* and trained on fakes, not laundering-pair features), spai-cvpr2025 (JPEG aug ablation 89.1 vs 91.0 AUC, Table 3).
  - Web check: DRIFT (arXiv 2606.06918, https://arxiv.org/abs/2606.06918, ECCV 2026 submission) learns robust/fragile subspaces with projection heads on a frozen VFM under one-class supervision and "physically plausible imaging transformations" - the closest idea; abstract only read, transformations and backbone not named. Effort (https://proceedings.mlr.press/v267/yan25b.html) is the SVD-subspace analogue on weights. No paper found that estimates the nuisance subspace from *real platform-shared pairs* and applies a closed-form projection before a supervised probe.
  - Difference: closed-form, no training of the backbone or heads beyond logistic regression, real-only paired estimation with actual platform pipelines; evaluated on a held-out laundering type (screenshot simulation) never used in estimation.

- **Evidence for / against:**
  - For: laundering mostly breaks thresholds and directions, not total separability (corvi-icassp2023 Table 1: Grag2021 avg AUC 90.8 but Acc 61.5 after resize+JPEG); robust-to-laundering CLIP-family methods are the most stable in B-Free's social simulation (c2pclip-aaai2025: FakeInversion AUC 74.1 -> 71.9, bfree-cvpr2025 Table 11), suggesting a mostly-semantic subspace survives laundering.
  - Against: in B-Free Table 11, most detectors fall to ~chance under a mild resize 0.7-1 + JPEG 70-100 (UnivFD 54.9 -> 49.7, FatFormer 68.2 -> 48.7); if the fake signal in frozen features lives in the same directions as laundering, removing them removes the signal. DRIFT reports a learned version works (abstract claim only).

- **Test plan (Mac-feasible):**
  - Data: shared eval set (docs/02-hypotheses.md, "Shared infrastructure"). Projection pairs (reals only, never eval devices): VISION D_train devices native <-> WA/FBH/FBL (~200 photos x 3 = ~600 pairs) + RAISE train split <-> synthetic laundering (JPEG q in {60,75,90} x resize {0.5,0.75,1.0}, ~1,200 pairs). Held out from estimation: screenshot-sim and Mavali "Medium" (75% + JPEG Q50, mavali-aigi-robustness-arxiv2025).
  - Model: cached frozen DINOv3 ViT-L/16 (fallback DINOv2 ViT-L/14-reg) CLS + mean-patch from 6 blocks; standardize with real-feature statistics; eigendecompose cov(Delta); P = I - U_k U_k^T; logistic regression on P f. k pre-registered from a real-only criterion (smallest k capturing 80% of Delta energy); also report k in {16, 64, 256}.
  - Baseline B0: identical probe, trained on clean + laundered copies (q75+0.5x, q90, 0.75x) without projection. Also report UnivFD and CommFor ViT-S (released weights) for context.
  - Metrics: AUROC/AP/TPR@1%FPR on held-out FLUX and LOGO mean over the 10 other generators, conditions clean / q75+0.5x / screenshot-sim / Medium; threshold drift = |FPR(WA or FBL reals) - FPR(native reals)| at a threshold fixed on native reals of eval devices.
  - **Pass:** FLUX and LOGO-mean AUROC at q75+0.5x >= B0 + 0.03, clean AUROC >= B0 - 0.01, screenshot-sim AUROC >= B0, threshold drift reduced by >= 50%. **Fail:** gain < +0.01 at q75+0.5x, or clean loss > 0.02.
  - Cost: projection and probe are seconds on CPU once features are cached; feature cache ~1-2 h on MPS (estimate from measured ResNet-50 141 img/s and ~15x the FLOPs for ViT-L; confirm with a 50-image pilot).

- **Status:** proposed
