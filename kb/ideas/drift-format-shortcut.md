# Are perturbation-drift ("robustness asymmetry") detectors partly measuring compression history?
- **id:** drift-format-shortcut
- **Gap addressed:** G8 (perturbation-sensitivity detectors never tested with compression-history-matched classes), G5 (dataset format bias)
- **Scores (1-5):** novelty 3 · expected gain on unseen/in-the-wild 2 (diagnostic; prevents adopting a shortcut) · Mac feasibility 5 · **product 30** (rank 4)

- **Hypothesis:** A training-free RIGID-style score (1 - cosine similarity between frozen DINOv2/DINOv3 CLS features of x and x + Gaussian noise) loses >= 0.10 AUROC when real and fake images share the same compression history (both JPEG q90, or both laundered q75+0.5x) compared with the format-mismatched setting (camera-JPEG reals vs never-compressed PNG fakes); the frozen supervised probe B0 loses < 0.05 under the same change.

- **Rationale:**
  - Training-free "real images are more stable under perturbation" detectors are attractive for unseen generators (no generator samples needed). But a never-compressed PNG and a camera JPEG differ in high-frequency content in a way that by itself could change feature stability under noise. The KB documents exactly this kind of shortcut for trained detectors: a GenImage-trained ResNet-50 calls 1024x1024 FFHQ "natural" 80.45% as PNG but 99.95% after JPEG80 (fakeorjpeg-arxiv2024 Table 2); DIRE's results were driven by JPEG reals vs PNG fakes (aeroblade-cvpr2024 Table 3, dire-iccv2023).
  - Most public evaluations pair JPEG reals (COCO, LAION, ImageNet) with PNG fakes (kb/datasets/communityforensics-eval.md "format-bias risk is real"; genimage-neurips2023). Synthbuster + RAISE PNG lets us build all three settings from the same images.
  - LAION-Mobile already shows the DINOv3 variant of RIGID at only AUC 0.624 on NTIRE 2026 content (laion-mobile-arxiv2026 Table 4) - consistent with, but not proof of, a fragile signal.

- **Closest prior work:**
  - KB: laion-mobile-arxiv2026 (RIGID, RIGID-DINOv3 evaluated), fakeorjpeg-arxiv2024, aeroblade-cvpr2024 (format-bias analysis of DIRE).
  - Web check: RIGID (He, Chen, Ho, arXiv 2405.20112, https://arxiv.org/abs/2405.20112) - training-free, noise-perturbation similarity in VFM space, claims robustness to corruptions; RA-Det (arXiv 2603.01544, https://arxiv.org/abs/2603.01544) - "robustness asymmetry", claims +7.81% average over 10+ detectors on 14 generators; DRIFT (arXiv 2606.06918) learns invariance subspaces. None of the abstracts read mentions controlling the compression history of real vs fake; I did not read their full experimental sections, so this is a gap in what I could verify, not proof they ignored it.

- **Evidence for / against:**
  - For: format shortcuts are documented for trained detectors (above); RIGID-DINOv3 is weak on modern content in LAION-Mobile.
  - Against: RIGID claims robustness to corruptions (abstract); a perturbation-stability score uses the image's own features as reference, which may cancel global compression differences.

- **Test plan (Mac-feasible):**
  - Data: 300 RAISE PNG reals (B-Free extended Synthbuster) + 300 VISION native JPEGs + 100 per generator from Synthbuster (9) + FLUX + SD3.5.
  - Settings: (i) mismatched = VISION JPEG reals vs PNG fakes; (ii) matched-PNG = RAISE PNG vs PNG fakes; (iii) matched-JPEG = RAISE and fakes both JPEG q90 (single compression); (iv) both q75 + 0.5x resize.
  - Score: RIGID-style drift with DINOv2 ViT-L/14 and DINOv3 ViT-L/16 CLS; noise sigma pre-registered grid {0.01, 0.02, 0.05} (pixel range [0,1]), report all; 4 noise draws per image. Reference: B0 probe (cached features, no extra cost).
  - Metric: AUROC per setting, per generator; bootstrap 95% CI.
  - **Supported if:** AUROC(i) - AUROC(iii) >= 0.10 for the best sigma, while B0's gap < 0.05. **Refuted if:** gap < 0.03 for all sigma (then drift scores are a legitimate training-free component for fusion).
  - Cost: (1 + 4) ViT-L forwards per image x ~2,000 images x 4 settings = ~40k forwards, ~1-1.5 h on MPS (estimate; pilot first).

- **Status:** proposed
