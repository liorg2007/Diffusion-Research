# Towards Universal Fake Image Detectors that Generalize Across Generative Models
- **id:** univfd-cvpr2023
- **Authors / venue / year:** Utkarsh Ojha*, Yuheng Li*, Yong Jae Lee (Univ. of Wisconsin-Madison) / CVPR 2023 / 2023
- **Links:** arXiv 2302.10174 · https://arxiv.org/abs/2302.10174 · https://github.com/Yuheng-Li/UniversalFakeDetect (repo README links to this; GitHub API reports org WisconsinAIVision/UniversalFakeDetect)
- **Family:** foundation-features
- **Status:** read-full

## Summary (plain language)
Instead of training a deep network to separate real from fake (which makes "real" a sink class and fails on new generator families), the authors freeze CLIP ViT-L/14 and train only a linear layer (or use nearest neighbour) on its last-layer image embedding. Trained on ProGAN images only, this detects unseen diffusion and autoregressive generators much better than a fully trained ResNet-50. It is the baseline almost every later CLIP-based detector compares against ("UnivFD" / "UFD").

## How it works
- Input -> CLIP ViT-L/14 visual encoder, **last layer** embedding as phi (Sec. 4) -> linear classifier (LC) or nearest neighbour (NN, k=1..9) to a real/fake feature bank. CLIP frozen; only the linear head is trained.
- Backbone: CLIP ViT-L/14 (OpenAI, 400M image-text pairs). Ablation of other backbones (CLIP ViT/RN50, ImageNet ViT) in Sec. 6 (ViT-L/14 best within CLIP).
- Preprocessing (from repo `validate.py`, read): `CenterCrop(224)` at test time, **no resize**. Training uses blur+JPEG augmentation (paper Sec. 5 / Table 2-3 note "ViT:CLIP (B+J 0.5)" is a fine-tuned baseline); the paper states the linear classifier is trained with jpeg+blur augmentation on ProGAN data (Fig. 10 discussion).

## Training data
ProGAN, 20 LSUN object classes, 720k images (360k real/360k fake) (Sec. 3.1, protocol of Wang et al. CNNDetection). Also repeated with LAION-real vs LDM-fake (Fig. 7).

## Reported results
Test sets: 19 generators; ~10k images per diffusion/AR domain in the paper (repo releases 1k per domain).
- Unseen generators, AP (Table 2, "Ours LC"): total mean AP 93.38 across all columns. Text (Sec. 6.1): LC reaches 99.31 mAP on GAN family and ~95.00 mAP on diffusion+autoregressive (LDM+Guided+Glide+DALL-E) vs 75.51 mAP for the best trained CNN (Blur+JPEG 0.5).
- Unseen generators, accuracy at threshold from ProGAN validation set (Table 3): LC total avg acc 81.38; NN k=9 84.25; trained ResNet-50 Blur+JPEG(0.1) 69.58, (0.5) 64.73; oracle-threshold (test-set calibrated) ResNet-50 76.26. Text: NN/LC reach ~82-84% avg acc on diffusion/AR vs 53-58% for trained CNNs.
- Per-generator weak spots in Table 3 LC: Deepfake 66.6, SITD 63.0, SAN 57.5, CRN 59.5, IMLE 72.0, Guided 70.03, LDM-200 with CFG 73.76, Glide 78-80, vs LDM 94.19/94.36, DALL-E 86.78.
- Perturbations (Fig. 10, JPEG and Gaussian blur curves, no table): "generally robust" for blur and JPEG; absolute AP still higher than CNN baseline on diffusion/AR; trained CNN more consistent across strengths. Exact numbers not in text -> **[unverified]** (figure only).
- Resize/social-media style perturbations: not evaluated in the paper.
- Threshold/calibration: Table 3 accuracy uses a threshold fixed on the ProGAN validation set; Table 5/6 split accuracy into real and fake. No FPR at fixed threshold on fresh real photos.

## Known weaknesses / failure modes
- From paper: accuracy (threshold-dependent) is far below AP on unseen families (81.38 acc vs 93.38 mAP, Tables 2-3) -> ranking is good but decision threshold does not transfer; authors acknowledge the need for threshold calibration (Table 3 note on Oracle).
- Not robust-by-design: frozen CLIP was not trained for blur/JPEG robustness (Fig. 10 discussion).
- Later papers:
  - Cozzolino et al. CVPRW 2024 (arXiv 2312.00195), Table 4/5: Ojha et al. mean AUC 84.1 on 18 generators, falls to 71.8 on cropped/resized/JPEG images; AUC 36.4 on one commercial-tool column and 66.2/97.5 on others (column mapping not confirmed).
  - B-Free (arXiv 2412.17671) via HTML fetch: Table 5 WildRF bAcc Facebook 48.8, Reddit 59.5, Twitter 56.0; Table 7 GenImage 65.5, FakeInversion 52.8, SynthWildX 52.5 (bAcc); Table 11 FakeInversion AUC 54.9 original -> 49.7 under social-network simulation (chance).
  - AIDE (ICLR 2025, arXiv 2406.19435) via HTML fetch: AIGCDetectBenchmark Table 3 UnivFD mean acc 78.43 (Midjourney 56.13, SD1.4 63.66, DALLE2 50.75); Chameleon Table 5 acc 57.22 (ProGAN-trained: fake 3.18 / real 97.83) and 55.62 (SD1.4-trained: fake 74.97 / real 41.09); JPEG QF=50 acc 68.68, blur sigma=4 acc 61.18 (Table 6).
  - Community Forensics (arXiv 2411.04125) via HTML fetch: Ojha et al. mAP 0.957 on its own test set but 0.592 on the Community Forensics eval set, 0.620 on Synthbuster, 0.797 on GenImage (Table 2); notes that freezing the backbone consistently hurts (Fig. 6a).
  - Zhou et al. (arXiv 2509.12995) Table 4: UnivFD 59.2 real acc / 29.8 fake acc on WebAIG-25.
  - Numbers from later papers were read via a fetch/summary tool of the arXiv HTML; spot-check before citing in a paper.

## Code + weights
https://github.com/Yuheng-Li/UniversalFakeDetect (org WisconsinAIVision). License: MIT (GitHub API spdx_id MIT; LICENSE file reads "MIT License ... (c) 2025 Wisconsin AI and Vision Lab"). Weights: `pretrained_weights/fc_weights.pth` in repo (linear head only); backbone is OpenAI CLIP ViT-L/14 downloaded separately. Publicly downloadable.

## Verified?
yes (paper numbers copied from arXiv 2302.10174 PDF text, Tables 2-3, Sec. 3.1, 6.1, Fig. 10 text; license and center-crop read from repo files). Later-paper numbers: partially (see above).

## Relevance to this project
- Establishes that frozen large-scale features beat trained CNNs across generator families, but the cross-family operating threshold is not preserved: AP is high, accuracy much lower.
- Last-layer CLIP is not good enough for in-the-wild data (near chance on Chameleon, WildRF, SynthWildX in later benchmarks); intermediate layers (RINE) and richer training data help.
- Center-crop 224 with no resize: a phone photo or downscaled image sees only a central crop or different content scale; resize behaviour is untested.
