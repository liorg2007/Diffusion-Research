# TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks
- **id:** truefake-ijcnn2025
- **Authors / venue / year:** Dell'Anna, Montibeller, Boato (Univ. of Trento; first names not verified). IJCNN 2025 (per search result; unconfirmed on arXiv page).
- **Links:** arXiv 2504.20658 · https://arxiv.org/abs/2504.20658 · code+data https://github.com/MMLab-unitn/TrueFake-IJCNN25
- **Family:** robustness-attacks
- **Status:** partially-verified

## Summary (plain language)
A 600k-image dataset of real and AI images (SD 1.5-3, SDXL, FLUX.1, StyleGAN1-3), of which 180k were actually uploaded to Facebook, X and Telegram and re-downloaded. It measures how real social-network processing hurts existing detectors.

## How it works
Detectors trained only on non-shared data (FFHQ/FORLAB reals; StyleGAN2 and SDXL fakes; 70/15/15 split); shared images used for test only. Platform processing observed: Facebook downscales width >720 px, JPEG QF 61-92 (varies per image), most disruptive; X downscales >1200 px, no change for <=768x768, QF 87; Telegram caps height 800 px, QF 85. Augmentations for retrained models: random scale, 300x300 crop, color jitter, blur, JPEG.

## Training data
See above. Dataset: 100k reals (FFHQ, FORLAB), 200k diffusion fakes, 120k GAN fakes; shared subset 20k real + 25k DM + 15k GAN per figure caption (counts per platform: 180k shared total).

## Reported results
- Non-shared mean TNR/TPR (Table III): R50-E2P 0.98/0.85; R50-ND 1.00/0.79; MISLNet 0.98/0.61; NPR 0.99/0.60; CLIP-D 0.97/0.93; P2G 1.00/0.91.
- Shared (Table IV): NPR TPR on SDXL via Facebook 0.21 (-0.79); MISLNet TNR on FFHQ via Facebook 0.41 (-0.58) (real images flagged as fake); CLIP-D best but losses ~10% on several classes on Facebook; R50-ND most resilient CNN.
- Fine-tuning on shared images not tested here (earlier TrueFace work reportedly found it helps).

## Known weaknesses / failure modes
Only one capture of platform behavior (platform pipelines change); faces dominate real set; no phone-ISP reals.

## Code + weights
Repo license CC-BY-4.0 (GitHub API). Dataset license not verified separately.

## Verified?
Partially: arXiv HTML via fetch summary.

## Relevance to this project
- Real platform JPEG QFs (61-92, 85, 87) and resize caps to reproduce laundering cheaply.
- Social sharing raises FPR on reals for some detectors, not only lowers TPR.
