# Where Did I Come From? Origin Attribution of AI-Generated Images (arXiv title: Alteration-free and Model-agnostic Origin Attribution of Generated Images)
- **id:** ronan-neurips2023
- **Authors / venue / year:** Zhenting Wang, Chen Chen, Yi Zeng, Lingjuan Lyu, Shiqing Ma / NeurIPS 2023 (per search results; arXiv page lists preprint only) / 2023
- **Links:** arXiv 2305.18439 · code https://github.com/1920309095/ronan
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
The "model-agnostic" origin attribution paper (RONAN). For any generator you hold (GAN, VAE, diffusion), optimise the generator's INPUT until its output matches the image; images made by that model reach a lower final error than others. A calibration by a reference model removes the effect of image complexity. It is the ancestor of LatentTracer.

## How it works
Reverse-engineer input i minimising MSE(M(i), x) with Adam; divide by the loss of a reference model (Consistency Model, ImageNet); Grubbs-style test with N=100 own images, alpha=0.05.

## Training data
None. Models tested: DCGAN, VAE, StyleGAN2-ADA, StyleGAN-XL, Consistency Model, ControlGAN, StackGAN-v2, Stable Diffusion 2.0.

## Reported results
- Intro: average TPR 95.70% at FPR ~5.00%.
- Belongings vs unseen data (Table 2): 94.0-98.0%; different architectures (Table 3) avg 95.3%; same architecture different data (Table 4) avg 93.0%.
- Stable Diffusion 2.0 case study (Sec. 5.3): 87.5% (18 TP, 3 FP, 2 FN, 17 TN; only 20+20 images). Calibration ablation (Table 6): 75.0% -> 87.5%.
- AEDR (Tables 1-2) and LatentTracer (Table 2) find RONAN near chance (~50%) on latent diffusion models.
- Perturbations: JPEG/resize NOT evaluated; only an Instagram "_1977" filter adaptive attack on DCGAN, 90.5% (Table 8).
- Runtime (appendix): 55.16 s (StyleGAN2-ADA), 152.83 s (Consistency Model); per-image unit unclear.

## Known weaknesses / failure modes
Slow, white-box, small SD experiment, no compression robustness, fails on LDMs in later comparisons (above).

## Code + weights
GitHub repo above; licence not read **[unverified]**.

## Verified?
Partially: fetch-tool summary of arXiv HTML; NeurIPS venue from search result titles.

## Relevance to this project
- Historical baseline only; superseded by LatentTracer/AEDR for latent diffusion. Shows the "reconstruction loss as membership test + complexity calibration" idea that AEDR reuses.
