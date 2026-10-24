# Explainable Synthetic Image Detection through Diffusion Timestep Ensembling (ESIDE)
- **id:** eside-arxiv2025
- **Authors / venue / year:** Yixin Wu, Feiran Zhang, Tianyuan Shi, Ruicheng Yin, Zhenghua Wang, Zhenliang Gan, Xiaohua Wang, Changze Lv, Xiaoqing Zheng, Xuanjing Huang (Fudan) / arXiv preprint (no venue stated) / 2025
- **Links:** arXiv 2503.06201 (v2) · https://arxiv.org/abs/2503.06201 · code https://github.com/Shadowlized/ESIDE
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Runs one DDIM inversion (noising) pass with a pretrained diffusion model, keeps intermediate-timestep latents, extracts CLIP features at each and ensembles per-timestep MLP classifiers (AdaBoost-style). An optional GPT-4o module writes explanations.

## How it works
T = 24 DDIM timesteps; classifiers at stride-3 timesteps (9); CLIP ViT-L/14 features into a 5-layer MLP (1024-512-256-128); AdamW lr 1e-4. Diffusion checkpoints only described as pretrained 256 and 512 models following prior work (names not given).
- **Weights needed:** pretrained diffusion UNet (unspecified) + CLIP ViT-L/14 + trained MLPs. Not VAE-only.
- **Passes per image:** one DDIM inversion trajectory (up to 24 UNet steps by my reading; "single inversion pass" per paper) + ~9 CLIP forwards. Authors admit inversion is time-consuming.

## Training data
GenImage subsets, one model per subset (Table 1); cross-generator test trained on SD1.4 (Table 2).

## Reported results
- **In-distribution:** Table 1 average hard/original ACC 95.89 / 98.91 (Midjourney 92.38/98.42, SD1.4 96.65/99.17, Wukong 91.50/97.25).
- **Unseen-generator results:** trained on SD1.4 -> average 75.95 / 64.97 across other subsets (Table 2). FLUX.1-dev (12,000 images, Appendix A.2, Table 9): ESIDE 94.26 / 99.58 vs DRCT 84.46 / 98.17, NPR 92.23 / 98.67 (training set for that test not confirmed **[unverified]**). SD3, DALL-E 3 not evaluated.
- **Laundering results:** Table 4 covers Gaussian blur (sigma 0.5-2.5), rotation (+-45 deg), brightness (0.3-1.8): average ACC ESIDE 80.63, FreqNet 78.31, DRCT 77.48, NPR 61.13. JPEG and resize NOT tested.
- **FPR on real photos:** not reported.

## Known weaknesses / failure modes
Slow inversion; multiple classifiers; explanation quality depends on GPT-4o; no JPEG/resize test.

## Code + weights
- **License:** no code license stated in fetched text **[unverified]**; paper uses the arXiv non-exclusive license.

## Verified?
Partially: arXiv HTML v2; numbers from Tables 1, 2, 4, 9.

## Relevance to this project
- One of the few diffusion-feature detectors with a FLUX test (Table 9), but without JPEG/resize evidence and with a full UNet; low priority for our budget.
