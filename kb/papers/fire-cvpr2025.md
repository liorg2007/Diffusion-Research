# FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error
- **id:** fire-cvpr2025
- **Authors / venue / year:** Beilin Chu, Xuan Xu, Xin Wang, Yufei Zhang, Weike You, Linna Zhou / CVPR 2025, pp. 12830-12839 / 2025
- **Links:** arXiv 2412.07140 (v3 Nov 2025) · https://openaccess.thecvf.com/content/CVPR2025/html/Chu_FIRE_Robust_Detection_of_Diffusion-Generated_Images_via_Frequency-Guided_Reconstruction_Error_CVPR_2025_paper.html · code https://github.com/Chuchad/FIRE
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Diffusion autoencoders reconstruct real photos worst in the mid-frequency band. FIRE reconstructs both the image and a version with the mid band removed, feeds the two error maps to a ResNet-50, and learns a refined frequency mask. The claim is better transfer to unseen diffusion models and robustness to perturbations.

## How it works
Image -> reconstruction using only the SD v1.5 VAE encoder/decoder (no denoising loop, no DDIM steps) -> |error| map. FFT -> shared encoder (4 conv 3x3 stride-2) with two decoders giving a mid-band mask and complement; predefined mid mask = ring of radii 40-120 px on 256x256 spectrum. A "pseudo-generated" image is made by removing the mid band; its reconstruction error is computed too. The two absolute error maps are concatenated and classified by ResNet-50. Loss weights 1/5, 1/5, 3/5 (mid-reconstruction alignment, mask refinement, CE). Adam lr 1e-4, bs 16, 100 epochs.
**Compute per image:** 2 VAE encode+decode passes (original + pseudo) + FFT + ResNet-50; no U-Net. Training time ~45 min/epoch on one A100-40G (README).

## Training data
DiffusionForensics: LSUN-Bedroom 40k real + 40k ADM train; ImageNet 40k + 40k ADM. Also self-collected: 10k LAION reals + 10k fakes (DALL-E 3, Kandinsky 3, Midjourney, SDXL, Segmind Vega), 1,000/class test.

## Reported results
- Table 1 (DiffusionForensics, AUC/ACC): FIRE avg 100.0/100.0 on LSUN-Bedroom and ImageNet; FakeInversion 100.0/99.6 and 99.9/99.7; DIRE 99.8/98.3 and 100.0/99.7; AEROBLADE 99.08/98.0 and 99.1/98.1. (In-distribution-like: near-saturated.)
- Table 2 (newer generators, self-collected; avg AUC/ACC): trained on LAION+DALL-E 3: FIRE 72.0/68.1 vs FakeInversion 68.8/63.5; trained on LAION+Kandinsky 3: 78.1/72.9 vs 70.5/65.6. i.e. unseen-generator performance is only ~68-73% ACC.
- Table 3 mask ablation: predefined mask #5 best (e.g., DALL-E 3 AUC 78.3, Kandinsky 3 74.5). Table 4: removing mask loss -> 68.3/63.1; removing mid-reconstruction loss -> 54.3/55.6; full 78.3/74.5.
- Perturbed: Fig. 7 (JPEG, crop+resize, blur, noise) numbers only in figure; text says FIRE beats others on most metrics but falls below FakeInversion under severe blur. Numeric values **[unverified]**.
- No GenImage results, no AP, no FPR-at-threshold reported.

## Known weaknesses / failure modes
- Severe blur hurts (loses mid/high frequency); predefined mask only ablated over a few fixed options; authors plan more robust frequency refinement.
- Headline in-distribution numbers (100/100) are saturated and not informative; the unseen-generator numbers (Table 2) are modest.
- Training on DiffusionForensics (ADM, LSUN/ImageNet) where DIRE already saturates; format-bias status of these training sets not discussed.

## Code + weights
https://github.com/Chuchad/FIRE - MIT (LICENSE file, GitHub license API). Pretrained checkpoints released via Google Drive links in README: "Imagenet with adm" and "Lsun_bedroom with adm" (Table 1 models); the repo ckpt/ dir only holds a placeholder. Pre-processed DiffusionForensics also on Google Drive. (README license badge links to an unrelated repo - ignore; the LICENSE file is MIT.)

## Verified?
Partially: numbers from fetch-tool summary of arXiv HTML v1; license from GitHub API; README read via raw file. Not checked against the CVPR camera-ready (v3 may differ).

## Relevance to this project
- Cheap (VAE-only) reconstruction + frequency mask; CPU-feasible at inference and weights are available under MIT.
- Unseen-generator accuracy is low (68-73% ACC in Table 2) and robustness evidence is graphical only; treat as a lead, not a solution.
