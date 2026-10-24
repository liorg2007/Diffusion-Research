# Training-free Detection of AI-generated images via Cropping Robustness (WaRPAD)
- **id:** warpad-arxiv2025
- **Authors / venue / year:** Sungik Choi, Hankook Lee, Moontae Lee / arXiv preprint (NeurIPS checklist present, acceptance not stated) / 2025
- **Links:** arXiv 2511.14030 (v1) · https://arxiv.org/abs/2511.14030 · code: not released ("after the paper gets accepted")
- **Family:** foundation-features
- **Status:** partially-verified

## Summary (plain language)
Real images keep their DINOv2 features stable under small high-frequency (Haar wavelet) perturbations more or less than generated ones; WaRPAD scores that sensitivity per patch and averages. No diffusion model is used, so it is not diffusion-specific, but it is a strong training-free baseline from the same group as HFI.

## How it works
Base score: cosine similarity between DINOv2 ViT-L/14 features of an image and a copy perturbed along Haar-wavelet high-frequency directions (alpha 0.1, 2-level Haar). Image rescaled (896 for GenImage/Deepfake-LSUN, 1344 for Synthbuster), split into 224x224 patches, patch scores averaged (mean/median best, Table 6). Backbone ablation (Table 8, Synthbuster): DINOv2 0.834, CLIP 0.802, SwaV 0.743, DINO 0.707.
- **Weights needed:** DINOv2 ViT-L/14 only (about 1.2 GB fp32, size from general knowledge **[unverified]**); NO VAE/UNet. Exceeds our remaining ~0.2 GB budget.
- **Passes per image:** 2 ViT-L forward passes per patch x 16 patches (896) or 36 (1344) (inference from the formula; paper gives no timing).

## Training data
None.

## Reported results
- **Unseen-generator results (AUROC):** Synthbuster mean 0.834 (Table 2; Firefly 0.927, GLIDE 0.999, SDXL 0.830, SDv2 0.775, DALL-E 3 0.422 i.e. below chance, DALL-E 2 0.930, Midjourney 0.702) vs AEROBLADE 0.703, RIGID 0.587. GenImage mean 0.946 (Table 3; Midjourney 0.810) vs AEROBLADE 0.928, trained AIDE 0.969. Deepfake-LSUN-Bedroom mean 0.934 (Table 4) vs Manifold Bias 0.920. FLUX and SD3 not evaluated.
- **Laundering results:** JPEG, center-crop+resize, Gaussian noise on GenImage, Figure 6 only (no numbers); text says it stays competitive while AEROBLADE degrades. Resize-only and blur not reported in the read part.
- **FPR on real photos:** not reported (AUROC only; real sets RAISE-1k, ImageNet, LSUN, WikiArt).

## Known weaknesses / failure modes
Extra compute from patches; backbone dependence; future generators with better high-frequency fidelity may defeat it; DALL-E 3 below chance (Table 2). Last 11.8k chars of the paper (appendix) not read.

## Code + weights
- **License:** paper CC BY 4.0; no code released at fetch.

## Verified?
Partially: arXiv HTML v1, first 100k of 111.8k chars; numbers from Tables 2-4, 6, 8.

## Relevance to this project
- A non-diffusion training-free comparison point; heavy for our hardware/budget (DINOv2-L, up to 72 forward passes).
- Evidence that DALL-E 3 is hard for training-free detectors (HFI, WaRPAD both weak).
