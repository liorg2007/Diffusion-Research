# Time Step Generating: A Universal Synthesized Deepfake Image Detector (TSG)
- **id:** tsg-arxiv2024
- **Authors / venue / year:** Ziyue Zeng, Haoyuan Liu, Dingjie Peng, Luoxu Jin, Hiroshi Watanabe (Waseda University) / arXiv preprint (a TSG++ version is reported at MM Asia 2025 per search snippet **[unverified]**) / 2024
- **Links:** arXiv 2411.11016 (v2) · https://arxiv.org/abs/2411.11016 · code https://github.com/NuayHL/TimeStepGenerating
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Instead of running DIRE's full inversion+denoising, TSG feeds the image once into a frozen pretrained diffusion UNet at a fixed timestep and uses the predicted-noise map as the feature for a ResNet-50 classifier. About one UNet pass per image.

## How it works
Feature F = eps_theta(I, t), t = 0 (best) or 50, from the unconditional ImageNet-256 ADM UNet (Dhariwal and Nichol), not fine-tuned. ResNet-50 classifier trained on the feature maps (resize 256, crop 224).
- **Weights needed:** the ADM 256x256 unconditional UNet (hundreds of MB; size not checked **[unverified]**) + trained ResNet-50. Not VAE-only; exceeds our 0.2 GB budget.
- **Passes per image:** 1 UNet forward (vs DIRE 20 DDIM inversion + 20 denoise). Table 2: 26.3 s vs 277.1 s for 100 images on one RTX A6000 (DIRE batch 50, TSG batch 5).

## Training data
GenImage subsets (BigGAN, VQDM, SD1.5, ADM, Wukong; ~330k images each), one classifier per subset; Table 4 also a mixed ADM+SD1.5+BigGAN classifier.

## Reported results
- **In-distribution:** Table 1 average ACC TSG(t=0) 94.9, TSG(t=50) 89.2 vs LaRE2 75.6, DIRE 61.8 (column order inferred **[unverified]**). Mixed-training TSG 100.0 on diff-based, BigGAN, un-bias test sets (Table 4).
- **Unseen-generator results:** cross-generator accuracies are in Figure 4 only (diagonal ~100%, off-diagonal not in text). Classifiers trained on BigGAN generalize poorly to diffusion. FLUX, SD3, DALL-E 3, Midjourney (beyond GenImage MJ) not evaluated.
- **Laundering results:** JPEG only, Figure 6 (cross-validation "high accuracy", values not in text); on Glide, SD1.4, Midjourney subsets filtered to JPEG quality > 96 (Table 3). No resize/social tests.
- **FPR on real photos:** not reported (accuracy only).

## Known weaknesses / failure modes
Per-subset trained classifiers; timestep choice unprincipled; GenImage-only evaluation inherits GenImage JPEG/size bias (see genimage-neurips2023). Speed measured on 100 images only.

## Code + weights
- **License:** no code license stated in fetched text **[unverified]**; paper CC BY 4.0. ADM weights from the guided-diffusion release (license not checked).

## Verified?
Partially: arXiv HTML v2; numbers from Tables 1-4; figure values not extracted.

## Relevance to this project
- Shows a single UNet-forward noise-prediction feature beats 40-step reconstruction detectors in-distribution, but evidence is GenImage-only with likely format bias.
- Not feasible under our weight budget unless the UNet is already available.
