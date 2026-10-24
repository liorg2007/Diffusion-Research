# LATTE: Latent Trajectory Embedding for Diffusion-Generated Image Detection
- **id:** latte-arxiv2025
- **Authors / venue / year:** Ana Vasilcoiu, Ivona Najdenkoska, Zeno Geradts, Marcel Worring (U. Amsterdam, Netherlands Forensic Institute) / arXiv preprint (no venue) / 2025
- **Links:** arXiv 2507.03054 (v1) · https://arxiv.org/abs/2507.03054 · code https://github.com/AnaMVasilcoiu/LATTE-Diffusion-Detector
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Instead of one reconstruction error, LATTE records how the latent changes along five points of the noising/denoising path of Stable Diffusion 2.1, refines these latents with transformer decoders that attend to ConvNeXt image features, and classifies. It improves on LaRE^2.

## How it works
Timesteps [981, 741, 521, 261, 1]; at each, one forward-noising + one single-step denoising correction with SD 2.1 (not a full reverse pass). ConvNeXt-base (timm convnext_base.fb_in22k, fine-tuned; frozen is worse, Table 9: 81.9 vs 90.9), L cross-attention decoders, avg pool + linear head. BCE, AdamW 1e-4, batch 32, 224x224; about 2 h/epoch on an H100.
- **Weights needed:** SD 2.1 (VAE + UNet) + ConvNeXt; trained head. Not VAE-only.
- **Passes per image:** 5 UNet evaluations (+ VAE encode) + ConvNeXt forward.

## Training data
GenImage subsets and DiffusionForensics subsets (Tables 1-2); train/test per benchmark.

## Reported results
- **Unseen-generator results:** GenImage average ACC 82.5% (Table 1; text conclusion says 82.2) vs LaRE^2 79.1; BigGAN 87.8 vs 74.0. DiffusionForensics Bedroom 85.7 vs 69.5 (Table 2a), ImageNet 93.9 vs 89.9 (2b), CelebA 91.1 vs 90.0 (2c). Ablations: visual-only 74.9, latent-only 53.2, full 90.9 (Table 6; the protocol behind 90.9 differs from Table 1 averages, not reconciled **[unverified]**). FLUX, SD3, current Midjourney not evaluated.
- **Laundering results:** JPEG, center-crop+resize, blur, noise in Figure 5 only (no numbers); authors state performance drops under strong post-processing but beats LaRE^2.
- **FPR on real photos:** not reported.

## Known weaknesses / failure modes
Strong post-processing, stylized/over-processed real photos give false positives (Fig. 6, qualitative); simple high-plausibility fakes missed; internal number inconsistencies noted above.

## Code + weights
- **License:** code license not stated; paper CC BY-NC-ND 4.0; depends on SD 2.1 (CreativeML Open RAIL-M), ConvNeXt via timm (Apache 2.0), GenImage (CC BY-NC-SA 4.0).

## Verified?
Partially: arXiv HTML v1; numbers from Tables 1-3, 5, 6, 9.

## Relevance to this project
- Shows latent-trajectory features add little on their own (latent-only 53.2 vs visual-only 74.9, Table 6): the diffusion signal needs the image branch. Cautionary for "diffusion signal alone" claims. Needs full SD 2.1.
