# HFI: A unified framework for training-free detection and implicit watermarking of latent diffusion model generated images
- **id:** hfi-arxiv2024
- **Authors / venue / year:** Sungik Choi, Sungwoo Park, Jaehoon Lee, Seunghyun Kim, Stanley Jungkyu Choi, Moontae Lee (LG AI Research) / arXiv preprint (no venue stated on fetched page) / 2024
- **Links:** arXiv 2412.20704 (v1) · https://arxiv.org/abs/2412.20704 · code: none found in fetched text **[unverified]**
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
The latent-diffusion autoencoder acts like a down-sample/up-sample filter that destroys (aliases) high-frequency detail. HFI measures how much the autoencoder reconstruction error changes after the image is slightly blurred; images that already came out of this autoencoder change differently from real photos. It is an AEROBLADE-style score with a "high-frequency influence" correction, and it needs no training.

## How it works
Score = inner product between the gradient of the reconstruction distance d(x, AE(x)) and the high-pass residual x - F(x); approximated (first-order Taylor, Eq. 3) by d(x, AE(x)) - d(F(x), AE(F(x))). F = 3x3 Gaussian blur, sigma 0.8 (box blur similar; DCT nulling worse, Tables 7, 12). Distance = LPIPS or the VGG layer-2 LPIPS variant of AEROBLADE (LPIPS2 best on average, 0.977, Table 8). Several AEs (SDv1.4, SDv2-base, Kandinsky 2.1, MiniSD) can be ensembled by taking the minimum score (Eq. 4). A Gaussian pre-filtered variant (B-HFI) is more robust.
- **Weights needed:** VAE (autoencoder) weights only (SD1.4 / SD2-base / Kandinsky 2.1 / MiniSD AEs) plus LPIPS VGG; no UNet. Exact file sizes not checked **[unverified]**.
- **Passes per image:** 2 AE encode+decode passes per AE (x and F(x)) + LPIPS; ensemble multiplies by number of AEs. Reported 0.255 s/sample on A100 (vs LatentTracer 14.65 s).

## Training data
None (training-free). Real: ImageNet (GenImage), RAISE-1k (Synthbuster), others per dataset below.

## Reported results
- **Unseen-generator results:** mean AUPR GenImage (Table 3): HFI_LPIPS2 0.977, HFI_LPIPS 0.966, AEROBLADE_LPIPS2 0.935, NPR 0.833, RIGID 0.812. Synthbuster (Table 4): HFI_LPIPS2 0.825, AEROBLADE 0.796, RIGID 0.601; weak on DALL-E 2 (0.425) and DALL-E 3 (0.648). DiffusionFace (Table 5) mean AUPR 0.751 vs AEROBLADE 0.710. FLUX and SD3 not evaluated. Note: GenImage numbers use the SD-family AEs which the generators partly share.
- **Laundering results:** only JPEG and cropping, on ImageNet vs SDv1.4 with the SDv1.4 AE (Figure 5, no numbers): holds up under small corruptions (esp. crop), degrades under severe ones; B-HFI improves. Resize and blur NOT tested.
- **FPR on real photos:** not reported (only AUROC/AUPR).
- Also reports near-perfect attribution of SD vs other LDMs (Table 9) and use as implicit watermark.

## Known weaknesses / failure modes
- Needs an AE related to the generator; BigGAN/ADM (no AE) and proprietary models (Firefly, DALL-E 3) weak. Simple-background real images can still be false positives (Fig. 4). Severe corruption degrades it. Later work (Fragile Reconstruction, arXiv 2604.12781) shows reconstruction-based detectors break under small adversarial perturbations (tested on AEROBLADE, not HFI).

## Code + weights
- **License:** paper CC BY 4.0 (arXiv page). No code repo or code license found in the fetched text **[unverified]**; AE weights carry their own (SD) licenses, not checked.

## Verified?
Partially: fetched arXiv HTML v1; numbers copied from Tables 3-5, 9 and text. Figure 5 values not available.

## Relevance to this project
- Direct, cheap upgrade of AEROBLADE (VAE-only, ~2x cost); shows the high-frequency/aliasing part of the VAE residual is the useful signal, which matches the D-014 VAE-trace direction.
- Laundering robustness is the open question (resize untested); would need our own `jpeg75_resize05` evaluation.
