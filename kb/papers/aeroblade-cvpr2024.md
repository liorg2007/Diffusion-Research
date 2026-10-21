# AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error
- **id:** aeroblade-cvpr2024
- **Authors / venue / year:** Jonas Ricker, Denis Lukovnikov, Asja Fischer / CVPR 2024 (per task brief; fetched arXiv page lists no venue **[unverified]**) / 2024
- **Links:** arXiv 2401.17879 (v2) · code https://github.com/jonasricker/aeroblade
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Latent diffusion models (Stable Diffusion, Kandinsky, Midjourney) decode images with a VAE decoder, so their outputs lie close to what that autoencoder can reproduce. Encode+decode a test image with the AE and measure the LPIPS distance: small distance suggests generated. No training and no diffusion-sampling are needed.

## How it works
Image -> AE encode/decode (one pass, no denoising) -> LPIPS distance (best: 2nd LPIPS layer, "LPIPS_2") -> threshold. With several AEs (SD1, SD2, Kandinsky 2.1) take the minimum distance (Delta-Min). Fully training-free; threshold chosen by the user. Deeper DDIM-based reconstructions did not help (paper).
**Compute per image:** 1 AE encode + decode per AE (x3 for SD1/SD2/KD2.1) + LPIPS. No timing reported in the paper. Independent measurement: Brokman et al. (ICLR 2025) report ~5.4 s/sample on an A100 for the official code (App. G.3; code was not batched).

## Training data
None (training-free). Evaluation data: 1,000 real LAION-5B (aesthetic score >= 6.5) center-cropped 512x512; 1,000 generated each for SD1.1, SD1.5, SD2.1, KD2.1 (prompts via CLIP Interrogator), Midjourney v4, v5, v5.1 (public Discord dataset, 1024x1024).

## Reported results
- In-distribution-ish (generators whose AE is known): AP (Table 1, LPIPS_2 Delta-Min): SD1.1 0.979, SD1.5 0.978, SD2.1 0.994, KD2.1 0.999, MJ4 0.999, MJ5 0.997, MJ5.1 0.996 (mean ~0.992). TPR@5%FPR (Table 3): 0.981, 0.963, 0.995, 0.999, 0.997, 0.993, 0.989.
- Baselines (Table 3 AP): Corvi et al. (trained) ~1.000; Ojha et al. 0.682-0.895; Gragnaniello 0.526-0.715; SeDID_Stat 0.376-0.783; DIRE ~0.46-0.51 with TPR@5%FPR 0.000.
- Perturbed (Fig. 7, Sec. 8.5): JPEG, center crop, blur, noise on 250+250 images; AEROBLADE beats Gragnaniello and Ojha in most settings, but is less robust than Corvi (trained on perturbed LDM images). Numbers only in figures **[unverified: no numeric values]**. LPIPS_4 least affected by perturbations. Fig. 22 uses best layer per perturbation (authors admit this is unfair).
- Reports TPR@5%FPR (a fixed-FPR number), but on LAION real images, not phone photos.

## Known weaknesses / failure modes
- Needs the generator's own AE for best results; Midjourney AE is not public so SD2 AE served as proxy (it worked well).
- Cannot attribute among models sharing an AE (SD1.1/1.5).
- Low-complexity images (logos, flat graphics) harder.
- Only LDM-family generators tested; pixel-space diffusion, GANs not in the AE manifold. Independent evaluation by Brokman et al. (ICLR 2025, Table 1) on ~20 generators (CNNSpot/UFD/GenImage mix incl. GANs): AEROBLADE avg AUC 0.444 / AP 0.492 / ACC 0.464, i.e. near chance on that broad mix. FIRE (CVPR 2025, Table 1) reports AEROBLADE ACC ~98% on DiffusionForensics averages (99.08 AUC /98.0 ACC on LSUN-Bedroom).
- Robustness to JPEG/crop/blur is only partial; AE reconstruction error is sensitive to low-level changes.

## Code + weights
https://github.com/jonasricker/aeroblade. No LICENSE file found (GitHub license API returned none). No weights of its own: uses HF-hosted AEs (README commands use repo IDs like runwayml/stable-diffusion-v1-5; HF hosting is inferred **[unverified]**) and LPIPS. Real data must be rebuilt from LAION URLs via img2dataset (links go stale); generated data from Zenodo.

## Verified?
Partially: AP/TPR numbers from fetch-tool summary of arXiv HTML (Table 1/3); first 100k of 112k chars read. Robustness figures not extracted.

## Relevance to this project
- The cheapest reconstruction detector (one AE pass), CPU-feasible; a good zero-shot baseline for LDM-family fakes.
- Expect it to fail on non-LDM generators and mixed benchmarks; also the source documenting DIRE's JPEG/PNG artifact.
