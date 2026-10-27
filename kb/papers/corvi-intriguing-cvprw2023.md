# Intriguing properties of synthetic images: from generative adversarial networks to diffusion models
- **id:** corvi-intriguing-cvprw2023
- **Authors / venue / year:** Riccardo Corvi, Davide Cozzolino, Giovanni Poggi, Koki Nagano, Luisa Verdoliva / CVPR Workshops 2023 (pp. 973-982) / 2023
- **Links:** arXiv 2304.06408 · https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_to_CVPRW_2023_paper.html · code: none found in text
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
An analysis paper (not an attribution classifier) showing that diffusion models, like GANs, leave periodic spectral/autocorrelation traces in noise residuals, whose spacing depends on the model's upsampling. It is the physical basis for spectral/residual attribution of latent diffusion models.

## How it works
Noise residuals (Zhang et al. denoiser, sigma=1) -> average spatial autocorrelation and average power spectrum over 1,000 images per source; radial/angular spectra.

## Training data
Generators in Table 1: BigGAN, StyleGAN2/3/T, GALIP; Taming Transformers, DALL-E Mini; Score-SDE, ADM, GLIDE, DALL-E 2, eDiff-I, Latent Diffusion, Stable Diffusion, DiT. 1,000 images per source.

## Reported results
- No accuracy numbers: paper reports no detection or attribution figures (Table 1 only lists model properties).
- Qualitative: latent-to-pixel upsampling factor 4 gives spectral peaks at multiples of 1/4, factor 8 at 1/8 (Fig. 3); generated images reproduce a JPEG Q=75 pattern if training data was JPEG (Sec. 4.2, Fig. 5); resize/blur/JPEG alter the autocorrelation and strong JPEG can hide artifacts (Fig. 4).

## Known weaknesses / failure modes
- Lab images without realistic post-processing (authors).
- Dataset artifacts transfer to generated images and can mislead fingerprint detectors; FFHQ interpolation makes its spectra resemble synthetic ones (authors).

## Code + weights
No code/data links in the fetched text.

## Verified?
Partially: ar5iv HTML fetch summary; figures not inspected.

## Relevance to this project
- Supports a spectral-fingerprint hypothesis for LDM attribution, but also warns that training-data/format traces (JPEG) are inherited, i.e. a confound for any attribution probe.
- Same-architecture models (shared VAE) will share peak spacing, limiting spectral attribution among them.
