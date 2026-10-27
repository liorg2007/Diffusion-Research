# Do GANs leave artificial fingerprints?
- **id:** marra-gan-fingerprints-arxiv2018
- **Authors / venue / year:** Marra, Gragnaniello, Verdoliva, Poggi; arXiv Dec 2018 (published at IEEE MIPR 2019 [unverified])
- **Links:** arXiv 1812.11842 · no official code located
- **Family:** camera-forensics
- **Status:** partially-verified

## Summary (plain language)
Treats each GAN like a camera: average noise residuals over many generated images to extract a stable 'fingerprint' à la PRNU, then attribute images to the right GAN by correlation. Real cameras (two RAISE Nikons) are included in the same experiment, so real and generated sources are attributed in one framework.

## How it works
PRNU-style: denoising filter -> residual = image minus estimate -> average 512 residuals = fingerprint; correlation with a query residual identifies the source (Section 3).

## Training data
Table I: 1,000 RGB 256x256 images per source; 512 for fingerprint and 488 for test. Sources: CycleGAN C1-C9, ProGAN G1-G6, StarGAN S1-S5, real cameras N1-N2 (RAISE Nikon D90, D7000). Forensics GAN Challenge: 1,000 images of varied size.

## Reported results
Same-vs-different GAN AUC 0.990 (CycleGAN) and 0.998 (ProGAN) (Sec. 3); overall attribution accuracy 90.3% before JPEG, 90.1% after QF=95 (Sec. 4, Figs. 7-8); fused with deep net on the GAN Challenge AUC 0.999. Cameras slightly weaker (lower PRNU energy). Resizing, blur and noise NOT tested (future work). Closed-set, GANs only, no diffusion models; no FPR on fresh photos.

## Known weaknesses / failure modes
Only GANs (no diffusion); only mild JPEG (QF 95); needs hundreds of images from each source; closed set.

## Code + weights
No official code located.

## Verified?
Partially: numbers from the arXiv HTML text via WebFetch summarisation; venue year not verified.

## Relevance to this project
- Direct precedent for 'generator = camera' fingerprinting; but needs 512 same-size residuals per source, which is easy for generators and hard for phones with 14 images.
- Gives a concrete template for a cheap PRNU-style baseline on our fake sets.
