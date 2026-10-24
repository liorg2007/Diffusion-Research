# Enhancing Frequency Forgery Clues for Diffusion-Generated Image Detection (F2C)
- **id:** f2c-arxiv2025
- **Authors / venue / year:** Daichi Zhang, Tong Zhang, Shiming Ge, Sabine Susstrunk / arXiv preprint (no venue stated) / 2025
- **Links:** arXiv 2511.00429 (v1) · https://arxiv.org/abs/2511.00429 · code: none found
- **Family:** low-level
- **Status:** partially-verified

## Summary (plain language)
Diffusion images differ from natural photos by an amount that grows from low to high frequency. F2C re-weights the Fourier spectrum with a frequency-selective filter that suppresses useless bands and boosts informative ones, then trains a ResNet-50 on the filtered image.

## How it works
Grayscale -> FFT -> multiply by w(f) (zero for f <= tau = 0.1; otherwise (e^(k(f)/2) - 1)/f with k(f) = -0.2 f^2 + 0.8 f - 0.05, Eqs. 5-10) -> inverse FFT -> ResNet-50, BCE. Grayscale beats RGB (Table II: 80.42/90.81 vs 63.86/72.33 avg ACC/AP).
- **Weights needed:** none pretrained; trains ResNet-50 (about 100 MB checkpoint).
- **Passes per image:** 1 FFT + 1 ResNet-50 forward; no diffusion model.

## Training data
About 40,000 ADM fakes and ImageNet reals (Sec. IV-A); tested on GenImage.

## Reported results
- **Unseen-generator results:** Table I GenImage (trained on ADM) F2C average ACC 99.91 / AP 100.0; Midjourney 99.90/99.99; vs DIRE AP 94.59, NPR 89.02. GAN images from DiffusionForensics (Table V) 99.0-99.8 ACC. Abnormally high; likely GenImage format/size bias (see genimage-neurips2023), no FLUX/SD3/DALL-E.
- **Laundering results:** JPEG (q 75/50/25), blur (sigma 1-3), noise in Figure 5 only (no numbers); text says "slight or no" drops; caption/text conflict on which generator was used. Resize not tested.
- **FPR on real photos:** not reported.

## Known weaknesses / failure modes
ADM-only training; empirically fitted kernel and tau; perturbation numbers missing; likely exploits GenImage dataset bias (our inference **[unverified]**).

## Code + weights
- **License:** none found.

## Verified?
Partially: arXiv HTML v1; Tables I-VI.

## Relevance to this project
- A frequency-only approach that claims diffusion-specific spectral differences; the near-100% GenImage score is a red flag for bias rather than a result to trust. Useful as a negative-control idea only.
