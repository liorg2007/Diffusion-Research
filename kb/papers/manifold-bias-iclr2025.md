# Manifold Induced Biases for Zero-shot and Few-shot Detection of Generated Images
- **id:** manifold-bias-iclr2025
- **Authors / venue / year:** Jonathan Brokman, Amit Giloni, Omer Hofman, Roman Vainshtein, Hisashi Kojima, Guy Gilboa / ICLR 2025 / 2025
- **Links:** arXiv 2504.15470 · code https://github.com/JonathanBrok/Manifold-Induced-Biases-for-Zero-shot-and-Few-shot-Detection-of-Generated-Images
- **Family:** reconstruction (zero-shot, diffusion score/manifold)
- **Status:** partially-verified

## Summary (plain language)
Generated images tend to sit near stable local maxima of a pretrained diffusion model's learned density (high curvature, small gradient). The method probes the image with small random perturbations through Stable Diffusion 1.4 and combines curvature, gradient and bias estimates into one score, requiring no training data. A few-shot variant mixes this with ZED-style experts.

## How it works
Image (resized 512) -> caption via LLaVA 1.5 -> SD1.4 VAE/U-Net; s spherical-noise perturbations at small diffusion time (README example: time_frac 0.01, num_noise 2) -> predicted noise/scores -> criterion C(x0) approximating a weighted curvature term minus a gradient-magnitude term plus a bias term; vectors mapped to CLIP space and compared by cosine similarity. Threshold calibrated on 1,000 real images. Few-shot: mixture of experts with a random forest on 1K labeled samples, combined with Cozzolino et al. 2024 (ZED).
**Compute per image:** s=64 perturbation passes through the SD U-Net by default (+ VAE, LLaVA captioning, CLIP); App. G.3: ~2.1 s/sample on an A100 with 16 perturbations, batch 4 (AEROBLADE official code 5.4 s/sample). Heavy for CPU / 4 GB GPU (LLaVA captioner alone is large).

## Training data
None for zero-shot (pretrained SD1.4 only; 1,000 real images to set the threshold). Few-shot: 1K labeled samples.

## Reported results
- Test set: aggregate of CNNSpot, UniversalFakeDetect and GenImage, ~20 generators (GANs, diffusion, Midjourney etc.); ~100k real + 100k fake in main text, App. F.1 mentions 200,000 samples (inconsistent **[unverified]**).
- Table 1 zero-shot averages: Proposed AUC 0.835 / AP 0.832 / ACC 0.741; AEROBLADE 0.444 / 0.492 / 0.464; RIGID 0.439 / 0.519 / 0.555. Top-10 generator accuracy 0.839 (RIGID 0.666, AEROBLADE 0.739).
- Few-shot (Fig. 6): improved separability, no exact numbers in text **[unverified]**.
- Robustness (Table 2): JPEG AUC 0.835 -> 0.79 (App. G.3: accuracy -3.45% on 16k-image reduced set); Gaussian blur k=3 AUC 0.822; SD v2-base 0.831, Kandinsky 2.1 0.826 as the base model; s 4->64 raises AUC 0.828 -> 0.835. No resize/screenshot test seen.
- Threshold fixed from 1,000 real images; FPR at that threshold not extracted.

## Known weaknesses / failure modes
- Authors: criterion derived for one diffusion model; works empirically on other generators but "no comprehensive theory"; ~2% drop with other base models; dips with smaller noise radii (-1.5% at /10, -2% at /100) and under JPEG.
- Comparison covers only AEROBLADE and RIGID (not ZED zero-shot, DRCT, etc.); AEROBLADE's near-chance score here contrasts with its 0.99 AP on LDM-only data, showing how benchmark mix matters.
- Cost and reliance on LLaVA captions.

## Code + weights
Official repo above; no LICENSE file in contents listing (README, environment.yml, example_images, manifold_bias_criteria.py). Uses HF diffusers/transformers weights (SD1.4, CLIP, LLaVA 1.5) downloaded at runtime; no separate checkpoint needed.

## Verified?
Partially: numbers from a fetch-tool summary of arXiv HTML (Table 1, Table 2, App. G.3); license by absence of file in GitHub contents listing.

## Relevance to this project
- Strongest peer-reviewed zero-shot number on a broad multi-generator mix (AUC 0.835), and mild JPEG degradation; but still ~0.74 ACC and needs SD + LLaVA per image.
- Suggests zero-shot scores need per-deployment threshold calibration on real photos.
