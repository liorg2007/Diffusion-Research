# Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection
- **id:** staypositive-icml2025
- **Authors / venue / year:** Anirudh Sundara Rajan, Yong Jae Lee (UW-Madison) / ICML 2025 (PMLR v267) / 2025
- **Links:** arXiv 2502.07778 · https://proceedings.mlr.press/v267/sundara-rajan25a.html · code: none found in fetched text
- **Family:** data-bias-benchmarks
- **Status:** partially-verified

## Summary (plain language)
Detectors can learn "this has a real-image feature, so real" or "this lacks a fake artifact" rules, which breaks on fakes that look real or are partly edited. Stay-Positive retrains only the final linear layer with non-negative weights so the detector can only vote "fake" from fake-linked features. Not diffusion-specific by itself, but it targets exactly the fragility of LDM-trained detectors on FLUX.

## How it works
Start from a trained ResNet-50 detector (Corvi or Rajan), freeze the feature extractor, retrain the last linear layer with weights initialised at zero and clamped to max(w, 0) after each SGD step (batch 2048, App. A.1). Full-network constrained training hurt real-image accuracy (Sec. 4).
- **Weights needed:** an existing ResNet-50 detector checkpoint (Corvi/Rajan, ~100 MB) only; no VAE/UNet at inference.
- **Passes per image:** 1 ResNet-50 forward.

## Training data
Base detectors: real LSUN/COCO, fake LDM images (Sec. 5, App. A.1). 

## Reported results
- **Unseen-generator results (AP, Table 1, no post-processing):** FLUX.1-dev Corvi 57.25 -> 99.33, Rajan 80.64 -> 90.50; aMUSEd Corvi 89.18 -> 99.80, Rajan 87.20 -> 98.11. Other eval generators (Sec. 5.2.1): SDv1.5, Midjourney v4/v5, Kandinsky 2.1, Playground v2.5, PixArt-alpha, LCM, Wuerstchen. SD3 and DALL-E not evaluated.
- **Laundering results:** post-processed fakes (Table 2, AP): Corvi FLUX 74.57 -> 94.23; Rajan FLUX 87.80 -> 91.85. WEBP compression and downsizing improved in Figures 4-5 (no numbers). Plain JPEG is not separately evaluated (only a training augmentation).
- Partial inpainting (Table 3): at 50% inpainting Corvi 6.13 -> 97.48 AP. GenImage VQDM (Table 4) AP 95.11 -> 99.26 (Rajan).
- **FPR on real photos:** not reported (AP/accuracy only); accuracy drops on some GAN sets (App. A.5: 99.96 -> 86.75).

## Known weaknesses / failure modes
Upsampled real images still raise the fake score (Fig. 7); needs a good base detector; GAN accuracy cost.

## Code + weights
- **License:** not stated in fetched text **[unverified]**; paper CC BY 4.0. Release of retrained weights not confirmed.

## Verified?
Partially: arXiv HTML v1; numbers from Tables 1-4; ICML venue from PMLR search result.

## Relevance to this project
- Cheap, orthogonal training trick (a constraint on the last layer) that could be applied to any VAE-trace detector head; directly addresses real-photo false positives.
- Good evidence that FLUX failure of LDM-trained detectors is a "real-feature reliance" problem rather than a missing signal.
