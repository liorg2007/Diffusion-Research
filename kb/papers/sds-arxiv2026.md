# Diffusion Model Attribution via Spectral Coupling of Denoiser Responses
- **id:** sds-arxiv2026
- **Authors / venue / year:** Pragati Shuddhodhan Meshram, Varun Chandrasekaran (UIUC) / arXiv preprint 2606.28092v1 (26 Jun 2026) / 2026
- **Links:** arXiv 2606.28092 · code https://github.com/Pragati-Meshram/SGS
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Fingerprints a white-box diffusion model by probing its denoiser with band-limited noise and measuring how energy moves across frequency rings. Distinguishes eight SD-family models, including ones sharing the same VAE, at ~99.9% accuracy. Closed-set and white-box only.

## How it works
Spectral Denoising Signatures: K=8 frequency rings, 3 amplitudes, 5 timesteps, 10 repeats -> 960-dim signature; shared SD1.5 KL-f8 VAE encodes all inputs; prototype cosine argmin (training-free, N=50) or LinearSVC. Probe uses empty text embedding.

## Training data
8 models: SD1.4, SD1.5, SD2.1, Dreamshaper-8, Realistic Vision v5, SDXL, PixArt-alpha, SDXL-Turbo. SD-Prompts dataset (Ntest=500 per model); MS-COCO prompts for cross-domain.

## Reported results
- Closed-set (Table 1, SD-Prompts): LinearSVC 99.98%, PCA-LR 99.62%, argmin 78.33%. Per-model (Table 3): 100% except Dreamshaper-8 99.8% (overall 99.9%).
- MS-COCO prompts (Table 1): LinearSVC 96.20%, argmin 78.40%.
- JPEG Q=25 (Table 2, SD-Prompts/COCO): LinearSVC 93.1 / 85.0; argmin 76.2 / 55.6. Crop 75%+resize: LinearSVC 90.6 / 87.5.
- Baselines (Table 3 overall top-1): RONAN 30.5%, LatentTracer 44.8%, SDS 99.9%; both baselines ~20-27% on the five shared-VAE models.
- Open-set: not evaluated (stated open problem). Table 2 uses 20 test images per model (n=160/distortion) vs 500 main.

## Known weaknesses / failure modes
- Closed-set, white-box (needs candidate weights); proprietary APIs excluded; no spoofing study (authors).
- Small test sets under distortion; 99.9 vs 99.98 inconsistency (reviewer).
- Fetched text did not show whether LoRA/fine-tunes are separated beyond Dreamshaper/Realistic Vision (which are SD-derived fine-tunes: 99.8-100%).

## Code + weights
https://github.com/Pragati-Meshram/SGS; license not stated in fetched text.

## Verified?
Partially: arXiv HTML fetch summary; 2026 preprint, not peer-reviewed.

## Relevance to this project
- Shows fine-tuned/community checkpoints (Dreamshaper, Realistic Vision) are separable from base SD with weight access, whereas VAE-trace/reconstruction methods are not (baselines ~20-27% on shared VAE).
- Needs white-box weights: not applicable to closed commercial or phone-ISP sources.
