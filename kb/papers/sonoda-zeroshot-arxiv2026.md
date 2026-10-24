# Efficient Zero-Shot AI-Generated Image Detection
- **id:** sonoda-zeroshot-arxiv2026
- **Authors / venue / year:** Ryosuke Sonoda, Ramya Srinivasan (Fujitsu) / arXiv preprint (no venue) / 2026
- **Links:** arXiv 2603.21619 (v1) · https://arxiv.org/abs/2603.21619 · code: none found
- **Family:** foundation-features
- **Status:** partially-verified

## Summary (plain language)
Training-free: add small high-frequency noise (one FFT) to the image and see how much CLIP's intermediate-layer embedding moves; real images change more than generated ones. Not diffusion-specific, but the most useful independent modern benchmark in this batch (OpenFake, 34 generators incl. FLUX, SD3.5, Midjourney 6/7, DALL-E 3), with runtimes for DTAD, WaRPAD, RIGID.

## How it works
Score = cosine similarity between CLIP ViT-L/14 embeddings (layer 13 CLS, 224x224, 16 patches of 14, tau = 0.5, lambda = 0.01 noise) of the image and its perturbed copy. Backbones (Table 6, OpenFake AUC): CLIP-L/14 L13 0.881, CLIP-B/32 0.765, DINOv2 0.680, DINOv3 0.502.
- **Weights needed:** CLIP ViT-L/14 (~0.9 GB fp32, size not checked **[unverified]**); no VAE/UNet.
- **Passes per image:** 1 FFT + 2 CLIP forward passes (paper says "a single forward pass", ambiguous).

## Training data
None.

## Reported results
- **Unseen-generator results (AUC avg):** OpenFake 0.881 vs DTAD 0.779 (Table 2); Semi-Truth 0.841 vs WaRPAD 0.779 (Table 3); GenImage 0.972 vs WaRPAD 0.956 (Table 4). DTAD wins on several FLUX generators in Table 2 (per-row numbers not read).
- Runtime (Table 5, A100, batch 8, seconds on OpenFake): ours 436.8, RIGID 1,519.5, WaRPAD 19,933.8, DTAD 276,861.7.
- **Laundering results:** blur, noise, center crop, JPEG at 3 intensities on OpenFake, Figure 4 only; text: DTAD near random under high JPEG; theirs degrades moderately under noise and JPEG. Resize not tested.
- **FPR on real photos:** not reported (AUC only).

## Known weaknesses / failure modes
AUC only, no threshold; fixed frequency decomposition; no code at fetch.

## Code + weights
- **License:** none stated; paper not checked for license **[unverified]**.

## Verified?
Partially: arXiv HTML v1; Tables 2-6.

## Relevance to this project
- Provides a modern-generator benchmark (OpenFake) and an honest cost comparison against DTAD/WaRPAD; may be a cheap reference baseline, though CLIP-L exceeds our weight budget.
