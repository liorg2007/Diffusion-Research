# Denoising Trajectory Biases for Zero-Shot AI-Generated Image Detection (DTAD)
- **id:** dtad-neurips2025
- **Authors / venue / year:** Yachao Liang, Min Yu, Gang Li, Jianguo Jiang, Fuqiang Du, Jingyuan Li, Lanchi Xie, Zhen Xu, Weiqing Huang / NeurIPS 2025 (poster; per neurips.cc page) / 2025
- **Links:** OpenReview https://openreview.net/forum?id=2h8vXbEufN · NeurIPS page https://neurips.cc/virtual/2025/poster/120117 · code https://github.com/Eleven4AI/DTAD · arXiv: none found
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Runs diffusion inversion on the test image and watches the denoising outputs along the trajectory; for generated images these outputs converge to the input image faster than for real images. The similarity between the input and the trajectory outputs is the authenticity score. No training on generated images.

## How it works
From the abstract + README: DDIM inversion (script `ddim_inversion.py`) saving denoising outputs, then a similarity score along the trajectory. Which checkpoint, number of steps, timesteps and similarity metric were NOT read (OpenReview PDF blocked) **[unverified]**.
- **Weights needed:** a pretrained diffusion model (UNet) - README does not say which **[unverified]**. Not VAE-only.
- **Passes per image:** a full DDIM inversion (many UNet steps). Independent timing in Sonoda and Srinivasan (arXiv 2603.21619, Table 5): DTAD 276,861 s on OpenFake vs 436 s for their method (A100, batch 8) - i.e. extremely expensive.

## Training data
None (zero-shot).

## Reported results
- README (author-reported, Acc / AP): ForenSynths 89.9 / 97.4, GenImage 89.6 / 97.3, New Generator set 92.9 / 97.8; average 90.6 / 97.5 vs strongest baseline NPR 82.5 / 87.7.
- **Unseen-generator results:** independent: Sonoda and Srinivasan Table 2 (OpenFake, 34 generators incl. FLUX variants) show DTAD AUC avg 0.779 and beating their method on several FLUX generators (per-generator numbers not read). 
- **Laundering results:** same independent paper (Fig. 4, no numbers) says DTAD falls near random under high JPEG compression. The original paper's robustness section not read **[unverified]**.
- **FPR on real photos:** not reported (not read).

## Known weaknesses / failure modes
Cost (above); JPEG fragility (independent, figure-only). Full limitations not read.

## Code + weights
- **License:** none stated on the GitHub page. Weights: README silent.

## Verified?
Partially: abstract (NeurIPS page) + README table; paper PDF not accessible. Treat as unverified beyond the abstract.

## Relevance to this project
- Genuinely diffusion-specific (trajectory convergence) and notably strong on FLUX per the independent benchmark, but needs a full UNet and fails under JPEG; the signal may be worth a one-step proxy only if a UNet is available.
