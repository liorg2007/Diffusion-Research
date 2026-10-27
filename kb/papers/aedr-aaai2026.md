# AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction
- **id:** aedr-aaai2026
- **Authors / venue / year:** Chao Wang, Zijin Yang, Yaofei Wang, Weiming Zhang, Kejiang Chen / AAAI 2026 (AAAI OJS vol. 40 per search result; arXiv page states no venue) / 2025-2026
- **Links:** arXiv 2507.18988 (v2, 28 Jan 2026) · https://ojs.aaai.org/index.php/AAAI/article/view/37930 · code https://github.com/wangchao0708/AEDR
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Passes the image through a candidate model's VAE twice. Images made by that VAE are already "on its manifold", so the second reconstruction barely improves on the first; the ratio of the two errors is the signal. A homogeneity (texture-flatness) calibration compensates for image complexity. One cheap pass pair per candidate, no optimisation.

## How it works
Loss ratio of two consecutive AE reconstructions (MSE; Eqs. 1-3), calibrated by a grayscale-homogeneity metric (l=32 grey levels; Eqs. 4-5); per-model threshold via kernel density estimation over 500 belonging-image signals (Eq. 6) with model-specific quantile alpha (Table 7). bf16 for FLUX/SD3.5, fp32 for others.

## Training data
None. Benchmark: 1,000 LAION-5B real + 1,000 images from each of 8 LDMs (SD1.5, SD2base, SD2.1, SDXL, SD3.5, FLUX.1-dev, VQDiffusion, Kandinsky 2.1), prompts from CLIP Interrogator.

## Reported results
- Attribution accuracy vs other models (Table 1) avg: AEDR 95.1%, LatentTracer 70.4%, RONAN 50.3%. Vs real LAION (Table 2): 96.9 / 66.7 / 52.2%.
- Per model (Table 7): SD1.5 98.48, SD2.1 99.66, SD2base 98.90, SD3.5 96.70, SDXL 99.44, FLUX 96.65, VQDM 90.61, KD2.1 82.93%.
- Loss ablation, SD2.1 (Table 5): MSE 99.40, SSIM 99.10, MAE 97.20, LPIPS 90.70%.
- Time (Table 3, hardware not stated): AEDR 0.06-1.25 s (SD1.5 0.27, SDXL 1.25) vs LatentTracer 12.6-162.9 s; 43-210x faster.
- Robustness: NOT evaluated in the AEDR paper (no JPEG/resize). See dna-arxiv2026 for JPEG QF75 collapse.

## Known weaknesses / failure modes
Weaker for quantised AEs (MoVQ 82.93%, VQ-VAE 90.85%, Table 4); thresholds not transferable across models; needs the target AE; models sharing an AE are inseparable by construction (as for any VAE test).

## Code + weights
Code URL above; code licence not read **[unverified]**. Uses public HF VAEs/pipelines of the eight models; FLUX.1-dev weights are under a non-commercial licence **[unverified, not read here]**.

## Verified?
Partially: arXiv HTML v2 via fetch-tool summary (Tables 1-7).

## Relevance to this project
- Cheapest strong attribution signal (two VAE passes) and directly usable with our sd-vae-ft-mse and SDXL VAE.
- Accuracy is on clean PNG-like outputs; DNA's follow-up shows it breaks under JPEG, which is the whole point of our wild-image focus.
