# Model Synthesis for Zero-Shot Model Attribution
- **id:** modelsynth-arxiv2023
- **Authors / venue / year:** Tianyun Yang, Juan Cao, Danding Wang, Chang Xu / arXiv preprint (v3, Jan 2025; venue not stated on fetched page) / 2023
- **Links:** arXiv 2307.15977 · code https://github.com/TianyunYoung/Model-Fingerprint
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Trains a fingerprint extractor only on thousands of synthetic tiny generators, then attributes images from real, unseen generators (GANs, VAEs, flows, diffusion, T2I). Also finds LoRA variants carry the same fingerprint as their base model.

## How it works
5,760 synthetic autoencoder generators (288 architectures x 20 seeds). ResNet50 extractor on denoised Fourier spectra, CE + triplet loss, 128x128 inputs with sliding window.

## Training data
Training: synthetic models only (plus optional fine-tune). Test (Table III): ProGAN ... StyleGAN3; VAEs; Glow; ADM, DDPM, LDM, PDNM; SD1.5, SD2.1, SDXL, Midjourney, GLIDE, DALL-E mini/2/3.

## Reported results
- Emerging models, identification (Table VIII): Only-Syn 93.20% acc / 93.11 F1; best baseline POSE 49.80% / 48.87; fine-tuned 94.02%.
- Verification (Table IX): Only-Syn 85.88% acc / 92.66 AUC; DNA-Det 69.64% / 77.54.
- Seen classical models: 96.71% identification (Only-Syn), 99.88% fine-tuned.
- LoRA (Sec. IV-D): 4 LoRAs each on SD1.5, PonyXL v6, SDXL 1.0, 10 images each; embeddings cluster with base model (t-SNE, qualitative, no numbers).
- Robustness (App. B): fairly robust to crop-resize; sensitive to JPEG; JPEG+crop augmentation helps but does not fully recover. No numbers extracted.

## Known weaknesses / failure modes
- Authors: JPEG sensitivity unresolved; synthesis targets CNN generators; cannot separate LoRA variants of one base.
- Reported timings inconsistent across sections.
- Spectra-based, so JPEG (the main laundering step) is the weak spot.

## Code + weights
https://github.com/TianyunYoung/Model-Fingerprint; license not checked **[unverified]**.

## Verified?
Partially: arXiv HTML fetch (last ~3.7k chars, incl. Appendix F, unread).

## Relevance to this project
- Best evidence that attribution to modern T2I (SD/MJ/DALL-E 3) can generalise zero-shot, but only on clean images.
- LoRA/fine-tune = base-model attribution only: fine-grained attribution is open.
