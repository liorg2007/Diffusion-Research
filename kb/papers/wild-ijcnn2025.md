# WILD: a new in-the-Wild Image Linkage Dataset for synthetic image attribution
- **id:** wild-ijcnn2025
- **Authors / venue / year:** Pietro Bongini et al. (IPLAB Catania; first author per fetch summary, full author list not read) / IJCNN 2025 VERIMEDIA Workshop (per iplab.dmi.unict.it page) / 2025
- **Links:** arXiv 2504.19595 · dataset https://www.kaggle.com/datasets/pietrob92/wild-in-the-wild-image-linkage-dataset (license not stated)
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Benchmark with per-generator labels and a closed/open split for source attribution, with prompt-matched images and 1-3 random post-processing steps to mimic in-the-wild conditions. Shows that closed-set attribution is easy on clean images and degrades with laundering.

## How it works
Benchmark paper. 10 closed-set commercial/online generators, 10 open-set generators, 1,000 images each; same 1,000 prompts across closed-set generators (so each prompt gives one image per generator). Post-processing used for testing only.

## Training data
Closed: Firefly, DALL-E 3, FLUX.1, FLUX 1.1 Pro, Freepik, Leonardo, Midjourney, SD 3.5 Large, SDXL Turbo, Starry AI. Open: DALL-E 1, Deep AI, Hotpot AI, Sana PAG, Stable Cascade, SD Attend&Excite, StyleGAN 1/2/3, Hunyuan. Head-and-shoulder portraits only. Operators: JPEG/WebP QF 50-100, crop, resize 0.4-2, rotation, contrast, blur, grayscale, super-resolution, JPEG AI.

## Reported results
- Closed-set balanced accuracy, plain / 1 / 2 / 3 steps (Table I): EfficientNetB4 100 / 84.43 / 73.03 / 59.93; ResNet50 99.90 / 84.87 / 73.53 / 60.63; CLIP+MLP 96.67 / 87.13 / 80.13 / 72.40; DE-FAKE 94.00 / 87.00 / 83.00 / 78.00; ViT classifier 95.80 / 93.40 / 91.50 / 88.67.
- Open-vs-closed ROC-AUC, plain (Table II): EfficientNetB4 0.997, Xception 0.992, ResNet50 0.985, ViT 0.890, CLIP+MLP 0.876.
- Open-set ROC-AUC under processing 1/2/3 steps (Table III): EfficientNetB4 0.872 / 0.763 / 0.659; ViT 0.857 / 0.829 / 0.801.
- Multi-class with rejection (Table IV): CRR plain 0.70 (Xception) / 0.52 (DE-FAKE) / 0.37 (ViT), at 3 steps 0.64 / 0.49 / 0.17.

## Known weaknesses / failure modes
- Faces only; open set not for training; no real/fake detection.
- Prompt-image linkage imperfect; some generator settings (e.g. Midjourney negative prompt) hard to reproduce.
- Random-operator mixture: not a specific social-media pipeline. No license on dataset page.

## Code + weights
Dataset on Kaggle (license not stated). Dataset size ~10k closed + 10k open images; not downloaded.

## Verified?
Partially: arXiv HTML v2 fetch summary; venue from workshop page via search.

## Relevance to this project
- Most relevant benchmark for our goal: per-generator labels, prompt-matched content (controls confound), laundering. Not downloaded (CLAUDE.md download rule).
- Key pattern: the best clean accuracy (CNNs, 100%) is the worst after laundering (59.9%); semantic ViT/CLIP degrade less.
