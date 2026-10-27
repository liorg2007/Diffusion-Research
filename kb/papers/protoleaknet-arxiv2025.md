# Proto-LeakNet: Towards Signal-Leak Aware Attribution in Synthetic Human Face Imagery
- **id:** protoleaknet-arxiv2025
- **Authors / venue / year:** Claudio Giusti, Luca Guarnera, Sebastiano Battiato / arXiv preprint (v2 15 Nov 2025) / 2025
- **Links:** arXiv 2511.04260 · code "after acceptance" (none at fetch)
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Encodes an image with the SD 2.1 VAE, re-applies a little forward diffusion noise (t in {0,5,10}) to expose residual generator traces, and classifies the source with a trained ResNet-18 + prototype head. Supervised, so not training-free, but built on the same VAE-latent idea.

## How it works
Image resized to 256x256 -> SD2.1 VAE -> forward noising at 3 steps -> ResNet-18 per step -> temporal attention -> 4 prototypes/class; open-set via Gaussian KDE on frozen embeddings. SDXL VAE gave near-identical ablation results (Top-1 82.55%, Macro AUC 98.09%).

## Training data
Closed-set face generators (face-centric data only; list not read).

## Reported results
- Closed set (Table 1): Macro AUC 98.13% (SuSy 98.02, LatentTracer 97.88); Top-1 82.60%.
- Post-processed (Table 2, 1-3 random steps of compression/crop/resize/rotation/blur/colour/SR): Macro AUC 96.14 / 94.62 / 92.25; Top-1 75.89 / 74.51 / 69.52%.
- Open set (Table 3): AUC 57.24 / 56.62% (near chance). Cost not reported.

## Known weaknesses / failure modes
Faces only; open-set weak; JPEG/resize not reported separately; no code.

## Code + weights
None released at fetch; licence n/a.

## Verified?
Partially: fetch-tool summary of arXiv HTML v2.

## Relevance to this project
- Shows a VAE-latent + small learned head can survive mixed post-processing (AUC 92-96) closed-set, yet fails open-set: caution for unseen generators.
