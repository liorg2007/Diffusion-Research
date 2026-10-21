# Synthbuster: Towards Detection of Diffusion Model Generated Images
- **id:** synthbuster-ojsp2024
- **Authors / venue / year:** Quentin Bammey (ENS Paris-Saclay). IEEE Open Journal of Signal Processing, vol. 5, pp. 1-9, 2024 (dataset deposited Nov 2023). DOI 10.1109/OJSP.2023.3337714.
- **Links:** no arXiv version found · https://researchportal.ip-paris.fr/en/publications/synthbuster-towards-detection-of-diffusion-model-generated-images/ · dataset https://zenodo.org/records/10066460 (DOI 10.5281/zenodo.10066460)
- **Family:** data-bias-benchmarks
- **Status:** read-abstract-only

## Summary (plain language)
Proposes a training-free detector based on spectral peaks in the Fourier transform of a residual image, and releases the Synthbuster dataset: 1,000 images from each of 9 generators, prompted to resemble RAISE-1k raw photos, never JPEG-compressed or resampled.

## How it works
Spectral analysis of a residual image (abstract). No training; details not read.

## Training data
None (detector); the dataset is for evaluation. See kb/datasets/synthbuster.md.

## Reported results
Abstract only: works "even under mild JPEG compression" and "generalizes relatively well to unknown models". No numbers read **[unverified]**.

## Known weaknesses / failure modes
The paper's own claim is only for mild JPEG. Dataset is evaluation-only; B-Free uses Synthbuster precisely because real (RAISE) and fake images are both uncompressed/raw-derived, avoiding format bias (bfree-cvpr2025 Sec 3.1).

## Code + weights
Dataset licence CC BY-NC-SA 4.0 (Zenodo API). Detector code not checked **[unverified]**.

## Verified?
Abstract from IP Paris portal; dataset facts from Zenodo API.

## Relevance to this project
Format-clean evaluation set of 9 generators; use its fakes paired with RAISE reals (RAISE reals not included).
