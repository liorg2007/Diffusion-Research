# Leveraging Frequency Analysis for Deep Fake Image Recognition
- **id:** freq-dct-icml2020
- **Authors / venue / year:** Joel Frank, Thorsten Eisenhofer, Lea Schonherr, Asja Fischer, Dorothea Kolossa, Thorsten Holz / ICML 2020 (arXiv comment: "Accepted to ICML 2020") / 2020
- **Links:** arXiv 2003.08685 · https://arxiv.org/abs/2003.08685 · https://github.com/RUB-SysSec/GANDCTAnalysis
- **Family:** low-level
- **Status:** read-full

## Summary (plain language)
GAN images show grid-like artifacts in the DCT (frequency) spectrum, caused by the up-sampling layers in generators. Feeding log-scaled DCT coefficients instead of pixels to a simple classifier separates real from fake very well on GAN data and slightly improves robustness to perturbations. It is a GAN-era paper; no diffusion models and no unseen-generator transfer test.

## How it works
Input image -> 2D DCT of the whole image -> log scaling and normalisation (Sec. 4) -> classifier. Classifiers compared: ridge regression, kNN, Eigenfaces, PRNU baselines, Yu et al.'s CNN, and a small custom CNN (about 170,000 parameters, ~1.9% of Yu et al.'s). Spectrum-weight heatmaps (Fig. 4) show that up-sampling (nearest neighbour / bilinear / binomial) leaves the artifacts.

## Training data
FFHQ vs. StyleGAN (16,000 each: 10,000 train; Sec. 4.1, 1024x1024); StyleGAN variants at 256x256 with different up-sampling; LSUN bedrooms and other GAN data sets (Tables 3-5). Models are trained and tested on the same generator.

## Reported results
- In-distribution: Table 1 (FFHQ vs StyleGAN, ridge): pixels 75.78%, DCT 100.00%. Table 2 (ridge, StyleGAN with nearest/bilinear/binomial up-sampling): pixel 74.77/62.13/52.64%, DCT 98.24/85.96/84.20%. Table 3 (CNN-DCT): 99.64% and 99.07% on the two data sets reported (pixel CNN 98.95/97.80).
- Perturbed (Table 5, LSUN bedrooms, accuracy; CD = trained on clean, tested on perturbed; PD = trained on perturbed): CNN-DCT CD: blur 61.42, crop 83.52, compression (JPEG quality ~U(10,75)) 71.86, noise 48.99, combined 67.76. CNN-Pixel CD: 60.56, 74.49, 68.66, 59.51, 65.98. With perturbed training (PD) CNN-DCT: blur 93.61, crop 98.83, compression 94.83, noise 89.56, combined 92.17.
- Unseen generators: not evaluated (no cross-generator table) . Fixed-threshold FPR / calibration on fresh real photos: not reported.

## Known weaknesses / failure modes
- From the paper: perturbation robustness for clean-trained models is poor (compression CD 71.86%, noise 48.99%, i.e. chance); only works well if perturbations are included in training. The DCT variant is not more robust to noise (authors say "all perturbations except noise"). Vulnerable to adversarial examples (Carlini & Farid, 2020, cited in Sec. 4.5).
- Heavier anti-aliasing in the generator removes artifacts (Table 2: bilinear/binomial 86/84% for linear model).
- Later evaluations: FreqNet (arXiv 2403.07240, Table 1) reports "Frank(2020)" trained on ProGAN with mean Acc 65.7 (1-class) / 75.0 (2-class) / 78.9 (4-class) on ForenSynths (DeepFake 52.4). NPR (arXiv 2312.10461, Table 4/5 context) lists Frank at mean Acc 58.8 on 1000-step diffusion models (Table 5) and mean Acc 56.8 over 28 models (Table 6).
- Corvi et al. (2211.00680, Table 1) evaluate a related spectrum detector ("Spec", Zhang et al.) and find chance-level performance (avg Acc 52.7) once images are resized and JPEG compressed; they do not evaluate Frank directly.

## Code + weights
https://github.com/RUB-SysSec/GANDCTAnalysis . License: MIT (LICENSE file read; Copyright 2021 Chair for Systems Security, Ruhr University Bochum). README says pre-trained models are downloadable from a Google Drive link; I did not download them [unverified that they are still live].

## Verified?
Tables 1, 2, 5 copied from PDF text extraction of the arXiv version; Table 3/4 only partially read (two data-set columns; labels of those columns not confirmed) [unverified column names]. License read from raw LICENSE.

## Relevance to this project
- Origin of the "up-sampling leaves spectral traces" idea; shows that spectral cues are fragile to JPEG/noise unless training includes them.
- Not a candidate detector for diffusion/in-the-wild; value is conceptual baseline.
