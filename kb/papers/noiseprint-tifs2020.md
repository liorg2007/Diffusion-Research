# Noiseprint: a CNN-based camera model fingerprint
- **id:** noiseprint-tifs2020
- **Authors / venue / year:** Cozzolino, Verdoliva; IEEE Trans. Information Forensics and Security 2020 (arXiv v1 Aug 2018)
- **Links:** arXiv 1808.08396 · IEEE TIFS 2020 [venue year from the paper's GitHub README, not the arXiv page] · https://github.com/grip-unina/noiseprint
- **Family:** camera-forensics
- **Status:** partially-verified

## Summary (plain language)
A Siamese CNN is trained so patches from the same camera MODEL (and same position in the image) map to similar 'noiseprints' while scene content is suppressed. The noiseprint is a model-level trace, used chiefly to localise forgeries where the trace is inconsistent. The authors say it does not identify individual devices.

## How it works
Fully convolutional net initialised from the Zhang et al. denoiser (DnCNN) weights, trained with a Siamese/contrastive scheme: 200 patches of 48x48 per minibatch in 50 groups of 4 patches from the same camera and position; ADAM; one network per JPEG quality factor.

## Training data
125 cameras, 70 models, 19 brands from Dresden (44 cams), Socrates (32), VISION (32), private (17); 100 cameras train, 25 validation; all JPEG (arXiv HTML text).

## Reported results
Forgery localisation, average over 9 datasets: MCC 0.403 (Table III), F1 0.444 (Table IV), AP 0.399 (Table V); best Splicebuster comparator MCC 0.365. Small source-ID test (3 models, 2 devices each): model ID 100% vs PRNU 77%; device ID 62% vs PRNU 70%. No AI-generated image evaluation. No FPR on fresh photos.

## Known weaknesses / failure modes
Authors: strong textures leak content into the noiseprint; very small images lack data; compression, resizing and blur weaken it (Fig. 9 discussion); only ~3-model source-ID test; trained on JPEG so raw data (Korus) ranks only third.

## Code + weights
https://github.com/grip-unina/noiseprint: copyright GRIP-UNINA 2019, 'used, reproduced and modified only for informational and nonprofit purposes' => NON-COMMERCIAL. TensorFlow 1.2.1 / Python 3.5 (old). Fetched README did not list weight files [verify before use].

## Verified?
Partially: numbers from the arXiv HTML text via WebFetch summarisation (may not match the TIFS camera-ready); licence from README.

## Relevance to this project
- Closest learned analogue of a 'real-camera trace'; trained on VISION cameras, so evaluating on VISION would leak.
- Old TensorFlow stack and non-commercial licence make it costly; a retrain on our own camera set would be the alternative.
