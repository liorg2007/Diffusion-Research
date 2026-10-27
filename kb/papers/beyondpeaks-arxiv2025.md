# Beyond Spectral Peaks: Interpreting the Cues Behind Synthetic Image Detection
- **id:** beyondpeaks-arxiv2025
- **Authors / venue / year:** Sara Mandelli, Diego Vila-Portela, David Vazquez-Padin, Paolo Bestagini, Fernando Perez-Gonzalez / arXiv preprint (venue not stated) / 2025
- **Links:** arXiv 2510.05633 · code https://github.com/polimi-ispl/beyond-spectral-peaks
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Tests whether detectors really use the decoder's periodic spectral peaks. Peaks are removed with a Fourier mask; most deep detectors barely change, but a simple linear peak-based detector collapses. The authors still caution against concluding peaks are irrelevant.

## How it works
Binary mask on the full Fourier spectrum on a PxP grid (disk dilation, DC kept; Eq. 1) applied to synthetic images and to real images "laundered" through SDXL/SD3.5/Flux.1 autoencoders. 7 detectors via official code plus a linear detector (Laplacian-of-Gaussian high-pass, 8x8 or 16x16 magnitude grid; Eq. 2).

## Training data
No new training. 1,000 synthetic images/generator (Wild dataset), 1,000 RAISE reals cropped 1024x1024; uncompressed only.

## Reported results
- Untouched TPR (Table 1): detectors 0.812-0.998, Linear-8 0.859, Linear-16 0.750.
- Peak removal, relative TPR change (Table 2): [3] -0.45/-0.51, [7] -0.01/-0.06, Linear-8 -0.83, Linear-16 -0.69/-0.97.
- Laundered real images (Table 3): Linear detectors -0.61 to -0.82; some deep detectors gain (+0.30 to +1.81).
- JPEG/resize: NOT tested; authors say the linear detector would likely fail under JPEG.

## Known weaknesses / failure modes
TPR only (no false alarms for deep detectors); no compression; mask may alter more than peaks.

## Code + weights
Repo above; licence not stated **[unverified]**.

## Verified?
Partially: fetch-tool summary of arXiv HTML.

## Relevance to this project
- Confirms decoder-upsampling peaks are a real but fragile cue; AE-laundered reals (a decoder pass on real images) are exactly the confound our own laundering tests target.
