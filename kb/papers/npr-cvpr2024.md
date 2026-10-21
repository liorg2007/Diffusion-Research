# Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection
- **id:** npr-cvpr2024
- **Authors / venue / year:** Chuangchuang Tan, Huan Liu, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, Yunchao Wei / CVPR 2024 (CVF open-access page exists; arXiv listing shows no venue) / 2024
- **Links:** arXiv 2312.10461 · https://arxiv.org/abs/2312.10461 · https://github.com/chuangchuangtan/NPR-DeepfakeDetection
- **Family:** low-level (local pixel relationships / up-sampling traces)
- **Status:** read-full

## Summary (plain language)
Generators build images with up-sampling layers, which make neighbouring pixels in a 2x2 block correlated in a characteristic way. NPR takes the differences between pixels inside each 2x2 block, feeds that to a tiny CNN, and trained on ProGAN only gets strong accuracy on many GAN and early diffusion test sets. Independent benchmarks later found it close to chance on modern generators and on compressed social-media images.

## How it works
Image split into l x l (l=2) grids; each pixel minus the grid reference pixel (j=1) gives the NPR map, which is the classifier input (Sec. 3). Classifier: lightweight ResNet-style CNN, 1.44M parameters, Adam lr 2e-4, batch 32, RTX 3090 (Implementation Details). Ablation of reference choice (Table 11 in arXiv version: block average / max worse than first pixel) [table number from the first-pass summary; not re-checked in PDF].

## Training data
ForenSynths ProGAN, 4-class setting (car, cat, chair, horse) in the repo's training command; real = LSUN. Repo: `--niter 50 --batch_size 32 --lr 0.0002`.

## Reported results
All Acc / AP %, trained on ProGAN only (arXiv PDF tables):
- Table 1 ForenSynths (8 GANs): mean 92.5 / 96.1 (ProGAN 99.8, StyleGAN 96.3, StyleGAN2 97.3, BigGAN 87.5, CycleGAN 95.0, StarGAN 99.7, GauGAN 86.6, DeepFake 77.4 Acc).
- Table 2 self-synthesised GANs (9): mean 93.2 / 96.6.
- Table 3 DiffusionForensics: mean 95.3 / 99.8. Table 4 Ojha diffusion sets (LDM/Glide/DALL-E mini variants, 100-200 steps): mean 95.2 / 97.4.
- Table 5 diffusion with 1000 steps plus Midjourney and DALL-E (from Discord): DDPM 88.5, IDDPM 77.9, ADM 75.8, Midjourney 77.4, DALL-E 80.7 Acc; mean 80.1 / 84.8 (LGrad 64.2, Ojha 66.4).
- Table 6 overall: 92.2 mean Acc (LGrad 80.5, Ojha 79.8). Caption says "38 sub-testsets", text says 28 models; the paper is inconsistent.
- JPEG / blur / resize: NOT evaluated in the paper (no robustness section, no limitations section). Fixed-threshold FPR / calibration on fresh real photos: not reported.

## Known weaknesses / failure modes
- From the repo README: odd-sized images need a 1-pixel crop; for AIGCDetectBenchmark the evaluation disables resize/crop (batch size 1). Nearest-neighbour up-sampling in ProGAN is the premise; the detector is bound to up-sampling traces.
- SPAI (CVPR 2025, Table 1, 13 modern generators, default SIDBench checkpoints, avg AUC over 5 real sources): NPR 45.4 (worst of 12 compared) with very low AUC on Flux 19.8, DALL-E 2 3.9, SD2 12.5, SDXL 18.1, MJv5 15.3, MJv6.1 19.8, Firefly 38.0; high only on DALL-E 3 97.1, GigaGAN 83.2, SD1.3 89.6. Authors: NPR "underperforms in all the rest generative approaches".
- B-Free (arXiv 2412.17671, Table 5, bAcc / NLL): NPR 42.7 / 22.5 average over Synthbuster generators, new generators and WildRF; DALL-E 3 0.6, Firefly 0.4 (inverted), Facebook 50.0, Reddit 78.3, Twitter 51.8; Table 7: GenImage 50.7, FakeInversion 87.0, SynthWildX 49.9. Authors conclude NPR and LGrad perform well on a single dataset and poorly elsewhere, suggesting dataset-bias exploitation.
- These later results used default released checkpoints; whether they apply the PNG-vs-JPEG preprocessing issue is not separated, so part of the failure may be format bias rather than compression per se.

## Code + weights
https://github.com/chuangchuangtan/NPR-DeepfakeDetection ; weights in repo root: `NPR.pth`, `model_epoch_last_3090.pth` (ProGAN-4class); GenImage sdv4 checkpoint on Google Drive. No LICENSE file in the repo (GitHub API license: null); README only says code borrows partially from CNNDetection (which is CC BY-NC-SA 4.0). Treat as unclear / possibly encumbered, not an open licence.

## Verified?
Tables 1-6 numbers copied from arXiv PDF text (v2); the 'Table 11 ablation' and exact venue acceptance not independently confirmed in PDF [unverified]. Repo file list from GitHub API.

## Relevance to this project
- Strong on same-family GAN/early-diffusion benchmarks yet fails on modern diffusion and social-media data in independent tests; a cautionary example of benchmark overfitting.
- Low-level up-sampling cues are destroyed or biased by JPEG/resizing; any low-level component we use must be tested on JPEG-matched real/fake data.
