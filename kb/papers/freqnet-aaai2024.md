# Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning (FreqNet)
- **id:** freqnet-aaai2024
- **Authors / venue / year:** Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, Yunchao Wei / AAAI 2024 (arXiv comment: "9 pages, 4 figures, AAAI24") / 2024
- **Links:** arXiv 2403.07240 · https://arxiv.org/abs/2403.07240 · https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection
- **Family:** low-level (frequency-domain learning)
- **Status:** read-full (arXiv version; may differ from AAAI camera-ready)

## Summary (plain language)
A very small network (1.9M parameters) that repeatedly applies FFT high-pass filtering to the image and to intermediate feature maps, plus convolutions on the FFT phase and amplitude. Trained on ProGAN only, it generalizes to 17 GAN test sets. It evaluates GANs only; no diffusion and no JPEG/resize tests.

## How it works
High-frequency representation of the image (HFRI): FFT, zero out the central W/4 x H/4 block, iFFT (Eq. 4-5). HFR of features across spatial (HFRFS) and channel (HFRFC) dimensions (Eq. 6). Frequency convolutional layer (FCL): convs on amplitude and phase between FFT and iFFT. Residual CNN from scratch, no pretraining. Adam, lr 2e-2 as printed in the paper (likely a typo for 2e-4; unverified), batch 32, 100 epochs, lr decay 20% each 10 epochs; RTX 3090.

## Training data
ForenSynths ProGAN, 1-class (horse), 2-class (chair, horse), 4-class (car, cat, chair, horse); real = LSUN. 

## Reported results
Acc / AP %, trained on ProGAN:
- Table 1 ForenSynths mean (8 GANs): 1-class 89.6/97.6 (Ojha 89.2/97.8; LGrad 86.3/92.7); 2-class 91.0/97.9; 4-class 91.5/98.5 (Ojha 89.1/98.3; LGrad 86.1/91.5). 4-class per generator Acc: ProGAN 99.6, StyleGAN 90.2, StyleGAN2 88.0, BigGAN 90.5, CycleGAN 95.8, StarGAN 85.7, GauGAN 93.4, DeepFake 88.9.
- Table 2 self-synthesised 9 GANs (4-class): mean 94.0 / 97.5 (Ojha 77.6/98.8; LGrad 68.6/80.8).
- Table 3: mean Acc over all 17 models 92.8 with 1.9M parameters (Ojha 83.0 with 304M; LGrad 76.8, 46.6M; F3Net 77.8). Face test: 98.7 / 99.0 / 99.5 on ProGAN / StyleGAN / StyleGAN2 (text).
- Ablation (Table 4): removing modules gives 82.0-87.8 mean Acc vs 91.5 full.
- JPEG / blur / resize robustness: NOT evaluated. Diffusion: NOT evaluated. FPR at a fixed threshold / calibration: not reported.

## Known weaknesses / failure modes
- From the paper: none discussed (no limitations section); the evaluation is limited to GANs.
- Not found in the primary sources I read: any independent JPEG/resize or diffusion evaluation of FreqNet. I did not locate one; note this gap [unverified]. Frequency methods in general lose accuracy under compression (Frank et al. Table 5; LGrad Table 4; Corvi Table 1 "Spec" chance after resize+JPEG), so FreqNet is expected to be fragile but this is not measured here.
- Protocol caveat: reuses ForenSynths real-vs-fake whose real images are JPEG and fakes are PNG; frequency-domain classifiers can latch onto that format difference.

## Code + weights
https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection ; checkpoint in repo: `4-classes-freqnet-v2.pth` (README test command uses `./4-classes-freqnet.pth`, a name mismatch). No LICENSE file in the repo listing -> licence unspecified.

## Verified?
Tables 1-4 and text copied from arXiv PDF extraction. Licence = absence of file in repo listing.

## Relevance to this project
- Shows how far a 1.9M-parameter frequency model goes on GAN benchmarks, but gives no evidence for diffusion or in-the-wild use.
- Useful only as an ablation baseline for "does explicit high-frequency emphasis help".
