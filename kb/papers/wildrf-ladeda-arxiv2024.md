# Real-Time Deepfake Detection in the Real-World (LaDeDa, Tiny-LaDeDa, WildRF)
- **id:** wildrf-ladeda-arxiv2024
- **Authors / venue / year:** Bar Cavia, Eliahu Horwitz, Tal Reiss, Yedid Hoshen. arXiv preprint, June 2024 (no peer-reviewed venue found in this pass).
- **Links:** arXiv 2406.09398 · https://arxiv.org/abs/2406.09398 · project https://vision.huji.ac.il/ladeda · code https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld
- **Family:** data-bias-benchmarks
- **Status:** partially-verified

## Summary (plain language)
This is the paper that introduces WildRF (the Reddit/Facebook/Twitter set; origin confirmed). It argues that detectors trained on lab benchmarks fail on images scraped from social networks, and proposes LaDeDa (classifies 9x9 patches, pools scores) and a distilled 4-layer Tiny-LaDeDa.

## How it works
Patch-level classifier (receptive field 9x9) with pooled image score; distillation to Tiny-LaDeDa. Not read in more detail.

## Training data
Standard ForenSynth/ProGAN protocol and WildRF "social" protocol: train on 1,200 real + 1,200 fake Reddit images; test on Reddit 750/750, X 340/340, Facebook 160/160.

## Reported results
- WildRF composition (Fig. 4/Sec 4): Reddit 2,150 real (2017-22) + 2,150 fake (2022); X 340+340 (2021-24); Facebook 160+160 (2021-24); sums to 5,300 although text says "around 5000".
- Collection: real images by authenticity hashtags (#photography, #nature, #nofilter...), fakes by #deepfake, #midjourney, #stablediffusion... Labels derive from hashtags, no separate verification described -> label noise possible.
- Table 5 (train Reddit, ACC / AP): LaDeDa Reddit 91.8/96.0, Twitter 83.3/92.8, Facebook 81.9/92.6, mean 85.7/93.7. Baselines mean ACC/AP: CNNDet 72.5/85.0, PatchFor 82.2/91.9, CLIP (Ojha linear probe) 79.1/92.5, NPR 81.9/91.6, Tiny-LaDeDa 82.5/91.6.
- Table 3 (train ProGAN, test WildRF): LaDeDa mean 68.3 ACC / 79.9 AP.
- Paper says WildRF "inevitably contains some biases"; platforms re-compress images so format is mixed.

## Known weaknesses / failure modes
Later critique (arXiv 2412.05676, via search snippet, not read) says WildRF includes many drawings/CG graphics rather than photographic fakes **[unverified]**. Small Facebook/X sets (320/680 images) give noisy estimates.

## Code + weights
Code repo with LICENSE file (contents not read **[unverified]**); WildRF via Google Drive link in the README; weights (LaDeDa, Tiny-LaDeDa, each trained on WildRF and ForenSynth) via Google Drive. No explicit WildRF licence found; images are scraped from social platforms, so treat as research-only.

## Verified?
Numbers from arXiv HTML v1 via fetch tool; repo structure from README.

## Relevance to this project
Only public benchmark of genuinely platform-processed real/fake images; use with caution given hashtag labels and small test sets.
