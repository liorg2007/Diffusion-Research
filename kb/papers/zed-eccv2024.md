# Zero-Shot Detection of AI-Generated Images
- **id:** zed-eccv2024
- **Authors / venue / year:** Davide Cozzolino, Giovanni Poggi, Matthias Niessner, Luisa Verdoliva / ECCV 2024 (stated by the code repo README; fetched arXiv page lists no venue) / 2024
- **Links:** arXiv 2409.15875 · project https://grip-unina.github.io/ZED/ · code https://github.com/grip-unina/ZED
- **Family:** reconstruction (zero-shot, lossless-coding / compression based)
- **Status:** partially-verified

## Summary (plain language)
Rather than using a diffusion model, ZED learns only from real images how to losslessly compress them at several resolutions. Fake images are "surprising" in a particular way: their coding cost deviates from what is expected, so thresholding a statistic of this gap flags them. No generated image is ever seen in training.

## How it works
SReC-style lossless coder: image downsampled by 2x2 average pooling into 4 levels; CNNs predict each finer level from the coarser one as a mixture of K=10 discrete logistics. Per level l in {0,1,2}: NLL (actual cost) and entropy H (expected cost); D(l)=NLL-H; feature Delta01 = D(0)-D(1); best statistic |Delta01|. Trained on real images only.
**Compute per image:** one pass through a lightweight multi-resolution CNN coder; paper calls it "computationally lightweight" but gives no timing/FLOPs. No diffusion model needed.

## Training data
Real only: default Open Images (~338k); variants LAION (~117k) and COCO (~106k), with/without JPEG augmentation (quality 80-100). COCO with augmentation (COCO*) gave the most uniform results.

## Reported results
- Table 2 (AUC, 30 test sets, 20+ generators, avg): |Delta01| 90.0%, Delta01 88.8%, |D(0)| 86.4%, D(0) 83.3%; best competitors Corvi2023 89.4%, Ojha2023 88.4%. Worst case |Delta01| = 65.1% on GLIDE-R, where |D(0)| reaches 89.4%. Claimed accuracy gain >3% over SOTA (accuracy table not in content read **[unverified]**).
- Table 4 (|Delta01| average AUC by real-image set): compressed COCO 92.2, ImageNet 85.9, LSUN 89.0, LAION 91.9; uncompressed FFHQ 89.4, RAISE 89.3 - authors state no significant difference between compressed and uncompressed.
- Resizing robustness not tested; JPEG robustness only via the real-set / augmentation studies.
- Threshold depends on dataset; they note only the proposed statistic keeps a near-single good threshold (Fig. 7).

## Known weaknesses / failure modes
- Authors: heavy compression/resizing may make statistics unreliable; fully generated images only; threshold calibration may need data; some generators (e.g., GLIDE) have anomalously low coding cost, so statistic choice matters; robustness to impairments is future work.
- Brokman et al. (ICLR 2025) used ZED inside a few-shot mixture-of-experts (see manifold-bias-iclr2025).

## Code + weights
https://github.com/grip-unina/ZED - custom GRIP-UNINA license (GitHub shows "Other"/NOASSERTION): "only for informational and nonprofit purposes" -> NON-COMMERCIAL. At fetch time the README said code is "Coming Soon" and the repo listed only README/LICENSE/docs; no weights visible. Recheck before planning to use it **[unverified current status]**.

## Verified?
Partially: via fetch-tool summary of arXiv HTML; license text read in repo LICENSE.md header.

## Relevance to this project
- Most robustness-aware of the zero-shot family (JPEG augmentation in training) and the cheapest at inference; requires only real images (could be trained on our own real photos, matching deployment).
- Open question for us: resize/screenshot robustness is untested; code availability uncertain.
