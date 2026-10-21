# A Sanity Check for AI-generated Image Detection (AIDE + Chameleon)
- **id:** aide-chameleon-iclr2025
- **Authors / venue / year:** Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, Weidi Xie. ICLR 2025 (confirmed by ICLR proceedings/poster pages in search results; arXiv lists none).
- **Links:** arXiv 2406.19435 (v3 15 Feb 2025) · https://arxiv.org/abs/2406.19435 · code https://github.com/shilinyan99/AIDE
- **Family:** data-bias-benchmarks
- **Status:** partially-verified

## Summary (plain language)
The authors collect "Chameleon", a set of AI images that two human annotators both judged real, and show that nine existing detectors mostly call them real. They also propose AIDE, which combines CLIP semantic features with low-level features from the most and least textured patches.

## How it works
AIDE: mixture of experts using CLIP-based semantics plus noise/frequency-style features from highest- and lowest-frequency patches (abstract); details beyond that not read.

## Training data
Evaluated with training on ProGAN (360k LSUN, Table 10), SD v1.4 (324k ImageNet-based GenImage, Table 11), or all GenImage (Chameleon Table 5).

## Reported results
- Chameleon (Table 5), AIDE overall / fake-acc / real-acc: ProGAN-trained 58.37 / 5.04 / 98.46; SD1.4-trained 62.60 / 20.33 / 94.38; all-GenImage 65.77 / 26.80 / 95.06. Other detectors are ~55-61 overall, e.g. NPR 57.29 (fake 2.20, real 98.70) ProGAN; 58.13 (2.43/100.00) SD1.4; 57.81 (1.68/100.00) GenImage. UnivFD with SD1.4: 55.62 (74.97/41.09). Most detectors call nearly everything real -> a threshold/calibration failure as much as a feature failure.
- AIGCDetectBenchmark mean 92.77 (Table 3, ProGAN-trained); GenImage mean 86.88 (Table 4, SD1.4-trained).
- JPEG robustness (Table 6, avg over 16 AIGCDetectBenchmark sets): 92.77 original -> 75.54 (QF95), 74.21 (QF90), 70.64 (QF75), 69.60 (QF50). Blur sigma 1-4: 81.88 -> 79.86.
- Chameleon (Table 1): 14,863 real, 11,170 fake. Real from Unsplash; fakes from ArtStation, Civitai, Liblib; images <448x448 removed; resolutions mostly >720P up to 4K; each image CC BY 4.0 or owner-permitted (Sec 3.2).

## Known weaknesses / failure modes
Paper itself: all detectors degrade under JPEG/blur. Chameleon fakes are selected by human adversarial filtering and come from community art platforms (stylistic/prompt bias, not phone photos); file formats not stated.

## Code + weights
Code repo; checkpoints via Google Drive linked from README Model Zoo. Chameleon: README says "only used for academic research. Commercial use in any form is prohibited"; not publicly downloadable - request by email (tattoo.ysl@gmail.com); paper Appendix E says a signed EULA is required. Repo code licence not read **[unverified]**.

## Verified?
Numbers from arXiv HTML v3 via fetch tool; README read via fetch tool. Not PDF-checked.
(verified vs PDF p.9, 2026-10-21) Table 5 AIDE rows (fake/real acc): ProGAN 5.04/98.46, SD1.4 20.33/94.38, All GenImage 26.80/95.06, i.e. real 94.4-98.5%, fake 5.0-26.8% (overall 58.37/62.60/65.77); NPR 2.20/98.70, 2.43/100.00, 1.68/100.00. The broader "94-100% real, 0-27% fake" range holds for AIDE, NPR, CNNSpot, Fusing, DIRE and LNP but NOT for all detectors: UnivFD fake 74.97/85.52 (real 41.09/41.56), FreDect real 87.12, PatchCraft real 92.82. ICLR 2025 confirmed by PDF header.

## Relevance to this project
- Shows high real-accuracy / near-zero fake-accuracy failure mode -> report fake recall at fixed real FPR.
- Chameleon is a hard but gated, non-commercial set; request early if wanted.
