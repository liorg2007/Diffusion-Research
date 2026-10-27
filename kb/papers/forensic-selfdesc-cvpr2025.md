# Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images
- **id:** forensic-selfdesc-cvpr2025
- **Authors / venue / year:** Tai D. Nguyen, Aref Azizpour, Matthew C. Stamm (Drexel) / CVPR 2025 / 2025
- **Links:** arXiv 2503.21003 · https://cvpr.thecvf.com/virtual/2025/poster/33420 · code: not found in fetched text
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Learns, from real images only, a set of linear predictive filters; each image is described by how well these filters fit it. The fitted parameters ('self-description') support zero-shot fake detection, open-set source attribution and clustering, without ever seeing a fake in training.

## How it works
K=8 learned 11x11 linear predictors (centre weight 0, weights sum to 1), trained with an energy loss + spectral-diversity term (lambda=1.0), 3 scales (Sec. 3, 5.1). Zero-shot detection = GMM on real self-descriptions. Attribution/clustering on the description vectors.

## Training data
Real-only training: COCO2017 100k, ImageNet-1k 100k, ImageNet-22k 100k, MIDB 22,329 (Table 8). Test: 24 generators from OSSIA, DMID, Synthbuster plus authors' own (Table 9): ProGAN, StyleGAN1-3, GigaGAN, GLIDE, SD 1.3-3.0, DALL-E, Midjourney, Firefly.

## Reported results
- Zero-shot detection AUC avg 0.960 (Tables 1-2); NPR best baseline 0.926; worst case 0.892 (SD1.5, COCO) .. 0.714 (GLIDE, IN-22k).
- Open-set attribution (Table 3; known IN-1k, StyleGAN, SG3, SD1.4, ProGAN; unknown SG2, SD3, DALL-E 3, Firefly): accuracy 0.964, AU-CRR 0.933, AU-OSCR 0.913; Fang et al. accuracy 0.988 but AU-OSCR 0.852; POSE AU-OSCR 0.608.
- Clustering (Table 4, N=8): acc/purity/NMI 0.78/0.77/0.69 at N clusters (UFD ties on acc 0.78).
- JPEG (Table 6, QF 50-100, ablation subset): avg zero-shot AUC 0.972 vs 0.986 uncompressed. Resize/other processing not evaluated.
- Fixed-FPR / phone-photo calibration: not reported.

## Known weaknesses / failure modes
- Authors: needs diverse high-quality real data; post-processed/social-media images left to future work.
- Reviewer: abstract wording 'slightly behind LGrad' understates 0.714 vs 0.776 on IN-22k.
- Real-source choice changes numbers (0.941-0.971 avg).
- Source mixes: unknown sources are still generator-specific but content domains differ across sets (not controlled in visible text).

## Code + weights
Not found in fetched text; license unknown **[unverified]**.

## Verified?
Partially: arXiv HTML fetch (first 100k of 113.8k chars; appendices E-F unread).

## Relevance to this project
- Real-only training matches our unseen-generator focus; attribution of unseen SD3/DALL-E 3/Firefly at AU-OSCR 0.913 is the strongest open-set diffusion number found, but clean/JPEG-only.
- Candidate for reproduction as a baseline for attribution + detection on phone-photo reals.
