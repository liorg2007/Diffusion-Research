# Attribution survey, part 1: learned / classifier-based source attribution (GAN background to 2026 diffusion)
Scope: "which generator made this image", closed-set and open-set. Sources: kb/papers/{yu-iccv2019, corvi-intriguing-cvprw2023, defake-ccs2023, pose-cvpr2023, fang-openset-bmvc2023, modelsynth-arxiv2023, forensic-selfdesc-cvpr2025, wild-ijcnn2025, rpa-arxiv2026, sds-arxiv2026}. Companion GAN/reconstruction entries (not mine): marra-gan-fingerprints-arxiv2018, ronan-neurips2023.
Grounding: numbers were read via fetch-tool summaries of arXiv HTML, not PDFs; all entries are partially-verified. Correction: POSE is CVPR 2023, not ICCV.

## 1. Accuracy levels reported
| Setting | Result | Source |
|---|---|---|
| GAN, closed-set, clean | 99.43 / 98.58 (CelebA/LSUN) | yu-iccv2019 Table 1 |
| T2I diffusion, 4-way, image+prompt | 0.936 (MSCOCO) | defake-ccs2023 Table 2 |
| 25 / 27 generators, caption-matched | 98.9% / 95.0% | rpa-arxiv2026 Table 1 |
| 8 SD-family models, white-box probe | 99.9% | sds-arxiv2026 Table 3 |
| Open-set, unseen architecture/seed (GAN) | AUC 84.17 / 68.15 | pose-cvpr2023 Table 4 |
| Open-set incl. unseen SD3/DALL-E 3/Firefly | AU-OSCR 0.913 | forensic-selfdesc-cvpr2025 Table 3 |
| Open-set, unseen SD (1 model) | CRR 0.806 | fang-openset-bmvc2023 Table 3 |
| Open-set, 17 known / 10 unseen | AU-OSCR 0.875 | rpa-arxiv2026 Sec. 4.4 |
| Zero-shot to emerging models (T2I incl.) | 93.20% ID acc (POSE 49.80%) | modelsynth-arxiv2023 Table VIII |

Under laundering:
- yu-iccv2019 (Tables 7/8): JPEG drops an unaugmented GAN classifier to 8.77% / 24.73%; attack-specific finetuning restores 88.02% / 92.40%.
- wild-ijcnn2025 (Table I, balanced acc, plain -> 3 random post-processing steps): EfficientNetB4 100 -> 59.93; DE-FAKE 94.00 -> 78.00; ViT 95.80 -> 88.67. Open-set ROC-AUC (Table III, 3 steps): EfficientNetB4 0.659, ViT 0.801.
- sds-arxiv2026 (Table 2): JPEG Q=25 93.1% / 85.0% (SD-Prompts / COCO), but only 20 test images per model per distortion.
- forensic-selfdesc-cvpr2025 (Table 6): JPEG QF 50-100 avg zero-shot AUC 0.972 vs 0.986; this is detection, not attribution; resize not tested.
- pose-cvpr2023 (Fig. 11) and rpa-arxiv2026 (Fig. 3): perturbation augmentation helps; numbers only in plots, not extracted.
- modelsynth-arxiv2023 (App. B): "sensitive to JPEG", no numbers read.

## 2. What signals work
- Spectral/residual: periodic peaks at 1/4, 1/8 from the latent-to-pixel upsampling (corvi-intriguing-cvprw2023 Fig. 3), but strong JPEG can hide them (Fig. 4); the same peaks mean models sharing a decoder look alike. Learned versions (POSE, modelsynth on Fourier spectra) are JPEG-sensitive.
- Learned low-level residuals: self-descriptions (real-only training) and a 64x64 raw-patch CNN reach the best open-set / caption-matched numbers on clean or lightly compressed data.
- Camera-forensics pretraining (fang-openset-bmvc2023) helps open-set rejection.
- VAE / denoiser traces: sds-arxiv2026 separates shared-VAE models that RONAN and LatentTracer cannot (20-27%, Table 3), but needs white-box weights.
- Semantic (CLIP/ViT): attribute well on content-matched data and degrade least under WILD processing (ViT 88.67 at 3 steps), but DE-FAKE's gain comes with prompt text, a content channel.
- Fine-tunes: LoRA variants fall onto their base model's fingerprint (modelsynth-arxiv2023 Sec. IV-D, qualitative); full-checkpoint fine-tunes (Dreamshaper, Realistic Vision) are separable only with weights (sds-arxiv2026 Table 3).

## 3. Same-prompt / content confounds
- Controlled: wild-ijcnn2025 (same 1,000 prompts for all closed-set generators; faces only), rpa-arxiv2026 DRAGON (same captions, default settings, Sec. 4.1), sds-arxiv2026 (empty-prompt probe; COCO cross-domain loses 3.8 pp).
- Not controlled: yu-iccv2019, pose-cvpr2023 and fang-openset-bmvc2023 use generators trained on different datasets, so content/domain can leak (the POSE "unseen dataset is easiest" result is suspect, reviewer inference); forensic-selfdesc-cvpr2025 shows no visible control; defake-ccs2023's prompts help partly via semantics.
- Training-data traces carry over: generators reproduce JPEG patterns in their training data (corvi-intriguing-cvprw2023 Sec. 4.2), a format confound for attribution.
- Cross-dataset transfer shows dataset cues: 96.6% -> 86.8% (rpa-arxiv2026 Sec. 4.5).

## 4. Three biggest open gaps
1. No attribution result under a realistic social/phone pipeline with real photos in the pool: WILD uses random operator mixes on faces; most others test JPEG alone. No paper reports FPR on fresh phone photos for attribution or open-set rejection.
2. Open-set separation of near-twins is weak: same architecture, different seed gives AUC 68.15 (POSE Table 4); LoRAs collapse to the base model; shared-VAE models defeat reconstruction methods unless weights are available. Black-box fine-grained attribution is open.
3. Content-controlled, laundering-controlled benchmarks with modern generators are scarce (WILD, DRAGON): the large accuracy numbers (95-99%) come from clean data; laundered numbers are plots, small tests, or 60-88%. Several 2026 results are unreviewed preprints.

## 5. Implications for us
- WILD (Kaggle, license unstated) and DRAGON are the candidate benchmarks; do not download without owner approval (size/licence rules).
- A cheap test: take our detector features and check whether the same features attribute generators after `social_sim`, with prompts matched.
