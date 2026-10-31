# Detecting Origin Attribution for Text-to-Image Diffusion Models (in RGB and Beyond)
- **id:** xu-origin-wacv2025
- **Authors / venue / year:** Katherine Xu, Lingzhi Zhang, Jianbo Shi / arXiv v1 Mar 2024, v3 Apr 2025; WACV 2025 (venue from the CVF supplemental URL; NeurIPS 2024 virtual-page title also seen) **[unverified]** / 2024-25
- **Links:** arXiv 2403.19653 · https://arxiv.org/abs/2403.19653
- **Family:** attribution
- **Status:** partially-verified (arXiv HTML via fetch summary)

## Summary (plain language)
Attributes images to 12 text-to-image generators, studies which hyper-parameters (seeds, etc.) are detectable, and shows that blurring/noising high-frequency detail barely hurts attribution; style (Gram-matrix) representations beat raw RGB.

## How it works
EfficientFormer attributor trained from scratch per input representation; also frozen CLIP ViT-B/16 or DINOv2 ViT-L/14 + linear probe/MLP (App. B).

## Training data
12 generators, 13-way task (chance 7.69%).

## Reported results
- Sec. 5: VGG Gram matrices from three layers: 92.80% accuracy vs RGB-trained (Sec. 5).
- Perturbations (Sec. 5, Fig. 7): Gaussian blur, bilateral filter, Gaussian noise, SDEdit: "modest decrease in classification accuracy"; a separate model trained per perturbed set.
- Table 5 (App. C.2): frozen CLIP+LP 70.15% vs DINOv2+LP 67.68%; fine-tuned 95.31% vs 96.67% (DINOv2 better).
- App. C.4: single-patch training, 2x2 patches 22.29% vs 7.69% chance. No patch-shuffle experiment.

## Known weaknesses / failure modes
Perturbation models retrained per perturbation (no transfer to unseen ones); no JPEG/resize; the style-vs-content statement is a suggestion, not a controlled test.

## Code + weights
Not checked.

## Verified?
Partially.

## Relevance to this project
- Closest prior for (d): "mid-level/style, not content" for attribution, via Gram/VGG and crops, not a patch-shuffle of ViT features.
- Frozen generic CLIP/DINOv2 probes are weak here (about 68-70%) next to trained-from-scratch (90%); our frozen mid-layer probe is a counter-datapoint.
