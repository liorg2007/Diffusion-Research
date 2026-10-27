# Progressive Open Space Expansion for Open-Set Model Attribution
- **id:** pose-cvpr2023
- **Authors / venue / year:** Tianyun Yang, Danding Wang, Fan Tang, Xinying Zhao, Juan Cao, Sheng Tang / CVPR 2023 (task brief said ICCV; arXiv page says CVPR) / 2023
- **Links:** arXiv 2303.06877 · code https://github.com/TianyunYoung/POSE
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Defines open-set model attribution: label images from known generators and reject images from unknown ones, even when unknown models differ only by random seed, architecture or training data. POSE simulates unknown-model traces with small augmentation networks during training.

## How it works
Feature extractor (DCT layer + small CNN, following DNA-Det) + MLP head. Tiny 2-conv augmentation nets, one per epoch, trained with reconstruction + diversity loss to create open-set samples with same semantics but different traces; task loss = CE + triplet (margin 0.3). Input resized to 128 px. Test: threshold on max softmax.

## Training data
Seen: 7 real sets (CelebA, Face-HQ, ImageNet, Youtube, LSUN-Bedroom/Cat/Bus), each with two seen fake models: 15 known classes. Unknown: 53 classes (10 seed, 22 architecture, 21 dataset). Five splits averaged. Mostly GANs; Stable Diffusion only as an extra 1k-sample test.

## Reported results
- Table 4 (avg 5 splits): closed-set acc 94.81; unseen-seed AUC/OSCR 68.15/67.25; unseen-architecture 84.17/81.62; unseen-dataset 88.24/85.64; unseen-all 82.76/80.50.
- Ablation (Table 6): base 90.68 acc / 73.78 AUC -> full 82.76 AUC.
- Stable Diffusion unseen test AUC 92.40 on 1k samples (Sec. 8).
- Robustness (Sec. 9, Fig. 11, split 1 only, plots not tables): without augmentation blur/JPEG/noise/crop/resize cut OSCR substantially; with perturbation augmentation drop within ~10% for JPEG QF ~80-100 and small blur.
- Fixed-FPR on real photos: not reported.

## Known weaknesses / failure modes
- Authors' own benchmark; robustness only on split 1.
- Unseen-seed (same arch, same data) is hardest (AUC 68.15) i.e. near-twin models are barely separable.
- Unseen-dataset 'easiest' case may partly reflect content differences (inference by reviewer, not stated by paper).
- Later work: model-synthesis (modelsynth-arxiv2023, Table VIII) finds POSE at 49.80% identification accuracy on emerging real models; forensic-selfdesc (CVPR 2025, Table 3) gets POSE AU-OSCR 0.608 on a diffusion/GAN mix.

## Code + weights
https://github.com/TianyunYoung/POSE (stated in abstract). License: not checked **[unverified]**.

## Verified?
Partially: ar5iv HTML fetch summary; robustness figure not inspected.

## Relevance to this project
- Reference definition of open-set attribution; shows the realistic ceiling (~83 AUC) when unknowns are near-identical to knowns.
- Trained on low-resolution (128 px) face/scene GANs, so transfer to modern T2I diffusion is weak per later papers.
