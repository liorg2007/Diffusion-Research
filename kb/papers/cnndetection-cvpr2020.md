# CNN-generated images are surprisingly easy to spot... for now
- **id:** cnndetection-cvpr2020
- **Authors / venue / year:** Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, Alexei A. Efros / CVPR 2020 (arXiv listing states accepted to CVPR 2020) / 2020
- **Links:** arXiv 1912.11035 · https://arxiv.org/abs/1912.11035 · https://github.com/peterwang512/CNNDetection
- **Family:** low-level (strictly a pixel-space CNN baseline; included because it defines the ProGAN-train / augmentation protocol that NPR, LGrad, FreqNet reuse)
- **Status:** read-full

## Summary (plain language)
A plain ImageNet-pretrained ResNet-50, trained only on ProGAN fakes vs. real LSUN photos, with blur and JPEG augmentation, detects fakes from many other GANs it never saw. It created the "ForenSynths" benchmark (11 CNN generators) and the standard protocol of training on ProGAN. It predates diffusion models; later work shows it fails on them.

## How it works
Input: RGB image, random 224x224 crop (center crop at test), no resizing of the crop. Model: ResNet-50 (ImageNet-pretrained) fine-tuned end to end as real/fake classifier (Sec. 3 / Appendix B.2). Augmentation variants (Sec. 3): No aug; Blur (50%, sigma~U[0,3]); JPEG (50%, quality~U{30..100}, OpenCV and PIL); Blur+JPEG (0.5); Blur+JPEG (0.1) (10% probability). Metric: average precision (AP), threshold-free.

## Training data
ProGAN trained per LSUN category: 20 classes, 720K training images and 4K validation images (text near Table 2); fake/real balanced. Ablations with 2/4/8/16 classes, a BigGAN-trained model and a deep-image-prior-trained model.

## Reported results
- Unseen GANs, AP, Table 2 (mAP over 11 test generators): No aug 90.1; JPEG only 93.0; Blur+JPEG(0.5) 90.8; Blur+JPEG(0.1) 92.6. Blur+JPEG(0.5) per-generator: ProGAN 100, StyleGAN 98.5, BigGAN 88.2, CycleGAN 96.8, StarGAN 95.4, GauGAN 98.1, CRN 98.9, IMLE 99.5, SITD 92.7, SAN 63.9, DeepFake 66.3.
- Class-count ablation (Table 2): 2-class 81.3, 4-class 85.4, 8-class 87.9, 16-class 91.4 mAP (with Blur+JPEG(0.5)).
- Perturbed: Figure 5 plots AP vs. test-time blur / JPEG; the text says no-augmentation training degrades across all datasets under both perturbations, and training with blur+JPEG is best or near best (exceptions: SAN super-resolution, DeepFake). Exact per-quality numbers are only in the figure (not extracted) [unverified].
- Threshold/calibration: AP only; no FPR at a fixed threshold, no calibration on fresh real photos. The paper notes (Sec. on evaluation) AP is chosen as it is threshold-free.

## Known weaknesses / failure modes
- From the paper: concluding discussion says robustness to compression/blur/resize is only "to some degree" and "much more work is needed" for in-the-wild social-media processing. Augmentation lowers performance on some sets (e.g. DeepFake 98.2 -> 66.3 under Blur+JPEG(0.5), Table 2; SAN 93.6 -> 63.9).
- Corvi et al. (arXiv 2211.00680, Table 1), "Wang2020" trained on ProGAN: on uncompressed PNG, avg Acc/AUC 59.5/78.6 over 12 generators (Stable Diff. 50.9/65.9, ADM 50.6/56.3, DALL-E 2 50.3/85.8); after random crop + resize to 200x200 + JPEG Q65-100, avg 55.8/75.0 (Stable Diff. 50.7/72.9). Accuracy at the fixed 0.5 threshold is near chance on diffusion models.
- B-Free (arXiv 2412.17671, Table 5/7): "CNNDetect" balanced accuracy ~50% on every diffusion generator and on WildRF Facebook/Reddit/Twitter (50.0/50.7/50.1) and SynthWildX (50.0) [numbers from bAcc columns].
- SPAI (CVPR 2025, Table 1): CNNDet. avg AUC 55.5 over 13 modern generators.

## Code + weights
Code: https://github.com/peterwang512/CNNDetection . License: LICENSE.txt read from repo = Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (NON-COMMERCIAL, ShareAlike). Weights: `weights/blur_jpg_prob0.5.pth` (and a 0.1 variant) fetched by `bash weights/download_weights.sh`; hosting location of the weights not confirmed [unverified]. Train/val/test sets on Google Drive links in the README (README notes the link was to be replaced in July 2024).

## Verified?
Numbers from Table 2 and text copied from the arXiv PDF text extraction (text of v-latest as fetched). License read from raw LICENSE.txt on GitHub. Fig. 5 values not read.

## Relevance to this project
- Defines the ProGAN-only training / Blur+JPEG augmentation recipe; shows a pixel-space CNN fails to transfer to diffusion output even before any in-the-wild processing.
- Useful as a negative baseline; do not expect it to give calibrated scores on unseen generators.
