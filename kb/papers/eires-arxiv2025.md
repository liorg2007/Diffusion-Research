# EIRES: Training-free AI-Generated Image Detection via Edit-Induced Reconstruction Error Shift
- **id:** eires-arxiv2025
- **Authors / venue / year:** Wan Jiang, Jing Yan, Xiaojing Chen, Ling Shen, Chenhao Lin, Yunfeng Diao, Richang Hong / arXiv preprint (no venue stated) / 2025
- **Links:** arXiv 2510.25141 (v2) · https://arxiv.org/abs/2510.25141 · code: "in supplementary material", no URL
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
A single VAE reconstruction error is hard to threshold. EIRES edits the image first (add object, erase, semantic replace) with an external image-editing model and measures how much the VAE reconstruction error shifts; real and generated images shift differently. Training-free, but needs an image editor on top of the VAE.

## How it works
Edits: Add, Erase, SemR (Multi-Edit, Table 4); default editor ERNIE-iRAG-Edit, FLUX.1 as alternative (Table 5). Reconstruction: SD v1.5 AE (ensemble, SDv2.1, Kandinsky 2.1 ablated, Table 6), LPIPS v2 (VGG 2-layer) distance (Table 7: 93.2 AP vs MSE 87.9, PSNR 79.6). Score = max over edits of the error change, threshold calibrated on real validation images. 512x512 input.
- **Weights needed:** SD VAE + LPIPS-VGG AND a large image-editing generative model (ERNIE-iRAG-Edit via API/compute, or FLUX.1) - far beyond our 0.2 GB budget.
- **Passes per image:** one editing-model run per edit type (3) + AE passes on original and each edit; no timing reported.

## Training data
None.

## Reported results
- **Unseen-generator results (GenImage, Table 2):** mean ACC 0.876 / AUROC 0.941 / AP 0.938 vs best training-free baseline HFI 0.835 / 0.929 / 0.914. Per-generator ACC: ADM 0.751, BigGAN 0.998, GLIDE 0.959, Midjourney 0.720, SD1.4 0.972, SD1.5 0.974, VQDM 0.699, Wukong 0.935. FLUX, SD3, DALL-E not evaluated as generators (FLUX only as editor; editor swap changes SD1.4 ACC 0.972 -> 0.896, Table 5).
- **Laundering results:** JPEG and crop only, Figure 5 (no numbers): AP "drops only slightly" with a dip at q=70. Resize not evaluated.
- **FPR on real photos:** not reported.
- No runtime numbers.

## Known weaknesses / failure modes
Assumes real/fake respond differently to edits; weaker on stylized content and hard-to-reconstruct real images; image-level only; cost of editing models. Midjourney and VQDM have the lowest ACC despite text calling them "strong".

## Code + weights
- **License:** paper CC BY-NC-ND 4.0; no public code found.

## Verified?
Partially: arXiv HTML v2 fetch; numbers from Tables 2-7; robustness figure values not extracted.

## Relevance to this project
- Shows that a single-AE residual can be made more separable by perturbing the input, but the editor requirement makes it infeasible for our budget; the cheap analogues are HFI (blur) and RDD (rotation/low-pass).
