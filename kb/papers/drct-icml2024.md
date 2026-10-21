# DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images
- **id:** drct-icml2024
- **Authors / venue / year:** Baoying Chen, Jishen Zeng, Jianquan Yang, Rui Yang / ICML 2024 (Spotlight per repo README), PMLR 235:7621-7639 / 2024
- **Links:** PMLR https://proceedings.mlr.press/v235/chen24ay.html · OpenReview oRLwyayrh1 · arXiv: none found · code https://github.com/beibuwandeluori/DRCT
- **Family:** reconstruction (reconstruction used for training-data construction, not as an inference feature)
- **Status:** read-full

## Summary (plain language)
DRCT takes real images and re-renders them through a Stable Diffusion model (so they look the same but carry the generator's fingerprint). A standard detector is then trained to separate real from these "hard fake" reconstructions with an extra contrastive loss. The detector at test time is an ordinary classifier - no diffusion at inference.

## How it works
Reconstruction stage: encode with SD VAE, noise + DDIM denoise, decode (Sec 3.2); the released script uses the SD inpainting pipeline with an all-zero mask and empty prompt, 50 steps default (data/diffusion_rec.py: steps=50, guidance 7.5). Training stage: 4 sample types - real, real-reconstructed, fake, fake-reconstructed; labels real=0, other three=1. Loss = lambda*margin contrastive (m=1.0, lambda=0.3) + (1-lambda)*BCE (Eq. 3-5). Backbones: ConvNeXt-base (Conv-B, all weights tuned) and UnivFD CLIP ViT-L/14 (frozen, last FC tuned). Input 224 (center crop); augmentations: flip, Gaussian noise, blur, rotation, JPEG random quality, brightness/contrast, grid dropout.
**Compute per image:** inference = one forward pass of the classifier (no diffusion). Training-data cost: 1 diffusion reconstruction (50 steps default) per training image; ablation (App. Table 8) shows steps 10/20/50/70 give similar accuracy on SDv1-DR, so 10 steps suffices for seen reconstructor.

## Training data
DRCT-2M: 2M images, 16 SD-family models (LDM, SDv1.4, SDv1.5, SDv2, SDXL, SDXL-Refiner, SD-Turbo, SDXL-Turbo, LCM-SDv1.5, LCM-SDXL, 3 ControlNet variants, 3 DR variants), real = MSCOCO, 256-1024 px, plus DRCT-2M-Wild (136k images from 8 online platforms). Main training: SDv1.4 (or SDv2) fakes + MSCOCO reals + their SDv1/SDv2 reconstructions; also GenImage/SDv1.4.

## Reported results
- Unseen generators, DRCT-2M (Table 1, ACC, trained on SDv1.4; threshold 0.5): Conv-B baseline avg 79.11 -> DRCT/Conv-B(SDv1) 90.79, DRCT/Conv-B(SDv2) 96.55; UnivFD 83.46 -> DRCT/UnivFD(SDv1) 90.49, (SDv2) 91.35. DIRE avg 71.23 (SDv1) / 72.55 (SDv2). Weak spots: DRCT/Conv-B(SDv1) on SDXL 83.87, SDXL-Turbo 70.04, SDXL-DR 75.39.
- GenImage trained on SDv1.4 (Table 2, ACC): Conv-B 74.98 -> DRCT/Conv-B 82.08; UnivFD 79.45 -> DRCT/UnivFD 89.49. DRCT/UnivFD: Midjourney 91.50, ADM 79.42, GLIDE 89.18, VQDM 90.03, BigGAN 81.67. (README errata: actual BigGAN ACC for DRCT/Conv-B is 59.81%.) Abstract/Intro: UnivFD 67.73 -> 87.95 avg (Fig. 1, differing protocol).
- In-the-wild (DRCT-2M-Wild, Table 3, ACC): DRCT/Conv-B 82.06 (SDv1) / 93.16 (SDv2); DRCT/UnivFD 87.23 / 96.90; all non-DRCT baselines < 65 (UnivFD 63.55, CNNSpot 58.89, DIRE 35.05/56.43).
- Perturbed (Fig. 5, resize 0.5-1.5 and JPEG QF 60-100, numbers only in figure): text says DRCT/Conv-B keeps ACC "up to 99%"; Conv-B more robust than UnivFD variant. Trained with the same JPEG/blur/noise augmentation for all compared methods.
- Ablation (Table 4, GenImage avg ACC, Conv-B): fake only 68.98; + real-rec 73.53; both 76.03; + fake rec 76.99; + contrastive loss 83.53.
- Metric is ACC at 0.5 threshold; no AP/AUC or FPR-at-fixed-TPR in the extracted text; no phone/social-media real photos.

## Known weaknesses / failure modes
- Paper: improvement on non-diffusion (GAN) images is less marked; only global generation evaluated, not local edits.
- Best numbers use a reconstructor (SDv2) matched to a newer family; with SDv1 reconstructor, SDXL-family and DR sets stay at 70-84 (Table 1).
- Real reconstructions are mostly SD/LDM-VAE fingerprints; may learn "VAE-decoder trace" cues, which would not transfer to non-LDM or pixel-space generators (hypothesis, not from the paper).
- Real images are all COCO (JPEG); fakes mostly PNG-generated - format bias is a risk, mitigated by JPEG augmentation (not verified for matched formats).

## Code + weights
https://github.com/beibuwandeluori/DRCT. No LICENSE file found (GitHub license API returned none). Pretrained weights: pretrained.zip on ModelScope (checkpoints like convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth, clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth); DRCT-2M on ModelScope (BokingChen/DRCT-2M). Builds on CNNDetection, FreDect, Gram-Net, DIRE, UnivFD code.

## Verified?
Yes for numbers: extracted text from the PMLR PDF (Tables 1-4, Sec 3-5, App. step ablation) and README. License: GitHub API only.

## Relevance to this project
- Best practical bet in this family: cheap inference, ready weights, JPEG/resize augmentation, wild test set (but ACC only, 0.5 threshold).
- Idea to borrow: use autoencoder/inpaint reconstructions of our own real photos as hard negatives/aligned pairs - and note the 10-step ablation makes this cheap enough for CPU/small GPU.
