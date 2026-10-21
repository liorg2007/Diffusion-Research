# Any-Resolution AI-Generated Image Detection by Spectral Learning (SPAI)
- **id:** spai-cvpr2025
- **Authors / venue / year:** Dimitrios Karageorgiou, Symeon Papadopoulos, Ioannis Kompatsiaris, Efstratios Gavves / CVPR 2025, pp. 18706-18717 / 2025
- **Links:** arXiv 2411.19417 · https://arxiv.org/abs/2411.19417 · CVF https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html · https://github.com/mever-team/spai
- **Family:** low-level (spectral model of real images; borderline with reconstruction/foundation features since it uses a ViT backbone)
- **Status:** read-full (CVF PDF); Fig. 3 numbers not extracted

## Summary (plain language)
A ViT is first pre-trained (by others, MFM) to reconstruct masked frequency bands of real images, so it models the "normal" spectrum of real photos. SPAI measures how badly that model reconstructs an image's spectrum and uses it, with an attention module that handles any resolution without resizing, to flag fakes. Trained on one Latent Diffusion set, it reports the best average AUC on 13 modern generators and more robustness to online perturbations than competitors.

## How it works
Frozen spectral model G = ViT-B/16 pre-trained by MFM (Xie et al.) with masking radius r=16. Spectral Reconstruction Similarity (SRS) between original and frequency-masked-reconstructed features; Spectral Context Attention (SCA) over patches of the full-resolution image (224x224 patches, K_training=4 random views at training; all patches at inference); D=1024. Training: 35 epochs, lr 5e-4, cosine decay, 5-epoch warmup, AdamW, batch 72, augmentation: random resize, crop, rotation, Gaussian blur and noise, JPEG; one L40S GPU (Sec. 4.1). Ablation: removing spectral pretraining drops AUC to 52.5 (Table 3).

## Training data
180K 256x256 fake images from Latent Diffusion + 180K real from Corvi et al. (arXiv 2211.00680).

## Reported results
- Table 1 (AUC averaged over 5 real-image sources; 1k images per generator, MJv6.1 631): SPAI avg 91.0 vs RINE 85.5, DMID 83.5, PatchCraft 80.4, UnivFD 67.3, LGrad 65.9, CNNDet. 55.5, FreqDet. 57.1, NPR 45.4. SPAI per generator: Glide 90.2, SD1.3 99.6, SD1.4 99.6, Flux 83.0, DALL-E 2 91.1, SD2 96.5, SDXL 97.4, SD3 75.9, GigaGAN 85.4, MJv5 94.5, MJv6.1 84.0, DALL-E 3 90.2, Firefly 96.0.
- Real sources (1k each): RAISE (DSLR), FODB (smartphone), ImageNet, COCO and Open Images test sets (generic online images). CORRECTED: was "RAISE and FODB (DSLR), plus smartphone sources (three more)"; PDF p.6 says only FODB is smartphone, the other three are ImageNet/COCO/OpenImages.
- Perturbed: Fig. 3 plots average AUC under JPEG and WebP (Q=85/70/50), Gaussian blur (k=3/5/7), noise (sigma 1/3/5) and resizing (85/70/50%) on two real sources (smartphone, DSLR); compared only against the two best competitors; text claims SPAI is best in all cases. Exact values only in the figure [unverified].
- Ablations (Table 3): without JPEG compression augmentation and distortions, AUC falls (89.1 without JPEG; 87.7 without both JPEG and noise/blur, vs 91.0). MFM backbone 91.0 vs CLIP ViT-B/16 87.6 and DINOv2 87.5 (Table 2).
- Fixed-threshold FPR / calibration: not reported (AUC only).

## Known weaknesses / failure modes
- Authors' limitation: compression and noisy channels remove the subtle spectral information any signal-based detector relies on; this "constrains what any detector ... can detect". Fig. 5 shows failures on derivative AI-generated images (e.g. edited).
- Weakest on SD3 (75.9), GigaGAN 85.4, Flux 83.0.
- Robustness comparison is against only two competitors; absolute robust AUC values not verified.
- Independent later evaluation: not located [unverified].

## Code + weights
https://github.com/mever-team/spai . README: source code and model weights released under Apache 2.0 (LICENSE file header = Apache License 2.0); weights checkpoint via Google Drive link in README; requires the separately downloaded MFM ViT-B/16 pretrained model (`mfm_pretrain_vit_base.pth`) which may carry its own licence. Third-party downloaded data keep their own licences.

## Verified?
Table 1-3 numbers copied from CVF PDF text; license from README and raw LICENSE.
(verified vs arXiv PDF p.6, 2026-10-21) Table 1: NPR avg AUC 45.4, LGrad 65.9, SPAI 91.0 over 13 generators (RINE 85.5, DMID 83.5, PatchCraft 80.4, UnivFD 67.3, CNNDet 55.5, FreqDet 57.1 and all SPAI per-generator values also match).

## Relevance to this project
- Best current evidence that a spectral-prior detector can be robust and resolution-agnostic, while old low-level detectors (NPR 45.4, LGrad 65.9) collapse on the same 13 generators.
- Trains on a single LDM set and uses JPEG/blur/noise augmentation; the relevant open question for us is robustness at realistic Q and on phone/screenshots (only partly covered by Fig. 3).
