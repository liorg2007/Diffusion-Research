# DIRE for Diffusion-Generated Image Detection
- **id:** dire-iccv2023
- **Authors / venue / year:** Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, Houqiang Li / ICCV 2023 (venue per task brief; the arXiv abstract page itself lists no venue **[unverified]** against the CVF page) / 2023
- **Links:** arXiv 2303.09295 · (ICCV open-access page not fetched) · code https://github.com/ZhendongWang6/DIRE
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Pass an image through a pretrained diffusion model (invert to noise, then regenerate) and look at the pixel-wise difference to the input. The authors observe generated images are reconstructed well (small residual) while real images are not. A ResNet-50 is trained on these residual images ("DIRE") to say real vs fake.

## How it works
Input image (256x256) -> DDIM inversion to noise -> DDIM reconstruction using an unconditional ADM trained on LSUN-Bedroom -> DIRE = |x - recon(x)| (Eq. 9) -> ResNet-50 binary classifier (BCE). Default S=20 DDIM steps for inversion and for reconstruction (Table 4 tests 5/10/20/50; more steps = higher accuracy). Taking the absolute value matters: without it SD-v1 ACC/AP falls 99.7/100 -> 87.0/93.0 (Table 6). Training crops 224 with random flip; test center crop. SD outputs resized to 256 bicubic.
**Compute per image:** ~2 x 20 = 40 U-Net (ADM, pixel-space 256x256) forward passes. The DIRE paper reports no timing; LaRE2 (CVPR 2024) states DIRE takes >2 s/image on a Tesla V100 (lare2-cvpr2024).

## Training data
Main setting: ADM subset of LSUN-Bedroom, 40k train / 1k val / 1k test per generator. Benchmark DiffusionForensics (Table 1): LSUN-Bedroom unconditional ADM, DDPM, iDDPM, PNDM; text-to-image LDM, SD-v1, SD-v2, VQ-Diffusion; ImageNet: conditional ADM (50k) and SD-v1 (10k test). Real: LSUN-Bedroom, ImageNet.

## Reported results
- In-distribution (Table 2): DIRE (ADM generator/reconstructor) averages 99.9% ACC / 100% AP over the 8 LSUN-Bedroom generators; retrained CNNDet*/Patchforensics*/F3Net* 86.5-91.6% ACC; ProGAN/FF++-trained baselines ~51-59% ACC.
- Cross-dataset (Table 3): train LSUN-Bedroom ADM, test ImageNet ADM 90.2 ACC / 97.9 AP; test ImageNet SD-v1 97.2 / 99.8.
- Perturbed: text states no performance drop under Gaussian blur (sigma 1,2,3) and JPEG (q 65, 30), while retrained CNNDet*/Patchforensics* degrade sharply under JPEG (no numbers extracted **[unverified]**).
- No FPR-at-fixed-threshold or fresh-real-photo calibration reported.
- "Unseen" here is mostly other diffusion models in the same scene domain (bedrooms) and same family; no phone photos / social media.

## Known weaknesses / failure modes
- **Format bias (key):** Ricker et al., AEROBLADE (CVPR 2024, arXiv 2401.17879) reproduced DIRE and report it suspect: DIRE training data likely stored real images as JPEG and generated ones as lossless PNG; their uncompressed DIREs were almost all classified as "generated". On their benchmark DIRE gets AP ~0.46-0.51 and TPR@5%FPR 0.000 (Table 3 of AEROBLADE, as summarized by fetch tool; exact per-generator values **[unverified]**).
- Related evidence that dataset-level JPEG/size confounds inflate detector scores: Grommelt et al., "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets" (arXiv 2403.17608, ECCV 2024 workshop) - studied on GenImage, not DIRE specifically; they report >11 pts cross-generator change for ResNet50/Swin-T after removing JPEG/size bias (from abstract summary).
- DRCT (ICML 2024) re-trained DIRE with an SD reconstruction: on GenImage trained on SDv1.4, DIRE avg ACC 71.24, with Midjourney 50.40, ADM 52.32, VQDM 50.10, BigGAN 49.99 (Table 2); on DRCT-2M-Wild avg 35.05 (SDv1 reconstructor) (Table 3).
- Only one reconstruction model (ADM/LSUN-Bedroom) in main experiments; no limitation section in the paper. Expensive (many full U-Net passes).

## Code + weights
https://github.com/ZhendongWang6/DIRE (archived 2025-07-08, read-only). No LICENSE file found (GitHub license API returned none; README shows none) - treat as no explicit license. Pretrained DIRE model and DiffusionForensics are on BaiduDrive / RecDrive (password "dire"); file names not listed. Built on guided-diffusion and CNNDetection.

## Verified?
Partially: numbers taken from a fetch-tool summary of the arXiv HTML (Tables 1-6 as cited); not cross-checked against the PDF. License: GitHub API + README only.

## Relevance to this project
- Cautionary tale: very high in-distribution numbers can come from format artifacts; always match JPEG/PNG state between real and fake.
- Too costly (40 pixel-space U-Net passes) for an Intel CPU / 4 GB GPU at scale; useful mainly as a baseline.
