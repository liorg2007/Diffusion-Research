# Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection
- **id:** lgrad-cvpr2023
- **Authors / venue / year:** Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Yunchao Wei / CVPR 2023, pp. 12105-12114 / 2023
- **Links:** CVF https://openaccess.thecvf.com/content/CVPR2023/html/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.html · arXiv: none found [unverified] · https://github.com/chuangchuangtan/LGrad
- **Family:** low-level (gradient maps from a frozen pretrained CNN)
- **Status:** read-full

## Summary (plain language)
Instead of classifying pixels, LGrad feeds a frozen pretrained CNN (best: the discriminator of a StyleGAN) the image, back-propagates to the input, and uses the resulting gradient map as an image-content-suppressed "artifact picture". A second ResNet-50 classifies that map. Trained on ProGAN only, it beat prior methods on unseen GANs, but it is GAN-only and is weak on JPEG and on diffusion.

## How it works
G = d sum(M(I)) / dI for a fixed transformation model M (Eq. 1-2); G is normalised to 0-255 and used as classifier input. Classifier: ImageNet-pretrained ResNet-50, Adam, lr 5e-4 decayed 20% every 10 epochs, batch 16, 100 epochs, last epoch reported (Sec. 4.2). Best transformation model in Table 5: StyleGAN-bedroom discriminator (mean Acc 86.3 / AP 92.7). Others: VGG16 76.0, ResNet50 73.4, CLIP-ResNet50 77.1, ViT 51.5.

## Training data
ForenSynths ProGAN (Wang et al.), 1/2/4-class settings plus a data-size study (4k-72k images, Table 6); face setting with 20,000 CelebA-HQ real + 20,000 ProGAN fakes. Images resized to 256x256 in the face experiments.

## Reported results
- Table 3 (ForenSynths, StyleGAN-bedroom transformation, Acc/AP): 1-class mean 86.3/92.7; 2-class 86.2/92.2; 4-class 86.1/91.5. Per generator (1-class): ProGAN 99.4, StyleGAN 96.0, StyleGAN2 93.8, BigGAN 79.5, CycleGAN 84.7, StarGAN 99.5, GauGAN 70.9, DeepFake 66.7 (Acc).
- Perturbed (Table 4, mean Acc/AP over the 8 sets): none 86.3/92.7; blur 78.5/83.2; cropping 85.0/91.9; JPEG 69.5/81.2; noise 69.1/79.4; combined 75.9/84.1. ProGAN column: JPEG 76.2 Acc vs 99.4 clean. (Perturbation parameters not extracted; see paper Sec. 4.) 
- Unseen diffusion: not in the paper (GAN-only). Fixed-threshold FPR / calibration: not reported.

## Known weaknesses / failure modes
- Authors: "may not perform well in detecting non-GAN generative models, such as deepfake" (Conclusion; DeepFake 66.7% Acc).
- JPEG costs ~17 points of mean Acc and noise ~17 (Table 4).
- Diffusion/modern generators: NPR paper (arXiv 2312.10461) Table 5 (1000-step diffusion incl. Midjourney/DALL-E): LGrad mean Acc 64.2 / AP 74.3; Table 6 overall 80.5 over 28 sets. SPAI (Table 1, 13 modern generators): LGrad avg AUC 65.9 (SD3 12.7, DALL-E 3 30.0, Firefly 42.0). B-Free (arXiv 2412.17671, Tables 5 and 7): LGrad bAcc 54.4 avg on Synthbuster/new generators/WildRF (Facebook 66.6, Reddit 57.8, Twitter 45.7), 46.3 on SynthWildX; authors argue LGrad and NPR show "non-uniform" behaviour, exploiting dataset biases.
- Needs a heavy transformation model (46.6M params in FreqNet's Table 3 comparison).

## Code + weights
https://github.com/chuangchuangtan/LGrad (files: CNNDetection, img2gad, img2gad_pytorch, train/transform scripts). No LICENSE file in the repo root (GitHub API reports none; all rights reserved by default; reuse permission unclear). Weight file names/hosting not confirmed [unverified].

## Verified?
Tables 3-6 copied from CVF PDF text extraction; later-paper numbers copied from their PDFs (SPAI, B-Free, NPR). License = absence of file in repo listing.

## Relevance to this project
- Content-suppressing gradient maps do not buy robustness to JPEG, and the method does not transfer to diffusion; treat as a GAN-era baseline.
