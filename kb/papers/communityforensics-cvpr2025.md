# Community Forensics: Using Thousands of Generators to Train Fake Image Detectors
- **id:** communityforensics-cvpr2025
- **Authors / venue / year:** Jeongsoo Park, Andrew Owens (Univ. of Michigan). CVPR 2025, pp. 8245-8257 (per GitHub README).
- **Links:** arXiv 2411.04125 · https://arxiv.org/abs/2411.04125 · code https://github.com/JeongsooP/Community-Forensics · weights https://huggingface.co/OwensLab/commfor-model-384 and .../commfor-model-224
- **Family:** data-bias-benchmarks
- **Status:** partially-verified

## Summary (plain language)
Rather than a cleverer model, the authors scale generator diversity: 4,803 generators (mostly Hugging Face latent-diffusion checkpoints) provide 2.7M fakes, paired with 2.7M real images. A plain ViT-S trained on it generalises well to 21 held-out generators including commercial ones (Midjourney, DALL-E 3, Firefly, FLUX, Imagen 3...).

## How it works
ViT-S pretrained with CLIP objective (LAION-2B / IN-21K / IN-1K), linear head + sigmoid, fine-tuned end-to-end (freezing hurts, Fig 6a). Input 224^2 or 384^2. AdamW lr 2e-5, wd 1e-2, batch 512, 52K iterations; modified RandAugment incl. JPEG, resize, crop, flip, rotate, shear, pad, cutout.

## Training data
5.4M images: 2.7M fake + 2.7M real. Fakes: 4,763 HF models (~403 images each, 1.9M) + 19 manual models (774K). Real: LAION, ImageNet, COCO, FFHQ, CelebA, MetFaces, AFHQ, Forchheim, IMD2020, LandscapesHQ, VISION. Most generators are latent diffusion (stated limitation).

## Reported results
- Table 2 (High-res ViT-S; mAP / acc): comprehensive eval 0.987 / 89.2%; public eval 0.994 / 91.2%; Synthbuster 0.974 / 90.8%; GenImage 0.998 / 95.7% (in-distribution for the GenImage baseline); mean 0.986 / 92.3%. 224^2 model: 0.971 mAP / 86.1% on comprehensive set.
- Baselines on comprehensive set: Wang et al. 0.537 mAP / 51.3%; Ojha et al. (UnivFD) 0.592 / 54.0%; GenImage-trained 0.912 / 81.8%; RED140 0.764 / 56.2%.
- Scaling: gains flatten beyond ~1,000 generators; 1,000 random models beat 10 popular ones at equal image count (Fig 1/4, Sec 4.3).
- Robustness: text claims better robustness to JPEG, blur, resize etc.; Fig 7 is qualitative, no numbers.
- Accuracy is at a fixed threshold (mAcc averaged per generator); no FPR on fresh independent real photos.

## Known weaknesses / failure modes
Authors: heavy latent-diffusion bias; error rates too high for critical use; dataset should not train classifiers for direct real-world use. Real data come from a fixed set of public datasets (LAION/ImageNet etc.) - real-source shortcuts possible.

## Code + weights
Code MIT (repo). Weights on HF OwensLab, MIT licence, safetensors, ViT, 21.8M params (384 model card); publicly downloadable. Datasets: OwensLab/CommunityForensics (cc-by-4.0, research only, ~1.08 TB), -Small (cc-by-nc-sa-4.0, 278K+278K, ~265 GB), -Eval (cc-by-nc-sa-4.0, ~206 GB). See kb/datasets/communityforensics-eval.md.

## Verified?
Numbers from arXiv HTML via fetch tool (the tool noted table layout was garbled and inferred column mapping); the CommunityForensics-Eval card's comparison table reproduces 0.987/0.892 (paper) and 0.971/0.861 for the 384/224 models, which cross-checks the comprehensive-set numbers. Licences read from HF cards / API.

## Relevance to this project
- A pretrained, permissively-licensed (MIT) detector baseline that can be run on a Mac (ViT-S, 21.8M params).
- Evidence that generator count, not just artifacts, drives unseen-generator generalisation.
