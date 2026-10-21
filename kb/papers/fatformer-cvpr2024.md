# Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection
- **id:** fatformer-cvpr2024
- **Authors / venue / year:** Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Jingdong Wang, Yao Zhao / CVPR 2024 (pp. 10770-10780, DOI 10.1109/CVPR52733.2024.01024 per CVF search result) / 2024
- **Links:** arXiv 2312.16649 · https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.html · https://github.com/Michel-liu/FatFormer
- **Family:** foundation-features
- **Status:** partially-verified (arXiv v1 HTML read via fetch tool; CVF PDF not opened)

## Summary (plain language)
FatFormer adapts CLIP rather than freezing it: small "forgery-aware adapters" inserted between ViT stages extract local image-domain and frequency-domain (wavelet) forgery traces, and a "language-guided alignment" module trains the image features to match learned text prompts for real/fake. Trained on ProGAN only, it beats the frozen-CLIP linear probe (UnivFD) on unseen GANs and diffusion models.

## How it works
- CLIP ViT (Table 4 reports ViT-B/16 and ViT-L/14; the main variant is **[unverified]** from the fetched text) with: Forgery-Aware Adapter (image forgery extractor = 2 convs + ReLU; frequency extractor = DWT with inter-/intra-band attention) and Language-Guided Alignment (patch-based enhancer for prompts, text-guided interactor, augmented contrastive objective).
- Training: Adam, lr 4e-4, 25 epochs, batch 256. **Images resized to 256x256 then cropped to 224x224** (resize used in the pipeline).

## Training data
ProGAN, 2-class (chair, horse) or 4-class (car, cat, chair, horse) (Wang et al. protocol).

## Reported results
(as extracted from arXiv HTML; table refs from the paper)
- Unseen GANs, mean ACC/AP, 4-class (Table 1): FatFormer 98.4 / 99.7 vs UniFD 89.1 / 98.3. 2-class: 95.8 / 99.2 vs 88.1 / 97.8.
- Unseen diffusion models (PNDM, Guided, DALL-E, VQ-Diffusion, LDM, Glide), 4-class, mean ACC/AP (Table 2): 95.0 / 98.8 vs UniFD 85.4 / 94.6.
- Ablation (Table 3e): w/o FAA 95.3/99.2; w/o LGA 91.5/98.1; neither 66.6/74.3.
- Robustness on GAN set (Appendix A.3, ACC/AP): blur 90.7/98.1 (UniFD 78.1/93.0); crop 98.2/99.7 (88.9/98.1); JPEG 95.9/99.2 (88.4/97.7); noise 88.0/96.5 (82.6/93.9). Combined perturbations only in Fig. 5.
- Note: C2P-CLIP's UniversalFakeDetect Table 2 lists FatFormer at 90.86 mAcc (a different test protocol: 19 subsets), so the 95/98 numbers are on the GAN/DM sets of the FatFormer protocol only.
- No FPR on fresh real photos; fixed-threshold details not extracted -> **[unverified]**.

## Known weaknesses / failure modes
- From paper (Conclusion): still room to improve on some diffusion models, e.g. Guided.
- Later papers: B-Free (arXiv 2412.17671, via HTML fetch): Table 5 WildRF bAcc Facebook 54.7, Reddit 69.5, Twitter 54.8; Table 7 GenImage 61.5, FakeInversion 59.7, SynthWildX 53.3 (bAcc, avg 58.2); Table 11 FakeInversion AUC 68.2 -> 48.7 under social-network simulation (largest drop of CLIP-based methods listed). Also Cozzolino/C2P-CLIP/AIDE comparisons: C2P-CLIP Table 3 GenImage FatFormer mAcc 88.9.
- Uses frequency (DWT) branch and 256-resize + 224 crop -> likely brittle to resizing/re-compression despite crop/JPEG robustness on GAN set.

## Code + weights
https://github.com/Michel-liu/FatFormer ; License Apache-2.0 (GitHub API spdx_id). Checkpoints linked from README (Baidu, OneDrive, Google Drive). CLIP ViT-L/14 weights must be downloaded from OpenAI (README).

## Verified?
partially: numbers from arXiv v1 HTML via fetch tool, repo license from GitHub API, venue from CVF listing. Not opened from CVF PDF.

## Relevance to this project
- Adapter-based tuning raises unseen-diffusion accuracy by ~10 points over linear probe on the standard ProGAN protocol, but that benchmark is saturated/format-biased; independent in-the-wild tests (B-Free) show it near chance.
- Illustrates risk of explicitly adding low-level (wavelet) branches to a CLIP model: gains on curated sets, weak under social-network processing.
