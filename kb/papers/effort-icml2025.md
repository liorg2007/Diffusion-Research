# Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection (Effort)
- **id:** effort-icml2025
- **Authors / venue / year:** Zhiyuan Yan, Jiangming Wang, Peng Jin, Ke-Yue Zhang, Chengchun Liu, Shen Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, Li Yuan / ICML 2025 (PMLR vol. 267, pp. 70268-70288 per mlanthology search result) / 2025
- **Links:** arXiv 2411.15633 · https://icml.cc/virtual/2025/poster/45843 · https://github.com/YZY-stack/Effort-AIGI-Detection
- **Family:** foundation-features
- **Status:** partially-verified (arXiv v4 HTML via fetch, last ~8k characters including Limitation section not read)

## Summary (plain language)
The paper argues that fine-tuning a pretrained CLIP for fake detection collapses its feature space to a low-rank, overfit subspace. Effort splits each attention weight matrix by SVD, freezes the top-r singular components (pretrained knowledge) and trains only the small residual, with orthogonality and singular-value constraints. Only 0.19M parameters are trained.

## How it works
- CLIP ViT-L/14 (307M params); SVD applied to the self-attention linear layers; frozen principal part + trainable residual of rank n-r; best n-r = 1 for synthetic detection (Table 6 ablation, 95.19 mAcc). Input 224. Crop/resize policy: not extracted -> **[unverified]**.
- Training for synthetic images: Adam, lr 2e-4, batch 48.

## Training data
Synthetic: ProGAN 20-class real/fake images (UniversalFakeDetect protocol) for Table 2/3; SDv1.4 for GenImage (Table 9). Deepfake variant: FaceForensics++ c23 (not relevant here).

## Reported results
- UniversalFakeDetect (19 subsets, Tables 2-3): mean mAP 99.41, mean Acc 95.19 (vs UniFD 81.38 acc / C2P-CLIP 93.79 from C2P-CLIP's tables; cross-paper comparison, same protocol).
- GenImage (train SDv1.4, Table 9): mean Acc 91.1, +1.6% over DRCT.
- Robustness (Appendix A.3 / Fig. 8, deepfake setting): block-wise distortion, contrast, JPEG at five levels; text says generally higher AUC than other methods; **no numbers extracted -> [unverified]**; this robustness study appears to be on the deepfake (face) benchmark, not AIGI.
- Fixed-threshold FPR on fresh real photos: not reported.

## Known weaknesses / failure modes
- Paper: limitation section (Appendix D) exists but was not read -> **[unverified]**. Impact statement notes misuse risk.
- Later papers: no independent in-the-wild evaluation found among B-Free (explicitly not evaluated), Community Forensics, AIDE (published earlier). The authors' own repo releases a Chameleon-trained checkpoint, suggesting default checkpoints were not enough for Chameleon (inference, not stated) -> **[unverified]**.
- Evaluated mostly on ProGAN-protocol and GenImage, both known to have format/size biases.

## Code + weights
https://github.com/YZY-stack/Effort-AIGI-Detection ; README badge states CC BY-NC 4.0 (non-commercial); **no LICENSE file in repo root** (GitHub API listing). Checkpoints on Google Drive: FF++-trained, GenImage (SDv1.4)-trained and Chameleon (SDv1.4)-trained; a `UniversalFakeDetect_Benchmark/` folder reproduces Table 2 (ProGAN-trained checkpoint location not confirmed -> **[unverified]**). CLIP backbone from Hugging Face/OpenAI.

## Verified?
partially: numbers from arXiv v4 HTML via fetch (first 100k of 108k characters); venue from ICML/mlanthology search; license from README.

## Relevance to this project
- Best-in-class benchmark numbers (95.19 acc on UFD) from a rank-1 CLIP update; shows regularising how much of the foundation model is changed is key.
- Non-commercial license: usable for research baselines only.
- Needs independent in-the-wild test; no JPEG/resize/phone numbers for AIGI setting are available from the paper text.
