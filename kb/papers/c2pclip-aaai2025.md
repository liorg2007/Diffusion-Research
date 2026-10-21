# C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection
- **id:** c2pclip-aaai2025
- **Authors / venue / year:** Chuangchuang Tan, Renshuai Tao, Huan Liu, Guanghua Gu, Baoyuan Wu, Yao Zhao, Yunchao Wei / AAAI 2025 (Vol. 39, DOI 10.1609/aaai.v39i7.32772 per AAAI OJS search result) / 2025
- **Links:** arXiv 2408.09647 · https://ojs.aaai.org/index.php/AAAI/article/view/32772 · https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection
- **Family:** foundation-features
- **Status:** partially-verified (arXiv v2 HTML via fetch tool; AAAI PDF not opened)

## Summary (plain language)
The authors observe that CLIP detects fakes by recognising concept words like "deepfake". They append a fixed category word to every caption (e.g. "Deepfake" for fakes, "Camera" for reals), train CLIP's image encoder with LoRA and a contrastive loss against those prompts, and at test time discard the text side. Result: an unchanged-size CLIP ViT-L/14 detector with better unseen-generator accuracy.

## How it works
- Backbone CLIP ViT-L/14 (following UniFD). Captions produced with ClipCap + category common prompt; text encoder frozen; image encoder tuned with LoRA on q/k/v projections (r=6, alpha=6, dropout 0.8); loss = contrastive + alpha(=8.0) x classification loss. Inference: image encoder + classifier only (no extra parameters).
- Training: Adam lr 4e-4, batch 128, **1 epoch**. Resolution: for GenImage, images smaller than 224 are duplicated and cropped to 224; crop details for UniversalFakeDetect protocol not given -> **[unverified]**. Resize at test time: not stated.

## Training data
UniversalFakeDetect protocol: ProGAN 4-class (horse, chair, cat, car). GenImage protocol: SD v1.4 subset.

## Reported results
- UniversalFakeDetect, 19 subsets: mAcc 93.79 ("Deepfake, Camera" prompt) / 93.00 ("Trump, Biden") vs UniFD 81.38 and FatFormer 90.86 (Table 2); mAP 98.66 / 98.02 vs UniFD 90.14, FatFormer 98.16 (Table 1). Paper claims +12.41% acc and +8.52% mAP over UniFD. (The paper text swaps the two prompt-pair values relative to the tables; tables used here.)
- GenImage (train SD v1.4), mAcc (Table 3): 95.8 / 94.7 vs UnivFD 88.8, FatFormer 88.9; claimed +7.0% over UniFD.
- Perturbation robustness: **not evaluated** in the paper. FPR on fresh real photos: not reported.

## Known weaknesses / failure modes
- From paper: analysis of caption/word frequency is incomplete (only stated limitation).
- Later papers: B-Free (arXiv 2412.17671, via HTML fetch): Table 5 WildRF bAcc Facebook 54.4, Reddit 68.4, Twitter 57.4 (NLL 0.97/0.67/0.91); Table 7 GenImage 75.5, FakeInversion 59.6, SynthWildX 57.4 (bAcc avg 64.2); Table 11 FakeInversion AUC 74.1 -> 71.9 under social-network simulation (most stable of the CLIP-based methods, but at 55-60 bAcc). Zhou et al. (arXiv 2509.12995) Table 4 WebAIG-25: C2P-CLIP real acc 84.7, fake acc 31.0 (avg 57.9).
- Large gains concentrated on the curated ProGAN/GenImage protocols; in-the-wild accuracy is low.

## Code + weights
https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection ; **No LICENSE file** in repo root listing (GitHub API returned no license) -> treat as all-rights-reserved/unknown. Pretrained models, text links and dataset: Google Drive folder linked in README; CLIP weights from HF openai/clip-vit-large-patch14.

## Verified?
partially: arXiv HTML via fetch tool; venue from AAAI OJS listing; repo contents from GitHub API.

## Relevance to this project
- Shows text-side prompt tricks give large benchmark gains with LoRA only, but independent in-the-wild tests show only modest improvement over UnivFD.
- Calibration (low NLL in B-Free tables) was better than other CLIP-based methods, though accuracy remained ~55-60.
