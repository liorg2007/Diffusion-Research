# LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection
- **id:** lare2-cvpr2024
- **Authors / venue / year:** Yunpeng Luo, Junlong Du, Ke Yan, Shouhong Ding / CVPR 2024 / 2024
- **Links:** arXiv 2403.17465 (v4, Feb 2025) · code https://github.com/luo3300612/LaRE
- **Family:** reconstruction
- **Status:** partially-verified

## Summary (plain language)
Instead of fully inverting and regenerating an image (slow, like DIRE), LaRE adds noise to the Stable Diffusion latent once and measures how well the denoiser predicts that noise (a one-step error map). This map is fused with image features through an attention module (EGRE) in a trained classifier, and is ~8x faster than DIRE.

## How it works
Image resized to 256 -> SD v1.5 VAE latent (32x32x4) -> noise added at timestep t=200, one U-Net step, squared error averaged over e=4 noise samples, prompt "a photo" (Eq. 5-7) -> LaRE map. EGRE: spatial attention with LaRE added to attention logits + channel gating, concatenated and an FC layer. Backbone CLIP-pretrained ResNet-50, 224 input, bs 48, lr 1e-4, one V100. One model per GenImage generator subset. t stable in [150, 300] (Fig. 6).
**Compute per image:** e=4 U-Net forward passes (SD1.5, 32x32 latent) + VAE encode + the ResNet-50. Paper: DIRE takes >2 s/image on V100 and LaRE is ~8x faster (Fig. 1c); no absolute LaRE timing given.

## Training data
GenImage (2,681,167 images: 1,331,167 real / 1,350,000 fake, 8 generators); official split 2,581,167 train / 100,000 val.

## Reported results
- GenImage Table 1 (ACC only): Midjourney 66.4, SDv1.4 87.3, SDv1.5 87.1, ADM 66.7, GLIDE 81.3, Wukong 85.5, VQDM 84.4, BigGAN 74.0; avg 79.1. Reproduced DIRE avg 67.2. Claimed gains up to 11.9% ACC / 12.1% AP over best SOTA.
- Internal inconsistency to note: ablation Table 2 full model "Model E" avg 85.9 ACC / 99.8 AP; Table 3 (trained on SDv1.5, averaged over 8 subsets) EGRE 84.5 / 99.5, LaRE alone 66.2/68.3, image only 73.1/92.3. Protocol differences between tables are not explained in the extracted text **[unverified]**.
- No JPEG/blur robustness results found in the content read. No fixed-FPR/calibration reported.

## Known weaknesses / failure modes
- Paper: LaRE alone is insufficient (information loss, 66.2 ACC); cross-generator generalization remains hard and varies by family (Fig. 4); degrades outside t in [150,300].
- Per-generator training (not one universal model); GenImage has known JPEG/size biases (Grommelt et al., arXiv 2403.17608), so GenImage-trained numbers may partly reflect format cues. Later papers (not checked) may re-evaluate it.

## Code + weights
https://github.com/luo3300612/LaRE - Apache-2.0 (listed on repo page; LICENSE file present). Pretrained weights: not mentioned on README **[unverified]**. README TODO says feature-extraction/training code release; extraction built on DIFT, training on LASTED.

## Verified?
Partially: via fetch-tool summary of arXiv HTML; license from repo page. Table references as given.

## Relevance to this project
- Shows one-step latent error is a cheaper signal than DIRE, but still needs SD1.5 U-Net passes (4/image) plus training on GenImage.
- Gains are modest cross-generator (Midjourney 66.4, ADM 66.7) - not a solution for unseen generators.
