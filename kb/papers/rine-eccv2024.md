# Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection
- **id:** rine-eccv2024
- **Authors / venue / year:** Christos Koutlis, Symeon Papadopoulos (CERTH-ITI) / ECCV 2024 (venue per task brief; the arXiv PDF text itself does not state it -> **[unverified]** from primary text) / 2024
- **Links:** arXiv 2402.19091 · https://arxiv.org/abs/2402.19091 · https://github.com/mever-team/rine
- **Family:** foundation-features
- **Status:** read-full

## Summary (plain language)
UnivFD uses only CLIP's last layer, which mostly encodes semantics. RINE takes the CLS token from every intermediate Transformer block of a frozen CLIP ViT-L/14, learns a weighting over blocks, and feeds the result to a small MLP. Low-level information in early/mid blocks helps detect fakes from unseen generators. Training takes about 1 epoch (~2-8 min).

## How it works
- Frozen CLIP image encoder; CLS tokens from all blocks stacked (K); a "Trainable Importance Estimator" (TIE) produces per-block importance weights (Q1, Q2 modules); weighted sum -> MLP head (two d'xd' ReLU layers + d'x1). An additional supervised contrastive loss (factor xi) pulls same-class features together.
- Best configs (Table 1): ViT-L/14 backbone in all of 1-, 2-, 4-class settings; trainable params 10.52M / 0.28M / 6.32M; 2 / 4 / 8 minutes.
- Training: batch 128, lr 1e-3, 1 epoch, Adam. Blur + JPEG augmentation with prob 0.5, random crop 224, flip. Validation/test: **center crop 224x224, no resizing** ("known to eliminate synthetic traces", Sec. 4). Accuracy computed at fixed threshold 0.5, no calibration.

## Training data
ProGAN images from 1 (horse), 2 (chair, horse) or 4 (car, cat, chair, horse) LSUN classes (Wang et al. protocol). A separate model trained on LDM data (from Corvi et al. DMimageDetection) is released.

## Reported results
- In-distribution/unseen mix, 20 test sets (ProGAN, StyleGAN(1/2), BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake, SITD, SAN, CRN, IMLE, Guided, LDM x3, Glide x3, DALL-E), ACC avg (Table 2): RINE 1-class 90.3, 2-class 87.7, 4-class 91.5 vs UFD (20-class) 80.9, LGrad 78.3-79.7, DMID 77.6. AP avg (Table 4 "full"): 98.6 / 98.7 / 98.8.
- Ablation (Table 4, ACC all generators, avg of 3 settings): full 89.8; without intermediate blocks (last layer only = UFD-like) 80.8; without TIE 87.3; without contrastive loss 88.4. Non-GAN generators only: full 86.6 vs w/o intermediate 75.5.
- Perturbations (Table 5, 1-class model, ACC avg over 20 sets, each perturbation applied with prob 0.5): blur 83.9, crop 88.8, JPEG compress 86.7, noise 85.6; combination "considerably reduces performance" (text; numbers not extracted -> **[unverified]**).
- Commercial diffusion, Synthbuster, ten-crop (Table 6, ACC/AP): ProGAN-trained: DALL-E 2 64.6/86.7; DALL-E 3 21.1/30.7; SD1.3 66.3/91.3; SD1.4 65.8/91.4; SD2 49.0/66.8; SDXL 53.3/73.4; Glide 42.6/55.7; Firefly 70.8/99.3; Midjourney 34.2/39.5. LDM-trained: DALL-E 2 89.8/96.2; DALL-E 3 47.2/32.5; SD1.3 96.4/100.0; SD2 93.5/98.5; SDXL 96.3/99.8; Firefly 85.3/93.4; Midjourney 92.4/97.4.
- No FPR on fresh real photos reported.

## Known weaknesses / failure modes
- From paper: ProGAN-only training fails on commercial generators (DALL-E 3 ACC 21.1, Midjourney 34.2, Table 6); training-data origin matters hugely (diffusion-trained jumps to 90%+). Performance dropped when training longer than 1 epoch in the 1- and 4-class settings (Sec. 5.4). Hyper-parameters chosen by test-set ACC+AP (Sec. 5.1) -> optimistic.
- Later papers: B-Free (arXiv 2412.17671, via HTML fetch) Table 5: RINE WildRF bAcc Facebook 52.8, Reddit 67.7, Twitter 56.0; Table 7 GenImage 69.1, FakeInversion 63.6, SynthWildX 56.2 (bAcc); Table 11 FakeInversion AUC 69.6 -> 62.4 under social-network simulation; B-Free Table 4: RINE Synthbuster bAcc 65.0 with ProGAN training vs 83.2 with authors' dataset. Numbers not directly re-verified in the B-Free PDF.

## Code + weights
https://github.com/mever-team/rine ; License: Apache-2.0 (GitHub API spdx_id). Checkpoints in repo `ckpt/`: model_1class_trainable.pth, model_2class_trainable.pth, model_4class_trainable.pth, model_ldm_trainable.pth (only the trainable part; CLIP backbone loaded separately). Publicly downloadable.

## Verified?
yes for paper numbers (copied from arXiv 2402.19091 PDF text, Tables 1-6) and repo listing/license (GitHub API); B-Free-reported numbers partially (fetch summary).

## Relevance to this project
- Intermediate-layer features are the single largest gain (ablation: +9 ACC pts) and are cheap to train; good candidate recipe for our own probe.
- Robust to crop/JPEG/noise individually but not to their combination; no resizing is applied anywhere, so resize/phone/screenshot pipelines are untested.
- Fixed-threshold 0.5 on unseen sets shows large ACC drops for new generators even when AP stays ~98 -> calibration problem remains.
