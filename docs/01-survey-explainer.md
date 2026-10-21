# Detecting diffusion-generated images: a survey for newcomers

*Built from `kb/index.md`, the 34 entries in `kb/papers/`, the 8 in `kb/datasets/` and `LOG.md`, as of 2026-10-21.*

**Citations.** `[npr-cvpr2024, T5]` means "the KB entry `npr-cvpr2024`, Table 5 of that paper". If a number about method X comes from a *different* paper's evaluation, the citation names that paper and says so in words, e.g. "SPAI's Table 1 reports NPR at 45.4" `[spai-cvpr2025, T1]`. Most KB entries are **partially verified**: they were read through fetch-tool summaries of the arXiv HTML, not checked against the PDF. Where the KB marks a number **[unverified]**, this document keeps the mark. No number here comes from outside the KB.

**What counts for this project.** We care about generators the detector never saw in training, and about real-world processing: JPEG, resizing, screenshots and phone photos from neural-ISP cameras. A near-perfect score on held-out images from the training generator tells us very little.

---

## 1. TL;DR

1. We want to tell whether an image came from a diffusion model, including generators and image processing that the detector never saw in training. Scores near 100% on a paper's own benchmark are common. They do not answer that question.
2. The research falls into five groups: low-level pixel or frequency artifacts; features from big pretrained vision models (CLIP, DINO); reconstruction by a diffusion model or its autoencoder; work on training data and benchmarks; and robustness and attack studies.
3. A recurring pattern is that a detector strong on its own benchmark falls to chance on someone else's. NPR scores 92.2% in its own paper but has an average AUC of 45.4 on 13 modern generators (SPAI's test). After a mild resize plus JPEG it gets 49.9% balanced accuracy, measured on one dataset (FakeInversion, B-Free Table 11).
4. The main cause is **shortcuts in the data**. In GenImage, the real photos are JPEG and the fakes are PNG, so detectors partly learn "is this a JPEG?". When AEROBLADE's authors re-tested DIRE's near-perfect detector, its result fell to chance; they suspect this format shortcut.
5. Processing usually breaks the **decision threshold** before it breaks the **ranking**. After resize plus JPEG, one detector still has AUC 90.8, but its accuracy at the default threshold falls to 61.5% (Corvi, Table 1).
6. False alarms on fresh real phone photos are almost never measured. The one study (LAION-Mobile, a preprint) found detectors flagging 17–91% of real phone photos at a threshold tuned on modern AI images. Only 212 of its photos came from 2021-or-later (neural-ISP) phones.
7. The most promising evidence is about **training data**. B-Free builds fakes from the real photos themselves and reports 96.3% balanced accuracy, including social-media images. Community Forensics trains on 4,803 generators and reports 89.2% accuracy on 21 unseen ones. Both numbers are self-reported.
8. **Foundation-model features** also help: CLIP intermediate layers (RINE), a small CLIP+SVM model (AUC 85.2 after crop/resize/JPEG) and a DINOv3 linear probe (94.5% on the authors' private set). That private set compares campus photos with Reddit Midjourney images, so the content differs between real and fake.
9. Among signal-based methods, SPAI (a spectral model of real photos) reports the best average AUC: 91.0 on 13 modern generators. Its robustness results exist only as plots.
10. Almost none of the strongest claims has been replicated independently. The same detector can look excellent or useless depending on which benchmark you use.
11. Adversarial attacks transfer between detectors. Perturbations built on other detectors drop a CLIP detector's AUROC from 0.95 to 0.62, and to 0.54 with an ensemble of other detectors (RAID, 16/255; 0.50 is the white-box case). On the commercial HIVE detector, an attack touching 20% of images cut real-image accuracy from 99% to 52%.
12. Licensing: B-Free, ZED, Effort, CNNDetection and most good evaluation datasets are **non-commercial**. MIT/Apache options include UnivFD, RINE, ClipDet, SPAI, FIRE, FatFormer, DMID and Community Forensics (weights).
13. What follows for us: test on real/fake sets with matched formats; report the detection rate at a fixed false-positive rate on fresh real photos (modern phones, shared copies); test laundering and screenshots explicitly.

---

## 2. One-page overview

### The five families

**Low-level artifact detectors** (CNNDetection, Frank-DCT, Corvi/DMID, LGrad, NPR, FreqNet, SPAI). Generators leave small statistical traces. Examples are periodic patterns from up-sampling layers, unusual noise residuals, and peaks in the frequency spectrum. A CNN is trained on these traces, or on a transformed view of the image such as DCT coefficients, gradients or pixel differences. They are cheap and work very well when the test generator resembles the training one. They are fragile, because JPEG, resizing and noise destroy exactly these traces.

**Foundation-model features** (UnivFD, RINE, ClipDet, FatFormer, C2P-CLIP, Effort, DINOv3 probe). A large pretrained encoder (CLIP ViT-L/14, DINOv2/v3) is kept frozen or tuned lightly, and a small head is trained on top. The large model already separates many kinds of images, so a linear head trained on one generator often transfers to others. Variants differ in which layer they read: the last layer is mostly about meaning, while intermediate layers keep more low-level detail. They also differ in how much of the backbone they tune (none, LoRA, adapters, rank-1).

**Reconstruction-based** (DIRE, AEROBLADE, LaRE², FIRE, Manifold-bias, ZED, DRCT). The idea is that a generator, or its autoencoder, reproduces its own outputs better than real photos. You reconstruct the image and measure the error, either thresholding it directly or feeding the error map to a classifier. Some methods need no training at all. Others are expensive, at up to about 40 U-Net passes per image (DIRE). All of them work best when the test generator shares the autoencoder used for reconstruction. ZED instead uses a lossless image coder trained only on real photos. DRCT uses reconstructions only to build training data.

**Data-bias and benchmark work** (B-Free, GenImage, Fake-or-JPEG, AIDE/Chameleon, Community Forensics, WildRF/LaDeDa, Synthbuster). These papers ask whether we are measuring the right thing. Some expose shortcuts in benchmarks (JPEG vs PNG, image size, content). Some build harder or cleaner test sets (Chameleon, WildRF, Synthbuster). Others change the training data rather than the model: B-Free makes fakes from the real images themselves, and Community Forensics uses thousands of generators.

**Robustness and attacks** (Carlini-Farid, RAID, Mavali et al., FPBA, TrueFake, LAION-Mobile). These test detectors against deliberate adversarial perturbations, real social-network processing (Facebook, X, Telegram) and false alarms on real phone photos.

### Why generalization is the hard part

1. **Shortcut learning: format, JPEG and size bias.** If all real training images are JPEG and all fakes are PNG, "has JPEG blocking" is a perfect feature in training and a useless one in the wild.
   - In GenImage, the real images are mostly JPEG quality 96 and 450–550 px, while the fakes are PNG at fixed sizes `[fakeorjpeg-arxiv2024]`. A ResNet-50 trained on GenImage-Midjourney classifies more real FFHQ images as real the harder they are JPEG-compressed: 80.45% as PNG, 100.0% at JPEG quality 60 `[fakeorjpeg-arxiv2024, T2]`. Removing the bias raises cross-generator accuracy by about 11 points (ResNet-50 71.68 → 82.74; Swin-T 74.09 → 85.83) `[fakeorjpeg-arxiv2024, T4]`.
   - DIRE's near-perfect results are suspected to come from JPEG reals vs PNG fakes. When AEROBLADE's authors re-ran it, it scored AP ~0.46–0.51 with TPR@5%FPR = 0.000 `[aeroblade-cvpr2024, T3]` (per-generator values [unverified] in `dire-iccv2023`).
2. **Generator-specific fingerprints.** A detector learns the traces of the generators it trained on, and those traces differ between families.
   - A ResNet-50 trained on SD 1.4 reaches 99.9% on SD 1.4 and 54.9% on Midjourney, 53.5% on ADM and 52.0% on BigGAN `[genimage-neurips2023, T3]`.
   - Corvi's LDM-trained detector gets 99.7% on Stable Diffusion but 52.9% on ADM and 50.4% on DALL-E 2 `[corvi-icassp2023, T2]`. B-Free's later test gives it 50.0% on DALL-E 3 `[bfree-cvpr2025, T5 via corvi-icassp2023]`.
3. **Thresholds miscalibrated by processing and domain shift.** AUC/AP measure ranking; accuracy needs a threshold. Processing and new generators move the score distribution, so a threshold chosen on training data stops working even when ranking survives.
   - Corvi: Grag2021 has AUC 90.8 but accuracy 61.5 after resize+JPEG `[corvi-icassp2023, T1]`. Calibrating with 2 real + 2 fake images per generator lifts average accuracy to 87.8 `[corvi-icassp2023, T2]`.
   - UnivFD: 93.38 mAP but 81.38% accuracy `[univfd-cvpr2023, T2–3]`.
   - In LAION-Mobile, the false-positive rate on the same real phone photos ranges from 0.2% to 39.1% for UnivFD, depending only on which AI set the threshold was tuned on `[laion-mobile-arxiv2026]`.
4. **Semantic and domain mismatch between real and fake sets.** If reals are DSLR landscapes and fakes are Midjourney art, a detector can learn the content or the source instead of the generator.
   - The DINOv3 result `[vfm-baseline-arxiv2025, T4]` compares private campus photos against Reddit Midjourney posts.
   - WildRF labels come from hashtags `[wildrf-ladeda-arxiv2024]`.
   - Chameleon fakes come from art communities, and the reals come from Unsplash `[aide-chameleon-iclr2025]`.
   - B-Free and ClipDet avoid this by building fakes from the same content as the reals (inpainting, or the same captions) `[bfree-cvpr2025; clipdet-cvprw2024]`.

On top of these four problems, **adversarial perturbations** transfer between detectors and survive JPEG Q50 and a 75% resize `[mavali-aigi-robustness-arxiv2025]`.

---

## 3. Methods, one by one

Each entry gives: **Intuition**, **Sketch**, **Results** (split into in-distribution / unseen generators / perturbed), and **Main weakness**. "ProGAN protocol" means training on ProGAN-vs-LSUN images (Wang et al. 2020) and testing on other generators.

### 3.1 Low-level artifact detectors

#### CNNDetection (Wang et al., CVPR 2020) — `cnndetection-cvpr2020`
- **Intuition.** A standard ResNet-50, trained on one GAN (ProGAN) with blur and JPEG augmentation, spots images from other GANs it never saw. It set up the "train on ProGAN, test on everything" protocol that later papers reuse.
- **Sketch.** `image → random 224 crop (no resize) → ResNet-50 (ImageNet init, fine-tuned; blur+JPEG aug) → score`
- **Results.**
  - *In-distribution:* ProGAN 100 AP (Blur+JPEG 0.5) `[T2]`.
  - *Unseen GANs:* mAP 90.8 (Blur+JPEG 0.5) and 92.6 (Blur+JPEG 0.1) over 11 generators; SAN 63.9, DeepFake 66.3 `[T2]`.
  - *Unseen diffusion (other papers):*
    - Corvi: avg Acc/AUC 59.5/78.6 over 12 generators; Stable Diffusion 50.9/65.9, ADM 50.6/56.3 `[corvi-icassp2023, T1]`.
    - SPAI: avg AUC 55.5 on 13 modern generators `[spai-cvpr2025, T1]`.
    - Community Forensics: 0.537 mAP / 51.3% on its 21-generator eval `[communityforensics-cvpr2025, T2]`.
    - B-Free: bAcc ~50 on every diffusion generator; WildRF 50.0/50.7/50.1 `[bfree-cvpr2025, T5/T7 via cnndetection]`.
  - *Perturbed:* after crop + resize to 200 px + JPEG Q65–100, avg 55.8/75.0 `[corvi-icassp2023, T1]`. The paper's own blur/JPEG curves exist only in Fig. 5 [unverified].
- **Main weakness.** Near chance on diffusion images. The licence is non-commercial.

#### Frequency/DCT analysis (Frank et al., ICML 2020) — `freq-dct-icml2020`
- **Intuition.** GAN up-sampling layers leave grid patterns that are invisible in pixels but obvious in the DCT spectrum. A simple classifier on log-DCT coefficients separates real from fake almost perfectly when trained and tested on the same GAN.
- **Sketch.** `image → 2D DCT (whole image) → log-scale + normalise → ridge regression or small CNN (~170k params) → score`
- **Results.**
  - *In-distribution:* FFHQ vs StyleGAN, ridge regression on pixels 75.78% vs on DCT 100.00% `[T1]`.
  - *Unseen generators:* not evaluated in the paper. FreqNet's paper reports a ProGAN-trained "Frank" at 65.7–78.9 mean Acc on ForenSynths `[freqnet-aaai2024, T1]`. NPR's paper reports 58.8 on 1000-step diffusion sets `[npr-cvpr2024, T5 context]`.
  - *Perturbed (LSUN):* trained clean, then tested on JPEG 71.86%, noise 48.99%, all perturbations combined 67.76%. Trained with perturbations, JPEG 94.83% and combined 92.17% `[T5]`.
- **Main weakness.** GAN-only. Robust only to the perturbations it was trained on, and heavier anti-aliasing removes the artifacts.

#### Corvi et al. / DMID (ICASSP 2023) — `corvi-icassp2023`
- **Intuition.** This was the first systematic check of GAN detectors on diffusion images. Latent/Stable Diffusion leave periodic noise fingerprints, but ADM and DALL-E 2 leave weak ones. The authors retrain a strong CNN on Latent Diffusion; that model ("DMID") became a standard baseline.
- **Sketch.** `image → ResNet-50 without first-layer down-sampling (Grag2021, trained on LDM, heavy aug) → logit  [optionally averaged with a ProGAN-trained model, then Platt-calibrated]`
- **Results.**
  - *ProGAN-trained, 12 generators:* PNG Acc/AUC 75.8/92.8; after resize+JPEG 61.5/90.8 `[T1]`. Ranking survives; the threshold does not.
  - *LDM-trained, resized+compressed test:* LDM 97.1/99.9, Stable Diffusion 99.7/100, ADM 52.9/81.9, DALL-E 2 50.4/74.2, average 66.9/89.2. Fusion plus Platt calibration (2 real + 2 fake per generator) gives 87.8/94.1 `[T2]`.
  - *Independent:*
    - B-Free reports DMID bAcc of Midjourney 100, SDXL 99.7, SD3.5 99.9, FLUX 63.7, DALL-E 2 50.1, DALL-E 3 50.0 and Firefly 51.0; WildRF average 75.6 `[bfree-cvpr2025, T5 via corvi-icassp2023]`.
    - SPAI reports avg AUC 83.5, with DALL-E 3 at 41.3 `[spai-cvpr2025, T1]`.
    - B-Free's appendix says performance under social-network simulation "drops substantially, except for DMID and our method" `[bfree-cvpr2025, T11 text]`.
- **Main weakness.** Transfers within the SD family but not to DALL-E/Firefly-like pipelines.

#### LGrad (CVPR 2023) — `lgrad-cvpr2023`
- **Intuition.** Pass the image through a frozen pretrained network (best: a StyleGAN discriminator) and back-propagate to the pixels. The gradient image suppresses content and highlights artifacts, and a second CNN classifies it.
- **Sketch.** `image → frozen StyleGAN discriminator D → ∂ΣD(I)/∂I → gradient map (0–255) → ResNet-50 → score`
- **Results.**
  - *Unseen GANs (ForenSynths, 1-class):* mean Acc/AP 86.3/92.7 `[T3]`.
  - *Perturbed:* JPEG 69.5/81.2, noise 69.1/79.4, combined 75.9/84.1 `[T4]`.
  - *Unseen diffusion (other papers):*
    - NPR's paper: mean Acc 64.2 on its Table 5 diffusion sets `[npr-cvpr2024, T5]`.
    - SPAI: avg AUC 65.9, with SD3 12.7 and DALL-E 3 30.0 `[spai-cvpr2025, T1]`.
    - B-Free: bAcc 54.4 average and 46.3 on SynthWildX `[bfree-cvpr2025, T5/T7 via lgrad]`.
- **Main weakness.** GAN-only by design. JPEG costs ~17 accuracy points. No licence file.

#### NPR (CVPR 2024) — `npr-cvpr2024`
- **Intuition.** Up-sampling makes the four pixels in each 2×2 block correlated in a characteristic way. NPR feeds the within-block pixel differences to a tiny CNN. It was the state of the art on GAN and early-diffusion benchmarks.
- **Sketch.** `image → 2×2 blocks → subtract each block's first pixel → NPR map → small ResNet (1.44M params) → score`
- **Results (trained on ProGAN 4-class).**
  - *Paper's own benchmarks:*
    - ForenSynths 92.5/96.1 `[T1]`.
    - DiffusionForensics 95.3/99.8 `[T3]`.
    - 1000-step diffusion + Midjourney + DALL-E: 80.1/84.8 `[T5]`.
    - Overall 92.2 mean Acc `[T6]`. The caption says "38 sub-testsets" and the text says 28: the paper contradicts itself.
  - *Unseen modern generators (independent):*
    - SPAI: avg AUC 45.4, worst of 12 detectors. Flux 19.8, SDXL 18.1, MJv5 15.3, DALL-E 2 3.9, but DALL-E 3 97.1 `[spai-cvpr2025, T1]`.
    - B-Free: avg bAcc 42.7; DALL-E 3 0.6 and Firefly 0.4 [unverified]; Table 7 GenImage 50.7, FakeInversion 87.0, SynthWildX 49.9 `[bfree-cvpr2025, T5/T7 via npr]`.
    - Chameleon: overall 57.29 but fake accuracy 2.20 `[aide-chameleon-iclr2025, T5]`.
    - WebAIG-25: 93.5 real / 34.9 fake `[vfm-baseline-arxiv2025, T4]`.
  - *Perturbed:*
    - FakeInversion under social-network simulation (resize 0.7–1 + JPEG 70–100): AUC/bAcc 91.6/87.0 → **43.3/49.9** `[bfree-cvpr2025, T11]`. This is one dataset under one setting, not a general number (see LOG.md).
    - TrueFake: TPR on SDXL images shared through Facebook drops to 0.21 `[truefake-ijcnn2025, T4]`.
    - The NPR paper itself has no robustness evaluation.
- **Main weakness.** Tied to up-sampling traces, which JPEG/resize destroy, and apparently to biases of individual datasets. No licence file (derived from CC BY-NC-SA code).

#### FreqNet (AAAI 2024) — `freqnet-aaai2024`
- **Intuition.** A very small network (1.9M parameters) that repeatedly high-pass filters the image and its feature maps in the Fourier domain, so it can only look at high-frequency artifacts.
- **Sketch.** `image → FFT, zero the centre (low) frequencies → iFFT → [conv + feature-space FFT high-pass + amplitude/phase convs] × N → score`
- **Results.**
  - *Unseen GANs:* 4-class mean Acc/AP 91.5/98.5 `[T1]`; 92.8 mean Acc over 17 GANs `[T3]`.
  - *Diffusion, JPEG, resize:* **not evaluated**. No independent evaluation was found.
- **Main weakness.** No evidence at all for diffusion or in-the-wild images. The ProGAN protocol's JPEG-real/PNG-fake split is a likely shortcut for a frequency model. No licence file.

#### SPAI (CVPR 2025) — `spai-cvpr2025`
- **Intuition.** A ViT pretrained to reconstruct masked frequency bands of *real* images learns what a normal photo spectrum looks like. SPAI measures how poorly that model reconstructs a test image's spectrum. It works on the full-resolution image patch by patch, without resizing.
- **Sketch.** `full-res image → 224 patches → frozen MFM ViT-B/16 (masked-frequency pretraining) → spectral-reconstruction similarity per patch → spectral context attention over all patches → score`
- **Results** (trained on 180K LDM fakes + 180K reals).
  - *Unseen modern generators:* avg AUC 91.0 over 13 generators and 5 real sources, including smartphone sources. Competitors: RINE 85.5, DMID 83.5, UnivFD 67.3, NPR 45.4. Weakest on SD3 75.9, Flux 83.0, MJv6.1 84.0 `[T1]`.
  - *Ablations:* removing JPEG augmentation gives 89.1. Removing spectral pretraining gives 52.5 `[T3]`.
  - *Perturbed:* JPEG/WebP, blur, noise and resize are shown only in Fig. 3, and only against two competitors [unverified].
- **Main weakness.** Self-reported and AUC-only, with no threshold or false-positive numbers. The authors note that compression removes the spectral information the method relies on.

### 3.2 Foundation-model features

#### UnivFD (Ojha et al., CVPR 2023) — `univfd-cvpr2023`
- **Intuition.** Do not train a deep network, which overfits to one generator's artifacts. Freeze CLIP and train only a linear layer on its image embedding. CLIP's broad features let a ProGAN-trained probe recognise diffusion images. Almost every later CLIP detector compares against this baseline.
- **Sketch.** `image → centre-crop 224 (no resize) → CLIP ViT-L/14 (frozen) → last-layer embedding → linear (or k-NN) → score`
- **Results** (trained on ProGAN).
  - *Unseen generators:* total mAP 93.38; ~95.00 mAP on diffusion/autoregressive models vs 75.51 for the best trained CNN `[T2]`.
  - *Accuracy at a threshold set on ProGAN validation:* 81.38 `[T3]`. Weak spots: Guided 70.03 and LDM-200-with-CFG 73.76 `[T3]`.
  - *Independent, unseen:*
    - ClipDet: AUC 84.1 on 18 generators `[clipdet-cvprw2024, T4]`.
    - SPAI: avg AUC 67.3 `[spai-cvpr2025, T1]`.
    - Community Forensics: 0.592 mAP / 54.0% `[communityforensics-cvpr2025, T2]`.
    - Chameleon: 57.22 overall (fake 3.18 / real 97.83) `[aide-chameleon-iclr2025, T5]`.
    - WebAIG-25: 59.2 real / 29.8 fake `[vfm-baseline-arxiv2025, T4]`.
    - WildRF bAcc: Facebook 48.8, Reddit 59.5, Twitter 56.0 `[bfree-cvpr2025, T5 via univfd]`.
  - *Perturbed:*
    - Crop/resize/JPEG: AUC 71.8 `[clipdet-cvprw2024, T5]`.
    - FakeInversion: AUC 54.9 → 49.7 under social-network simulation `[bfree-cvpr2025, T11 via univfd]`.
    - JPEG QF50: accuracy 68.68 `[aide-chameleon-iclr2025, T6 via univfd]`.
    - Transfer attack (ε=16/255, source K24): AUROC 0.95 → 0.62; N-model ensemble → 0.54; white-box → ~0.5 `[raid-arxiv2025, T2]` (corrected after PDF check, kb/verification-log.md).
  - *False positives on phone photos:* 0.2% at a ProGAN-EER threshold vs 39.1% at an NTIRE-2026-EER threshold `[laion-mobile-arxiv2026]`.
- **Main weakness.** Ranking transfers but the threshold does not. The last layer is mostly about meaning and near chance in the wild. The fixed centre crop means resize behaviour is untested.

#### RINE (ECCV 2024) — `rine-eccv2024`
- **Intuition.** CLIP's last layer is mostly about meaning, while earlier blocks keep low-level information. RINE takes the CLS token from *every* block of a frozen CLIP, learns how much each block matters, and trains a small MLP. Training takes about one epoch, roughly 2–8 minutes.
- **Sketch.** `image → centre-crop 224 → CLIP ViT-L/14 (frozen) → CLS token of every block → learned per-block weights → weighted sum → MLP → score`
- **Results.**
  - *Unseen (20 sets, ProGAN-trained):* ACC 90.3 / 87.7 / 91.5 for 1/2/4 classes vs UFD 80.9; AP 98.6–98.8 `[T2, T4]`.
  - *Ablation:* last layer only gives 80.8 vs 89.8 for the full model `[T4]`.
  - *Commercial generators (Synthbuster, ProGAN-trained):* DALL-E 3 21.1/30.7, Midjourney 34.2/39.5. LDM-trained: SDXL 96.3, Midjourney 92.4, DALL-E 3 47.2 `[T6]`.
  - *Independent:*
    - SPAI: avg AUC 85.5 `[spai-cvpr2025, T1]`.
    - B-Free: WildRF bAcc 52.8/67.7/56.0; GenImage 69.1 `[bfree-cvpr2025, T5/T7 via rine]`.
    - B-Free: Synthbuster bAcc 65.0 with ProGAN training vs 83.2 when trained on B-Free's data `[bfree-cvpr2025, T4 via rine]`.
  - *Perturbed:* individually, blur 83.9, crop 88.8, JPEG 86.7, noise 85.6 `[T5]`; combined perturbations "considerably reduce" performance (numbers [unverified]). Under social-network simulation, FakeInversion AUC goes 69.6 → 62.4 `[bfree-cvpr2025, T11 via rine]`.
- **Main weakness.** Hyper-parameters were tuned on test accuracy, so results are optimistic. Never resizes. ProGAN training fails on commercial generators.

#### ClipDet ("Raising the Bar", CVPRW 2024) — `clipdet-cvprw2024`
- **Intuition.** Penultimate-layer CLIP features plus a linear SVM need only 1k–10k training pairs. The reals are COCO photos and the fakes are generated by Latent Diffusion from the same captions, so content is matched. Adding resized/compressed copies to that small training set buys robustness.
- **Sketch.** `image → resize 224 (released code) → CLIP ViT-L/14 (open_clip, CommonPool; frozen) → penultimate layer → linear SVM → log-likelihood-ratio score`
- **Results.**
  - *Unseen (18 GAN/diffusion/commercial generators):* avg AUC 89.8 (10k) vs Ojha 84.1, Corvi 84.2, NPR 76.8 `[T4]`.
  - *Perturbed (random crop, resize 200 px, JPEG Q65–100):* 10k 78.1, 10k+ (augmented reference set) 85.2, 1k+ 83.2. Ojha 71.8, Corvi 70.8, NPR 51.0, LGrad 49.4 `[T5]`.
  - *Fusion with Corvi:* +3.6 AUC, +7.4 accuracy `[T7]`.
  - *Inconsistency:* the text claims +6.8 AUC over the best competitor, but the table gap is 5.6.
- **Main weakness.** Accuracy at a fixed threshold is well below AUC (Fig. 3). No independent evaluation. Possible JPEG(COCO)/PNG format bias in the reference set is not discussed [unverified].

#### FatFormer (CVPR 2024) — `fatformer-cvpr2024`
- **Intuition.** Instead of freezing CLIP, insert small adapters that extract local and wavelet-frequency forgery traces. Then align image features with learned "real" and "fake" text prompts.
- **Sketch.** `image → resize 256 → crop 224 → CLIP ViT + forgery-aware adapters (conv + DWT frequency branch) ↔ learned real/fake text prompts → score`
- **Results.**
  - *Unseen (ProGAN 4-class):* GANs Acc/AP 98.4/99.7 `[T1]`; diffusion 95.0/98.8 vs UniFD 85.4/94.6 `[T2]`.
  - *Perturbed GAN set:* JPEG 95.9/99.2 (App. A.3).
  - *Other protocols:* C2P-CLIP lists FatFormer at 90.86 mAcc on the 19-subset protocol `[c2pclip-aaai2025, T2]` and 88.9 on GenImage `[c2pclip-aaai2025, T3]`.
  - *Independent, in the wild:* WildRF bAcc 54.7/69.5/54.8 and a Table 7 average of 58.2. FakeInversion AUC goes 68.2 → 48.7 under social-network simulation, the largest drop among CLIP methods `[bfree-cvpr2025, T5/T7/T11 via fatformer]`.
- **Main weakness.** Gains on curated benchmarks disappear in the wild. The resize step and wavelet branch are likely brittle to re-compression.

#### C2P-CLIP (AAAI 2025) — `c2pclip-aaai2025`
- **Intuition.** CLIP partly recognises fakes through concepts like "deepfake". During training, a fixed category word ("Deepfake" / "Camera") is appended to every caption, and the image encoder is tuned with LoRA to match it. The text side is discarded at test time.
- **Sketch.**
  - *Training:* `image → CLIP ViT-L/14 + LoRA(q,k,v) ↔ frozen text encoder("caption + Deepfake/Camera") → contrastive + classification loss`
  - *Inference:* `image → LoRA-CLIP image encoder → linear → score`
- **Results.**
  - *Unseen (UFD protocol, 19 subsets):* mAcc 93.79, mAP 98.66 `[T2, T1]`. GenImage 95.8 `[T3]`. Robustness not evaluated.
  - *Independent:*
    - WildRF bAcc 54.4/68.4/57.4; Table 7 average 64.2 `[bfree-cvpr2025, T5/T7 via c2pclip]`.
    - FakeInversion AUC 74.1 → 71.9 under social simulation: the most stable CLIP method, but at ~55–60 bAcc `[bfree-cvpr2025, T11 via c2pclip]`.
    - WebAIG-25: 84.7 real / 31.0 fake `[vfm-baseline-arxiv2025, T4]`.
  - *Inconsistency:* the paper's text swaps the two prompt-pair values relative to its tables.
- **Main weakness.** Big benchmark gains, modest gains in the wild. No licence file.

#### Effort (ICML 2025) — `effort-icml2025`
- **Intuition.** Fully fine-tuning CLIP squeezes its features into a narrow, overfit subspace. Effort decomposes each attention weight with SVD, freezes the main components, and trains only a rank-1 residual, about 0.19M parameters.
- **Sketch.** `CLIP ViT-L/14: W = UΣVᵀ → freeze top-r components, train rank-1 residual (orthogonality + singular-value constraints) → linear → score`
- **Results.**
  - *Unseen (UFD protocol):* mean Acc 95.19, mAP 99.41 `[T2–3]`. GenImage (SD1.4-trained): 91.1 `[T9]`.
  - *Robustness:* evaluated only in the face-deepfake setting (Fig. 8) [unverified].
  - *False positives on real phone photos:* 3.6% at a ProGAN-EER threshold, 91.3% at an NTIRE-EER threshold `[laion-mobile-arxiv2026]`.
- **Main weakness.** All results are on the ProGAN/GenImage benchmarks, which are known to be biased. Licence CC BY-NC 4.0 per the README badge (**non-commercial**); there is no LICENSE file.

#### Modern VFM linear probe / DINOv3 (arXiv 2509.12995) — `vfm-baseline-arxiv2025`
- **Intuition.** Newer vision foundation models (DINOv3, Perception Encoder, SigLIP-2…) with just a linear probe beat specialised detectors on in-the-wild sets. Some of that gain may come from the test images being in the model's pretraining data. To check, the authors built a post-cutoff set, WebAIG-25; only DINOv3 stays strong on it.
- **Sketch.** `image → DINOv3 (frozen, largest weights) → [CLS] → linear → score` (layer, resolution and resize handling [unverified]). Trained on GenImage SD1.4 with no augmentation.
- **Results.**
  - *WebAIG-25:* DINOv3 97.5 real / 91.4 fake / 94.5 avg. CLIP 46.7/71.1, PE 44.4/91.7, DINOv2 40.9/54.2 `[T4]`.
  - *Chameleon:* 93.7/90.5/92.4 `[T1]`. *WildRF:* avg 96.4 `[T2]`. *SocialRF / CommunityAI:* 93.3 / 95.3 `[T3]`.
- **About WebAIG-25.** It is the paper's **internal** set, not a public benchmark (LOG.md). The fakes are post-cutoff r/midjourney posts; the reals are private campus photos. Size and threshold are not stated [unverified]. Content and source therefore differ systematically between the classes.
- **Main weakness.** Preprint only; no code found; baselines were copied from other papers rather than re-run. Language-supervised models label many real photos as fake (CLIP real accuracy 46.7). Low FPR on diverse real photos has not been shown.

### 3.3 Reconstruction / diffusion-based detectors

#### DIRE (ICCV 2023) — `dire-iccv2023`
- **Intuition.** Invert an image to noise with a diffusion model, regenerate it, and compare with the input. Generated images come back almost unchanged and real ones do not. A ResNet-50 classifies the absolute-difference image.
- **Sketch.** `image (256) → DDIM inversion with ADM (20 steps) → DDIM reconstruction (20 steps) → |x − x̂| → ResNet-50 → score`  (~40 U-Net passes per image; >2 s/image on a V100 per LaRE²)
- **Results.**
  - *In-distribution (8 LSUN-Bedroom generators):* 99.9 ACC / 100 AP `[T2]`.
  - *Cross-dataset:* ImageNet ADM 90.2/97.9, ImageNet SD-v1 97.2/99.8 `[T3]`.
  - *Independent:*
    - AEROBLADE: AP ~0.46–0.51 and TPR@5%FPR 0.000 `[aeroblade-cvpr2024, T3]`.
    - DRCT: GenImage avg 71.24, with Midjourney 50.40 and ADM 52.32 `[drct-icml2024, T2 via dire]`; 35.05 on DRCT-2M-Wild `[drct-icml2024, T3]`.
    - LaRE²: reproduced average 67.2 `[lare2-cvpr2024, T1]`.
    - LAION-Mobile: FPR on real phone photos at τ=0.5 was 98.1% (on 738 images) `[laion-mobile-arxiv2026]`.
  - *Perturbed:* the paper says there is no drop under blur or JPEG, but gives no numbers [unverified].
- **Main weakness.** The headline results are suspected to be a JPEG-real/PNG-fake shortcut. It is also very expensive. No licence file.

#### AEROBLADE (CVPR 2024) — `aeroblade-cvpr2024`
- **Intuition.** Latent diffusion images are produced by a VAE decoder, so that autoencoder reproduces them almost perfectly. One encode/decode pass plus a perceptual distance gives a detector with no training.
- **Sketch.** `image → VAE encode→decode (SD1, SD2, Kandinsky-2.1 AEs) → LPIPS (layer 2) distance → min over AEs → threshold`
- **Results.**
  - *In-distribution-ish (LDM-family fakes vs LAION reals):* AP 0.978–0.999, mean ~0.992 `[T1]`; TPR@5%FPR 0.963–0.999 `[T3]`.
  - *Broad ~20-generator mix including GANs:* AUC 0.444 / AP 0.492 / ACC 0.464 `[manifold-bias-iclr2025, T1]`.
  - *DiffusionForensics:* 99.08 AUC / 98.0 ACC `[fire-cvpr2025, T1]`.
  - *Perturbed:* figures only (Fig. 7) [unverified].
- **Main weakness.** Works only when the generator's autoencoder, or a close proxy, is available. Near chance outside the LDM family. Error is sensitive to low-level changes. No licence file.

#### LaRE² (CVPR 2024) — `lare2-cvpr2024`
- **Intuition.** A cheaper DIRE: add noise to the SD latent once and measure how well one denoiser step predicts it. The error map is then fused with image features by attention. The paper reports it is ~8× faster than DIRE.
- **Sketch.** `image → resize 256 → SD1.5 VAE latent → add noise (t=200), 1 U-Net step × 4 noise draws → error map → attention-fused with CLIP-ResNet-50 features → score`
- **Results.**
  - *GenImage cross-generator:* avg ACC 79.1, with Midjourney 66.4 and ADM 66.7 `[T1]`.
  - *Internal inconsistency:* the ablation tables give 85.9 `[T2]` and 84.5 `[T3]` for what seems to be the full model; the protocol difference is unexplained [unverified].
  - No robustness or threshold results.
- **Main weakness.** Modest unseen-generator accuracy. Trained on GenImage, which has the JPEG/size bias. Weights not mentioned [unverified].

#### ZED (ECCV 2024) — `zed-eccv2024`
- **Intuition.** Train a lossless image compressor on *real photos only*. It knows how many bits a real image "should" cost; fakes cost an unexpected amount. No fake image is needed for training.
- **Sketch.** `image → 4-level pyramid (2×2 average pooling) → CNN coder predicts each level from the coarser one → (actual cost − expected cost) per level → |Δ01| = |D(0) − D(1)| → threshold`
- **Results.**
  - *Unseen (30 test sets, 20+ generators):* avg AUC 90.0 vs Corvi 89.4 and Ojha 88.4. The worst case is GLIDE-R at 65.1 `[T2]`.
  - *Real-set robustness:* compressed COCO 92.2, ImageNet 85.9, uncompressed RAISE 89.3. The authors see no significant compressed/uncompressed difference `[T4]`.
  - Resize not tested.
- **Main weakness.** Code was "coming soon" when fetched [unverified current status]. Licence is **non-commercial**. The threshold depends on the dataset.

#### DRCT (ICML 2024) — `drct-icml2024`
- **Intuition.** Pass real photos through Stable Diffusion so they look identical but carry SD's fingerprint. These "hard fakes" force the classifier to learn the fingerprint rather than content. At test time it is an ordinary classifier with no diffusion step.
- **Sketch.**
  - *Training:* `real → SD inpainting, empty mask (reconstruction) → hard fake; train ConvNeXt-B (or frozen CLIP ViT-L/14 + FC) with BCE + contrastive loss`
  - *Inference:* `image → crop 224 → classifier → score`
- **Results.**
  - *Unseen (DRCT-2M, SD-family, trained SDv1.4):* avg ACC Conv-B 79.11 → DRCT/Conv-B(SDv2) 96.55; UnivFD 83.46 → 91.35. The SDv1-reconstructor model is weak on SDXL-Turbo (70.04) `[T1]`.
  - *GenImage:* DRCT/UnivFD 89.49, with Midjourney 91.50 and ADM 79.42 `[T2]`.
  - *In the wild (DRCT-2M-Wild):* 87.23 / 96.90 vs all baselines below 65 `[T3]`.
  - *Independent:*
    - Mavali reports benign AUC of DRCT-CLIP 92.3 and DRCT-ConvB 83.8 `[mavali-aigi-robustness-arxiv2025, T1]`.
    - RAID found DRCT-ConvNeXt the most attack-resistant (white-box F1 ~0.29) `[raid-arxiv2025, T2]`.
  - *Perturbed:* only in Fig. 5.
- **Main weakness.** Mostly SD-family training. COCO reals are JPEG while fakes are mostly PNG, so a format shortcut is possible. Accuracy is reported only at threshold 0.5. No licence file.

#### FIRE (CVPR 2025) — `fire-cvpr2025`
- **Intuition.** Diffusion autoencoders reconstruct real photos worst in the mid-frequency band. FIRE compares reconstruction errors with and without that band, using a learned frequency mask, and classifies the two error maps.
- **Sketch.** `image → SD1.5 VAE enc/dec → |err₁| ; image with mid-band removed (learned FFT mask) → VAE → |err₂| → concat → ResNet-50 → score`
- **Results.**
  - *In-distribution (DiffusionForensics):* 100.0/100.0 `[T1]`, saturated.
  - *Unseen newer generators:* AUC/ACC 72.0/68.1 trained on DALL-E 3 and 78.1/72.9 trained on Kandinsky 3 `[T2]`.
  - *Perturbed:* Fig. 7 only [unverified].
- **Main weakness.** Unseen-generator accuracy is only 68–73%. MIT licence, weights released.

#### Manifold-induced biases (ICLR 2025) — `manifold-bias-iclr2025`
- **Intuition.** Generated images sit near sharp local peaks of a diffusion model's learned density. Probing with tiny random perturbations measures curvature and gradient, which gives a training-free score.
- **Sketch.** `image → LLaVA caption → SD1.4 VAE+U-Net, s small noise perturbations at small t → curvature/gradient/bias terms → CLIP-space score → threshold set on 1,000 reals`
- **Results.**
  - *Unseen (~20 generators, GAN + diffusion):* AUC 0.835 / AP 0.832 / ACC 0.741, vs AEROBLADE 0.444 and RIGID 0.439 `[T1]`.
  - *Perturbed:* JPEG AUC 0.79, blur 0.822 `[T2]`.
  - *Cost:* ~2.1 s/sample on an A100.
  - *Inconsistency:* the test-set size is given as 100k+100k in the main text and 200,000 in an appendix [unverified].
- **Main weakness.** Expensive (SD + LLaVA per image). Compared only with AEROBLADE and RIGID. No licence file.

### 3.4 Data-bias and benchmark work

#### B-Free (CVPR 2025) — `bfree-cvpr2025`
- **Intuition.** Make each fake *from* a real photo: regenerate the whole image, or inpaint an object, with SD 2.1. Real and fake then share content, format and size, so the only difference is the generator trace. Heavy augmentation is added, and a DINOv2 ViT is fine-tuned end to end.
- **Sketch.**
  - *Training data:* `COCO real → SD2.1 inpainting (empty mask = full regen, or object mask) → 6 fakes/real; + blur/JPEG/scale/cut-out/noise aug`
  - *Detector:* `image → 504×504 crops (multi-crop average) → DINOv2-reg4 ViT (fine-tuned) → score`
- **Results.**
  - *Unseen generators + WildRF:* avg bAcc 96.3, NLL 0.10; WildRF Facebook 95.6, Reddit 86.2, Twitter 97.3 `[T5]`.
  - *GenImage-unbiased / FakeInversion / SynthWildX:* average bAcc 90.4 `[T7]`.
  - Reports accuracy at a **fixed 0.5 threshold** plus calibration metrics (ECE, NLL), which is rare in this field.
  - Its Table 11 is the source of the NPR collapse described in §3.1.
- **Main weakness.** Training fakes come from one generator family. No independent evaluation was located. Weights are **non-commercial** (custom GRIP-UNINA licence).

#### GenImage (NeurIPS 2023 D&B) — `genimage-neurips2023`
- **Intuition.** A million-image benchmark: ImageNet reals plus 8 generators. The protocol is to train on one generator and test on the others.
- **Sketch.** `train on {SD1.4 fakes, ImageNet reals} → test on {MJ, SD1.5, Wukong, ADM, GLIDE, VQDM, BigGAN}`
- **Results.** A ResNet-50 trained on SD1.4 scores SD1.4 99.9 and SD1.5 99.7, but Midjourney 54.9, ADM 53.5 and BigGAN 52.0, average 72.1 `[T3]`. The degradation table's values were inconsistent and are not reported [unverified].
- **Main weakness.** JPEG-real/PNG-fake and size bias `[fakeorjpeg-arxiv2024]`; old, low-resolution generators. **Non-commercial** licence.

#### Fake or JPEG? (arXiv 2024) — `fakeorjpeg-arxiv2024`
- **Intuition.** Shows that GenImage detectors partly learn "JPEG and ~500 px means real". Rebuilds the benchmark with matched JPEG quality and size, then retrains.
- **Sketch.** `GenImage → {compress fakes to JPEG96, constrain sizes} → retrain ResNet-50 / Swin-T → cross-generator test`
- **Results.**
  - A ResNet-50 trained on Midjourney calls more real FFHQ images real as JPEG gets stronger: PNG 80.45 → JPEG60 100.0 `[T2]`.
  - Training on JPEG96 only improves JPEG95 testing from 53.91 to 67.17 `[T3]`.
  - Cross-generator accuracy improves after debiasing: +11.06 (ResNet-50) and +11.74 (Swin-T) `[T4]`.
- **Main weakness.** Still uses GenImage's old generators. Not peer-reviewed. Licence [unverified].

#### AIDE + Chameleon (ICLR 2025) — `aide-chameleon-iclr2025`
- **Intuition.** Chameleon contains AI images that two human annotators both judged real. Nine detectors mostly call them real. AIDE combines CLIP semantics with noise features from the most- and least-textured patches.
- **Sketch.** `image → [CLIP semantic features] + [low-level features of highest/lowest-frequency patches] → mixture of experts → score`
- **Results.**
  - *Chameleon (trained on all GenImage):* AIDE 65.77 overall, 26.80 fake accuracy, 95.06 real accuracy. Other detectors score ~55–61 overall with fake accuracy near 0–3, e.g. NPR 2.20 `[T5]`.
  - *Benchmarks:* AIGCDetectBenchmark 92.77 `[T3]`; GenImage 86.88 `[T4]`.
  - *JPEG:* QF95 75.54, QF50 69.60 `[T6]`.
- **Main weakness.** Chameleon is gated (email plus EULA), academic-only, and drawn from art communities, so it is not phone photos. The AIDE repo licence was not read [unverified].

#### Community Forensics (CVPR 2025) — `communityforensics-cvpr2025`
- **Intuition.** Do not invent a cleverer model; add generator diversity. The training set has 2.7M fakes from 4,803 generators (mostly Hugging Face latent-diffusion checkpoints) and 2.7M reals. A plain ViT-S fine-tuned on it generalises to commercial models.
- **Sketch.** `image → ViT-S (CLIP-pretrained, fine-tuned end-to-end; RandAugment incl. JPEG/resize) → linear → sigmoid`
- **Results.**
  - *21 unseen generators, including 11 commercial:* 0.987 mAP / 89.2% `[T2]`.
  - *Other sets:* Synthbuster 0.974 / 90.8; GenImage 0.998 / 95.7 `[T2]`.
  - *Baselines on the same set:* Wang 0.537 / 51.3, Ojha 0.592 / 54.0 `[T2]`.
  - Gains flatten beyond ~1,000 generators. Robustness is shown only qualitatively (Fig. 7).
- **Main weakness.** Mostly latent-diffusion fakes. Reals come from fixed public datasets, including VISION, so real-source shortcuts are possible. Self-reported. Code and weights are **MIT**; the datasets are non-commercial.

#### LaDeDa / WildRF (arXiv 2024) — `wildrf-ladeda-arxiv2024`
- **Intuition.** Introduces WildRF: 5,300 real and fake images actually posted on Reddit, X and Facebook, labelled by hashtags. Also proposes LaDeDa, a 9×9-patch classifier with pooled scores, and its distilled 4-layer version Tiny-LaDeDa.
- **Sketch.** `image → 9×9 patches → patch CNN → pool patch scores → score`
- **Results.**
  - *Trained on Reddit:* mean ACC/AP 85.7/93.7. NPR 81.9/91.6 and the CLIP probe 79.1/92.5 under the same protocol `[T5]`.
  - *Trained on ProGAN, tested on WildRF:* 68.3/79.9 `[T3]`.
- **Main weakness.** Hashtag labels may be noisy, and a later critique says the set includes drawings/CG [unverified]. The Facebook and X test sets are tiny (320 / 680 images). Licence not read [unverified].

#### Synthbuster (IEEE OJSP 2024) — `synthbuster-ojsp2024`
- **Intuition.** A training-free detector based on spectral peaks of a residual image. The paper also releases 1,000 never-compressed PNG fakes from each of 9 generators, prompted to resemble RAISE DSLR photos.
- **Sketch.** `image → residual → Fourier spectrum → peak statistics → threshold`
- **Results.** Abstract only. It claims to work "even under mild JPEG compression"; no numbers were read [unverified].
- **Main weakness.** Only the abstract was read. RAISE reals are not included. **Non-commercial** licence (CC BY-NC-SA 4.0).

### 3.5 Robustness and attack studies

#### Carlini & Farid (CVPRW 2020) — `carlini-farid-cvprw2020`
- **Intuition.** Forensic CNNs are far more fragile than ImageNet classifiers. Flipping the lowest bit of a few percent of pixels, or adding a 1% sticker, defeats them.
- **Sketch.** `fake image → gradient of target (or surrogate ResNet-18) → L2 / lowest-bit / patch perturbation → detector says "real"`
- **Results.**
  - Flipping ~2% of pixels misleads the detector on 71.3% of fakes, and ~4% on 89.7% (Fig. 2a).
  - A 1% patch takes AUC 0.966 → 0.085 (Fig. 3b).
  - Black-box transfer takes AUC 0.96 → 0.22 (Fig. 3d).
  - The attacks survive PNG and JPEG saving (JPEG quality [unverified]).
- **Main weakness.** Pre-diffusion targets only. No code. The paper is inconsistent about whether perturbations are "10×" or "16×" smaller than for ImageNet.

#### Mavali et al. (arXiv 2024/25) — `mavali-aigi-robustness-arxiv2025`
- **Intuition.** Realistic black-box attacks (transfer, ensemble, diverse-input, universal, query-based) still work after social-media-style degradation, including against a commercial detector.
- **Sketch.** `image → surrogate ensemble → PGD L∞ ε=8/255 (diverse input) → [downsample 75% + JPEG Q50] → target detector`
- **Results.**
  - *Benign AUC:* DRCT-CLIP 92.3, UnivFD 81.5, Corvi 78.1 `[T1]`.
  - *Under the diverse-input attack:* average AUC 4.1 `[T2]`.
  - *HIVE (commercial):* AUC 99.0 → 73.8 on Synthbuster; real accuracy 99% → 52% `[T5]`.
  - *Defence:* a robust-CLIP backbone raises attacked AUC from ~0 to 69.0–75.7 but costs 3–7 benign AUC `[T6]`.
  - Per-degradation table numbers [unverified].
- **Main weakness.** Preprint; repo not resolvable [unverified].

#### FPBA (IEEE TMM 2025) — `fpba-tmm2025`
- **Intuition.** A transfer attack that perturbs both the frequency and spatial domains, using a Bayesian ensemble built from one surrogate to mimic unknown victims.
- **Sketch.** `image → surrogate + Bayesian MLP heads, random spectrum transforms → L∞ ε=8/255, 10 steps → victim`
- **Results.**
  - Success rate 99.6% white-box and 50.5% black-box over 13 detectors `[T1]`.
  - 99.2% even against detectors trained with blur+JPEG `[T8]`.
  - Cross-generator transfer only 12.7–28.0% `[T6]`.
  - The authors' adversarial-training attempt did not converge.
- **Main weakness.** Black-box success is only ~50%. Code not verified.

#### RAID (arXiv 2025) — `raid-arxiv2025`
- **Intuition.** A ready-made set of 72k adversarial images built against a 7-detector ensemble. It lets you estimate a new detector's transfer robustness without running attacks.
- **Sketch.** `D3 images → PGD on ensemble (excluding target) → released adversarial set → evaluate held-out detector`
- **Results.**
  - *Transfer:* AUROC UnivFD-style (O23) 0.95 → 0.62 (K24 source; 0.54 N-model ensemble); DRCT-ConvNeXt 0.94 → 0.59 `[T2]` (corrected after PDF check).
  - *Held-out baselines:* DINOv2 accuracy 83.1 → 54.6 `[T4]`.
  - *Commercial (50 pairs):* Sightengine AI-image accuracy 0.66 → 0.00 `[T7]`.
  - *Epsilon:* the released set uses ε=8/255; the experiments report 16–32/255 (mapping [unverified]).
- **Main weakness.** Only SD-family/IF generators; large budgets. **MIT** licence.

#### TrueFake (IJCNN 2025) — `truefake-ijcnn2025`
- **Intuition.** Upload real and fake images to Facebook, X and Telegram for real, download them again, and measure what the platforms' processing does to detectors.
- **Sketch.** `image → [Facebook: ≤720 px, JPEG QF 61–92 | X: QF 87 | Telegram: ≤800 px, QF 85] → detector`
- **Results.**
  - *Before sharing:* NPR TNR/TPR 0.99/0.60 `[T3]`.
  - *After Facebook:* NPR TPR on SDXL 0.21 `[T4]`. MISLNet TNR on real FFHQ 0.41, meaning real photos are flagged as fake `[T4]`.
  - The CLIP-based detector is best but loses ~10% on several classes.
- **Main weakness.** One snapshot of platform behaviour; face-heavy reals; no phone-ISP reals.

#### LAION-Mobile (arXiv 2026) — `laion-mobile-arxiv2026`
- **Intuition.** The only study in the KB that directly measures false alarms on ~1M real smartphone photos, identified via EXIF. It tests 12 detectors.
- **Sketch.** `LAION images with phone EXIF → filter/dedup → 9,115 scored → FPR at thresholds transferred from ProGAN / NTIRE-2026 sets`
- **Results.**
  - *Modern AI content (NTIRE 2026):* AUC 0.42–0.62; best is RIGID-DINOv3 at 0.624 `[T4]`.
  - *FPR at τ=0.5:* from 0.0% to 98.1% (DIRE).
  - *FPR at the NTIRE-EER threshold:* 17.4% (CNNDetection) to 91.3% (Effort).
  - The hypothesis that heavier-ISP phones raise FPR was not supported.
  - Only 212 images (0.03%) came from phones released in 2021 or later.
- **Main weakness.** Preprint; mostly 2014–2020 phones; EXIF "real" labels can be forged.

---

## 4. Comparison table

"Self" means reported by the method's own paper; other numbers come from the KB entry cited. **NC** marks a non-commercial licence. "none" means no LICENSE file was found, so reuse rights are unclear.

| Method | Family | Unseen-generator performance | Robustness (JPEG / resize / social) | Code / weights | License |
|---|---|---|---|---|---|
| CNNDetection | low-level | ~chance on diffusion: avg AUC 55.5 on 13 modern gens (SPAI T1); 0.537 mAP (CommFor T2) | crop+resize+JPEG avg Acc/AUC 55.8/75.0 (Corvi T1) | code + weights script | **CC BY-NC-SA 4.0 (NC)** |
| Frank DCT | low-level | not evaluated (GAN-only) | clean-trained JPEG 71.86, noise 48.99 (T5) | code; weights GDrive [unverified live] | MIT |
| DMID (Corvi) | low-level | MJ/SDXL/SD3.5 99.7–100 bAcc but DALL-E 3 50.0 (B-Free T5); avg AUC 83.5 (SPAI T1) | resized+JPEG avg 66.9/89.2 (T2); exempted from the "drops substantially" remark under social sim (B-Free T11 text) | code; weight location [unverified] | Apache-2.0 |
| LGrad | low-level | avg AUC 65.9 (SPAI T1); bAcc 54.4 (B-Free) | JPEG mean Acc 86.3→69.5 (T4) | code; weights [unverified] | none |
| NPR | low-level | avg AUC 45.4 (SPAI T1); avg bAcc 42.7 (B-Free T5) | FakeInversion bAcc 87.0→49.9 (B-Free T11); FB SDXL TPR 0.21 (TrueFake T4) | code + weights in repo | none (derived from NC code) |
| FreqNet | low-level | GAN-only, 92.8 over 17 GANs (self T3) | not evaluated | code + checkpoint | none |
| SPAI | low-level (spectral) | avg AUC 91.0 on 13 modern gens (self T1) | Fig. 3 only [unverified] | code + weights (+ MFM backbone) | Apache-2.0 |
| UnivFD | foundation | mAP 93.38 (self T2); AUC 67.3 (SPAI T1); 0.592 mAP (CommFor T2) | AUC 71.8 processed (ClipDet T5); FakeInversion AUC 49.7 (B-Free T11) | code + linear-head weights | MIT |
| RINE | foundation | ACC 91.5 (self T2); AUC 85.5 (SPAI T1); Synthbuster bAcc 65.0 (B-Free T4) | JPEG 86.7 (self T5); FakeInversion AUC 62.4 (B-Free T11) | code + checkpoints | Apache-2.0 |
| ClipDet | foundation | avg AUC 89.8 on 18 gens (self T4) | crop/resize/JPEG AUC 85.2 with "+" ref set (self T5) | code + weights (git LFS) | Apache-2.0 |
| FatFormer | foundation | diffusion ACC 95.0 (self T2); in-the-wild avg bAcc 58.2 (B-Free T7) | GAN-set JPEG 95.9 (self); FakeInversion AUC 68.2→48.7 (B-Free T11) | code + checkpoints | Apache-2.0 |
| C2P-CLIP | foundation | UFD mAcc 93.79 (self T2); in-the-wild avg bAcc 64.2 (B-Free T7) | not evaluated (self); FakeInversion AUC 74.1→71.9 (B-Free T11) | code + GDrive weights | none |
| Effort | foundation | UFD Acc 95.19 (self T2–3); GenImage 91.1 (self T9) | AIGI not evaluated; phone-photo FPR 3.6–91.3% by threshold (LAION-Mobile) | code + GDrive checkpoints | **CC BY-NC 4.0 (NC)**, README badge only |
| DINOv3 probe | foundation | WebAIG-25 avg 94.5 (self T4, private, confounded); WildRF 96.4 (self T2) | not evaluated | code not found | n/a (DINOv3 licence unchecked) |
| DIRE | reconstruction | AP ~0.46–0.51 (AEROBLADE T3); GenImage 71.24 (DRCT T2) | paper claims no JPEG drop [unverified]; format bias | code (archived) + Baidu weights | none |
| AEROBLADE | reconstruction | AP ~0.99 on LDM family (self T1); AUC 0.444 on 20-gen mix (Manifold T1) | Fig. 7 only [unverified] | code; uses public AEs | none |
| LaRE² | reconstruction | GenImage avg ACC 79.1 (self T1) | not evaluated | code; weights [unverified] | Apache-2.0 |
| ZED | reconstruction (zero-shot) | avg AUC 90.0 on 30 sets (self T2) | compressed vs raw reals similar (T4); resize untested | code "coming soon" [unverified] | **GRIP-UNINA, NC** |
| DRCT | reconstruction (training) | DRCT-2M-Wild 96.90 (self T3); GenImage 89.49 (self T2) | Fig. 5 only; most attack-resistant in RAID | code + ModelScope weights | none |
| FIRE | reconstruction | ACC 68.1–72.9 on newer gens (self T2) | Fig. 7 only [unverified] | code + GDrive weights | MIT |
| Manifold-bias | reconstruction (zero-shot) | AUC 0.835 on ~20 gens (self T1) | JPEG AUC 0.79 (self T2) | code; HF weights at runtime | none |
| B-Free | data-bias | avg bAcc 96.3 (self T5); 90.4 on 3 wild sets (self T7) | WildRF 86.2–97.3 bAcc (self T5) | code + weights | **GRIP-UNINA, NC** |
| Community Forensics | data-bias | 89.2% / 0.987 mAP on 21 unseen (self T2) | qualitative only (Fig. 7) | code + HF weights | MIT (datasets **NC**) |
| AIDE | data-bias | Chameleon 65.77, fake acc 26.80 (self T5) | JPEG QF50 69.60 (self T6) | code + GDrive weights | repo [unverified]; Chameleon academic-only |
| LaDeDa | data-bias | ProGAN-trained → WildRF 68.3 ACC (self T3) | Reddit-trained 85.7 ACC (self T5) | code + GDrive weights | [unverified] |
| Synthbuster detector | data-bias | no numbers read [unverified] | "mild JPEG" claim only | not checked | dataset **CC BY-NC-SA 4.0** |

**Non-detector resources**

| Resource | What it is | License |
|---|---|---|
| GenImage | 1.35M fakes from 8 generators + ImageNet reals | **CC BY-NC-SA 4.0** |
| Synthbuster (data) | 9 × 1,000 PNG fakes; RAISE reals not included | **CC BY-NC-SA 4.0** |
| B-Free extended Synthbuster | RAISE-1k resized PNG + FLUX + SD3.5 + LDM | **NC** (GRIP licence) [data licence unverified] |
| Community Forensics Eval | 51,836 images, 21 held-out generators | **CC BY-NC-SA 4.0** |
| Chameleon | 26k human-filtered images | **academic-only, gated** |
| WildRF | 5,300 social-media images | none found [unverified] |
| VISION | real phone JPEGs + WhatsApp/Facebook copies | CC BY-SA 4.0 (commercial use allowed) |
| RAISE | raw DSLR reals | **NC** |
| Open Images V7 | Flickr reals | CC BY 2.0 per image (check each) |
| RAID | 72k transfer-adversarial images | MIT |
| TrueFake | 600k images incl. platform-shared copies | repo CC-BY-4.0 (dataset unverified) |
| LAION-Mobile | phone-photo IDs + scores | HF tag truncated [unverified] |

---

## 5. How to read results critically: a checklist

1. **Are real and fake in the same format and size?** If reals are JPEG and fakes PNG (GenImage, ProGAN/ForenSynths, COCO-vs-generated), part of the score is format detection `[fakeorjpeg-arxiv2024; dire-iccv2023]`. Prefer format-matched sets (Synthbuster with RAISE PNG; B-Free extended) or match the formats yourself.
2. **Ranking metric or threshold metric?** AUC/AP can stay ~90 while accuracy at the deployed threshold is at chance (Corvi T1: AUC 90.8 vs Acc 61.5). Ask which threshold was used and where it came from: ProGAN validation, a fixed 0.5, or an oracle tuned on the test set.
3. **Is the FPR on fresh real photos reported at a fixed threshold?** Almost never. LAION-Mobile shows FPR on the same real photos swinging between 0.2% and 39.1% depending only on the threshold source. Report TPR at a fixed FPR (e.g. 1%) on independent real photos, including modern phones and platform-shared copies.
4. **Are the "unseen" generators really unseen?** Check whether training and test generators share a family or an autoencoder: SD1.4 → SD1.5 is not unseen (GenImage T3: 99.7). Also check for exposure through the foundation model's pretraining data (vfm-baseline's motivation for WebAIG-25).
5. **Do real and fake differ in domain?** Different sources, content or eras (campus photos vs Reddit Midjourney; Unsplash vs ArtStation; hashtag labels) let the detector learn source instead of generator.
6. **Is the test benchmark independent of the authors?** Compare self-reported numbers with third-party ones. The same detector varies hugely across benchmarks: UnivFD scores 93.38 mAP in its own paper, 67.3 AUC on SPAI's test and 0.592 mAP on Community Forensics.
7. **Is real-world processing tested, and how hard?** "Robust to JPEG" often means one augmentation family, applied once, shown only in a figure. Look for resize + JPEG combinations, real platform pipelines (TrueFake), screenshots (no KB paper tests these) and phone ISPs.
8. **Are baselines re-run or copied?** Copied numbers mix protocols (FatFormer: 95.0 in its own protocol vs 90.86 in C2P-CLIP's).
9. **Is the test set big enough?** WildRF Facebook has 320 images; WebAIG-25's size is not stated; RAID's commercial test uses 50 pairs.
10. **Were hyper-parameters tuned on the test set?** RINE chose them by test ACC+AP.

---

## 6. Glossary

- **AUROC / AUC**: probability that a random fake scores higher than a random real. 0.5 is chance; below 0.5 means the scores are inverted. Threshold-free: it measures ranking only.
- **AP / mAP**: average precision, the area under the precision-recall curve; mAP is its mean over test sets. Also threshold-free.
- **Accuracy (Acc)**: fraction correct at a specific threshold, often 0.5. It mixes ranking quality with calibration.
- **Balanced accuracy (bAcc)**: the mean of real-class accuracy and fake-class accuracy, so class imbalance cannot inflate it. 50 is chance. B-Free reports it at a fixed threshold.
- **Real acc / fake acc (TNR / TPR)**: accuracy on each class separately. "98% real acc, 2% fake acc" means the detector calls almost everything real.
- **TPR@1%FPR**: the share of fakes caught when the threshold is set so that only 1% of reals are flagged. It is the practical deployment number. The KB mostly has TPR@5%FPR (AEROBLADE T3); no KB paper reports TPR@1%FPR.
- **FPR@95%TPR**: the share of reals falsely flagged when 95% of fakes are caught. No KB paper reports it.
- **EER**: equal error rate, the threshold where FPR equals FNR. LAION-Mobile moves EER thresholds from one dataset to another.
- **Calibration / ECE / NLL**: whether a score of 0.8 really means an 80% chance of fake. ECE is expected calibration error; NLL is negative log-likelihood. **Platt scaling** fits a logistic function to the scores using a few labelled examples (Corvi T2).
- **Linear probe**: freeze a pretrained network and train only a linear classifier on its features (UnivFD, DINOv3 probe).
- **Frozen backbone / fine-tuning**: frozen means the pretrained weights are not changed; fine-tuning updates them all (B-Free, Community Forensics).
- **LoRA**: low-rank adaptation. Small trainable low-rank matrices are added to frozen weights (C2P-CLIP). **Adapter**: a small trainable module inserted between frozen layers (FatFormer).
- **CLS token / intermediate layers**: a ViT's summary token. Taking it from every block (RINE) keeps low-level information that the last layer discards.
- **CLIP / DINOv2 / DINOv3**: large pretrained vision encoders, trained on image-text pairs (CLIP) or self-supervised on images (DINO).
- **VAE / latent diffusion (LDM)**: Stable Diffusion runs diffusion in a compressed latent space; a VAE decoder turns the latent into pixels. That decoder leaves traces, which AEROBLADE, FIRE and LaRE² exploit.
- **Pixel-space diffusion (ADM, GLIDE, DALL-E 2)**: diffusion run directly on pixels, with no VAE; these often defeat LDM-trained detectors.
- **DDIM inversion**: running a diffusion sampler backwards to find the noise that would generate an image (DIRE).
- **Reconstruction error**: the difference between an image and its regeneration by a model; LPIPS is a perceptual distance used for it.
- **Up-sampling artifact**: periodic or checkerboard correlations left by generator up-sampling layers. Visible in the DCT/FFT spectrum (Frank) or in neighbouring pixels (NPR).
- **Noise residual / fingerprint**: image minus a denoised version of it; averaging residuals shows generator-specific periodic patterns (Corvi).
- **DCT / FFT / DWT**: frequency transforms (cosine, Fourier, wavelet).
- **JPEG quality factor (QF / Q)**: from 1 to 100; lower means stronger compression. Platforms use roughly 61–92 (TrueFake).
- **Format bias / shortcut learning**: the model learns an incidental difference between classes, such as JPEG vs PNG, size or content, instead of the real signal.
- **ProGAN protocol / ForenSynths**: train on ProGAN-vs-LSUN, test on other generators (Wang et al. 2020). Many papers reuse it, and it is GAN-heavy.
- **In-distribution vs unseen**: whether the test generator, or its family, was in the training data.
- **Zero-shot / training-free detector**: needs no fake training images (AEROBLADE, ZED, Manifold-bias). Few-shot uses a handful.
- **Neural ISP**: modern phone image pipelines (multi-frame fusion, learned denoising and sharpening) that may produce "generated-looking" statistics in real photos. Only 212 such photos (2021+ phones) appear in LAION-Mobile.
- **EXIF**: camera metadata embedded in image files; it can be forged.
- **Laundering**: post-processing (resize, recompress, screenshot, social upload) that strips traces from a fake, whether deliberately or not.
- **Adversarial perturbation (ε = 8/255)**: a tiny, crafted pixel change, bounded by ε per pixel, that flips the detector's decision. **PGD** is the standard iterative attack.
- **White-box / black-box**: the attacker does or does not have access to the target model's gradients.
- **Transfer attack**: craft the perturbation on surrogate models and apply it to an unseen target (RAID, Mavali, FPBA).

---

## 7. Open problems, tied to KB evidence

1. **FPR on modern neural-ISP phone photos is essentially unmeasured.** LAION-Mobile has only 212 photos from 2021+ phones (0.03%), and VISION covers 2015–2017 devices. No KB paper tests a detector on recent flagship-phone JPEG/HEIF output, or on screenshots.
2. **Thresholds do not transfer.** Every family shows a large gap between AUC/AP and accuracy on unseen data: Corvi T1, UnivFD T2–3, RINE T6 (AP 99.3 vs ACC 70.8 on Firefly), ClipDet Fig. 3. Corvi's 4-image Platt calibration (T2) hints that small target-domain calibration works, but nobody calibrates on fresh real photos and reports FPR.
3. **The best new claims are unreplicated and possibly confounded.**
   - DINOv3's 94.5 comes from a private set where reals and fakes differ in domain, and the probe was trained on biased GenImage without augmentation `[vfm-baseline-arxiv2025]`.
   - B-Free, Community Forensics, SPAI and Effort have no independent in-the-wild evaluation in the KB, apart from LAION-Mobile's FPR test of Effort.
4. **Generalization across families is still the failure point.**
   - DMID gets 50.0 on DALL-E 3 `[bfree-cvpr2025, T5]`.
   - ProGAN-trained RINE gets DALL-E 3 21.1 `[rine-eccv2024, T6]`.
   - AEROBLADE gets AUC 0.444 on a mixed set `[manifold-bias-iclr2025, T1]`.
   - FIRE gets 68–73% on newer generators `[fire-cvpr2025, T2]`.
   - DRCT with an SDv1 reconstructor gets 70.04 on SDXL-Turbo `[drct-icml2024, T1]`.
   - Reconstruction and LDM-trained detectors are tied to the VAE family they saw.
5. **Resize and screenshot robustness is barely studied.**
   - UnivFD and RINE never resize (centre crop only); ClipDet resizes to 224.
   - SPAI's resize results are figure-only.
   - B-Free's social simulation uses only a mild 0.7–1.0 resize.
   - Screenshots appear nowhere in the KB.
6. **Robustness to processing and to attacks have not been combined.**
   - Transfer attacks survive JPEG Q50 / 75% resize `[mavali-aigi-robustness-arxiv2025]`.
   - Blur/JPEG training does not defend `[fpba-tmm2025, T8]`.
   - The robust-CLIP defence costs 3–7 benign AUC.
   - Adversarial training did not converge in FPBA.
7. **Benchmarks give contradictory verdicts on the same detector.** NPR on DALL-E 3 is AUC 97.1 in SPAI T1 but bAcc 0.6 [unverified] in B-Free T5. NPR on FakeInversion is 87.0 bAcc but 50.7 on GenImage (B-Free T7). We have no shared, format-matched, modern, phone-inclusive benchmark.
8. **Data and licensing.**
   - The cleanest evaluation data (Synthbuster, B-Free extended, CommFor-Eval, RAISE, Chameleon) is non-commercial or gated.
   - VISION (CC BY-SA 4.0) is the only licence-clean phone-photo source in the KB, and it is old and already used in Community Forensics' training data, so it is not independent of that detector.

---

## Appendix: contradictions and caveats found in the KB

**NPR results**
- *DALL-E 3:* SPAI T1 gives AUC 97.1, while B-Free T5 gives bAcc 0.6 [unverified]. Taken at face value these conflict: an AUC near 97 is incompatible with near-total misclassification unless thresholds, checkpoints or real sets differ. SPAI used SIDBench default checkpoints; B-Free's setup is not detailed in the KB.
- *"49.9":* balanced accuracy on FakeInversion only under one simulation (B-Free T11), not a general figure (LOG.md). On FakeInversion without simulation, NPR scores 87.0 bAcc.
- *Paper's own count:* NPR's Table 6 caption says 38 sub-testsets; the text says 28.
- *WildRF:* bAcc 50.0/78.3/51.8 (B-Free T5), mean ACC 81.9 when trained on Reddit (LaDeDa T5), and avg 63.5 (vfm T2). These use different training protocols, so they are not directly comparable.

**Self-reported vs third-party benchmark numbers**
- *UnivFD across benchmarks:* 93.38 mAP (self), 84.1 AUC (ClipDet), 88.4 AUC (ZED's table), 81.5 AUC (Mavali), 67.3 AUC (SPAI), 0.592 mAP (Community Forensics). The spread comes from benchmark choice, not a single error.
- *FatFormer:* 95.0 in its own protocol vs 90.86 in C2P-CLIP's.
- *Index wording:* the index row for FatFormer says "near chance on social-network data". B-Free's Reddit figure (69.5) is above chance, so the wording overstates it slightly.
- *AEROBLADE:* AP ~0.99 (self, LDM-only) vs AUC 0.444 (Brokman, mixed set) vs ~98 ACC on DiffusionForensics (FIRE T1). All three are plausible because they are different test sets.

**Inconsistencies inside individual papers or KB entries**
- *LaRE²:* average 79.1 (T1) vs 85.9 (T2) vs 84.5 (T3) within one paper.
- *ClipDet:* text claims +6.8 AUC; the table gap is 5.6.
- *C2P-CLIP:* text swaps the prompt-pair values relative to its tables.
- *Carlini-Farid:* "10×" vs "16×".
- *Manifold-bias:* test-set size 100k+100k vs 200,000.
- *WildRF size:* 5,300 counted vs "around 5000" in the text.
- *Community Forensics:* mAcc 0.892 (paper) vs 0.893 (dataset card).
- *Effort KB entry:* says no independent in-the-wild evaluation exists, but `laion-mobile-arxiv2026` does evaluate Effort (FPR on real phone photos).
- *DIRE:* the paper claims no drop under JPEG/blur, while AEROBLADE's reproduction finds a format shortcut and near-chance performance.
