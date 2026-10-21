# Phase 2: gaps and hypotheses

**Written:** 2026-10-21. **Inputs:** kb/index.md, the 34 kb/papers entries, the 8 kb/datasets entries, kb/SCHEMA.md, LOG.md, DECISIONS.md, plus targeted web searches (URLs listed at the end).
**Goal reminder:** a new method, or a measurable improvement, for UNSEEN generators and in-the-wild images (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution wins do not count. Full-scale targets: FPR <= 1% on fresh phone photos, AUROC >= 0.95 after social-media laundering, and robustness to transfer attacks.

Numbers tagged [unverified] in the KB are treated with caution below. Every number cites a kb id and table. Web-only facts cite a URL I opened, and most of those I read only at abstract level, as noted in each idea file.

---

## 1. Gaps in the literature

**G1: The decision threshold does not transfer, and FPR on fresh real photos is almost never reported.**
- Ranking survives shift but the threshold breaks. Grag2021 keeps an average AUC of 90.8 after resize+JPEG while accuracy falls to 61.5 (corvi-icassp2023 Table 1). Platt scaling with 2 real + 2 fake images per generator lifts fused accuracy from 64.5 to 87.8 (Table 2).
- UnivFD reaches 93.38 mAP but only 81.38 accuracy at a threshold tuned on ProGAN validation data (univfd-cvpr2023 Tables 2-3).
- On Chameleon, detectors get 95-100% accuracy on reals and 0-27% on fakes (aide-chameleon-iclr2025 Table 5).
- On real phone photos, UnivFD's FPR is 0.2% at the ProGAN-EER threshold and 39.1% at the NTIRE-EER threshold. Across detectors it ranges from 17% to 91% (laion-mobile-arxiv2026).
- Almost every KB entry's "Reported results" section says FPR on fresh real photos was *not reported*. B-Free reports ECE and NLL, but only on fixed test sets (bfree-cvpr2025).
- Web: a 2026 AAAI paper recalibrates logits on a target set but targets missed fakes, not FPR control (arXiv 2602.01973). The only FPR-guarantee conformal work I found is for text (arXiv 2505.05084).

**G2: Laundering robustness is bought only through augmentation, and screenshots are untested.**
- Under B-Free's mild social simulation (resize 0.7-1 plus JPEG 70-100) on FakeInversion, AUCs fall as follows (bfree-cvpr2025 Table 11):
  - NPR: 91.6 to 43.3
  - FatFormer: 68.2 to 48.7
  - UnivFD: 54.9 to 49.7
  - RINE: 69.6 to 62.4
  - C2P-CLIP: 74.1 to 71.9
- Real Facebook sharing cuts NPR's TPR on SDXL to 0.21, and it also *raises FPR*: MISLNet TNR on FFHQ falls to 0.41 (truefake-ijcnn2025 Table IV).
- AIDE drops from 92.77 to 70.64 at QF75 (aide-chameleon-iclr2025 Table 6). CLIPDet's post-processed AUC is 78.1 without the "+" augmented reference set and 85.2 with it (clipdet-cvprw2024 Table 5).
- None of the KB papers estimates laundering invariance from *real paired* platform data. VISION provides such pairs (kb/datasets/vision-phone-photos.md).
- Screenshots (re-render, resample, PNG re-save) appear in no KB paper, and a web search found no study of them.

**G3: No FPR data on modern neural-ISP phones.**
- LAION-Mobile is the only phone-scale study. Only 212 of its images (0.03%) come from phones released in 2021 or later. Its "heavier ISP raises FPR" prediction was not supported, but that test is underpowered for modern devices (laion-mobile-arxiv2026 Sec 5.4, Fig 4, Table 1).
- VISION phones date from 2015-2017.
- SPAI uses smartphone real sources but reports only AUC (spai-cvpr2025 Table 1).
- The DINOv3 result rests on a single private campus real set (vfm-baseline-arxiv2025 Table 4).
- No study splits FPR by capture mode (night, portrait, super-resolution zoom); my web search found only anecdotes.

**G4: Frozen modern self-supervised backbones look strong, but the evidence is confounded or untested in the wild.**
- A frozen DINOv3 linear probe scores 97.5 real / 91.4 fake accuracy on WebAIG-25. Fakes there are Reddit Midjourney posts and reals are private campus photos, a source confound (vfm-baseline-arxiv2025 Table 4).
- Intermediate CLIP blocks add 9 accuracy points over the last layer (rine-eccv2024 Table 4).
- DINOv2 trails a spectral MFM backbone, 87.5 vs 91.0 AUC (spai-cvpr2025 Table 2).
- Web: frozen DINOv3 with token selection (arXiv 2511.22471) and intermediate-layer search (arXiv 2605.04358, which finds CLIP > DINOv2) exist. Neither abstract mentions phone FPR or laundering.

**G5: Training-data bias decides the outcome, and fixes are expensive or LDM-centric.**
- On GenImage, reals are JPEG and fakes PNG. Removing the size and JPEG bias adds about 11 pp cross-generator (fakeorjpeg-arxiv2024 Table 4). A ResNet-50 calls FFHQ "natural" 80.45% of the time as PNG vs 99.95% at JPEG80 (Table 2).
- DIRE's apparent strength came from format bias (aeroblade-cvpr2024 Table 3; dire-iccv2023).
- B-Free (SD2.1 inpainting) and DRCT (multi-step diffusion reconstruction) need generation passes (bfree-cvpr2025, drct-icml2024).
- Web: VAE-reconstruction alignment is already published (Rajan et al. ICLR 2025; DDA NeurIPS 2025; arXiv 2605.14486). What is open is whether it transfers to non-LDM generators and to phone FPR.

**G6: Training-free reconstruction scores are LDM-specific and fragile.**
- AEROBLADE reaches AP 0.979-0.999 on LDMs (aeroblade-cvpr2024 Table 1) but only 0.444 AUC on a 20-generator mix (manifold-bias-iclr2025 Table 1).
- The manifold criterion gets 0.835 AUC but costs about 2 s per image on an A100 plus LLaVA captioning (manifold-bias-iclr2025).
- FIRE reaches 68-73% accuracy on unseen newer generators (fire-cvpr2025 Table 2).
- ZED's code was not released when fetched (zed-eccv2024).
- Web: the double-reconstruction ratio already exists as AEDR, which targets attribution.

**G7: Adversarial transfer robustness is studied apart from laundering and calibration.**
- RAID transfer attacks drop UnivFD-style AUROC from 0.95 to 0.50, and a held-out DINOv2 baseline's accuracy from 83.1 to 54.6 at eps 16/255 (raid-arxiv2025 Tables 2, 4).
- Attacks at eps 8/255 survive 75% resize + Q50 (mavali-aigi-robustness-arxiv2025). The RobustCLIP defense costs 3-7 AUC on clean data (Table VI).
- Blur/JPEG training gives no defense (fpba-tmm2025 Table VIII).
- No KB paper measures attack, laundering and a calibrated FPR together.

**G8: Perturbation-drift ("robustness asymmetry") detectors have never been tested with matched compression history.**
- RIGID-DINOv3 reaches only 0.624 AUC on NTIRE 2026 content (laion-mobile-arxiv2026 Table 4).
- Format shortcuts are documented for trained detectors (G5).
- Web: the RIGID, RA-Det and DRIFT abstracts do not mention controlling real vs fake JPEG history. I did not read their full experimental sections.

---

## 2. Ranked hypotheses

Scores run from 1 to 5. **Novelty** reflects the web check: 1 = essentially published, 3 = a new combination or a test nobody has run, 5 = new mechanism. **Gain** = expected effect on unseen-generator / in-the-wild / FPR metrics. **Feasibility** = on this Intel Mac with the AMD 4 GB GPU, at most 5 GB of data, and no cloud.

| Rank | id | One-line hypothesis | Gaps | N | G | F | N×G×F |
|---|---|---|---|---|---|---|---|
| 1 | [laundering-nuisance-projection](../kb/ideas/laundering-nuisance-projection.md) | Project out feature directions that move under *real* WhatsApp/Facebook/JPEG+resize pairs before a frozen-DINOv3 linear probe: ≥ +0.03 AUROC on held-out FLUX at q75+0.5x vs an augmentation-only probe, with ≤ 0.01 clean loss | G2, G4 | 3 | 3 | 5 | **45** |
| 2 | [worst-env-conformal-fpr](../kb/ideas/worst-env-conformal-fpr.md) | A real-only, max-over-device-group 99th-percentile threshold keeps leave-device-out FPR ≤ 1.5% and leave-platform-out FPR ≤ 3%, at ≤ 30% TPR cost vs pooled conformal | G1, G3 | 2 | 4 | 5 | **40** |
| 3 | [neural-isp-fpr-audit](../kb/ideas/neural-isp-fpr-audit.md) | Night, portrait and zoom photos from a 2021+ phone exceed 2× the VISION-calibrated FPR | G3, G1 | 4 | 3 | 3 | **36** |
| 4 | [drift-format-shortcut](../kb/ideas/drift-format-shortcut.md) | RIGID-style noise-drift AUROC drops ≥ 0.10 once real and fake share JPEG history, while the supervised probe drops < 0.05 | G8, G5 | 3 | 2 | 5 | **30** |
| 5 | [canonicalize-vs-transfer-attack](../kb/ideas/canonicalize-vs-transfer-attack.md) | A frozen DINOv3 probe outside the attacked ensemble loses ≤ 0.10 AUROC on RAID at eps 8/255, and test-time q75+0.5x recovers ≥ 50% of the loss | G7 | 2 | 2 | 4 | **16** |
| 6 | [aligned-vae-frozen-probe](../kb/ideas/aligned-vae-frozen-probe.md) | A probe trained only on real vs JPEG-matched VAE round-trips reaches FLUX AUROC ≥ 0.85 laundered, and adding these pairs to LOGO training adds ≥ 0.02 | G5, G4 | 1 | 3 | 4 | **12** |
| 7 | [vae-double-recon-ratio](../kb/ideas/vae-double-recon-ratio.md) | The AEDR-style L1/L2 ratio, taken as the min over an SD1.5/SDXL VAE bank, beats AEROBLADE by ≥ 0.05 AUROC, holds ≥ 0.65 laundered, and adds ≥ 0.02 in fusion | G6 | 2 | 2 | 3 | **12** |

(The tie at 12 is broken by feasibility. The aligned-VAE probe also shares most of its pipeline with the required B0 baseline.)

**Novelty check:**
- Nothing here is a new detection *mechanism* at the level of a CVPR main-track paper.
- **#1** is closest to DRIFT (arXiv 2606.06918, one-class learned robust/fragile subspaces; only the abstract was read). Its distinctive parts are a closed-form projection, real platform-shared pairs, and a supervised probe.
- **#2** is standard conformal methodology applied where no image-detection paper has applied it. A non-peer-reviewed HF model card already does device-held-out "budget" calibration on VISION and shows it fails on another real set, which is a warning, not prior art for the guarantee.
- **#3** is a measurement, the most novel item, but it depends on the owner supplying photos.
- **#6 and #7** are published ideas (Rajan ICLR 2025 / DDA NeurIPS 2025; AEDR). They are kept only because they are cheap ablations and fusion inputs.
- One idea was dropped: "multi-layer frozen DINO probe" as a standalone contribution. It is crowded (arXiv 2511.22471, 2605.04358, vfm-baseline-arxiv2025), so it becomes the baseline B0.

---

## 3. Recommendation for Phase 4: test #1, #2 and #3, with #4 as a free add-on

1. **laundering-nuisance-projection (#1)** is the only *method* idea with honest novelty ≥ 3 that targets the AUROC-after-laundering criterion. It costs seconds once features are cached, and both outcomes are informative. If it fails, the failure measures how much of the frozen fake signal lives in laundering-sensitive directions, which decides whether to invest in laundering-invariant features at all.
2. **worst-env-conformal-fpr (#2)** is the only idea that directly attacks the FPR ≤ 1% criterion, which no KB detector controls. It reuses the same cached scores and is detector-agnostic: it wraps B0, UnivFD and CommFor alike. It also turns VISION's 35 devices × 4 versions into a proper leave-device-out and leave-platform-out evaluation.
3. **neural-isp-fpr-audit (#3)** is the "fresh phone photos" environment that #2's guarantee must survive. **Owner action needed now:** about 250 photos from their own 2021+ phone (default, night, portrait, ≥ 5x zoom), plus WhatsApp-sent copies. They stay local and are never committed. Without them, #2's claim stops at 2017-era phones.
4. **Add-on:** drift-format-shortcut (#4) adds about 1 hour of noise-perturbed forwards on the same images. It should run before anyone proposes a training-free drift score for fusion.

Deferred:
- #5 needs a RAID download whose structure has not been checked. Run it after B0 is fixed, as the brief's transfer-attack check.
- #6 and #7 need VAE timing on this GPU, which is unknown. Run a 20-image pilot when idle.

**Decision rule after Phase 4:**
- If #1 passes and #2 passes on VISION but #3 shows a ≥ 2× FPR jump, the next step is calibrating on modern-phone data, not more detector work.
- If #1 fails with clean AUROC loss > 0.02, drop invariance-by-projection and fall back to augmentation plus calibration (#2).

---

## 4. Minimal shared infrastructure

**Data, at most 5 GB in total.** Sizes are estimated from KB file sizes.
- VISION (CC BY-SA 4.0): 18 native photos from each of 35 devices = 630 photos (about 2.0 GB at about 3.16 MB each), plus their WA, FBH and FBL copies (about 0.4 GB, estimate).
  - Device split: D_train = 10 devices, used for probe training, projector pairs and the aligned arm. D_eval = 25 devices, used for conformal calibration and test folds.
- RAISE-1k resized PNG from B-Free extended Synthbuster: 300 images, range-fetched (about 0.5 GB). Split 150 train / 50 val / 100 test.
- Synthbuster: 100 PNG × 9 generators, range-fetched (about 1.2 GB): DALL-E 2, DALL-E 3, Firefly, MJ v5, SD1.3, SD1.4, SD2, SDXL, GLIDE.
- B-Free extended: FLUX 150 + SD3.5 150, range-fetched from sd3_flux.zip (about 0.5 GB, estimate).
- Not counted against the budget: the owner's phone photos (#3) and an optional RAID sample (#5).

**Splits.**
- **FLUX is held out entirely.** It is never used in training, in choosing C or k, or in calibration.
- LOGO runs over the other 10 generators: per generator, 60 train / 40 test, and the held-out generator uses all 100 for test.
- Reals are format-matched for AUROC. RAISE PNG pairs with PNG fakes. When VISION JPEG reals are scored against fakes, the fakes get a single JPEG at matched quality: q90, or the VISION median QF estimated from quantization tables.
- No image appears in more than one split. Laundered copies inherit their source's split.

**Laundering conditions.** Each is applied to both classes.
- C0: clean.
- C1: JPEG q75 + 0.5× resize (the primary condition, from the brief).
- C2: JPEG q90 (format match).
- C3: Facebook-like: downscale to width ≤ 720 + JPEG with QF drawn from 61-92 (truefake-ijcnn2025).
- C4: Mavali "Medium": 75% resize + Q50 (mavali-aigi-robustness-arxiv2025).
- C5: screenshot-sim: display resample by {0.5, 0.67, 1.25}, bilinear interpolation, 8-bit sRGB, PNG save, 2-5% border crop.
- **C4 and C5 are never used for fitting** (no projector, probe or threshold). They are held-out laundering types.
- VISION WA/FBH/FBL copies are real platform laundering.

**Feature cache.** Stored as fp16 .npy, keyed by (image_id, condition, view).
- **Backbones:**
  - DINOv3 ViT-L/16 via transformers 4.57.6, which already ran on MPS with random weights (D-004). The weights are gated; check the licence first [unverified]. Fallback: DINOv2 ViT-L/14 with registers.
  - OpenAI CLIP ViT-L/14, for UnivFD (last layer + released head, MIT) and RINE-style baselines.
- **Tokens:** CLS + mean patch token from blocks {4, 8, 12, 16, 20, 24} (12,288-d per view). For CLIP, also all 24 CLS tokens (for RINE).
- **Views:**
  - V1: resize the short side to 224 + center crop.
  - V2: native-resolution center 224 crop. This is UnivFD/RINE's own protocol.
  - V3 (subset ablation): mean over 4 native-resolution tiles.
- **Volume:** about 16.5k image-conditions × 2 views ≈ 33k forwards per backbone. That is about 1 hour at an estimated ~10 img/s for ViT-L on MPS, extrapolated from the measured ResNet-50 141 img/s and a ~15× FLOP ratio. **Pilot 50 images first.** About 0.8 GB on disk per backbone.

**Released-detector scores to cache once,** for every image and condition:
- UnivFD (MIT)
- CommFor ViT-S-384 (MIT, HF OwensLab)
- RINE LDM checkpoint (Apache-2.0)
- B-Free (non-commercial, research use only; heavier at 504-pixel crops, so run it on the test splits only)

**Baseline B0, used by #1-#5:** logistic regression on DINOv3 multi-layer V1 features, trained LOGO on clean + C1/C2-laundered copies. Fake training images get a random q85-95 JPEG with 50% probability to match camera-JPEG history.

**Metrics,** reported per generator, LOGO mean, and FLUX, for each condition:
- AUROC, AP, balanced accuracy at the chosen threshold, FPR@95%TPR, TPR@1%FPR
- For thresholds: empirical FPR with Clopper-Pearson CI on VISION D_eval (leave-device-out, leave-platform-out) and on the owner's phone photos
- Bootstrap 95% CIs throughout

---

## 5. Sources opened in this phase (web)
- Rajan et al., Aligned Datasets Improve Detection of Latent Diffusion-Generated Images (ICLR 2025): https://arxiv.org/abs/2410.11835v3
- Dual Data Alignment (arXiv 2505.14359): https://arxiv.org/abs/2505.14359
- Reduce the Artifacts Bias (arXiv 2605.14486): https://arxiv.org/abs/2605.14486
- Rethinking Cross-Generator Image Forgery Detection through DINOv3 (arXiv 2511.22471): https://arxiv.org/abs/2511.22471v1
- Intermediate Representations are Strong AI-Generated Image Detectors (arXiv 2605.04358): https://arxiv.org/abs/2605.04358
- AEDR (arXiv 2507.18988; AAAI OJS listing): https://arxiv.org/html/2507.18988v2 · https://ojs.aaai.org/index.php/AAAI/article/view/37930
- DRIFT (arXiv 2606.06918): https://arxiv.org/abs/2606.06918
- RA-Det (arXiv 2603.01544): https://arxiv.org/abs/2603.01544
- RIGID (arXiv 2405.20112): https://arxiv.org/abs/2405.20112
- Forensic Self-Descriptions (CVPR 2025; real-only training; noted as prior art for real-only detection, not used in a hypothesis): https://arxiv.org/abs/2503.21003
- Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated (arXiv 2602.01973): https://arxiv.org/abs/2602.01973v1
- Conformal FPR bounding for machine-generated text (arXiv 2505.05084): https://arxiv.org/pdf/2505.05084
- slope-scope HF model card (VISION held-out-device calibration; not peer-reviewed): https://huggingface.co/GPTchatly/slope-scope
- Search-result-only, not opened: PRNU false matches for AI images (Forensic Sci. Int. 2026): https://dfrws.org/wp-content/uploads/2026/03/Boon-or-Bane-Source-Camera-Identificati_2026_Forensic-Science-International.pdf · reproduction study of aligned training (arXiv 2512.21562)
