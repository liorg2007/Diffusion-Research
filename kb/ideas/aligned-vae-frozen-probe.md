# Generator-free training: frozen DINOv3 multi-layer probe trained only on real photos vs their (JPEG-history-matched) VAE round-trips
- **id:** aligned-vae-frozen-probe
- **Gap addressed:** G5 (training-data bias / alignment), G4 (frozen modern SSL backbones)
- **Scores (1-5):** novelty 1 · expected gain on unseen/in-the-wild 3 · Mac feasibility 4 · **product 12** (rank 6; but its LOGO-trained sibling B0 is required infrastructure)

- **Hypothesis:** A logistic probe on frozen DINOv3 multi-layer features trained only on real photos vs their SD1.5-VAE and SDXL-VAE round-trips (round-trips of JPEG reals re-saved at the source JPEG quality; zero generator samples) reaches AUROC >= 0.85 on the never-seen FLUX generator after JPEG q75 + 0.5x resize, >= UnivFD + 0.10; and adding these aligned pairs to the leave-one-generator-out training set (B0) raises held-out FLUX AUROC by >= 0.02.

- **Rationale:**
  - Content/format/size alignment removes shortcuts: B-Free's content-aligned SD2.1 inpainting fakes reach 96.3% average bAcc and WildRF 95.6/86.2/97.3 (bfree-cvpr2025 Table 5); fixing GenImage's JPEG/size bias adds ~11 pp cross-generator (fakeorjpeg-arxiv2024 Table 4); DRCT's reconstructed hard negatives lift UnivFD 79.45 -> 89.49 on GenImage (drct-icml2024 Table 2).
  - A VAE encode/decode is one pass, feasible on the Mac, unlike DRCT's 10-50 denoising steps or B-Free's inpainting.
  - Freezing the backbone: B-Free reports linear probing did poorly for its low-level-oriented data (per web-search snippet of arXiv 2412.17671; not verified in the PDF), which is a reason to use *intermediate* layers (rine-eccv2024 Table 4: +9 ACC from intermediate blocks) rather than the last layer.

- **Closest prior work (this is largely done - honest novelty 1):**
  - KB: bfree-cvpr2025, drct-icml2024, aeroblade-cvpr2024 (VAE reconstruction as a training-free score), fire-cvpr2025 (VAE-only reconstruction error).
  - Web check: Rajan, Ojha, Schloesser, Lee, "Aligned Datasets Improve Detection of Latent Diffusion-Generated Images", ICLR 2025 (arXiv 2410.11835, https://arxiv.org/abs/2410.11835v3) trains a detector on real vs LDM-autoencoder reconstructions - the core idea. Dual Data Alignment (arXiv 2505.14359, NeurIPS 2025 spotlight per arXiv comments, https://arxiv.org/abs/2505.14359) adds frequency alignment because reconstructions restore high frequencies that JPEG reals lost - our "JPEG-history matching" is their point. arXiv 2605.14486 (https://arxiv.org/abs/2605.14486) argues single-VAE negatives are too homogeneous and adds SRGAN-restoration negatives with LoRA experts on a frozen backbone.
  - What remains: (a) frozen modern SSL backbone + linear probe instead of trained/LoRA networks (cheapest possible); (b) explicit test on non-LDM / undisclosed-architecture generators (DALL-E 2/3, Firefly, GLIDE in Synthbuster) and a held-out VAE family (FLUX 16-channel) after laundering; (c) calibrated FPR on phone photos. These are evaluation contributions, not a new method.

- **Evidence for / against:**
  - For: alignment literature above; SD-family transfer is strong for LDM-trained detectors (corvi-icassp2023 Table 2: LDM-trained Grag2021 SD 99.7 acc after resize+JPEG).
  - Against: LDM-trained detectors fail on DALL-E 2/3 and Firefly (corvi-icassp2023 Table 2: ADM 52.9, DALL-E 2 50.4 acc; bfree-cvpr2025 Table 5 via DMID: DALL-E 3 50.0, Firefly 51.0, FLUX 63.7 bAcc); a search-result snippet (not opened) of a reproduction study, arXiv 2512.21562, describes uneven cross-generator generalization for aligned training (table not read).

- **Test plan (Mac-feasible):**
  - Data: reals for training = RAISE train split (PNG, ~300) + VISION D_train devices native (JPEG, ~200); fakes for training = their round-trips through SD1.5 VAE and SDXL VAE at a 512x512 center crop (~1,000 VAE passes; round-trips of JPEG reals re-saved as JPEG at the source QF estimated from quantization tables; PNG reals -> PNG). FLUX VAE held out (FLUX is the held-out generator). Eval = shared eval set.
  - Model: cached DINOv3 multi-layer features, logistic regression; arms: A = aligned-only; B = B0 (LOGO on generator samples); C = A + B.
  - Baselines: UnivFD, CommFor ViT-S, B-Free (released weights).
  - **Pass:** A: FLUX AUROC at q75+0.5x >= 0.85 and >= UnivFD + 0.10; C: FLUX AUROC >= B + 0.02 (clean and laundered). Report non-LDM generators separately (expectation: A weak there). **Fail:** A < 0.75 on FLUX laundered, or C <= B.
  - Cost: VAE passes ~1,000 at 512^2 on MPS/CPU - time unknown on this GPU; pilot 20 images first (budget <= 2 h); licences of VAE weights to be checked before download [unverified].

- **Status:** proposed
