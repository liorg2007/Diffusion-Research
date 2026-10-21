# Evading Deepfake-Image Detectors with White- and Black-Box Attacks
- **id:** carlini-farid-cvprw2020
- **Authors / venue / year:** Nicholas Carlini, Hany Farid. CVPR Workshops 2020 (venue per CPS-VO archive record; not independently confirmed from CVF page).
- **Links:** arXiv 2004.00622 · https://arxiv.org/abs/2004.00622 · code: none released (footnote 1)
- **Family:** robustness-attacks
- **Status:** partially-verified

## Summary (plain language)
Shows that a real/fake classifier for GAN images (Wang et al.'s ResNet-50) can be driven to near-zero accuracy by tiny perturbations, even flipping only the lowest bit of a few percent of pixels, and that a transfer attack from a different surrogate works without access to the target. Foundational template for later attacks on diffusion detectors.

## How it works
Targets: Wang et al. ResNet-50 (AUC 0.96-0.97) and Frank et al. frequency classifier. Threat models: white-box and black-box (transfer from authors' ResNet-18 surrogate, AUC 0.85, to Wang et al.). Inputs 224x224 RGB in [0,1].
Attacks: (1) L2 distortion-minimizing, Adam 1000 iters, binary search over c, threshold at 5% FPR. (2) Lowest-bit L-inf (max change 1/255): ~2% pixels flipped -> 71.3% fakes misclassified; ~4% -> 89.7% (Fig 2a). (3) Loss-maximizing with 40% pixel lowest bits flipped: AUC 0.966 -> 0.27 (Fig 3a). (4) Universal 24x24 patch (1% of image): AUC 0.966 -> 0.085 (Fig 3b, 4a). (5) StyleGAN latent-space universal vector: AUC 0.99 -> 0.17 (Fig 3c).

## Training data
Wang et al.: 720k train / 4k val, half real half ProGAN, with blur + JPEG augmentation. Eval: 94,036 images, 10,000 fakes per white-box attack. Surrogate training size is stated inconsistently in the paper (1M vs 1M each class).

## Reported results
- L2: distortion 0.02 -> TPR 50%; 0.1 -> TPR 0.1%.
- Frank et al.: 50% lowest bits flipped -> TPR 99% -> 0%.
- Black-box transfer (Fig 3d): target AUC 0.96 -> 0.22; source 0.85 -> 0.003.
- Perturbation needed ~10x (Sec 3 text says 16x in 3.1; inconsistent) smaller than for ImageNet classifiers.
- Compression: lowest-bit attacks survive both PNG and JPEG saving (Fig 3a solid=JPEG, dashed=PNG) [JPEG quality not extracted: unverified]. Wang et al. stays AUC >0.94 under blur/JPEG laundering without attack; resize/recompress/crop can cut TPR by >10 points.

## Known weaknesses / failure modes
Pre-diffusion (ProGAN/StyleGAN) only; the detector is a 2020 CNN. Perturbations show 8x8 JPEG-grid structure, hinting at JPEG training bias. Later work (Mavali 2024, RAID 2025) re-tests on diffusion-era detectors.

## Code + weights
No code released by design. Target model weights are from Wang et al. (not covered here).

## Verified?
Partially: numbers from ar5iv HTML via fetch summary; PDF not read directly.

## Relevance to this project
- Establishes that even 1-bit/1% perturbations break CNN forensic detectors; PNG-vs-JPEG saving does not rescue.
- Their transfer surrogate recipe (cheap ResNet-18) is a model for our black-box attack test.
