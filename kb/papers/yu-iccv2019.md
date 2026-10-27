# Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints
- **id:** yu-iccv2019
- **Authors / venue / year:** Ning Yu, Larry Davis, Mario Fritz / ICCV 2019 (per task brief; fetched ar5iv page states no venue) **[unverified]** / 2019
- **Links:** arXiv 1811.08180 · code https://github.com/ningyu1991/GANFingerprints
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Trains a CNN to say which GAN (or real) produced an image and shows that its features act as per-image and per-model fingerprints. Attribution is near-perfect for clean images, but collapses under JPEG/noise unless trained with those perturbations.

## How it works
CNN attribution classifier over real + K GANs; final-layer features = image fingerprint, final-layer weights = model fingerprint (Sec. 3.1). A separate AutoEncoder/residual variant visualises fingerprints (Sec. 3.3). Robustness 'defence' = fine-tune on the known attack.

## Training data
ProGAN, SNGAN, CramerGAN, MMDGAN trained on CelebA and LSUN-bedroom, 128x128; ten ProGAN seeds for seed-level attribution. Unconditional GANs only.

## Reported results
- Closed-set (Table 1, real+4 GANs, CelebA/LSUN): Ours 99.43/98.58; PRNU baseline 86.61/67.84; kNN 28.00/36.30.
- Seed variants (Table 3): Ours 99.14/97.04 (CelebA/LSUN).
- Perturbed, before -> after attack-specific finetuning (Tables 7/8, CelebA / LSUN): blur 49.64->97.20 / 74.48->96.68; crop 46.80->98.28 / 86.20->97.30; JPEG 8.77->88.02 / 24.73->92.40; noise 9.14->93.02 / 11.80->95.30; combination 19.31->72.64 / 24.44->83.42.
- Open-set: not addressed.

## Known weaknesses / failure modes
- Classifier 'completely fails' on noise and JPEG before finetuning (paper's words); defence assumes the attack type is known.
- Only 4 GAN architectures and 2 datasets; white-box GANs.
- Authors claim black-box applicability but do not test it.

## Code + weights
Code at https://github.com/ningyu1991/GANFingerprints. License: not checked **[unverified]**.

## Verified?
Partially: numbers from ar5iv HTML fetch (fetch-tool summary), not PDF.

## Relevance to this project
- Strongest early evidence that laundering (JPEG: 8.8% / 24.7% unaugmented) destroys fingerprint attribution; augmentation helps only for known attacks.
- Seed-level attribution works on clean GAN images, hinting that fingerprints are weight-specific, not only architecture-specific.
