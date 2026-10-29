# Final run (D-020): challenging the "mid-level rendering-style" attribution signal

**Registered:** 2026-10-29 20:59, before any data is downloaded and before any result exists. **Builds on:** H-A1 (experiments/020-attr-foundation; critique.md), with the same data, caption-family folds (results/folds_ha1_main.json), T-mix probe protocol and bootstrap. **Primary condition:** jpeg75_resize05. **Metric:** 8-way balanced accuracy (bacc, chance 0.125) with a 1,000-replicate caption-family bootstrap.

**Claim under test (C).** After laundering, the identity of the generator is carried mainly by mid-depth features of pretrained vision transformers. These encode local "rendering style" rather than semantics, and laundering-aware training makes them transfer to launderings never seen in training.

## Five challenges (each has a pre-registered verdict; nothing counts as a "PASS" for the project, these are findings)

**R1. Generality across backbones.**
- **Backbones** (all on disk except DINOv2-S/L): DINOv2-small, DINOv2-base (reference), DINOv2-large, OpenAI CLIP ViT-L/14, LAION CLIP ViT-L/14.
- **Layers:** CLS + mean-patch features at every block.
- **Input:** the same fov512 image, then each model's standard 224 centre crop and normalisation, identical for all classes.
- **Probe:** one T-mix probe per block, under the H-A1 protocol.
- **Verdict "general"** if, for ≥ 3 of the 4 non-reference backbones, the best block at relative depth 0.25–0.75 beats the last block by ≥ 0.10 bacc (paired CI lo > 0).
- **"Not general"** if this holds for ≤ 1 of them. Otherwise "partly general".
- **Report:** the full layer profile per backbone, and the best block's bacc compared with DINOv2-base.

**R2. Mechanism: what image information carries it.** DINOv2-base, all blocks, T-mix probe retrained per manipulation (the manipulation is applied to train and test alike, after laundering):
- (a) **Patch shuffle:** a random permutation of 32×32 tiles of the 224 input, which destroys global layout and semantics while keeping local texture.
- (b) **Low-pass:** Gaussian blur, σ = 1.5 px.
- (c) **High-pass residual:** image minus its blur, re-centred.
- (d) **Grayscale:** removes colour.
- **Pre-registered predictions** (style = local texture): (a) keeps ≥ 70% of the above-chance bacc of the best mid block; (b) keeps < 50%; the mid-over-last advantage persists under (a).
- **Verdict per prediction:** confirmed / refuted, with CIs. If (a) fails, the claim "local rendering style" is refuted, and the signal needs global layout or semantics.

**R3. Laundering stress map.**
- **Grid:** JPEG quality {95, 85, 75, 60, 45, 30} × resize factor {1.0, 0.75, 0.5, 0.35, 0.25}, applied to test images.
- **Probes:** the frozen T-mix best-mid-block probe of DINOv2-base, and the same probe trained with an extended T-mix that adds q45 and resize 0.35.
- **Report:** the bacc surface, the "breaking point" (where bacc < 0.50), and how far extended augmentation moves it.
- **Prediction:** smooth decay, not a cliff, and extended augmentation moves the breaking point by ≥ 1 grid step.

**R4. Scaling with data** (for future, larger-resource work).
- **Learning curves:** training images per generator n ∈ {15, 30, 60, 90, all ≈ 96}, 5 seeded subsamples by caption family, on the best mid block of DINOv2-base and of the best other backbone.
- **Fit:** error = a · n^(−b) + c by least squares on the subsample means, with a bootstrap CI for b and c.
- **Report:** the extrapolated bacc at n = 1,000 and 10,000 per generator, stated explicitly as an extrapolation with its CI, and whether the curve has plateaued (CI of b includes 0).

**R5. Novelty check (literature).**
- **Question:** has anyone published (a) a mid-layer advantage for generator *attribution* (not detection) in ViT/CLIP/DINO features, or (b) laundering-augmented attribution transferring to unseen launderings?
- **Output:** a verdict per sub-claim (novel / partially known / known), with citations checked on the primary page.

## Validity
- **Same everything:** same folds, rows, laundering code path and probe as H-A1.
- **Gates:** sha256 dedupe; a permutation control at chance; D-008 GPU vs CPU on 32 images per new backbone, where the tolerance is the larger of 1e-3 and twice the CPU run-to-run spread (the H-A6 critic's finding).
- **Selection:** block choice by inner CV on training folds for any headline number; whole profiles are reported, so nothing is cherry-picked.
- **Code freeze:** selftest committed before real-data results; afterwards bug fixes only, logged.
- **Critic and report:** one opus/high critic for the whole run; a fable/high addendum to docs/10 and a "future research" section.
