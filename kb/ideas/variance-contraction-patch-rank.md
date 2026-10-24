# Variance contraction: diffusion images have a lower effective rank of mid-level patch-token features than camera photos of matched texture
- **id:** variance-contraction-patch-rank
- **Gap addressed:** a diffusion-specific cue in the low-to-mid frequency range that laundering should preserve; training-free, no VAE needed
- **Scores (1-5):** novelty 3 · expected gain on unseen/in-the-wild 2 · feasibility 5 · **product 30** (rank 3 in docs/06-diffusion-hypotheses.md)

- **Hypothesis:** The participation ratio (effective rank) of the covariance of DINOv2-base block-8 patch tokens within one image is lower for diffusion images than for camera photos AFTER regressing out a texture/complexity covariate, giving a texture-residualised macro AUROC >= 0.75 at jpeg75_resize05 on 6 unseen Synthbuster generators, with a clean-to-laundered drop <= 0.05.

- **Mechanism (why diffusion images differ):**
  - Sampling with a learned denoiser (and classifier-free guidance) pulls samples toward high-density regions, so generated images under-represent real-image variability. Cozzolino, Poggi, Verdoliva (arXiv 2608.12155, abstract opened) report that regenerated images have "lower variance and effective dimensionality" than real data in a latent analysis, and that useful cues sit in the low-to-mid frequency range. INP-X proves a VAE variance-contraction theorem (inpx-arxiv2026, Theorem 3.2).
  - Low-to-mid frequency structure is what jpeg75_resize05 keeps, so a per-image dimensionality statistic of mid-level features should move less under laundering than high-frequency traces.
  - The E4 pair design separates two sources: if the contraction is a VAE effect, real vs VAE(real) should show it; if it is a sampling effect, only generator images will.

- **Closest prior work:** multiLID, Lorenz et al., ICCVW 2023 (arXiv 2307.02347; search-result abstract, not opened): local intrinsic dimensionality on untrained ResNet-18 features + a trained random forest. Cozzolino 2608.12155 (analysis, not a detector, no JPEG/resize per abstract). Here: training-free per-image statistic on frozen SSL patch tokens, tested under laundering with a texture-residualised score and real-photo FPR. Not found in a web search (docs/06 sources).

- **Evidence for / against:**
  - For: the above analyses; DINOv2 mid-level features carry generation cues (E4 F2 beats F1, 011/pilot/tables.md).
  - Against (strong): per-image feature rank is dominated by scene complexity and texture. Our H4 pilot showed a training-free DINOv2 drift score was below chance and mostly measured texture (AUROC 0.292; Laplacian 0.778 the other way, 006/pilot/tables.md, pilot_addendum/tables.md, via docs/05 §2). Zoom differs between arms (D-013). Expected gain is therefore low unless the residualised score survives.

- **Test plan:**
  - *Pilot now (no download, GPU minutes):* VISION test natives + WA/FBH (1,074) vs 6 Synthbuster x 120; arms fov512 (label) and orig; clean / jpeg75_resize05 / social_sim. Extract DINOv2-base patch tokens (blocks 4/8/12) once; statistic = participation ratio (sum lambda)^2 / sum lambda^2 of the token covariance. Covariates: Laplacian texture, grey-level std, JPEG QF estimate. Score = residual of a linear regression of the statistic on the covariates, fitted on train-split NATIVES ONLY.
    - Pair control: the same statistic on E4's real vs own round-trip pairs (VAE-only contraction).
  - *Validity gates:* raw statistic's |Spearman| with Laplacian among natives reported; if the residualised score's AUROC is < 0.60 at clean, UNINFORMATIVE.
  - **Pass:** residualised macro AUROC >= 0.75 laundered (CI lo >= 0.70), drop clean->laundered <= 0.05, >= texture baseline + 0.10, non-SD group >= 0.70, FPR at the natives' 1% threshold <= 2% on WA/FBH. **Fail:** residualised AUROC < 0.65 laundered.
  - *Weights:* none (DINOv2-base cached). *Compute:* < 30 min GPU [estimate].
  - *Phase-5 reach:* NO on its own; at best a fusion feature.

- **Status:** proposed
