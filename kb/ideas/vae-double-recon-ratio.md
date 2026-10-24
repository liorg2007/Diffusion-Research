# Training-free VAE double-reconstruction ratio over a VAE bank as a complement to a frozen probe
- **id:** vae-double-recon-ratio
- **Gap addressed:** G6 (training-free reconstruction scores are LDM-specific and fragile; complexity confound)
- **Scores (1-5):** novelty 2 · expected gain on unseen/in-the-wild 2 · Mac feasibility 3 · **product 12** (rank 7)

- **Hypothesis:** The score s(x) = min over a VAE bank {SD1.5, SDXL} (FLUX VAE excluded because FLUX is the held-out generator; reported only as a labelled ablation) of MSE(x, R(x)) / MSE(R(x), R(R(x))) separates fakes from RAISE reals with mean AUROC >= AEROBLADE (LPIPS_2, min over the same bank) + 0.05 on the 11 eval generators, keeps AUROC >= 0.65 after JPEG q75 + 0.5x, and adds >= 0.02 AUROC on held-out FLUX when fused (logistic stacking) with the B0 probe.

- **Rationale:**
  - AEROBLADE's single-pass error is near perfect on LDM images (AP 0.979-0.999, aeroblade-cvpr2024 Table 1) but near chance on broad mixes (AUC 0.444 in Brokman et al., manifold-bias-iclr2025 Table 1), and the authors note low-complexity images are hard. Dividing by the second-pass error normalises for image complexity: for a real image the first pass loses much and the second little; for an LDM output both are small.
  - It is a cheap, generator-sample-free signal orthogonal to semantic features; fusion of orthogonal cues helped before (clipdet-cvprw2024: +3.6 AUC fusing with Corvi on original images, Table 7 text).

- **Closest prior work (largely done):**
  - KB: aeroblade-cvpr2024, fire-cvpr2025 (VAE-only error + frequency mask; 68-73% ACC on unseen newer generators, Table 2), lare2-cvpr2024, manifold-bias-iclr2025.
  - Web check: AEDR, "Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction" (AAAI proceedings per OJS listing https://ojs.aaai.org/index.php/AAAI/article/view/37930, year not checked; arXiv 2507.18988, https://arxiv.org/html/2507.18988v2) uses exactly t = L1/L2 of two consecutive AE reconstructions (with a homogeneity calibration) for per-model attribution, and its Table 2 reports 96.9% average accuracy against LAION real images; no JPEG/resize evaluation in the content read. So the score is AEDR's; the only new parts would be using it as a min-over-bank universal detector, testing non-LDM generators, laundering, and fusion.

- **Evidence for / against:**
  - For: AEDR Table 2 (96.9% vs LAION reals, per target model); AEROBLADE's strong LDM numbers.
  - Against: reconstruction scores rely on fine detail that JPEG/resize destroys (aeroblade-cvpr2024: robustness only partial, Fig. 7; fire-cvpr2025: severe blur hurts); non-LDM generators are not on any VAE manifold; per-image cost = 2 VAE passes per VAE in the bank.

- **Test plan (Mac-feasible):**
  - Data: 300 RAISE PNG reals + 100 per generator (11 generators), 512x512 center crops; clean and q75+0.5x.
  - Compute: 2 passes x 2 VAEs x ~1,400 images x 2 conditions = ~11k VAE passes (+ FLUX-VAE ablation on a 300-image subset); time unknown on this GPU - pilot 20 images; if > 1 s/pass, cut to 50 images per generator and drop the ablation (budget: one overnight run).
  - Baseline: AEROBLADE-style LPIPS_2 single-pass min-over-bank on the same crops.
  - **Pass:** mean AUROC >= AEROBLADE + 0.05 (clean), >= 0.65 at q75+0.5x, fusion gain >= +0.02 on held-out FLUX over B0 alone. **Fail:** any of the three missed.

- **Pilot result (2026-10-24, FAIL; experiments/009-vae-aligned/pilot/, critique.md):**
  - The AEDR-style ratio with sd-vae-ft-mse only scores macro AUROC 0.601 clean. That beats single-pass L2 error (0.319, which is below chance because it is mostly texture, rho 0.85), but loses to the Laplacian texture baseline (0.778). Laundered it falls to 0.318.
  - It works only for SD2 and Midjourney v5 (0.98 each; MJ's VAE family is [unverified]), and collapses once laundered (SD2 0.19).
  - Drop this idea in its current form.
- **Status:** pilot failed; dropped
