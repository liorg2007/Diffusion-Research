# Latent-grid periodicity under rescaling: the VAE's 8-px upsampling grid survives non-dyadic downscaling, and jpeg75_resize05 erases it because 0.5x maps it onto the JPEG block-grid harmonics
- **id:** latent-grid-periodicity-rescale
- **Gap addressed:** why decoder traces vanish under laundering (E4); traces that survive downscaling; a training-free, physically interpretable score whose real-photo FPR can be reasoned about
- **Scores (1-5):** novelty 3 · expected gain on unseen/in-the-wild 3 · feasibility 5 · **product 45** (rank 2 in docs/06-diffusion-hypotheses.md)

- **Hypothesis:** In SD-family fakes, a residual periodicity at period 8·s px (s = total resize factor since decoding) stays detectable after resize s plus JPEG q75 whenever 1/(8s) is NOT a harmonic k/8 of the JPEG block grid (oracle-s peak AUROC vs VISION natives >= 0.80 for s in {0.6, 0.7, 0.8}), and collapses (<= 0.65) when it is (s = 0.5, or s = 1 with JPEG), so the D-009 laundering condition is close to a worst case for latent-grid traces.

- **Mechanism (why diffusion images differ):**
  - SD/SDXL-style decoders turn a latent with stride 8 into pixels through repeated x2 upsampling stages, leaving weak periodic correlations at periods 8, 4 and 2 px. Diffusion and VQ models show "anomalous regular patterns in the autocorrelation" and Fourier peaks (Corvi et al., CVPRW 2023, arXiv 2304.06408 - search-result abstract, not opened); corvi-icassp2023 shows peaks in averaged DnCNN residual spectra (Fig. 2). Camera photos have no 8-px generation grid (their periodicities are the CFA at 2 px and the JPEG block grid at 8 px of the *capture* resolution).
  - Antialiased downscaling by s keeps a component at frequency 1/(8s) cycles/px whenever 1/(8s) < 0.5, i.e. for every s > 0.25. So downscaling alone should not erase the grid fundamental.
  - The following JPEG works on an 8x8 grid of the *laundered* image, whose own block artefacts and quantisation sit at k/8 cycles/px. At s = 0.5 the grid fundamental lands on 2/8 = 0.25; at s = 1 on 1/8. There the weak decoder peak is masked by (and quantised together with) the much stronger block-grid harmonic. At s = 0.7 (period 5.6 px, f = 0.179) or s = 0.6 (f = 0.208) it falls between harmonics, in low DCT AC indices that q75 keeps.
  - E4's F3 periodicity features were measured exactly at k/8 bins (experiments/011-vae-trace/config.json), so its laundered collapse (F3 pair AUROC 0.527 orig, 011/pilot/tables.md) is consistent with this collision but cannot distinguish it from erasure.

- **Closest prior work:** synthbuster-ojsp2024 (spectral analysis of a residual, claims robustness only to mild JPEG; abstract only); corvi-icassp2023 (residual spectra peaks); freq-dct-icml2020 (GAN up-sampling grids in DCT; clean-trained models drop to 72% under JPEG, Table 5); spai-cvpr2025 (learned spectral model); f2c-arxiv2025 (spectral band re-weighting). Classic resampling forensics (Popescu-Farid; Kirchner) detect *interpolation* periodicity, and a search found the remark that "JPEG block artifacts severely interfere with spectrum-based methods" (search snippet, unopened). Web check (two searches, URLs in docs/06): no paper found that predicts where the latent grid lands after rescaling, scans for it at unknown scale, or tests the dyadic-collision prediction. Novelty is the falsifiable mechanism + protocol consequence, not the spectral feature itself.

- **Evidence for / against:**
  - For: F3 (which includes periodicity at k/8) transfers to SD-family fakes at 0.719 clean in `orig` with zero generator training (011 transfer table); FIRE's mid-band mask and Cozzolino 2608.12155 point to non-semantic spectral cues.
  - Against: the grid amplitude may be below q75 quantisation steps everywhere; a real photo's own capture-JPEG grid (VISION natives QF ~95, 009/pilot/critique.md F2) is also rescaled to period 8s - the same frequency as the decoder grid (format confound, must be controlled); DALL-E 3/Midjourney/Firefly may use upscalers with other periods; an adversary can pick a dyadic resize on purpose.

- **Test plan:**
  - *Pilot now (no download, CPU minutes):* VISION test natives (358) + WA/FBH copies and 6 Synthbuster generators x 120, `orig` 512 crop, then resize s in {1.0, 0.8, 0.7, 0.6, 0.5} (antialiased, both classes) then JPEG q75; plus clean, social_sim, and `fov512` reported with its predicted collisions (SDXL 1024 px -> s = 0.5 then 0.25 under laundering; SD2 768 px -> s = 0.667 then 0.333, f = 0.375 = 3/8).
    - Score O (oracle s): peak prominence (log power of the residual's row/column-averaged 1-D spectrum at 1/(8s) minus the median of +-2..4 neighbouring bins; residual = x - median3x3). Score S (blind): max prominence over periods 2.5-16 px excluding +-1 bin around k/8 and 0.5.
    - **JPEG-history-matched control (mandatory):** fakes first saved as JPEG at the QF of a randomly paired native (88-100) before resizing, so both classes carry a capture-JPEG grid at 8s.
    - **E4 add-on:** re-score E4's pairs at jpeg75_resize07 (existing round-trips; DINOv2 features only).
  - *Validity gates:* SD-family fakes clean (no resize, PNG) must show the peak at 1/8 +- 1 bin in >= 60% of images, else UNINFORMATIVE (no grid to track); texture baseline; per-generator reporting.
  - **Pass:** O-AUROC (SD2+SDXL vs natives) >= 0.80 at each of s = 0.6/0.7/0.8 with JPEG q75, <= 0.65 at s = 0.5, mean(non-dyadic) − mean(dyadic) >= 0.15; JM control within 0.05; beats texture by >= 0.10; S >= 0.80 SD-family and >= 0.70 all-6 at social_sim; FPR of S at the 99th-percentile threshold of clean natives <= 2% on WA/FBH copies. E4 add-on reported descriptively (a pair AUROC >= 0.70 at 0.7x vs 0.598 at 0.5x would show E4's collapse is partly a dyadic artefact). **Fail:** O < 0.70 at all non-dyadic s, or the dyadic gap < 0.05 (no collision effect).
  - *Weights:* none. *Compute:* numpy FFTs, < 1 h CPU; the E4 add-on about 10 min GPU [estimate].
  - *Phase-5 reach:* NO at jpeg75_resize05 by its own prediction. Value: a fusion feature for social_sim-type laundering, a mechanistic explanation of E4, and a protocol recommendation (add non-dyadic resize conditions, and report dyadic ones as adversarial).

- **Status:** proposed

- **Pilot result (2026-10-24, UNINFORMATIVE; experiments/013-latent-grid/pilot/, critique.md):**
  - The validity gate failed: the 1-D 1/8 peak is absent in SD2/SDXL fakes, and O is at chance at every scale.
  - Critic (post hoc): the 8-px lattice IS present in about 70% of SD fakes, but mostly as period 2–4 harmonics and off-axis. VISION camera-JPEG grids sit on the same lattice, so grid energy cannot separate them (0.44–0.47).
  - The dyadic-collision mechanism is not supported: the trace decays smoothly with downscaling.
  - Deprioritise frequency-exact grid features for laundered targets. Any future grid score needs a 2-D fold, an off-period control and a grid-free real reference.
- **Status (updated):** pilot uninformative; mechanism not supported (post hoc)
