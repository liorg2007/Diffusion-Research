# E4 PILOT critique (SD-VAE decoder trace), independent critic (opus/high), 2026-10-24

**Rating: SOUND. I found no bug or leakage, every reported number reproduces exactly, and the three pre-registered rule outcomes are
correct (not detectable in orig, does not survive laundering, trace-specific at clean). Three readings need correcting, though:**
- **orig is weaker than fov512 because both members are JPEG-coded at the source QF, not because of zoom.**
- **The 0.99 paired win-rate is not VAE-specific.** A plain Gaussian blur gets the same win-rate.
- **The 0.5× resize is the main thing that erases the trace.** JPEG 75 on its own leaves most of it.

**Reference-based scoring (x vs VAE(x)) is a real lead, but only at native scale on clean images:** clean orig 0.850 vs 0.764 for the single-image probe.

PILOT: none of this counts toward the stop rule. Every check below is post hoc and descriptive. Scripts, logs and JSON are in
`pilot/critic_checks/`:
- `c1_cpu.py`: integrity, F3 per-feature AUROCs, score structure;
- `c2_gpu.py`: the Q2, Q3, blur and Q4 checks. It ran sd-vae-ft-mse on the GPU for about 3,500 images, held in memory only, and DINOv2 on CUDA;
- `c3_reference.py`: reference-based AUROCs;
- `c4_fold.py`: a phase-folded latent-grid check.

The c2 checks have no D-008 CPU-vs-CUDA guard of their own. Their features match the 004/011 caches (max |Δ| 3.5e-5), and they reproduce the reported pair AUROCs to 4 decimals.

## Reproduction and integrity (question 1): no bug found
- **Reproduction.** Refitting the clean-primary probes gives these test pair AUROCs (`c2_gpu.json` `reproduce_clean`), all equal to `tables.md`:
  - orig: F2 0.8587, F3 0.6248, F4 0.8624;
  - fov512: F2 0.945, F3 0.8815, F4 0.9618.

  Recomputing the `jpeg75_resize05` features from the images gives the reported primary and matched values exactly (`c2_gpu.log`).
- **Pairing.** In 40 random test pairs per arm, the median |real − own round-trip| is 4.0 (orig) and 5.2 (fov512), against about 70 for a random other real (`c1_cpu.json`).
- **Format and QF.**
  - Test round-trips in orig carry the source QF in 40 of 40 sampled files.
  - JPEG-coding the raw VAE output at that QF reproduces the cached file bit-exactly on 32 images (max |Δ| = 0).
  - The 009 train round-trips regenerate exactly (`check_009_cache.json`).
  - Our features of them match 009's to within 3.6e-5.
- **Conditions.** Both pair members go through the same `apply_condition` call, keyed by the real's sha, so the seeds are identical. The conditions used here are deterministic anyway.
- **Separation, CV and bootstrap.**
  - Train and test devices are disjoint (10 vs 25 devices), and no `device:vidx` group is shared between them.
  - Grouped CV uses the photo group for both members of a pair.
  - `PairPlan` resamples devices and gives a real and its round-trip the same weight. Its point value equals plain Mann-Whitney (self-test passes, `ALL OK`).
- **F3.**
  - **Correct as specified.** The peaks sit at exact FFT bins 28, 56, 84 and 112 of the 224 crop. The fy=0 row measures x-periodicity. The neighbours are ±2..4 bins.
  - **Underpowered** (see F5).
  - F3 picks C = 10, the edge of the grid; so does F1 in fov512. This is minor.
- **notes.md** is a complete design note, but it has no results or interpretation section (it was probably cut off). The lead should add a pointer to `tables.md` and to this critique's corrections.
- The rule logic matches Amendment 11. Choosing the best family on test is disclosed. The `matched` protocol is correctly labelled as not pre-registered.

## Findings, ranked

### F1 (HIGH, question 2): orig < fov512 is mostly the source-QF JPEG coding, not zoom
- **fov512 with BOTH members JPEG-coded at the source QF:**
  - F4 falls from 0.962 to **0.871** (primary) and 0.865 (matched);
  - F2 falls from 0.945 to 0.862 (`c2_gpu.json` `Q2_fov512_bothJPEGatQF`).

  This roughly matches orig's 0.862 / 0.859. Once the coding is the same, zoom accounts for only about 0.01.
- **orig with raw (PNG) round-trips:**
  - F4 0.841 with the JPEG-trained probe; 0.902 with a probe trained on PNG pairs;
  - F3 0.625 → **0.925** (`Q2_png_roundtrips`).

  Do not use these as estimates. Here the real carries JPEG blocking and the round-trip does not, because the VAE removes it. This is the format cue Amendment 8 exists to block. For comparison, coding only the round-trip in fov512 inflates F4 to 0.975.
- **Dose evidence inside orig:**
  - Spearman(paired margin, QF) = +0.38 for F4.
  - FBH has the lowest QF (median 74) and the lowest per-pool AUROC (0.81). Native (QF 95) and WA (QF 92) are higher (0.87, 0.90).
  - The margin also tracks the real's sharpness: Spearman with Laplacian = +0.49. Pooled AUROC by real-sharpness quintile rises from 0.76 to 0.96 (`c1_cpu.json`).
- **Why F3 in particular collapses in orig.** The latent stride (8 px) is the JPEG block stride. On the full image, the phase-folded 8×8 residual energy gives:
  - fov512: real 1.00, round-trip 2.22, pooled AUROC **0.911** from this single scalar;
  - orig: reals already sit at 1.38 because of their own JPEG grid; AUROC 0.664 (`c4_fold.json`).

  In JPEG-coded reals, a grid-type VAE cue cannot be separated from JPEG blocking.

### F2 (HIGH, question 3): "erased by jpeg75_resize05" holds, and the 0.5× resize does most of it
Pair AUROC for F4, primary / matched (`c2_gpu.json` `Q3_components`). Both members get the condition.

| condition | orig | fov512 |
|---|---|---|
| clean | 0.862 | 0.962 |
| jpeg(90) | 0.820 / 0.839 | 0.874 / 0.864 |
| jpeg(75) | 0.782 / 0.789 | 0.802 / 0.800 |
| resize(0.5) | 0.666 / 0.774 | 0.638 / 0.830 |
| jpeg75_resize05 | 0.598 / 0.670 | 0.600 / 0.737 |

- **Resize is the main killer.** Training on matched data recovers much of the loss from resize alone, but not from resize followed by JPEG.
- **Matched training still fails the 0.75 rule** in both arms; fov512 reaches 0.737, CI lo 0.695 (`tables.md`).
- **Likely mechanism:**
  - After ×0.5, the latent grid has a period of 4 px. That lattice lies inside the 8-px JPEG block lattice, so JPEG 75 adds energy at exactly the trace's frequencies.
  - Grid energy (`c4_fold.json`, fov512): 0.911 clean, 0.639 after resize only, 0.567 after resize then JPEG.
- **The win-rate survives.** Under matched training it is still 0.97–0.98. The per-pair shift is still there, but it is small next to the spread between images: shift / sd(real) is about 0.5–0.7, against 1.3–2.3 at clean (`c1_cpu.json`).

### F3 (HIGH, question 4): the 0.99 win-rate is not a VAE signature, and reference scoring helps only at native scale on clean images
- **Blur gets the same win-rate.** On real vs Gaussian-blurred real (no VAE), the clean F4 probe gets win-rate **0.97–0.99** in both arms (`blur_control`). The 1-d Laplacian gets 0.97.

  So "VAE(x) scores higher than x" holds for any mild low-pass. It says nothing about whether reals and fakes differ in how much they move.
- **Reference-based test.** This used 358 test natives vs 60 fakes per generator. VAE(x) is the raw output, so the two classes are handled the same way (`c3_reference.json`).

  | score | clean orig | laundered orig | clean fov512 | laundered fov512 |
  |---|---|---|---|---|
  | pixel reconstruction error (AEROBLADE-like, L1, not LPIPS), macro | 0.358 | 0.309 | 0.382 | 0.356 |
  | probe change −Δs = −(s(VAE(x)) − s(x)), F4, macro | **0.850** [0.80, 0.89] | 0.562 | 0.800 | 0.451 |
  | single-image s(x), F4, same sample | 0.764 | 0.648 | 0.887 | 0.635 |

  - **Pixel reconstruction error is inverted:** fakes reconstruct *worse*. Soft camera-JPEG reals are easy to reconstruct; SD2 gets 0.39 and SDXL 0.29. Its prior is low.
  - **The probe change does carry an idempotence signal at clean orig.** SD-family 0.925: SD2 0.972, MJ-v5 0.960, SDXL 0.878. The median Δs is 0.03 for SD2 and 0.23 for MJ-v5, against 3.91 for reals. DALL-E 2/3 move like reals (0.65, 0.76). Δs therefore behaves like a decoder-family fingerprint.
  - **Under laundering it falls below the single-image probe.** It also loses at fov512, where the fakes have been resampled.

### F4 (MEDIUM): "trace-specific" passes against the Laplacian, but a matched-blur control is less flattering in orig
- **orig:**
  - blur r=0.5 (Laplacian ratio 0.82, close to the VAE's 0.84) gives F4 0.69;
  - blur r=0.8 (ratio 0.52) gives 0.83, against 0.86 for the VAE.
- **fov512:** blur gives 0.66 / 0.73, against 0.96 for the VAE. Here the probe is clearly VAE-specific.
- The 1-d Laplacian with a fixed sign is a weak control. For pooled discrimination it is close to chance by construction (0.55).

### F5 (MEDIUM): F3 is correct but underpowered
- **Single FFT bins on the axes of a 224 crop.**
  - In fov512 the best single peaks reach 0.79 (B, x-Nyquist) and 0.70 (G); in orig they stay about 0.5 (`c1_cpu.json`).
  - The phase-folded grid energy over the full 512 image reaches 0.911 as one scalar, more than F3 manages with 120 features (0.882).
- **F3 has one real strength.** Even with a pair AUROC of 0.625, the orig F3 probe transfers to native-scale **SDXL at 0.889** (`results.json` per_generator). The fov512 F3 probe gets 0.519, because resampling the fakes erases their own decoder grid.

## Recommendations (pre-registerable)
1. **E5, reference-based idempotence.**
   - Score: Δ = s(VAE(x)) − s(x), plus the change in phase-folded grid energy.
   - Data: native-scale crops, unseen generators, all D-009 conditions including `social_sim`.
   - Primary comparison: Δ vs the single-image probe on the same images, per decoder family.
   - Pre-register the expected outcome: ≥ +0.05 macro at clean, and "fails under ×0.5 laundering" as the honest prior.
   - Treat it as a component (a decoder-family flag), not as the laundering-robust detector.
2. **Laundering components as standard conditions.**
   - Add `resize(0.5)`, `jpeg(75)` and `jpeg(90)`.
   - Add a non-sub-lattice resize such as ×0.75: the latent stride becomes 6 px, which no longer coincides with JPEG's 8 px.
   - Pre-register the hypothesis that grid cues survive JPEG-only and non-integer resizes but not ×0.5 followed by JPEG.
3. **A matched-blur negative control for every "trace" claim.** Real vs blur at a per-image matched Laplacian ratio. The trace detector must beat it by ≥ 0.10 pooled AUROC.
4. **Replace the F3 peaks with phase-folded grid statistics.**
   - Use the full image and the known phase.
   - For laundered inputs, search over scale and phase (fold at p = 8·f).
   - Test them on PNG or resampled reals (fov512, RAISE-like) so the JPEG grid does not confound them.
5. **Rules on pooled metrics only.** Base every decision on pooled, deployable AUROC. Report win-rate as a diagnostic only, always next to a blur control.
6. **Deprioritize pixel-error AEROBLADE.** If it is tested at all, it needs LPIPS and complexity normalisation, pre-registered as low-prior: pixel error scored 0.31–0.38 here.
7. **Phase 5.** The pre-registered consequence stands: single-image VAE-trace probes are deprioritized for laundered targets. The clean, native-scale Δ result (F3) should be carried into the hypotheses as a bounded positive lead.
