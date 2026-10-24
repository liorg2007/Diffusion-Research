# H-D2 PILOT critique (latent-grid periodicity under rescaling): independent critic (opus/high), 2026-10-24

**Rating: procedurally SOUND, interpretation needs correcting.**
- The UNINFORMATIVE label is correct under the pre-declared gate.
- I found no bug and no leakage. The reported gate, O-AUROCs, JM and E4 add-on numbers reproduce.
- **The notes' reading "no 8-px grid in SD2/SDXL fakes" is wrong.** A phase-locked 8-px grid is present in about 70% of them.
  - The pre-registered score O looked only at the grid's weakest component: the 1/8 fundamental, on the frequency axes.
  - The reals it was compared against (VISION natives) carry a capture-JPEG grid of equal strength on the same 8-px lattice.
- **Post-hoc grid checks do not support the dyadic-collision mechanism either.** JPEG q75 erases the decoder grid at non-dyadic scales just as much as at 0.5×.
- **The E4 add-on is computed correctly, but a two-point comparison (0.5× vs 0.7×) cannot separate "dyadic" from "less downscaling".** The grid checks show a smooth, monotone decay with s and no dip at 0.5.

All checks below are post hoc and descriptive (PILOT, D-014, no stop-rule weight). Every check runs on CPU only, with no downloads and no GPU. Scripts and JSON are in
`pilot/critic_checks/`:
- `c1_spectra.py`: 2-D lattice spectra, fold z7/z8/z9, the harmonic split of the fold, and a crop-alignment check;
- `c2_rescale.py`: oracle-period fold under resize and resize+JPEG at s ∈ {0.5, 0.625, 0.75, 0.875, 1};
- `c3_gate.py`: the gate re-read against a grid-free real reference;
- `c4_gridshape.py`: a classifier on grid shape vs grid energy, plus a JPEG-matched variant.

These are point estimates without bootstrap CIs. With n = 120–358 per group, treat AUROC differences below about 0.05 as noise.

## Reproduction and integrity (question 3): no bug, no leakage
- **Gate threshold.** Recomputed from the images with `run.py:image_feats`, it is 1.50998, identical to `tables.md` (1.510). The share of SD fakes above it is 0.000 (`c3_gate.json`).
- **Crop alignment.** All 240 orig SD2/SDXL crops equal the raw file's `[(H−512)//2, (W−512)//2]` centre crop bit-exactly, and every offset is a multiple of 8, so the latent-grid phase is 0 (`c1_spectra.json` `crop_check_orig_SD`). The claim in `config.json` is correct.
- **Are the Synthbuster SD2/SDXL images at native generator resolution?** No file carries metadata (PNG, no text chunks). Every size is a multiple of 64 and matches the SDXL aspect buckets (832×1152, 1536×640, …, 26 sizes per generator), not a 768² or 1024² grid. They do not look resized:
  - in the plain crop, the phase-folded energy at p = 8 is 2.7 (SD2) and 2.6 (SDXL), against 1.1 at p = 7 and p = 9 (`c1`);
  - a non-integer resample by the dataset authors would have smeared that fixed-phase 8-px lattice.

  So the inputs are fine: native latent-grid geometry is present at phase 0.
- **fov512 natives.** They get the s_pre of a seeded random SD fake (`run.py:s_pre_for`), so O is compared at matched frequencies. This is correct.
- **positive_control.py.** Columns 32–39 are the R-channel F3 peaks x1..4/8 and y1..4/8 (`011/run.py:f3_peak_index`). This is correct.
- **E4 add-on.**
  - It reproduces 011 at `jpeg75_resize05` exactly (max diff 0).
  - D-008 passes: max |Δlogit| is 2.7e-5 (orig) and 2.4e-4 (fov512), both ≤ 1e-3 (`d008_guard.json`).
  - Both pair members go through the same `load_cond` call.
  - DINOv2 takes a 224 *centre crop* (`crop_mode="crop"`), not a resize. So at 0.7× the crop sees the trace at 1.4× the linear resolution of 0.5×, over a smaller field of view. This matters for question 2.
- **Minor issues (no effect on the label):**
  - In fov512, O is NaN for Firefly at `jr050`, because f* > 0.5. It enters no reported macro.
  - `kb/datasets/synthbuster.md` says "all PNG", but `001-eval-set/notes.md` found that the Firefly files are JPEG at q95 with a .png name. I re-checked: all 120 estimate as q95. Firefly's per-generator O (0.56–0.63) and its z8 (median 62) are that JPEG grid, not a generator trace. The lead should fix the KB line.

## Findings, ranked

### F1 (HIGH, question 1): the gate failure is real for the 1/8 fundamental, but "no grid" is wrong. The grid sits at the harmonics and off the axes, and VISION natives carry an equal JPEG grid on the same lattice
**Evidence (`c1_spectra.json`, `c3_gate.json`, clean orig 512 crops, gray median residual):**
- **The grid exists.** Fold energy at the latent period vs off-period controls, median z7 / z8 / z9:

  | group | z7 | z8 | z9 |
  |---|---|---|---|
  | SD2 | 1.09 | 2.73 | 1.06 |
  | SDXL | 1.18 | 2.63 | 1.14 |
  | MJ-v5 (also latent diffusion) | 1.09 | 2.88 | 1.04 |
  | DALL-E 2 (pixel cascade, a negative control) | 1.08 | 1.05 | 1.10 |
  | grid-free reals (fov512 natives) | 1.04 | 1.00 | 1.03 |

  Within-image test, z8 > 1.5 × max(z7, z9):
  - SD2 71%, SDXL 72%, MJ-v5 76%;
  - DALL-E 2 2.5%, fov natives 2.8%.

  On the group-mean 2-D spectrum, the lattice points (k/8, l/8) stand above their neighbourhood in SD2, SDXL and MJ-v5: most by 0.1–0.7 log10, the highest by 0.97 (SD2) and 1.43 (SDXL). The control lattice, shifted by half a step, sits at about 0.00.
- **The energy is not where O looked.** The share of the 8×8 fold's energy at harmonic 1 (period 8) is only 0.10 (SD2, SDXL). The rest sits at harmonics 2–4 (periods 4, 2.7 and 2 px), and 54–57% of it lies off the axes (cross terms).
  - O uses row/column-averaged 1-D profiles, which see only the fy = 0 / fx = 0 lines, and only at 1/8.
  - On the axes, the per-image 1/8 prominence of SD fakes barely beats even grid-free reals: AUROC 0.56–0.59.
  - The 2-D lattice mean reaches 0.86–0.92, and z8 0.94–0.95 (`auroc_fake_high`).
- **Even with a fair reference the fundamental fails the gate.**
  - Against the 99th percentile of grid-free fov512 natives (O threshold 0.93 instead of 1.51), only 5.8% (SD2) and 3.3% (SDXL) of fakes pass (`c3`).
  - The gate failure is therefore not an artefact of the threshold. It is a property of the specified statistic: the hypothesis assumed the fundamental dominates, and it does not.
- **The reference is also confounded.** Clean orig VISION natives (camera JPEG, QF mostly 95–97) have z8 = 3.31, *higher* than SD fakes, and a 2-D lattice mean of 0.169 (SDXL 0.159). Their capture-JPEG grid sits on the same 8-px lattice at a similar harmonic mix (k1 share 0.115).
  - Every grid-energy statistic is therefore at chance or inverted against VISION natives: z8 0.44–0.47; DALL-E 2 vs natives 0.09.
  - The idea file listed this format confound under "Against". It is not a nuisance; it dominates.
  - The JM control could not reveal it, because it only adds a JPEG grid to fakes that already have a grid, and O was at chance.
- **Residual filter, window and resolution are not the cause.**
  - The fold statistic uses the same median residual and finds the grid.
  - n = 512 puts 1/8 at exact bin 64, and the Hann window plus a ±1-bin maximum is adequate.
  - The cause is the choice of frequencies (fundamental, axes only) and of reals (JPEG-gridded).

**Reconciliation with the E4 critic.** Its folded-grid 0.911 and F3 4/8 peaks of 0.64–0.74 were measured on *paired* sd-vae-ft-mse round-trips against fov512 reals. Those reals were resampled about 6× down, so their JPEG grid was gone, and the round-trips were PNG.
- That setting has a grid-free reference. The fold z sums all 64 phase cells, so it captures the harmonics and the cross terms.
- On the same 150 pairs here: z8 AUROC 0.858, lattice at harmonic 4 0.793, 1/8 axes 0.56 (`c1`). This matches 011 (0.911 on the full set; F3 1/8 at 0.52–0.55, 4/8 at 0.64–0.74).
- **What they picked up:** the decoder's period-2/4 (x2-upsampling) structure, mostly at Nyquist and off the axes, seen against reals that had no grid at all. The 1/8 fundamental was weak there too.

### F2 (HIGH, questions 1 and 4): on the pre-registered metric, the dyadic-collision mechanism finds no support
H-D2 itself was not tested; the gate stops it. A fold statistic at the oracle period stands in for it. With s = 0.5, 0.625, 0.75, 0.875 and 1 the period p = 8s is an integer (4–8). The score is d = z_p − mean(z_{p−1}, z_{p+1}) (`c2_rescale.json`).

**Paired positive control** (150 fov512 VAE round-trips vs their own reals), d AUROC:

| | s = 1 | 0.875 | 0.75 | 0.625 | 0.5 |
|---|---|---|---|---|---|
| resize only | 0.83 | 0.78 | 0.70 | 0.62 | 0.60 |
| resize + JPEG 75 | 0.63 | 0.60 | 0.52 | **0.50** | 0.47 |

- With resize alone, the grid fades smoothly with s. There is no dip at 0.5.
- **JPEG q75 erases it at p = 5 just as at p = 4**, although period 5 shares no harmonic with the 8-px block grid: folding at 5 averages any 8-periodic pattern to a constant.
- The mechanism in the idea file ("between harmonics, in low AC indices that q75 keeps") is therefore not what happens: after q75 the round-trip grid is below the quantisation step everywhere.

**SD fakes (native scale) vs grid-free fov512 natives,** d AUROC:

| | s = 1 | 0.875 | 0.75 | 0.625 | 0.5 |
|---|---|---|---|---|---|
| resize only | 0.93 | 0.85 | 0.83 | 0.78 | 0.74 |
| resize + JPEG 75 | 0.80 | 0.78 | 0.74 | 0.69 | 0.64 |

- The decay is again smooth, with no dyadic dip.
- The full-generator grid is stronger than the round-trip grid (SD median d after `jr(75, 0.625)`: 0.46, against −0.24 for round-trips) and survives partly.
- **Against the real VISION natives it reaches only 0.56–0.60 at every s.** Their capture grid is rescaled by the same s (resize-only natives d = 0.86 at p = 5).

**Conclusion.** Under laundering, the latent grid is (a) weak after q75, (b) decays smoothly with downscaling and (c) is not separable from a JPEG photo's own rescaled block grid by energy. The dyadic-collision story is unsupported by this alternative statistic. It should not be carried forward as an explanation of E4.

### F3 (MEDIUM, question 2): the E4 add-on is sound, but it shows "less downscaling helps", not "0.5 is special"
- **The numbers are correct:**
  - F2 primary 0.597 → 0.672, with non-overlapping CIs [0.579, 0.620] and [0.645, 0.704];
  - matched 0.670 → 0.703 [0.671, 0.741].
- **The pre-declared descriptive threshold (≥ 0.70 for the primary probe) is not met.** The notes state this correctly.
- **The comparison has a confound.** DINOv2 takes a 224 centre crop without resizing. So 0.7× gives the probe the trace at 1.4× the linear resolution (more cycles per pixel survive the antialias filter) and a smaller field of view. Any trace whose strength falls with downscaling improves at 0.7×, dyadic or not:
  - `c2` resize-only: 0.60 → 0.70 between s = 0.5 and 0.75;
  - the E4 critic: resize, not JPEG, is the main killer of the probe.
- **F3 staying flat (0.527 → 0.513) is uninformative about collision.** F3 reads the fixed k/8 bins of the 224 crop. At 0.7× the grid sits at 1/5.6 (bin 40), which no F3 bin covers, so F3 cannot gain by construction.
- **Reading:** the trace survives laundering in proportion to the resolution kept. The reported 0.07 gain is the expected effect of s = 0.7 vs 0.5. It is not evidence that 0.5× is adversarially worse than a smooth trend predicts.
  - Separating the two needs at least three s values around 0.5 (for example 0.45 / 0.5 / 0.55), or a fine sweep. The prediction would then be "AUROC(0.5) lies ≥ 0.05 below the linear interpolation of its neighbours".

### F4 (MEDIUM, question 4): grid *shape* separates SD fakes from JPEG natives where grid *energy* cannot, but it is not yet diffusion-specific
- **Setup** (`c4_gridshape.json`, exploratory, clean orig only):
  - logistic regression on the phase-invariant |DFT| of the 8×8 fold (34 values, normalised by std(r)) plus log z8;
  - 5-fold CV grouped by device for natives and by image for fakes.
- **Results:**
  - SD2+SDXL vs natives: **0.885** (z8 alone 0.539);
  - after JPEG-saving the fakes at a sampled native QF (JM): **0.854** (z8 alone 0.560).
- **Caveats:**
  - Generators are in-distribution (fakes split by image, not leave-one-generator-out).
  - All 6 generators reach 0.81, including DALL-E 2 (0.75), which has *no* grid. So part of the signal is "camera JPEG/CFA structure present" (period-2 CFA energy lands on the same lattice), not "decoder grid present".
  - It is clean-only. Under any resize, the lattice moves to 8s and the phase-folded shape must be rebuilt at an unknown period.
- **Status:** a lead for a native-scale, unlaundered *component* only, not a laundering-robust detector.

### F5 (LOW): what the pilot's secondary numbers mean
- The z8/zs AUROCs of 0.55–0.65 at resized conditions (`tables.md`) do not come from the latent grid surviving.
  - `c2` shows natives' capture-grid d falls from 0.86 to 0.14 after resize+q75, while SD fakes keep 0.46.
  - So JPEG q75 wipes a QF95 capture grid somewhat more than a decoder grid. That asymmetry is weak and depends on the natives' QF, so it is not a deployable cue.
- O in fov512 (0.55–0.63) is close to its texture baseline (0.61–0.64) and is descriptive only, as labelled.

## Recommendations (pre-registerable)
1. **Keep the label UNINFORMATIVE and correct notes.md's interpretation:**
   - "No 1/8 fundamental, but an 8-px lattice is present: z8 2.6–2.7 vs 1.1 at p = 7/9, in about 70% of SD fakes, energy at periods 2–4 px and off-axis."
   - "VISION natives carry an equal-energy capture-JPEG lattice."
   - "Post-hoc fold checks show no dyadic collision: q75 erases the round-trip grid at p = 5 as at p = 4."
   - Mark H-D2's mechanism as *not supported (post hoc)*, not merely untested. Do not rerun H-D2 with a fixed-up score as a "confirmation"; that would be HARKing.
2. **Any grid statistic in future must specify:**
   - the full 2-D lattice or the phase fold (all harmonics, cross terms included);
   - a within-image off-period control (z_p vs z_{p±1});
   - at least one grid-free real reference (PNG/raw-derived reals, e.g. RAISE, listed in `data/manifest_raw.csv` but not on disk here) alongside the JPEG phone reals;
   - a gate that tests grid *existence* within-image, not "above the 99th percentile of JPEG photos".
3. **Laundering protocol:**
   - Replace the "dyadic is adversarial" recommendation with a **monotone resize sweep** (s = 0.45, 0.5, 0.55, 0.625, 0.75, 0.875) for every trace claim.
   - Pre-register a collision test as "AUROC(0.5) ≥ 0.05 below the interpolation of 0.45 and 0.55". The prior from `c2` is that it fails.
4. **Grid shape (F4), if pursued, as a new pre-registered hypothesis with a low prior:**
   - leave-one-generator-out;
   - a JM control at native QF (here it held at 0.854);
   - a CFA/no-CFA control (demosaiced-then-resampled reals vs PNG fakes);
   - a matched-blur control.
   - Expected reach: clean native-scale images only. Not Phase-5 laundering.
5. **Implications for diffusion-specific methods on our data:**
   - Decoder-grid cues are real at native scale. They are confounded with JPEG history whenever the reals are phone JPEGs, and they do not survive q75 after any downscale (`c2`: 0.47–0.52 for s ≤ 0.75 in the paired control).
   - Single-image periodicity features should be **deprioritised for the laundered targets**, in line with E4 (011 critique, recommendation 7).
   - Remaining budget should go to cues that degrade gracefully with resolution (the DINOv2 families: 0.67 at 0.7×), not to frequency-exact traces.
6. **KB hygiene:** fix `kb/datasets/synthbuster.md` ("all PNG": Firefly is JPEG q95 under .png names). Add a negative-result entry: the latent grid sits at the harmonics, not the fundamental, and it is masked by the capture-JPEG lattice.
