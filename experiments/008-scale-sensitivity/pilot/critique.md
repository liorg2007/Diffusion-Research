# E1 pilot critique (independent critic, opus/high, 2026-10-24)

**Rating: SOUND execution, no bug found. The effects are real, but three of the headline readings need reframing:**
- "B0 not sensitive" describes the retrain-per-arm recipe, not a deployed probe.
- The 23.4% is a fragile point estimate of a robust inflation of at least 5×.
- Zoom and resampling cannot be separated, by construction.

Scope: CPU re-analysis of `pilot/scores_long.csv.gz`, `results.json` and E1's cached B0 models (no refit, no GPU). Scripts and outputs are in `pilot/critic_checks/` (`c1_sizes_means.py` … `c4_dose.py`, `*.out`, `sizes.csv` = raw-file sizes read from headers). This is PILOT data: exploratory, not counted toward the stop rule.

## Findings, ranked

### F1 (high, interpretation): zoom and resampling are collinear per image, and the saved data cannot separate them
- In `fov512`, each image's zoom-out factor *equals* its resampling factor, k = short side / 512.
- Raw sizes (`critic_checks/c1.out`):

  | Group | Median short side | k |
  |---|---|---|
  | VISION native | 2448 (range 720–3936) | 4.8 (1.4–7.7) |
  | WA | 960 | 1.9 |
  | FBH | 1536 | 3.0 |
  | DALL-E 2/3 | 1024 | 2.0 |
  | Midjourney v5 | 896 | 1.75 |
  | SD2 | 768 | 1.5 |
  | SDXL | 832 | 1.6 |
  | Firefly | 1792 | 3.5 |

- The available handles are only partial (`c4.out`):
  - **Per-device dose-response within natives** is weak: Spearman(k, Δscore) is between −0.10 and +0.06 for RINE and ClipDet, and for CommFor clean. CommFor laundered is the exception at −0.47. There are 25 devices, 17 of them have k ≥ 4, only one has k < 2, and k is confounded with camera model.
  - **Firefly** has the largest fake-side downscale (k 3.5). It shows the largest CommFor score drop: median Δ −6.8 clean and −12.4 laundered, against −1.4 to −5.7 for the others. Its laundered AUROC goes 0.978 → 0.733. But Firefly is also the only JPEG-source generator, so this mixes trace removal, format grid removal and zoom.
  - **WA reals** (k 1.9) are scale-matched to the fakes in *both* arms. That makes them the best available control. ClipDet clean: Δ is +0.008 on WA but −0.127 on natives. So the ClipDet effect comes from the native photos' k ≈ 4.8. CommFor laundered: the WA Δ (+0.112) is larger than the native Δ (+0.062). That points to a fake-side loss of traces as cumulative downscaling grows, not to a real/fake zoom mismatch. RINE laundered is mixed: the WA Δ is +0.114, about half the native +0.228. Roughly half of RINE's effect is fake-side and half is native-specific (see F3).
- **Conclusion:** the Amendment 7 caveat stands exactly as written. A third arm is needed (R1).

### F2 (high, interpretation): each move has a mechanism tied to the detector's own preprocessing
Native protocols are from `002-baselines/scorers/*.py` docstrings and `kb/papers/*`.

- **RINE: `CenterCrop(224)`, no resize** (kb/papers/rine-eccv2024.md: "no resizing").
  - Because 003's crop512 is a centre crop, the RINE input in `orig` is the raw image's central 224 px at 1:1. That is RINE's native protocol (single crop rather than ten-crop).
  - In `fov512`, RINE instead sees a k-times-downsampled view.
  - After `jpeg75_resize05` (512 → 256), its 224 crop covers 87.5% of the window at 0.5× scale. So for RINE, laundering is itself a 2× zoom-out.
- **ClipDet: `Resize(224, bicubic)` of the whole input.**
  - The `orig` arm feeds it a 1:1 phone close-up of about 1/5 of the scene. `fov512` feeds it the whole scene, which is its native, published protocol (kb/papers/clipdet-cvprw2024.md).
  - Both classes move the right way under `fov512`: natives' median score goes −1.12 → −1.66, and SDXL / MJ / SD2 fakes rise by +0.68 / +0.37 / +0.16 (`c1.out`, `c4.out`).
  - This is consistent with the texture confound. On ClipDet's input, natives' median |Laplacian| is 0.040 vs 0.075 for fakes in `orig`, and 0.065 vs 0.083 in `fov512`. Spearman(score, Lap) on natives is −0.19 clean and −0.54 laundered (`tables.md`). Smooth, zoomed phone pixels look "fake" to ClipDet, and laundering makes them smoother still: laundered `orig` natives' median score is +0.16, above several generators.
- **CommFor: `Resize(440)` + `CenterCrop(384)`.**
  - Its native protocol resizes the whole image, which `fov512` approximates.
  - Clean: 0.995 in both arms. Its traces survive the 1.5–3.5× downscale.
  - Laundered: the 256-px input is *upsampled* 1.72× to 440. In `fov512` the fakes have been downscaled 2k in total (4–7×), and their median score falls from positive to negative for every generator (`c1.out`).
  - So CommFor's +0.062 is mostly fake-side trace loss under cumulative resampling. That is a real robustness limit, not a real-photo zoom artefact.
- **B0 (center_crop 224, retrained per arm)** learns whichever scale it is trained on, so the per-arm Δ is small. See F4.

### F3 (high, Q2): RINE laundered 0.945 > clean 0.813 is real, not a bug. It comes from DALL-E 3 and real-photo scale
- **The macro is driven by one generator.** DALL-E 3 goes 0.166 (clean `orig`) → 0.935 (laundered `orig`). Without DALL-E 3, the macro is 0.943 clean vs 0.946 laundered (`c2.out`).
- **Clean DALL-E 3 is inverted** (median −2.40, natives −0.75). This matches RINE's own LDM-trained Synthbuster result, DALL-E 3 AP 32.5 (kb/papers/rine-eccv2024.md, Table 6). It also matches 003's official run: DALL-E 3 0.215 (`003-baselines-eval/tables.md`).
- **After laundering, four of six generators collapse** to a near-constant logit of about −0.2 (fakes' interquartile range −0.29…−0.17). The natives go *down*, from −0.75 to −0.99. So the laundered AUROC is driven by the real side: VISION phone pixels at 1:1 still carry some "real" cue after ×0.5 + q75. In `fov512` (k ≈ 9.6 cumulative), natives rise to −0.41, into the fake cluster, and the AUROC is 0.717.
- **This replicates in an independent run with a different real pool.** In 003 official, RINE VISION-native goes 0.823 → 0.893 under laundering, while scale-matched RAISE (833 px) goes 0.882 → 0.652. Scores are reproducible: the 007 re-score max |d| is 4e-16, and D-008 on `fov512` passed at 3.6e-5.
- **Reading:** RINE's laundered gain is a VISION-native-at-1:1 effect. It does not generalise to scale-matched reals. Never cite it as robustness. The physical mechanism (which sensor or ISP cue survives) is [unverified].

### F4 (high, Q5): B0 "not sensitive" holds for the recipe; a fixed `orig`-trained probe is zoom-fragile, while an `fov512`-trained one is not
- **The pre-registered B0 Δ compares two probes**, each retrained in its own arm. As specified that is correct, but it does not measure a deployed detector.
- **Fixed-probe rows** (extra, not pre-registered; `tables.md`):

  | Probe | Condition | AUROC on `orig` | AUROC on `fov512` | Δ [95% CI] |
  |---|---|---|---|---|
  | `b0_trained_orig` | clean | 0.990 | 0.969 | +0.021 |
  | `b0_trained_orig` | laundered | 0.981 | 0.923 | +0.059 [+0.041, +0.077] |
  | `b0_trained_fov512` | clean | 0.969 | 0.980 | −0.011 |
  | `b0_trained_fov512` | laundered | 0.979 | 0.980 | −0.001 |

- **What moves:** under the `orig` probe, test natives' median logit jumps from about −8 to about 0 when the same photos arrive at `fov512` (`c3.out`). Fakes move only +0–2. So the `orig` probe partly encodes "1:1 phone close-up = real". This is the D-013 confound inside B0 itself.
- **The asymmetry also holds for thresholds:**
  - `fov512`-trained, threshold transferred to `orig`: FPR 1.1% clean and 1.3% laundered. With an oracle threshold at 1% within-arm FPR on the source arm's test natives, the cross FPR is 0.3%.
  - `orig`-trained: 23.4% (F5).
- **Cost:** the `fov512` probe has lower TPR at 1% FPR: 0.683 within-arm, vs 0.900 for the `orig` probe.

### F5 (medium, Q3): the cross-zoom FPR inflation is robust; the 23.4% figure is not

Where the threshold comes from:
- It is set from 43 calib natives drawn from only **10 devices** (calib and test devices are disjoint; `c3.out`).
- So it is a single photo's score. For the `dalle2` LOGO model, the calib max is 2.67 and the second-highest is −0.40.

B0 fixed `orig` probe, `orig` threshold applied to `fov512` test natives, clean:

| Threshold rule | Cross FPR |
|---|---|
| calib max (reported) | 0.234 |
| 2nd-highest calib native | 0.498 |
| 3rd-highest | 0.533 |
| device-bootstrap of calib (95% range) | [0.234, 0.703] |
| oracle threshold at 1% / 2% / 5% within-arm FPR | 0.17 / 0.28 / 0.58 |
| all-129 calib reals | 0.048 |

On the all-129 variant:
- It does not agree in magnitude. Its threshold is the second-highest of 129 scores, set by the WA/FBH copies. B0 `orig` scores those higher: within-arm WA FPR is 8.4%.
- It does agree in direction: within 0.3%, cross 4.8%, about 16×.

Other cells:

| Cell | Reported | Range across rules | Reading |
|---|---|---|---|
| CommFor clean, `fov512` → `orig` | 6.4% | 4.7–15.9% (2nd/3rd calib, oracle 1–5%, all-129) | robust 3–9× |
| RINE laundered, `orig` → `fov512` | 31.6% | 29–42%; all-129 gives 3.9% (WA-driven again) | the largest, unflagged in the log |
| B0 laundered | 2.2% | oracle 1% → 9.6% | understates, because laundering inflates the calib max |

Two further points:
- The Clopper–Pearson CIs ignore both device clustering and threshold uncertainty. The threshold uncertainty dominates.
- **Use:** report the inflation as "≥ 5×, plausibly 20–50% FPR". Do not quote 23.4% as a number.

### F6 (medium): "RINE clean not sensitive" is a cancellation
- The macro Δ is −0.000, but the per-generator Δ (`orig` − `fov512`) is large in both directions: MJ +0.150, SD2 +0.080, SDXL +0.026, DALL-E 2 +0.021, DALL-E 3 −0.197, Firefly −0.080 (`c1.out`).
- The macro-only label hides scale dependence of up to 0.2.

### F7 (low): checks on bugs and leakage. Nothing changes a label
- **Pairing:** OK. Scores are joined by sha256, and row order is identical across arms (code_review).
- **Bootstrap:** OK. One plan per pool, shared by arms and detectors.
- **Texture sign:** +1 in every (input, condition) cell (`results.json` texture_direction). The sign never flips, so code-review F3 is moot.
- **D-008:** OK.
  - `orig` re-check vs 007 max: 1.8e-15.
  - `fov512` CUDA vs CPU max: 8.1e-5.
- **B0 = 005 A0:** reproduction diff 0.0 in both arms.
- **Minor, pre-existing (004 split):**
  - 49 of 305 test RAISE prompt ids recur across generators, and the bootstrap resamples fakes by image rather than by prompt. The CIs are slightly narrow.
  - 88 prompt ids are shared between train and test fakes. This is content leakage that can favour B0, equally in both arms.

## Recommendations (pre-registerable, before any official E1 result)

1. **R1: separate zoom from resampling with a third arm in the official E1.** `du2` = the `orig` crop, bicubic-downsampled ×0.5 and upsampled back to 512, for both classes. It keeps `orig`'s field of view but erases resampling history.
   - `orig` vs `du2` = resampling at fixed zoom.
   - `du2` vs `fov512` ≈ zoom.
   - Keep the RAISE (833 px) and WA real pools as scale-matched controls in every row, and pre-register per-pool Δ.
2. **R2: comparing to published numbers.** Compare each detector in the arm closest to its native protocol:
   - RINE in `orig`;
   - CommFor and ClipDet in `fov512`.

   Restate the Phase 3 ClipDet figure (0.780) as "crop protocol; 0.907 at whole-image resize (pilot)". Never cite RINE's laundered VISION number as robustness (F3).
3. **R3: D-013.** The evidence favours `fov512` as primary *for training and calibrating our own probes*:
   - a fixed `fov512` probe is scale-stable (AUROC 0.969–0.980, cross FPR ≈ 1%);
   - a fixed `orig` probe is not (laundered 0.923, cross FPR ≥ 5×).

   Keep `orig` as a mandatory secondary arm for Phase 3 continuity. The owner decides.
4. **R4: H2 and H3 FPR program.** Treat input scale as an environment:
   - pre-register a scale-shift row for every reported threshold: calibrate in arm A, report test FPR in arm B, in both directions;
   - in H2's worst-environment conformal, include scale variants of the calib reals (e.g. `orig` + `fov512` + `du2`) as environments;
   - for H3, calibrate at the same scale protocol as deployment, and report the ≥ 5× zoom photos separately.
5. **R5: threshold reporting.** Calib natives cover only 10 devices.
   - Pre-register a device-bootstrap interval for every FPR.
   - Pre-register a rule that a max+eps threshold (n < 100) is reported together with the 2nd-highest-score variant.
   - Pre-register per-generator max |Δ| as a secondary, next to the macro Δ (F6).
