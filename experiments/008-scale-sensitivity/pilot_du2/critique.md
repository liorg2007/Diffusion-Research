# E1b (Amendment 10, `du2` arm) critique (independent critic, opus/high, 2026-10-24)

**Rating: EXECUTION SOUND, DESIGN CONFOUNDED.**
- Execution: no bug. The data match the committed code bit-exactly, the Deltas are paired and reproduce, and D-008 passes.
- Design: `du2` is not a clean "fixed zoom, `fov512` resampling" arm. It adds a strong class-differential blur, so the pre-registered labels cannot be read at face value.
- Under a bandwidth-matched control:
  - 1 of the 4 labels survives: the B0 `orig`-probe, laundered, zoom-dominated.
  - CommFor laundered flips from resampling-dominated to zoom-dominated.
  - RINE laundered moves from mixed to zoom-dominated.
  - ClipDet is path-dependent (unidentified).

Scope: CPU only. Inputs and methods:
- re-analysis of `pilot_du2/scores_long.csv.gz` and `results.json`;
- image statistics on the saved crops;
- one control arm, `du2m`, which is **exploratory and NOT pre-registered** (F2). It rebuilds the 358 test natives in memory from the RAW files (no PNG written) and scores them on CPU with CommFor, ClipDet and RINE, plus the fixed B0 probes.

Everything is in `pilot_du2/critic_checks/`:

| Question | Scripts | Outputs |
|---|---|---|
| Q3 | `c1_regen.py` | `c1_regen.out` |
| Q1 | `c2_blur.py` | `c2_blur.out`, `c2_blur.csv.gz` |
| Q1/Q2 | `c3_sides.py` | `c3_sides.out`, `.json` |
| Q1 control | `c4_du2m.py` | `c4_du2m.log`, `c4_scores.csv` |
| Q1 control | `c5_du2m_analysis.py` | `.out`, `.json` |
| Q1 control | `c6_du2m_blur.py` | `.out` |

This is PILOT data: exploratory, not counted toward the stop rule.

## Findings, ranked

### F1 (high, Q1): `du2` builds a class-differential blur cue that neither `orig` nor `fov512` has
**Why it happens.** Each image is downsampled by its own factor k = short side / 512 and then upsampled by the same k (median k, E1 `sizes.csv`):

| Group | Median k |
|---|---|
| VISION natives | 4.78 |
| FBH | 3.00 |
| WA | 1.88 |
| Fakes | 1.5–3.5 (median 2.0) |

- `fov512` keeps the downsampled pixels, so both classes look sharp.
- `du2` re-expands them. The surviving passband is about 0.5/k cycles per pixel: about 0.10 for natives against about 0.25 for fakes. The resampling *history* therefore becomes a visible, k-proportional *blur*.

**Measured on the 512-px arm images** (`c2_blur.out`, test rows). The statistic is the AUROC of the statistic for fakes against the pool's reals, where >0.5 means the fakes are sharper:

| Statistic, clean | `orig` | `du2` | `fov512` |
|---|---|---|---|
| natives, mean \|Laplacian\| | 0.789 | **0.907** | 0.601 |
| natives, spectral energy >0.125 cyc/px | 0.610 | **0.875** | 0.586 |
| natives, energy >0.25 cyc/px | 0.689 | **0.839** | 0.587 |
| WA (k matched to fakes), energy >0.125 | 0.538 | 0.475 | 0.585 |
| FBH (k 3.0), energy >0.125 | 0.618 | 0.691 | 0.589 |

- Laundered shows the same pattern. Natives' energy >0.25 gives AUROC 0.589 / **0.838** / 0.589.
- Natives keep a median 1.2% of their `orig` >0.25 cyc/px energy in `du2`, against 7.2% for fakes.
- At the detectors' actual inputs, the texture-only baseline rises to 0.86–0.89 in `du2`, from 0.68–0.78 in `orig` and about 0.61 in `fov512` (`tables.md`). This confirms the lead's concern.
- The cue is native-specific and tracks k. It is absent for WA (0.475–0.55) and intermediate for FBH.

**Consequence: the decomposition is valid algebra but not a clean factorisation.** Let U be the effect of the `du2`-only artefact. Then Δ_resample = true_r + U and Δ_zoom = true_z − U. U cancels in E1's total but not in |Δ_r| − |Δ_z|, which is what the labels test.

### F2 (high, Q1): the bandwidth-matched control `du2m` shows that U is large, and it changes 3 of the 4 labels
**Design.** `du2m` = the natives down/up by a fixed k = 2 (the fakes' median), then the plain 512 centre crop. The fakes keep their `du2` scores.

**It removes the spectral cue** (`c6_du2m_blur.out`), with AUROCs for fakes against natives:
- energy >0.125: 0.875 → 0.547 clean, and 0.762 → 0.562 laundered;
- Laplacian: 0.907 → 0.684, which is at or below the `orig` natives' intrinsic 0.79 / 0.69.

**Pipeline self-checks** (`c4_du2m.log`, `c5_du2m_analysis.out`):
- The in-memory per-image-k pipeline reproduces the stored `du2` PNGs bit-exactly.
- CPU scores of 8 `du2` natives match the cached CUDA `du2` scores to ≤ 4.7e-5 for every detector and for the B0 probes.
- B0 features match the 004 `du2` cache to a scaled 4e-6.

**Results**, native pool, same paired cluster bootstrap plan as `run_du2` (E1 `Plan`, seed 0, 2000 draws):

| Row (E1-flagged) | AUROC `orig` / `du2` / **`du2m`** / `fov512` | U = `du2` − `du2m` | Reported label (`du2`) | Label via `du2m` (Δ_r′ / Δ_z′) |
|---|---|---|---|---|
| CommFor, laundered | 0.891 / 0.808 / **0.882** / 0.828 | −0.074 [−0.102, −0.051] | resampling-dominated (+0.082 / −0.020) | **zoom-dominated** (+0.008 / +0.054; abs diff −0.046 [−0.084, −0.007]) |
| RINE, laundered | 0.945 / 0.855 / **0.934** / 0.717 | −0.079 [−0.099, −0.058] | mixed (+0.089 / +0.139) | **zoom-dominated** (+0.011 / +0.217; −0.207 [−0.250, −0.164]) |
| ClipDet, laundered | 0.524 / 0.470 / **0.665** / 0.742 | **−0.195** [−0.230, −0.156] | zoom-dominated (+0.054 / −0.271) | **resampling-dominated** (−0.141 / −0.077; +0.064 [+0.014, +0.110]) |
| ClipDet, clean | 0.780 / 0.867 / **0.919** / 0.907 | −0.052 [−0.072, −0.033] | mixed (−0.087 / −0.040) | **resampling-dominated** (−0.139 / +0.013; +0.127 [+0.088, +0.164]) |
| B0 fixed `orig` probe, laundered | 0.981 / 0.977 / **0.977** / 0.923 | **+0.000** [−0.001, +0.002] | zoom-dominated (+0.004 / +0.055) | **zoom-dominated** (+0.004 / +0.054; −0.050 [−0.069, −0.032]) |
| B0 fixed `fov512` probe (both conditions) | 0.969–0.980 in every arm | ≤ 0.007 | not flagged | mixed / stable |

**Caveat on `du2m`.** It is not the pre-registered quantity either:
- its natives carry a k = 2 history instead of their own k ≈ 4.8, so Δ_z′ also contains the natives' extra 2 → 4.8 downsampling;
- the fakes keep per-image k between 1.5 and 3.5.

So where `du2` and `du2m` disagree, the honest label is **"path-dependent: not identifiable with this design"**, not the `du2m` label.

**Which labels survive:**
1. **B0 `orig`-probe, laundered: zoom-dominated. Survives.**
   - U = 0.000 and both paths agree.
   - Per pool (`tables.md`), the zoom term scales with the real/fake zoom mismatch: FBH +0.037 and WA +0.008.
   - Natives' median logit stays at −3.66 → −3.82 from `orig` to `du2`, then jumps to +1.09 in `fov512` (`c3_sides.out`).
   - So the "1:1 phone close-up = real" shortcut (E1 critique F4) is about zoom, not resampling history.
2. **CommFor, laundered: the reported "resampling-dominated" does not survive. It flips to zoom/presentation scale.** Three independent lines of evidence:
   - `du2m` gives the zoom label.
   - The swap decomposition (`c3_sides.out`, AUROC with reals from one arm and fakes from another; the fakes are the same images in every pool) gives Δ_r = −0.003 (fake side) + 0.085 (real side). The fakes' laundered scores do not move from `orig` to `du2` (e.g. DALL-E 2 median 3.80 → 3.77). The real-side part tracks the natives' k: Spearman(k, `du2` − `orig` score) = +0.35, with median shifts of +0.93 for k < 4 against +2.68 for k ≥ 4. It also tracks the pool's k: real-side Δ_r is WA +0.015, FBH +0.031, natives +0.085.
   - The scale-matched pools already give the zoom label in the reported table: WA +0.014 / +0.098 and FBH +0.028 / +0.071.

   The genuine effect is fake-side and appears only in the `fov512` presentation: fake-side Δ_z = +0.307. The fakes' median laundered scores fall by 3.2 to 11.4 logits only in the `du2` → `fov512` step (for example Firefly 3.96 → −7.47). This **corrects E1 critique F2.** The CommFor loss is not trace loss from cumulative resampling *history*. It comes from showing the fakes at reduced scale before laundering.
3. **RINE, laundered: "mixed" was produced by the artefact.**
   - Without the artefact, `du2m` gives zoom-dominated, about 95% of the total.
   - The zoom part is real-side and pool-independent: real-side Δ_z = +0.126 natives, +0.138 WA, +0.151 FBH.
   - Reading: reals presented at reduced scale lose RINE's "real" cue after laundering. This matches E1 critique F3.
4. **ClipDet (both conditions): not identifiable.**
   - Its reported "zoom-dominated" rests on a `du2` AUROC depressed by U = −0.195. That exceeds the about 0.16 that the label could absorb before flipping, given a total of −0.217.
   - ClipDet's laundered score rises in `du2` for *both* classes (natives +0.81; fakes +0.16 to +1.54 in median), consistent with "smooth = fake" (E1 critique F2).
   - What can be said: a mild, class-matched band-limit at fixed zoom already recovers most of ClipDet's crop-protocol deficit (laundered 0.524 → 0.665; clean 0.780 → 0.919). So "ClipDet is bad on `orig` because of zoom" is **not** established. Pixel-level statistics matter at least as much.

### F3 (medium, Q2): the Deltas, pairing and CIs are computed correctly
- **Pairing.** The sha256 row order is asserted identical across the 3 arms (`stage_score`). One bootstrap `Plan` per pool is shared by all arms and rows, so replicate b uses the same device and image weights everywhere. The CIs of Δ_r, Δ_z and |Δ_r| − |Δ_z| are therefore paired.
- **Reproduction.** An independent recompute from `scores_long` reproduces every point Δ_r and Δ_z in all pools, rows and conditions to ≤ 7e-16 (`c3_sides.out`). Δ_r + Δ_z = E1's Δ is asserted in code.
- **Labels.** Labels are given only to E1-flagged rows (CommFor laundered, RINE laundered, ClipDet both conditions, `b0_trained_orig` laundered), with the pre-registered CI rule.
- **Minor issues** (none changes a label):
  1. The percentile CI of |Δ_r| − |Δ_z| is not centred when one Δ is near 0. For example, `b0` clean has point +0.008 with CI [−0.015, +0.011]. Only unlabelled rows are affected.
  2. The `texture_*` rows print a label although they are not pre-registered detectors. They should be shown as diagnostics only.
  3. The transfer-FPR "95% CIs" are percentile intervals of a calib-max threshold, and the point estimate sits at the edge of them. For example, the B0 `fov512` probe has TPR 0.394 [0.353, 0.950]. A bootstrap of an extreme order statistic is not a valid CI. Label these intervals "range under device resampling".
  4. As in E1 F6, macro Δ hides per-generator moves. ClipDet laundered per-generator Δ_r ranges from −0.25 (Firefly) to +0.22 (SDXL).

### F4 (low, Q3): the data were produced by the committed code. Provenance has one gap
- **Regeneration** (`c1_regen.out`). 54 stratified rows (6 per group × 9 groups, including every real pool and every generator, with native sizes up to 4784×2704) were regenerated with the committed `lib/preprocess.py`:
  - pixel-identical (max |Δ| = 0) to `data/processed/du2512`;
  - also identical to `crop512` and `fov512` in their default modes.
- **Default modes are preserved.** `git diff HEAD~1` shows only a new `elif mode == "du2"` branch. The `crop` and `fov` paths are untouched.
- **Prep report.** 2220 of 2220 OK, raw sha256 match 2220/2220, zero images with short side < 512.
- **Timeline.** File mtimes in the chain:

  | Event | Time |
  |---|---|
  | `preprocess.py` and `fov_prep.py` last modified | 04:52:51 |
  | prep report written | 04:55:27 |
  | `run_du2.py` last modified | 04:59:00 |
  | run start | 05:00:34 |

  No file in the chain was modified after the run. The working tree equals HEAD.
- **Gap.** `results.json` `git_head` = 428a066, the commit *before* E1b. The run used uncommitted code. The bit-exact regeneration closes this gap for the pilot.

### F5 (low, Q4): D-008 and B0 rows are sane
**D-008:**
- `du2` CUDA vs CPU: CommFor 1.2e-4, RINE ≤ 1.6e-5, ClipDet ≤ 2.8e-5.
- 004 feature guard: logit 9.0e-5, scaled features ≤ 5.2e-5.
- My CPU re-score of 8 `du2` natives vs the cached scores: ≤ 4.7e-5.

**B0:**
- The fixed `orig` and `fov512` probes re-score E1's cached A0 exactly (diff 0.0, `run.log`).
- The fixed `fov512` probe is flat across `orig`, `du2`, `du2m` and `fov512` (0.965–0.980), and its native and real-pool medians are about −1.2 in every arm.
- The `orig` probe's natives' median moves −8.69 → −5.93 → −0.58 (clean, `orig` → `du2` → `fov512`). The clean `du2` step is k-dependent: +1.60 for k < 4 vs +3.37 for k ≥ 4. So B0 reacts to the blur when not laundered, but not after laundering (U = 0.000).
- `b0_trained_du2` (extra) is plausible, but it was trained on data that carry the blur cue. Do not use it.

## What can be concluded (Q5)
**D-013.** E1b adds one robust fact:
- the fixed `orig`-trained B0 loses AUROC under scale change because of **zoom**, not resampling history (F2.1);
- a fixed `fov512`-trained probe is stable on every arm, including both blur arms (F5).

This strengthens E1 critique R3: train and calibrate our own probes in `fov512`, and keep `orig` as a mandatory secondary arm. Nothing in E1b argues against it. Do **not** use the pretrained-detector labels for D-013. The owner decides.

**Comparing published numbers.** E1 critique R2 stands: compare each detector in the arm closest to its native protocol (RINE in `orig`; CommFor and ClipDet in `fov512`). E1b adds three points:
- **Never compare or quote any `du2` number.** It is an artificial upsampled arm.
- CommFor's laundered drop in `fov512` (0.891 → 0.828) is a presentation-scale effect on fakes, and its native whole-image-resize protocol incurs that effect too. Quote the `fov512` figure as its protocol-matched laundered number, and describe it as scale fragility, not "resampling-history fragility".
- For ClipDet, do not explain the crop-vs-whole-image gap as "zoom" (F2.4).

## Recommendations (pre-registerable, before any official E1/E1b result)
1. **R1: an arm-validity gate.** For any synthetic arm, pre-register a texture-only check: the Laplacian and >0.125 cyc/px spectral AUROC (fakes vs each real pool) on the arm and at each detector's input. An arm whose texture AUROC exceeds max(`orig`, `fov512`) by > 0.05 is declared confounded, and its decomposition is reported as descriptive only. `du2` fails this gate (0.875 vs 0.610 / 0.586).
2. **R2: replace or augment `du2` with a bandwidth-matched pair.** For example:
   - `du2m` = down/up by a common fixed factor (k = 2) for **both** classes;
   - plus `fov512m` = the matching zoom arm.

   Label a detector only if the per-image-k path (`du2`) and the common-k path agree in direction. Otherwise label it "path-dependent".
3. **R3: report the real-side / fake-side swap decomposition** (reals from arm a, fakes from arm b) next to every Δ. Since the fakes are shared across pools, this separates real-photo-scale effects from fake-trace effects at no extra GPU cost.
4. **R4: require the scale-matched pools to agree.** WA, and RAISE in the official run, are the control. A native-pool label counts only if WA/RAISE do not show the opposite label. If they do, report "pool-dependent". In this pilot that rule alone would have caught CommFor.
5. **R5: reporting hygiene.**
   - Present `texture_*` rows as diagnostics, without labels.
   - Rename the transfer-FPR bootstrap intervals "device-resampling range" and report the 2nd-highest-score threshold beside them (E1 critique R5).
   - Report per-generator max |Δ_r| and max |Δ_z| next to the macro values.
   - Record the actual code commit, and commit before running.
