# Phase 4 pilot synthesis: what the pilots taught us

**Date:** 2026-10-24. **Status:** every result here is a PILOT. The data are VISION phone photos plus 6 Synthbuster generators. RAISE, SD3.5, FLUX and B-Free were not available, because grip.unina.it is down. Per plan Amendment 2, **no pilot counts toward the 3-failed-hypotheses stop rule**, so that count is still 0.
**Sources:** README "Current status", LOG.md (2026-10-23 09:49 onward), DECISIONS.md (D-009, D-012, D-013), docs/02-hypotheses.md, docs/04-phase4-plan.md (Amendments 1–10 and clarifications), and each experiment's `pilot*/tables.md` and `critique.md`. All paths below are relative to the repo root.

**Terms used below:**
- **AUROC:** the probability that a random fake scores higher than a random real. 0.5 is chance, 1.0 is perfect, and below 0.5 means the score points the wrong way.
- **FPR:** the share of real photos wrongly flagged as fake. **TPR@1%FPR:** the share of fakes caught when the threshold allows 1% FPR.
- **Laundered:** `jpeg75_resize05`, meaning the image is resized ×0.5 and then saved as JPEG at quality 75. Both classes get this.
- **B0:** our baseline. It is a logistic "probe" (a linear classifier) on frozen DINOv2-base features.
- **LOGO:** leave-one-generator-out. Each generator is scored by a probe that never saw it.
- **Calibration:** choosing the threshold from real photos only.
- **`orig` arm:** D-009's 512-px centre crop with no resize. **`fov512` arm:** resize the short side to 512, then crop. This roughly matches field of view (zoom) between the classes.
- **Critic:** an independent opus/high reviewer, one per result.

---

## 1. Bottom line

None of the six pilots answered its hypothesis. Each one was *uninformative* for a reason we can name, and none found a code bug that changed a verdict. Taken together, though, they found something more important than any single verdict. **Much of what our detectors "see" on this data is zoom and texture, not generation traces.** Phone crops in `orig` are about 2.5–3× more zoomed in than fake crops, so they are smoother. Several scores and probes learned "smooth 1:1 phone close-up = real". Thresholds then fail when the input scale or the domain changes:
- FPR is inflated ≥5× across scale;
- B0 calls 77% of RAID web reals fake.

The pilot AUROCs of about 0.98 are therefore ceiling effects, not progress. On the official Phase 3 data, laundered AUROC is 0.875 at best (B0), and no method reaches 0.95. None of the three Phase-5 targets has been met or properly tested. The most useful decision now is D-013. Training and calibrating our own probes in `fov512` removes the real-pool gaps and the threshold drift, and gives a scale-stable probe. The cost is TPR at 1% FPR.

## 2. What each pilot tested and found

Each pilot is given as: its label under the pre-registered rule, the critic's reading, and the one number that matters.

**H1: laundering-nuisance projection** (`experiments/005-nuisance-projection/pilot/`, `pilot_v2/`)
- *Tested:* whether removing the feature directions that move when *real* photos are laundered helps a probe under laundering. The test arm is A2; the comparison arm A1 is augmentation only.
- *Pre-registered:* FAIL. A2 − A1 = −0.0072 [−0.0104, −0.0044] at `jpeg75_resize05` (`pilot/tables.md`).
- *Critic:* uninformative, because there was no headroom. A1 was already 0.988, so the largest possible gain was +0.012, below the +0.03 bar. The small loss is real: the k-sweep worsens steadily with k. So laundering directions do overlap directions the probe uses. Whether those directions carry generator traces or the zoom shortcut is unknown (`pilot/critique.md`).
- *pilot_v2* (grouped CV, balanced probe, device bootstrap): both arms are `UNINFORMATIVE_NO_HEADROOM`.
- *Number that matters:* **A1 = 0.988** laundered (`pilot/tables.md`).
- *Side finding:* in `fov512`, B0's threshold drift between native and WhatsApp/Facebook copies nearly disappears without any projection. Mean drift is 0.197 in `orig` and 0.004 in `fov512` (`pilot_v2/{orig,fov512}/tables.md`).

**H2: worst-environment conformal threshold** (`experiments/007-worst-env-conformal/pilot/`, `pilot_v2/`)
- *Tested:* whether taking the worst calibration group's 99th percentile keeps FPR ≤1.5% on unseen phones without losing more than 30% TPR.
- *Pre-registered:* INCONCLUSIVE for B0, because the TPR ratio was 0.59.
- *Critic:* uninformative. With about 43 reals per device, "worst device's 99th percentile" is simply the pooled maximum. The test therefore compared a 1% operating point with a ≈0.1% one.
- *pilot_v2:* re-run with platform groups of 358 reals. It gives NOT_SUPPORTED in both arms: at matched TPR, worst-group is no better than pooled. That verdict rule was adopted after the labels were seen, which Amendment 4 discloses.
- *Number that matters:* plain pooled calibration already transfers across VISION devices. **Leave-one-device-out FPR is 1.02% [0.51%, 1.83%]** (M2, `pilot/tables_scores_long.md`). The weak spot is WhatsApp: 3.07% in `orig`. WhatsApp pooled FPR at 1% is 3.31% in `orig` and 1.58% in `fov512` (`pilot_v2/*/tables_v2_scores_long.md`).

**H4: does a training-free drift score ride on file format?** (`experiments/006-format-shortcut/pilot/`, `pilot_addendum/`)
- *Tested:* a RIGID-style score (how much DINOv2 features move under small noise) with and without matched JPEG history.
- *Pre-registered:* REFUTED in `orig` and INCONCLUSIVE in `fov512`.
- *Critic:* uninformative. The score is **below chance**, so its gaps mean nothing. It mostly measures input texture, because smooth phone crops drift more (`pilot/critique.md`).
- *Number that matters:* **AUROC 0.292 [0.230, 0.362]** (`pilot/tables.md`). A plain Laplacian texture measure gives 0.778 in the opposite direction (`pilot_addendum/tables.md`). Once texture is regressed out, the score is about 0.51.
- *Addendum:* B0 itself is scale-dependent. Trained on `orig` and tested on `fov512` (laundered), it scores 0.923 against 0.980 within `fov512`.

**E1: zoom sensitivity of the published detectors** (`experiments/008-scale-sensitivity/pilot/`)
- *Status:* exploratory, pre-registered as Amendment 7.
- *Labels:* scale-sensitive means AUROC(`orig`) − AUROC(`fov512`) is at least 0.03 in size:
  - ClipDet clean: −0.127;
  - laundered: CommFor +0.062, RINE +0.228, ClipDet −0.217;
  - B0: not sensitive.
- *Critic* (`pilot/critique.md`): sound, but three readings must change.
  1. The log's "23.4% cross-zoom FPR" rests on one calibration photo. Quote it as **"FPR inflated ≥5× across scale"**. RINE laundered reaches 31.6%.
  2. RINE's laundered 0.945 > clean 0.813 is a **DALL-E 3 artefact** (0.166 → 0.935). Without DALL-E 3 it is 0.943 vs 0.946. Never cite it as robustness.
  3. "B0 not sensitive" holds only if B0 is retrained in each arm.
- *Number that matters:* a **fixed `orig`-trained B0 drops 0.981 → 0.923** (laundered) when the same photos arrive at `fov512`. A fixed `fov512`-trained B0 stays at 0.979–0.980, with cross-scale FPR about 1.1% (`pilot/tables.md`, critique F4).

**E1b: is it zoom or resampling?** (`experiments/008-scale-sensitivity/pilot_du2/`)
- *Status:* exploratory, Amendment 10. A `du2` arm (down, then back up) was added to separate the two.
- *Critic:* execution sound, design confounded. `du2` blurs phone photos (downsampled about 4.8×) far more than fakes (about 2×). So it adds its own cue, and the critic says **no `du2` number may be quoted**. Of the labels, only one survives a bandwidth-matched control: **the fixed `orig`-trained B0's laundered loss is caused by zoom**. CommFor and RINE flip to zoom, and ClipDet is not identifiable (`pilot_du2/critique.md`).
- *Number that matters:* the `du2` artefact term is **0.000 [−0.001, +0.002] for the fixed `orig` B0**, against −0.074 to −0.195 for the published detectors (critique F2). That is why only the B0 label is trustworthy.

**E2 #6: aligned-VAE probe** (`experiments/009-vae-aligned/pilot/`)
- *Tested:* a probe trained only on real photos vs their own SD-VAE round-trips. It never sees a generator. (The SDXL VAE failed the D-008 guard, so the bank is SD-VAE only.)
- *Pre-registered:* PILOT INCONCLUSIVE, 0.778 [0.731, 0.817] laundered (band 0.75–0.85).
- *Critic:* sound, and the label is correct. There is no format, JPEG-grid, zoom or texture shortcut. But **it is not a VAE-trace detector**: it rides a "not straight from a camera" axis that fails on RAID web reals (0.52). In `fov512` it gives 0.681, so D-013 would likely turn it into a FAIL.
- *Number that matters:* on unseen-device photos vs *their own* round-trips, **A separates them at only 0.66 clean / 0.56 laundered** (`pilot/critique.md` F1).
- *Positive side:* at the VISION 1%-FPR threshold, A flags only 2.1% of RAID reals, against 67% for B0.
- *C vs B* (adding aligned fakes to B0): no headroom. Descriptively it is −0.044, because the aligned fakes share VISION content.

**E2 #7: VAE double-reconstruction ratio**
- *Pre-registered:* PILOT FAIL. *Critic:* FAIL is correct.
- *Number that matters:* **0.601 clean, which loses to the texture baseline's 0.778**. Laundered it is 0.318, and even SD2, whose own VAE is in the bank, collapses to 0.186 (`pilot/tables.md`, critique F4). Dropped.

**E3 #5: canonicalization vs RAID transfer attacks** (`experiments/010-raid-transfer/pilot/`)
- *Tested:* whether a probe outside the attack ensemble survives RAID attacks at ε = 8/255, and whether test-time JPEG + resize ("canonicalization") recovers the loss.
- *Pre-registered:* UNINFORMATIVE. B0's clean AUROC of 0.643 is below the 0.80 gate.
- *Critic:* the label is correct and there is no bug. B0 calls 77% of RAID web reals fake (median logit +4.29). That is a camera-photo prior learned from 99 VISION reals.
  - ClipDet's "attack drop" is fully reproduced by random-sign noise of the same size.
  - For CommFor, at most about 0.065 of its 0.233 drop is image-specific.
  - Canonicalization nearly cancels the attack inside canonicalized space (residual drop 0.007–0.061), but it costs about as much clean AUROC (0.08–0.16). Without recalibration, RINE and ClipDet then flag 35–44% of reals.
- *Number that matters:* **CommFor, which is not in the attack ensemble, falls 0.952 → 0.719** at ε 8 (`pilot/tables.md`).

## 3. Cross-cutting lessons

1. **Zoom/scale confound.** Median short sides (`manifest_raw.csv`, Amendment 3):
   - VISION natives 2448 px;
   - fakes 768–1024 px (Firefly 1792);
   - WhatsApp 960, Facebook-HQ 1536, RAISE 833.

   A 512 crop without resize therefore shows phones much closer up than fakes. The effects:
   - B0's real-pool gaps in `orig`: native 0.992 / WhatsApp 0.966 / Facebook-HQ 0.982. In `fov512` they vanish: 0.979 / 0.979 / 0.980 (`005/pilot_v2`).
   - Threshold drift between photo versions (§2, H1).
   - Probably part of Phase 3's VISION 0.962 vs RAISE 0.876 gap. That gap was first read as a format shortcut; the lead's log says it "likely re-explains" it.

   Any number computed under D-009 `orig` carries this confound.
2. **Texture is a hidden variable.** Smoothness alone separates the classes in `orig` (Laplacian AUROC 0.778). It explains:
   - the inverted RIGID score;
   - the below-chance single-pass VAE error (ρ 0.85 with texture);
   - ClipDet's "smooth = fake" behaviour;
   - the `du2` artefact.

   Every score now needs a trivial texture baseline next to it.
3. **Thresholds do not transfer across scale or domain, even when ranking does.**
   - Cross-scale FPR inflation is ≥5× (E1).
   - Laundering mostly moves the *reals*: their logits rise by +2.5 to +5 (`007/pilot/critique.md` §4).
   - Canonicalization breaks RINE's and ClipDet's calibration (E3).
   - B0 flags 77% of out-of-domain reals (E3).

   A threshold is only valid for the scale, channel and domain it was calibrated on.
4. **Validity gates paid for themselves.** Each of these turned a misleading "FAIL" or "REFUTED" into an honest "uninformative":
   - the headroom rule (H1; #6 C vs B);
   - the score-validity gate (H4, #7);
   - the clean-AUROC gate (E3);
   - the pair-trace check (#6);
   - the arm-validity gate proposed for E1b.
5. **Our probe has a camera-photo prior.** B0 was trained on only 99 VISION reals. It learned "camera photo = real", and partly scene content too:
   - it flags 77% of RAID reals;
   - its scores correlate with each RAID id's content (Spearman 0.19–0.50, `010/pilot/critique.md`);
   - 88 prompt ids are shared between train and test fakes (`008/pilot/critique.md` F7).

   Part of its ~0.98 pilot AUROC is therefore content, not artefacts.
6. **Pre-run code reviews caught real bugs before any result.**
   - E3's clean control differed from RAID's own crop by 64–83 grey levels, and 52 of 100 reals had black-border padding (`010/code_review.md`).
   - The official H4 code was missing the B0-JM arm and silently skipped missing caches (`006/code_review_official.md`).
   - The E2 gate was misstated, and the `fov512` aligned fakes had a format cue (`009/code_review.md`).
   - Two runs (E1b, E3) used uncommitted code. Bit-exact regeneration closed the gap, but the rule should be: commit before running.

## 4. Where we stand against the Phase-5 targets

| Target | Status | Best evidence | Untested |
|---|---|---|---|
| FPR ≤1% on fresh phone photos | **Not shown.** On 2015–17 VISION phones, pooled calibration gives 1.02% [0.51, 1.83] leave-one-device-out (`007/pilot/tables_scores_long.md`). WhatsApp is 3.07% (`orig`), or about 1.5–2.0% on every platform in `fov512` (`007/pilot_v2/fov512`). Default thresholds are unusable: B0 17.0%; B-Free 17.3% and ClipDet 10.1% in Phase 3 (`003/notes.md`). | Calibration transfers across *old* phones at one scale. | Modern neural-ISP phones (H3). Calibration sets of ≥100 reals per group: we have 43 natives from 10 devices. Scale-shifted inputs. |
| AUROC ≥0.95 after laundering | **Not met on honest data.** The pilot's 0.98 is a ceiling: no RAISE, no FLUX, VISION-only reals, zoom and content shortcuts. Phase 3 official: B0 0.875 (LOGO 0.861), FLUX 0.813 laundered (`003/notes.md`). | — | Held-out FLUX, SD3.5 and RAISE in the pilot settings. |
| Robustness to RAID transfer attacks | **Untested in proper form.** Our B0 fails the validity gate (0.643). CommFor, outside the ensemble, falls 0.952 → 0.719 (`010/pilot/tables.md`). | Canonicalized space is nearly attack-invariant, but at an equal clean cost. | A probe trained in-domain and on canonicalized inputs, at fixed FPR, with noise controls. |

## 5. Recommendations

### (a) D-013: should `fov512` be primary?
**Recommendation:** make `fov512` primary *for training and calibrating our own probes, and for every FPR/threshold claim*. Keep `orig` as a mandatory secondary arm for continuity with Phase 3. Score published detectors in the arm closest to their own protocol (RINE in `orig`; CommFor and ClipDet in `fov512`, per `008/pilot/critique.md` R2), and report both arms. The owner decides.

*For `fov512`:*
- It removes the real-pool gaps (§3.1).
- It collapses threshold drift (0.197 → 0.004).
- WhatsApp FPR falls (3.31% → 1.58%).
- Texture leakage is lower (0.778 → 0.607).
- A fixed `fov512` probe is stable across scales (0.979–0.980, cross-scale FPR about 1.1%), while a fixed `orig` probe loses 0.059 through zoom (E1, E1b).
- Real deployments receive images at arbitrary resolutions, so normalizing scale is the honest setting.

*Against `fov512`:*
- It resamples every image, which may erase genuine low-level traces (D-013).
- TPR@1%FPR falls: B0 0.900 → 0.683 in E1 (`008/pilot/tables.md`); 0.879 → 0.517 in `005/pilot_v2`. Part of the `orig` TPR is probably the zoom shortcut, but we cannot tell how much without RAISE [unverified].
- Idea #6 likely flips to FAIL (0.681).
- The CIs are wider.
- It breaks direct comparison with the Phase 3 tables.

### (b) Ideas to carry into the official run
- **H4 first, as planned.** The code is ready (Amendment 6): validity gate, texture baseline, the RAISE contrast M − (ii), and `fov224`/`fov512` arms.
- **H1:** run it with the headroom rule and RAISE. Expect FAIL or INCONCLUSIVE (critic). Its drift motivation is largely moot if `fov512` is primary.
- **H2:** keep it as an *evaluation protocol* (LODO × LOPO, device bootstrap, groups ≥100, α sweep), not a method claim. Add input scale as an environment, with a "calibrate at one scale, test at the other" row for every threshold (E1 R4).
- **#6:** run only under the critic's official design: FLUX held out, RAISE reals, ≥30 devices, SDXL round-trips on CPU (about 1.8 h per arm), a pair-trace gate, a laundering-aware arm, and the label arm declared up front.
- **#5:** run only with an in-domain probe plus a canonicalized-input arm, noise controls, and ≥300 unpadded test reals (`010/pilot/critique.md`).
- **#7:** dropped.
- **E1/E1b:** repeat with RAISE as the scale-matched pool. Add an arm-validity gate and bandwidth-matched arms (`du2m`/`fov512m`), and never quote `du2`.

### (c) What the official run needs
- **grip.unina.it data:** RAISE 400, SD3.5/FLUX 300, B-Free weights. Then `env/bootstrap_data.py`, the SETUP_WSL step-6 reproduction, and H4 first.
- **Owner decisions:**
  1. D-013 (above).
  2. About 250 photos from a 2021+ phone for H3: default, night, portrait, ≥5× zoom, plus WhatsApp copies. This is now the main FPR ≤1% test.
  3. RAID originals for #5. All of them are about 0.9 GB (`kb/datasets/raid.md`), but only about 0.2 GB is left under the 9 GB cap (about 8.8 GB used, LOG 01:30). A few-thousand-id subset might fit; the size is my estimate from per-file means and must be checked against the cap.
  4. A larger backbone (DINOv2-L / DINOv3), which exceeds the cap (D-007).

### (d) Candidate directions for the Phase-5 method (PROPOSALS, not yet tested)
1. **A scale-normalized, environment-calibrated probe.** Train in `fov512`, optionally with multi-scale augmentation. Calibrate on reals from several scales and channels treated as environments, and report the FPR inflation under scale shift as a headline metric. This targets the FPR goal directly, and is motivated by E1/E1b.
2. **Diversify the real side.** Train on reals beyond camera photos (RAISE, platform copies, modern phones, and in-domain web reals if approved) to remove the camera-photo prior. #6's aligned negatives could serve as a component, since their real-side prior was much weaker (2.1% vs 67% flagged on RAID).
3. **Train in canonicalized space.** Canonicalized inputs nearly cancel ε-8 attacks (E3). A probe trained and calibrated there might keep the attack invariance without the clean cost [unverified].
4. **Shortcut audits as acceptance criteria.** A method passes only if it beats a texture baseline, passes an arm-validity check, and holds its FPR on a scale-shifted and an out-of-domain real set.

## 6. Summary table

| Idea | Pilot label (pre-registered → critic) | Status | Next step |
|---|---|---|---|
| #1 H1 nuisance projection | FAIL → uninformative (no headroom) | Official code ready | Official run with RAISE and the headroom rule |
| #2 H2 worst-env conformal | INCONCLUSIVE; v2 NOT_SUPPORTED → uninformative | Reframed as protocol | Official LODO × LOPO with scale-shift rows |
| #3 H3 neural-ISP audit | not run | Blocked | Owner's ~250 modern-phone photos |
| #4 H4 drift format shortcut | REFUTED / INCONCLUSIVE → uninformative (texture) | Official code ready | Run first when grip returns |
| #5 canonicalize vs RAID | UNINFORMATIVE (gate) → confirmed | Untested in proper form | In-domain + canon-trained probe; owner decides RAID download |
| #6 aligned-VAE probe | INCONCLUSIVE → not a VAE-trace detector | Official design written | FLUX held out, RAISE, ≥30 devices, pair-trace gate |
| #7 VAE ratio | FAIL → confirmed | Dropped | None |
| E1 / E1b scale diagnostics | scale-sensitive / `du2` confounded | Pilot done | Official repeat with RAISE and bandwidth-matched arms |

Stop-rule count: **0 failed hypotheses.** Pilots do not count (Amendment 2).
