# Critique: OFFICIAL H4 run (006-format-shortcut/official)

**Rating: REFUTED stands. I found no bug or deviation that changes the verdict.** The finding "JPEG on the reals does not give RIGID a shortcut" is robust: the M−ii gap's CI upper bound is ≤ +0.008 in all 12 orig/fov512 cells. Two things are fragile or need care:
- The label cell passes the validity gate only because of Firefly.
- RIGID *is* sensitive to format. The effect is 0.02–0.06 AUROC, and its sign depends on the scale arm.

Separately, most of B0's VISION−RAISE gap comes from **caption twins shared across splits**, not from format or zoom.

Critic: opus/high, 2026-10-25, CPU only. Scratch checks and their outputs are in `official/critic_checks/` (`c1_recompute`, `c2_vision_raise_gap`, `c3_manipulation_pixels`, `c4_twin_bootstrap`, each a `.py` with a matching `.txt`).

## Answers in one line each
1. **Bugs or deviations.** None changes the verdict. The settings, best σ, gate, FLUX test-only, strict mode, D-008 and pairing all match the pre-registration (details in F6).
2. **Is REFUTED the right reading?** Yes, as a label. As a conclusion it holds only in the narrow form: no real-side JPEG shortcut ≥ 0.03 at the 512-crop scales. The gate's "beats texture" pass depends on Firefly. The manipulation did change the reals: PSNR 37 dB, Laplacian +3%, and the RIGID score rose for 70% of RAISE reals.
3. **Why the VISION−RAISE gap persists in fov512.** RAISE reals whose caption-twin fakes are in the data score much worse. RAISE reals with no twin reach 0.935, near VISION's 0.953. A scale-matched VISION pool (WhatsApp copies) stays at 0.914 (orig) / 0.939 (fov512). Zoom and format explain little of the gap.
4. **Stop rule.** Yes: count it as failed hypothesis 1 of 3. It is an official run with a pre-registered prediction, and that prediction was falsified.
5. **H1/H2.** Pre-register a twin-disjoint sensitivity analysis. Confirm H1's primary pools. Decide how UNINFORMATIVE labels count before reading any H1 output. Details in the Recommendations section.

## Findings (ranked)

### F1. HIGH (project-wide; not the RIGID label): the VISION−RAISE gap is mostly a caption-twin content effect
- **Background.** Every fake file is named after the RAISE image whose caption prompted it (`src_path`, e.g. `synthbuster/dalle2/r001d260dt.png`). The 003 splits do not group by RAISE id:
  - 181 of the 199 RAISE test reals have a twin among the fakes;
  - 126 of them have a twin among the **B0-train** fakes.
- **B0 clean macro AUROC on RAISE test reals by twin status** (`c4_twin_bootstrap.txt`; image bootstrap, 2,000 draws):

  | Twin status | n | orig | fov512 |
  |---|---|---|---|
  | twin in train AND test | 70 | 0.786 [0.728, 0.841] | 0.754 |
  | twin in train only | 56 | 0.883 | 0.862 |
  | twin in test only | 55 | 0.888 | 0.853 |
  | no twin | 18 | **0.935** [0.909, 0.961] | 0.918 |

  - No twin minus any twin: +0.088 [+0.054, +0.122] (orig) and +0.101 [+0.024, +0.163] (fov512).
- **Why this points to training.** A real's own twin is at most 1 of the ~60–75 fakes per generator, so the direct pair can move its AUROC by about 0.017 at most. The 0.15 drop for "train and test" twins therefore means the probe learned the twin's *content* during training.
- **Controls that rule out the alternatives:**
  - *Scale.* WhatsApp copies of the same VISION photos have a RAISE-like short side (960 px vs 833) and stay at 0.914 (orig) / 0.939 (fov512) (`c2`; tables.md "Per real pool").
  - *Generic content proximity.* Each real's nearest-fake cosine was computed with twins excluded. Within quartiles of that proximity, RAISE stays at 0.83–0.86 while VISION is at 0.93–0.97 (`c2`).
  - *Format.* B0-JM moves the gap by only −0.009 to −0.015. JPEG(90) on the RAISE reals *lowers* B0's AUROC (M−ii = −0.020).
- **Caveats:**
  - The no-twin group is small (n = 18).
  - Twin status is not random: with independent sampling I would expect about 39% of reals without a twin, not 9% (my estimate: 0.88⁶ × 0.85, assuming 120 of 1,000 RAISE-1k ids per Synthbuster generator and 150 for SD3.5). So the no-twin group may differ in other ways.
  - This is strong evidence, but not yet a controlled test.
- **Implications:**
  - The Phase 3 VISION−RAISE gap (0.086) and the B0-JM target were probably aimed at a split artifact.
  - RIGID is training-free, so the H4 label is unaffected (beyond the ≤ 0.017 direct-pair bound). The M−ii contrast is within RAISE anyway.

### F2. MEDIUM (how robust the label is): the label cell passes the gate only because of Firefly
- **Effect of each generator on the gate margin** (RIGID AUROC(ii) − texture AUROC(ii), σ = 0.02, orig; `c1_recompute.txt`, texture with the run's direction +1):

  | Generator dropped | Gate margin |
  |---|---|
  | none (label cell) | +0.081 |
  | Firefly | −0.002 (gate fails, matching the UNINFORMATIVE without_firefly row) |
  | each of the other 6 | stays between +0.045 and +0.153 |

- **Firefly** has RIGID 0.864 against texture 0.284 on setting (ii). Its crops are the most zoomed-in (1,792 px short side) and are JPEG q95, so they are smooth. Smooth inputs drift more, which is the pilot's mechanism. The texture baseline has one global sign, which is wrong for Firefly, so the macro "beats texture" comparison overstates RIGID.
- **Evidence that RIGID is still valid on RAISE:**
  - texture-residualized RIGID AUROC(ii) is 0.679 with Firefly and 0.665 [0.616, 0.714] without;
  - fov512, where zoom is matched, passes the gate in both Firefly variants (0.590 vs 0.501; 0.567 vs 0.524).
- **The conclusion does not depend on the gate.** The M−ii gap is negative, with CI upper bound ≤ +0.008, in all 12 orig/fov512 cells (post and pre-LN, all three Firefly variants). The residualized M−ii is also negative (orig σ 0.02: −0.055). Under any reading, no cell gets near the +0.03 needed to avoid REFUTED.

### F3. MEDIUM (interpretation): RIGID is format-sensitive, with a sign that depends on scale
- **Real-side JPEG, M−ii** (best σ):

  | Arm | M−ii | Reading |
  |---|---|---|
  | orig | −0.042 | JPEG reals look *more fake* |
  | fov512 | −0.012 | about neutral |
  | fov224 | +0.037 to +0.040 (CI lower +0.017) | the hypothesized direction, above the 0.03 refute line |

  - fov224 is UNINFORMATIVE only because RIGID is at chance there (AUROC(ii) 0.516).
- **Fake-side JPEG, ii−iii:** +0.020 (orig), +0.037 (fov512), +0.050 (fov224).
- **Wording.** Write "REFUTED: no real-side JPEG shortcut of ≥ 0.03 at 512-crop scales". Do *not* write "format-insensitive". Amendment 6 already says "a legitimate fusion component" does not follow. RIGID's weakness is clear from three numbers:
  - AUROC(ii) ≤ 0.62;
  - AUROC(i) is 0.27 (inverted on phones);
  - TPR at the 1%-FPR operating point is ≤ 0.037 in every arm (tables.md, operating-point table).

### F4. Manipulation check: M did change the reals, mildly
- PSNR of jpeg(90) vs clean on 40 RAISE test crops: median 37.2 dB, range 29.8–42.9 (`c3`).
- Mean Laplacian of the reals: ×1.033 (orig), ×1.055 (fov512), ×1.064 (fov224).
- RIGID score per RAISE real (orig, σ 0.02): it rose for 70% of images, by +0.0046 on average (0.14 SD) (`c1`).
- B0 M−ii: −0.020 [−0.025, −0.015]. Texture: −0.012.
- The raw RAISE files show no 8×8 JPEG grid (blockiness 1.000, against 1.048 for VISION and 1.039 for Firefly; `c3`), so M is a first compression.
- q90 is mild. It is the right size for "camera JPEG vs PNG" (VISION's quality is [unverified]), but stronger compression was not tested.

### F5. B0-JM fails, and it hurts the compressed real pools
- The criterion is not met in any arm (gap 0.088 / 0.098 / 0.085).
- B0-JM costs WA −0.017 and FBH −0.023 (orig), and −0.026 to −0.033 (fov512). The CIs exclude 0.
- JPEG-augmenting only the fakes creates a reverse "JPEG artifacts → fake" cue. Do not adopt it. This supports H1's symmetric (both-class) augmentation.

### F6. LOW: deviations and process (none changes the verdict)
- **Verified correct:**
  - settings M, ii, iii, iv (config);
  - best σ = argmax AUROC(ii); the gate checks CI lower bound > 0.5 AND beats texture; REFUTED requires gap < 0.03 at all σ (`h4.py:146-170`);
  - FLUX is test-only (asserted) and enters neither the texture direction nor training;
  - strict mode was on (no skip or drop lines in `run.log`);
  - D-008 passed on CUDA for every arm and condition: max scaled Δfeature 1.8e-5, max Δscore 2.1e-6, pipeline-vs-cache 0.0. The 004 extraction guards had max |Δlogit| ≤ 3.6e-5;
  - M−ii is paired on the same RealSet and FakeSet;
  - the code was committed before the run (cac9b81);
  - my recompute from `scores_long` matches every label-cell point estimate.
- **Minor items:**
  - (a) The B0-JM "LOGO-macro clean loss" is implemented as the mean over the VISION-native and RAISE pools. It is moot, because the gap criterion fails.
  - (b) The 0.086 Phase 3 reference comes from a different B0 recipe; this run's B0 gap is 0.097.
  - (c) RAISE is bootstrapped per image (4 photographers, 3 cameras), so the gate's CI is understated. The margin above 0.5 is 0.072, which is probably safe.
  - (d) The 004 `verify003` step was SKIPPED, because the 003 cache does not exist on this PC. This is mitigated by the 0.0 pipeline identity and the step-6 reproduction.
  - (e) The 004 `config_fov512.json` and `config_fov224.json` were first committed together with the results.

### F7. Stop rule
- H4 counts as **1 of 3**: it is an official pre-registered hypothesis, and its prediction (gap ≥ 0.10) was falsified.
- Its outcome was informative, but by the letter of CLAUDE.md a refuted hypothesis is a failed one.
- If both H1 and H2 fail, the project stops and reports. That is why R3 below matters.

## Recommendations (to log as a dated amendment BEFORE any H1 output is read; H1 is already running)
- **R1. Twin-disjoint sensitivity analysis** (H1, H2, B0; CPU, cached features).
  - Re-fit each probe without the train fakes whose RAISE id equals any calib or test RAISE real.
  - Also report RAISE per-pool AUROC split by twin status.
  - Secondary only. Primary endpoints stay as they are.
- **R2. H1 primary pools.**
  - Confirm `--primary-pools vision_native`, the pre-registered default, together with `--per-pool` and `--headroom-rule`.
  - Do not switch to RAISE because it has headroom. H4 has already exposed B0's per-pool numbers: (iv-p) native 0.911 vs (iv) RAISE 0.786.
  - Keep RAISE as a per-pool secondary.
- **R3. Fix the stop-rule counting now:**
  - UNINFORMATIVE and NO_HEADROOM labels do not count as failures;
  - decide whether the H-D1 critic's "miss ≥ 0.10 below the fail line = FAIL" applies to H1/H2. It is legitimate only if it is logged before their results.
- **R4. Robustness reporting for every gate-type decision.**
  - Report it with and without Firefly, plus a drop-one-generator analysis.
  - Flag "generator-fragile" if a single generator flips it.
- **R5. H2:** add the RAISE test reals (n = 199) as a descriptive held-out domain group, with FPR at the native-calibrated thresholds. `005/run.py:350` already computes this. It has no verdict role.
- **R6. Record in KB and README** that H4 is REFUTED in the narrow sense (F3 wording), that RIGID is not a fusion component, and that F1 is an open protocol issue affecting every RAISE number since Phase 3.
