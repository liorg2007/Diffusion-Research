# Critique of 021-attr-residual (H-A2): opus/high critic, 2026-10-27

**Verdict:** the label **H-A2 = PASS (via ii)** follows docs/08 and docs/09 exactly. I found no leak and no bug that changes a scored number.
- The device result is real sensor PRNU, not scene content.
- It is a replication of textbook PRNU (Lukas 2006 / Chen 2008) on 2010s iPhones. It is not a new finding, and it says nothing about diffusion images.
- The generator arm (i) carries no evidence:
  - it ties the M1 texture baseline;
  - its signal is locked to the image's top-left corner (the "phase-locked" artefact explained in §3), so a crop of a few pixels destroys it.
- **Stop-rule count:** H-A2 adds 0 failures.

Critic checks (CPU, reporting only, judged pipeline untouched, about 11 min): `critic/dev_checks.py` → `critic/dev_checks.json`; `critic/gen_checks.py` → `critic/gen_checks.json`.

## 1. Leaks and content
**Leave-one-photo-out (code read: run.py `lopo_K`, `test_worker`, `acc_worker`).**
- Each arm has its own per-device sums: nat, wa and fbh are accumulated separately. So the WA and FBH copies of a test photo never enter the native fingerprint, and vice versa.
- The test photo's own term is subtracted in its own arm. For nat→WA, the native copy is subtracted.
- `verify_lopo_subtraction` matches an explicit re-sum (float32 error |dN| 1.4e-3, negligible). Selftest: a planted leak inflates PCE and the LOPO gate raises. **No leak found.**

**Near-duplicate scenes (dev_checks.json "thumb", "highpass_thumb", 197 canonical natives).**
- Within-device pairs with thumbnail correlation > 0.9: 1 (D15_179 / D15_197, r = 0.96). Cross-device pairs: 0.
- For only 10–12% of photos is the most similar photo from the same device (chance about 6.6%). Scene clustering by device is weak.
- How duplicated a photo's scene is does not predict its true-device PCE:
  - Spearman ρ is 0.07 (p 0.30) on thumbnails and 0.04 (p 0.61) on high-pass thumbnails.
  - The 20% most duplicated photos have a lower median true PCE: 994 vs 1151 (thumbnail), 861 vs 1171 (high-pass).

**Direct content probe (single-photo fingerprints).**
- D15_179 scored against a fingerprint built from its near-duplicate D15_197 gives zero-shift PCE −0.27, and the peak is not at zero shift. Hand-held shots are misaligned, so content does not line up at zero shift.
- Non-duplicate same-device single-photo fingerprints already reach PCE 144–333 for D05 and D19. Sensor noise alone carries the match.
- The other "top" pairs are not real duplicates (high-pass r ≤ 0.15); their PCE matches non-duplicates (202, 245 vs 144, 146; 48 vs 333).
- Correlation at a 1–2 px shift is 0.1–0.5 of the zero-shift value, and negative at 2 px. That fits demosaic/JPEG-coloured PRNU, not broad scene edges.
- Caveat on the notes' argument: the "peak at zero shift" check alone would NOT rule out content. Aligned content also peaks at zero. The evidence is the probe above plus the same-model result.

**Other possible cues.**
- **Size:** all 14 devices are 3264×2448, so size cannot carry the label.
- **Orientation:** D33's portrait direction was chosen with the labelled device's fingerprint. Every test then tries 0° and 180° against every candidate, which covers both k = 1 and k = 3. The choice is neutral at test time.
- **Same-model devices are distinct:** the largest same-model false PCE is 38.3 (native arm) vs a true-device median of 1091. One physical device under two IDs would give PCE in the thousands.

**M1 for devices (missing from the run; gate 2 applies to every method).**
- I computed it (dev_checks.json "m1_dev"; 11 run.py statistics, leave-one-photo-out logistic regression, C = 1): 14-way top-1 0.346 (chance 0.071); within-model per set 0.657 / 0.643 / 0.643 / 0.857.
- PRNU's CI lower bound beats it by 0.62 (top-1) and by ≥ 0.14 (within-model). **Gate 2 holds for (ii).** The notes' line "No M1-type baseline gates (ii)" should be replaced by these numbers.

**Uncertainty is understated.**
- FPR 0 [0, 0] is a degenerate bootstrap.
  - There are 197×13 = 2,561 false pairs per arm, so the rule-of-three 95% upper bound is about 0.0012.
  - The largest false PCE is 49.9 (native) and 55.7 (nat→WA), just under the threshold of 60. "FPR 0" is fragile at this threshold.
- The within-model CI [1.000, 1.000] resamples photos. It covers only 9 devices in 4 sets.

## 2. Label vs the pre-registration
**Labels follow the rules.**
- PASS-ii needs native top-1 ≥ 0.70 and within-model ≥ 0.70. The run has 0.985 and 1.000.
- docs/08 and docs/09 decision 2: "PASS if (i) or (ii)". So H-A2 = PASS. All six gates hold (gate 2 for (ii) now shown above). This is not INVALID.

**(i) is not on the KILL side.**
- 0.522 sits in the INCONCLUSIVE band [0.30, 0.60), above its midpoint 0.45 and 0.08 below PASS. Its CI [0.478, 0.567] excludes the KILL threshold of 0.30.
- docs/09 decision 1 counts failures per hypothesis, and the hypothesis label is PASS.

**Stop-rule count from H-A2: 0.**

**Honesty note for README/LOG.** The PASS is carried entirely by the device arm. The generator question, which is the one tied to the project goal, got no positive evidence:
- (i) minus M1 = −0.008 [−0.072, 0.057];
- M0 raw-file metadata (0.674) beats the residual fingerprint (0.522).

Write it as "PASS via (ii) only; (i) uninformative", not as a plain PASS.

## 3. The unexplained 0.523 at jpeg75_resize05 (matched fingerprints): explained
**What a mean-residual template can use.** It only picks up patterns locked to the crop origin, which here is the image's top-left corner. Random content averages out.

**Test (gen_checks.json).** Fingerprints come from (0,0) training crops. Test crops are taken at offsets of the laundered image, using the same seed-0 family split.

| condition | crop offset (0,0) | (4,4) | (3,5) |
|---|---|---|---|
| clean | 0.522 (reproduces run) | 0.204 | 0.153 |
| jpeg75_resize05, matched | 0.523 (reproduces run) | 0.125 | 0.140 |
| resize(0.5) only, matched | 0.570 | 0.194 | 0.115 |

**Reading.**
- Both the clean 0.522 and the matched 0.523 come from a corner-anchored, phase-locked pattern. That can be a periodic generator or upsampler lattice, a JPEG-history grid, or decoder border artefacts.
- An exact ×0.5 downscale from the same origin keeps the pattern aligned (period 8 → 4, same phase). JPEG q75 only weakens it.
- Clean-train templates fail because the pattern is at the wrong scale; social_sim's random factor in [0.7, 1.0] misaligns it (0.211).
- Most of the signal comes from three generators (matched recall): Firefly 0.94 (JPEG-history grid), Flux 0.97 and Midjourney v5 0.75. Firefly's fingerprint has the largest share of energy on the 8-px lattice (0.023 vs ≤ 0.009 for the others).

**Consequences.**
- The prediction "near chance" was right for the realistic arm (clean fingerprints: 0.114).
- 0.523 is an oracle result: it needs the exact laundering and an untouched image origin.
- A shift of 3–5 px, as in any crop or screenshot, removes even the clean signal. This generator fingerprint is not usable in the wild without a shift and scale search.
- **Side note:** the jpeg(75)-alone arm of my check returned NaN fingerprints. A likely cause is a 0/0 in `wiener_denoise` on a constant crop [unverified]. The judged conditions are all finite, so this does not affect the run. Treat that arm as void.

## 4. Novelty and how notable the device result is
**Replication, not novelty.**
- The method is the Lukas (lukas-tifs2006) and Chen (chen-tifs2008) MLE fingerprint with PCE and the threshold of 60, on VISION (vision-eurasip2017).
- Device-level, same-model separation by PRNU is the classic claim of that literature. For VISION, the only kb datum is a non-peer-reviewed blog: no cross-device outliers well above 60 on VISION's 2010–2015 devices (vision-eurasip2017.md).

**What is mildly notable.**
- It works with 13–14 natural photos per device, below the 20–512 images in the literature (docs/07 part 3 §3). That contradicts our docs/07 part 3 §6.1 ("NOT feasible"), which should be corrected.
- Likely reasons:
  - full-resolution 8 MP native JPEGs from iPhone 4s–6 sensors (median true PCE 1,091);
  - VISION shot each photo hand-held, so content does not align.

**Platforms.** Through WhatsApp and Facebook copies, TPR at PCE 60 falls to 0.37 / 0.38, while top-1 stays at 0.93 / 0.87 because argmax needs no threshold. This degradation is known in the literature.

**Modern-phone caveat (decisive for the project goal).**
- Recent Samsung default-mode fingerprints have an FPR at PCE 60 of 0.66–1.0 (samsung-prnu-arxiv2025, Fig. 3b).
- Apple portrait mode adds a shared pattern (sdnp-tifs2026).
- Our phones are 2010s iPhones only. No result here transfers to neural-ISP phones.
- Scope: PRNU needs a reference set from the claimed device; it is not an AI-image detector and gives no generator attribution.

## 5. Requested edits (for the lead; I edited nothing)
1. notes.md / tables.md: add device M1 (0.346; within-model 0.64–0.86), the rule-of-three FPR bound (≤ 0.0012), and the largest false PCE (49.9 / 55.7).
2. notes.md: replace "unexplained" for the 0.523 with the crop-offset result in §3. Add that the clean generator signal is crop-fragile (0.522 → 0.15 at a (3,5) offset).
3. README/LOG wording: "H-A2 PASS via (ii) (PRNU replication on 2010s iPhones); (i) INCONCLUSIVE, ties M1, corner-locked; counts 0 failures."
