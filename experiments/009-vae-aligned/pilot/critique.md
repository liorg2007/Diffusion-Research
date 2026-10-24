# E2 PILOT critique (ideas #6 and #7), independent critic (opus/high), 2026-10-24

**Rating: SOUND. No bug or leakage found, all reported numbers reproduced, and both labels are correct (#6 INCONCLUSIVE, #7 FAIL).
Some readings need correcting, though. #6 A's 0.78–0.82 is not a format, zoom or texture shortcut. But under laundering,
A barely separates its own training pairs (0.59), so the laundered 0.78 comes from a correlated "not straight from a camera"
axis rather than from a VAE trace. That axis does not transfer to RAID's non-VISION domain (0.52). #7 works only for
generators whose decoder matches the bank, and only on un-resized images. C − B < 0 means that content-matched negatives
penalise cues B relies on. It is not a regularisation artefact.**

PILOT: none of this counts toward the stop rule. Every check below is post hoc (not pre-registered) and descriptive. They run on
CPU from saved outputs, except that the C2 control re-ran sd-vae-ft-mse on CPU for 80 images. Scripts and JSON outputs are in
`pilot/critic_checks/` (`c1`–`c6`). Shared helper: `common.py`, which imports `run.py` and restricts the bank to sd-vae-ft-mse.

## Reproduction
- A was refit on CPU from the cached features. Max |Δscore| vs `cache/scores6_orig.npz` = 5.9e-8, chosen C = 0.001 (`c1_A_shortcuts.json`).
  The A, B, C and C_all_reals per-generator AUROCs match `results.json` (`c3_C_vs_B.json`). The #7 macro AUROCs match (`c4_seven.json`).
- The SDXL guard2 Δlogit was reproduced with **CPU** DINOv2 features: clean 0.0148, laundered 0.1216 (`c5_sdxl_guard.json`). So it is
  not a feature-device artefact.

## Findings, ranked

### F1 (HIGH, reading of #6). The laundered 0.778 is not a VAE-trace detection; A rides on a correlated axis
- **The VAE trace is mostly gone under laundering.** Reals vs their own round-trips:
  - in-sample AUROC on the train pairs: 0.85 / 0.82 / 0.88 (native / FBH / WA) clean, but only **0.59** at `jpeg75_resize05`, in every pool (`c1`, `in_sample_train_auroc_per_pool`);
  - out-of-fold, device-grouped (`c2`): **0.645** clean, **0.546** laundered;
  - on 80 UNSEEN-device test natives vs their own fresh sd-vae round-trips (CPU, `c2_paired_controls.json`): **0.658** clean as PNG
    and 0.666 as JPEG at the source QF; 0.566 / 0.560 laundered.
  - On those same 80 natives, A scores the Synthbuster fakes at 0.805 clean / 0.803 laundered.
- **A separates generator fakes better than the very fakes it was trained to detect** (0.80 vs 0.66 clean, 0.80 vs 0.56 laundered,
  same natives). **Yet A still scores the test fakes at 0.778.** So under laundering, A's direction must separate test fakes from test natives by
  something other than the trace it was trained on. That something is a content/statistics difference between camera photos and
  generated images that happens to lie along the "native vs VAE(native)" axis in DINOv2 space.
- **Clean is similar, only weaker.** A uses heavy regularisation (C = 0.001) and gets 0.82 on test vs about 0.85 in-sample.
- **What this means for the claim.** "Generator-agnostic signal" is too strong. The fair claim is that A learns a
  "not-camera-native" direction. That direction transfers to Synthbuster within the VISION domain (other family 0.796 clean, 0.759 laundered).
- **Out-of-domain check (`c6_raid_reals.json`).** A on RAID originals (48 unpadded non-VISION reals vs 4×100 RAID-generator fakes, 256 crop):
  - pooled AUROC **0.52** (per generator 0.42–0.62);
  - B0 (A0, full-train refit) gets 0.64 on the same set;
  - caveats: tiny n, a 256 px / q75 domain, and the RAID generator mapping is [unverified].
  - **Positive side:** A flags only 2.1% of RAID reals at the VISION-native 1%-FPR threshold, vs **67%** for B0. A's real-side prior
    is far less camera-specific than B0's (median shift +0.19 vs +12.7 logits). This matches the E3 critic's finding that B0 calls
    77% of RAID reals fake.

### F2 (MEDIUM, Q1). Format, JPEG history and zoom are NOT what drives A
- **QF matched.** Estimated QF:
  - train natives 96 (88–100), test natives 95 (88–100);
  - Firefly test fakes are JPEG at QF 95 exactly; every other generator is PNG;
  - WA 92, FBH 71–91 (`c1`, `qf`).
  - Aligned fakes are re-saved at the source QF, so the format intent of Amendment 8 is met.
- **JPEG grid phase.** The blockiness phase was measured on the crops. The fraction with a crop-aligned 8×8 grid is similar for
  reals and their aligned fakes: native 41% vs 38%, WA 58% vs 51%, FBH 98% vs 100%. A's score does not depend on phase among test
  natives (median −0.124 aligned vs −0.151 not), and phase-aligned Firefly images score *lower* (`c1`, `jpeg_phase`, `A_by_phase77_*`).
  No grid shortcut.
- **Double JPEG.** The VAE wipes most of the first JPEG's traces, so an aligned fake is effectively "smooth content + one fresh JPEG
  at QF ~95". The only test fakes with a similar history are Firefly's (single JPEG at QF 95). Firefly is A's best non-SD generator:
  - orig arm: 0.913 clean;
  - fov512 arm, where the round-trips are PNG and Firefly is downsampled: 0.761;
  - Spearman(A, Laplacian) within Firefly is 0.44, the largest of any generator.
  - So part of Firefly's lead plausibly comes from the shared "fresh JPEG on smooth content" signature. This matters for interpretation
    (Firefly ≠ evidence of semantic generalisation), but it does not inflate the macro score much: dropping Firefly gives ~0.80 clean.
- **QF trend.** Spearman(A, QF) among test natives = −0.11 (weak).
- **Fresh-JPEG control on unseen-device natives (`c2_paired_controls.json`).** Re-saving each native crop as JPEG at its own QF
  (no VAE) moves A not at all: paired AUROC 0.502 clean, 0.500 laundered, median Δscore ≈ 0.
  - The VAE round-trip saved as PNG gives 0.658 and saved as JPEG 0.666. So the JPEG re-save adds nothing; the fresh JPEG is not a learned cue.
- **Zoom (`c1`, `zoom_cross_arm`).** The orig-trained A applied to fov512 features of the *same* images:
  - zoomed-out (and resampled) natives look slightly *more real* (AUROC 0.477), and so do fakes (0.410);
  - natives at fov512 vs fakes at orig gives 0.859, above the within-arm 0.819.
  - So "zoomed-out = fake" is not supported; the effect runs the other way.
  - Downsampling makes any image look more real to A. A resized fake therefore partly evades A, which is one route to the laundered drop.
  - Zoom and resampling are confounded here, as in E1.
- **Per pool.** A trained on natives only (99 vs 99) gives macro 0.831 clean / 0.769 laundered, about the same as the full A.
  WA-only gives 0.752 / 0.742 and FBH-only 0.763 / 0.710. So the compressed pools are not the source of the signal.
  - Test-split WA/FBH reals (unseen devices): macro AUROC vs fakes 0.815 / 0.768 clean.
  - At the native 1%-FPR threshold, 1.7% / 3.6% of them are flagged clean and 0.3% / 0.3% laundered. FBH shows mild FPR inflation.
- **Texture.** A's score is uncorrelated with the Laplacian among test natives (ρ = −0.05 clean, −0.17 laundered).
  - The within-quintile AUROC rises with texture (0.75 → 0.92), but the matched (0.797) and stratified (0.836) values stay near the overall 0.819.
  - The texture lesson from H4 does not undermine A in the orig arm.
- **fov512 arm.** A is lower there (0.778 clean / **0.681** laundered, i.e. FAIL territory) and was fit at C = 10, reaching an in-sample AUROC of 1.0.
  - The PNG round-trips of 4.8×-downsampled reals are trivially separable, so A overfits a pair-specific cue.
  - The orig label is the pre-registered one. But D-013 (fov512 as the primary arm) would turn #6 into a likely FAIL.
  - Report the arm dependence next to the label.

### F3 (MEDIUM, Q2). The SDXL guard failure is real but benign; the SD-only fallback is applied cleanly
- **What the pixel differences are.** GPU vs CPU x1 differ by ±1 uint8 in 17,873 of 25.1 M values, and by ±2 in 2 (`c5`).
  - The decomposition shows that JPEG quantisation amplifies these flips. Max |Δlogit| per condition: clean 0.015, resize 0.066,
    jpeg(75) **0.175**, jpeg75_resize05 0.122.
  - This is chaotic amplification of LSB flips by a discontinuous operator, not an accelerator defect.
  - Relative to B0's logit spread (std 7.9 on test), the max Δ is 0.2% (clean) to 2% (laundered) of one std. The #7 ratio Spearman is 0.9989.
- **The rule was applied correctly.** Under the pre-registered rule (Amendment 8 clarification 2), dropping SDXL was required.
  - Note: almost any ±1-LSB perturbation would fail a 1e-3 logit tolerance after JPEG q75. The downstream guard cannot pass for a VAE
    with any rounding difference.
- **Fallback is clean.** Only `sd-vae-ft-mse` exists in `cache/rt/{orig,fov512}`, in `cache/feats/*` (keys) and in `cache/r7/*`.
  - `n_al = 297` is asserted. The #7 "min over bank" equals the sd-vae column in every table row.
  - `set_bank` runs before vae/feat/analyze (`run.log` 04:53, 05:33).
- **Reporting nits:**
  - `set_bank` mutates `CFG["vaes"]` in place, so `results.json["config"]["vaes"]` lists only sd-vae, while `config.json` (pre-registration) has both.
  - `six_A` and `seven` in the saved config still say "594 = 297 × 2 VAEs" and "min over the 2 VAEs".
  - `notes.md` STATUS still says "NO result yet".
  - Fix in `notes.md`/README when merging. No number changes.
- **What the fallback costs:**
  - #6 lost half its aligned negatives, and with them its only non-SD1 decoder.
  - #7 lost the SDXL decoder: the SDXL ratio AUROC is 0.44 clean (`results.json` per generator), where the AEDR premise needs the generator's own AE.

### F4 (MEDIUM, Q3). FAIL is the right label for #7, and robust to the below-chance baseline
- **Why single-pass sits below chance.** L2 reconstruction error is a texture meter: Spearman(log e1, log Laplacian) = 0.85 clean,
  0.97 laundered (`c4`). Zoomed-in camera crops are smooth (median Laplacian 0.031 vs higher for fakes), so they reconstruct better.
  This is the H4 mechanism again.
  - Texture-residualised single-pass (log e1 regressed on log Laplacian over all rows, no labels) gives 0.578 clean, and within
    quintiles 0.610.
  - So criterion 1 (ratio ≥ single + 0.05) was passed only because of the texture inversion. Against the residualised single (0.578),
    the ratio (0.601) would miss +0.05.
- **The ratio is much less texture-bound** (ρ = 0.23 clean) and works for exactly two generators:
  - **SD2 0.977 and Midjourney v5 0.978**, with median ratio 1.40 / 1.44 vs 2.33 for natives;
  - DALL-E 2/3, Firefly and SDXL get 0.37–0.44 (`results.json` per generator; residualised: 0.41–0.51).
  - The SD2 result is the AEDR premise working (SD2 uses the SD kl-f8 AE).
  - The MJ-v5 result says its outputs lie close to the SD kl-f8 decoder manifold. That is an interesting [unverified] architectural
    hint; worth a KB line, not a claim.
- **The laundered failure is decisive and does not depend on the bank.** Laundered AUROC is 0.318, and SD2 itself, with its own AE in
  the bank, collapses to **0.186**. In the fov512 clean arm (downsampled 768→512), SD2 is 0.418 and MJ 0.442.
  - Any resize takes an image off the decoder manifold. Even a complete bank could not meet "≥ 0.65 laundered".
- **Is the comparison fair?** Mostly yes: ratio and single are both L2 at the same resolution with the same bank. The texture
  baseline is "strong" (0.778) only because of the dataset's real/fake texture confound. "Beats texture" is a high bar, but FAIL does
  not depend on it.
- **Validity gate.** It was correctly applied with the clarified rule: CI lower bound 0.540 > 0.5.

### F5 (MEDIUM, Q4). C − B < 0 comes from content-matched negatives, not regularisation
- CV picked C = 0.01 for C vs 10 for B (`run.log`). But that is not the cause:
  - B refit at C = 0.01 still gives 0.991 / 0.985;
  - C refit at C = 10 is *worse*: 0.874 / 0.767;
  - C at C = 10 with the aligned fakes down-weighted to 1/3 of the generator-fake weight gives 0.893 / 0.807 (`c3_C_vs_B.json`).
  - The aligned negatives actively conflict with B's decision direction.
- **Mechanism.** The aligned fakes have VISION *content*, and their native sources are b0_train reals in the same CV group. So the
  probe must stop using the content/style axes that separate VISION photos from Synthbuster prompts, which B uses heavily (B ≈ 0.99).
  - Evidence that B leans on such cues: B flags 67% of RAID reals (F1) and 77% per E3.
- **Laundering widens the gap** (−0.044 vs −0.020): C's aligned fakes include WA/FBH round-trips without the matching reals.
  - Adding those reals (C_all_reals) fixes the gap in fov512 (0.959 vs 0.926) but not in orig (0.932 vs 0.941). This part is not fully explained.
- **Reading.** Within this in-distribution benchmark, B's headroom comes partly from content cues. C − B < 0 here is not evidence
  against alignment for unseen generators; it is uninformative by the headroom rule, as labelled.

### F6 (LOW, Q5). Leakage, CV, bootstrap and guard order: OK
- **Train/test separation.**
  - Train reals come from 10 devices and test natives from 25 other devices (0 overlap).
  - Aligned fakes are made only from train reals. A never sees a generator image.
  - All 120 fakes per generator are used for A, which is legitimate because A never trains on them. For B/C, all of the held-out generator's fakes are excluded from training.
- **CV groups.**
  - `device:vidx` ties the native, WA and FBH copies and their round-trips together (`source_groups`, `al_group = tile(grp[tr])`).
  - C selection only; test devices are disjoint anyway.
- **Bootstrap.** Reals are resampled by device (25 clusters) and fakes by image, stratified per generator, in one paired plan.
  - The CIs are conditional on one training draw of 10 devices, so training-set variance is not in them. With 10 train devices, the true uncertainty of A is larger than [0.731, 0.817].
  - Synthbuster's six generators share prompts (one fake per prompt per generator), so the macro over generators is not six independent samples. This is minor.
- **Guards.** Order was timing → guard (pixel) → guard2 (downstream) → vae → feat → analyze. `vae` and `analyze` refuse to run without the
  guard, and the probe guard runs before `results.json` is written (`run.log`).
  - The probe-guard max |Δlogit| is 1.5e-4 (fov512 A); the orig items are ≤ 2.6e-5.
- **B reproduces 005 A0** to 1e-4 (orig) / 8e-4 (fov512).

## Recommendations (Q6): an official #6 test, all pre-registerable
1. **Held-out generator = FLUX** (once grip/Synthbuster-FLUX or another FLUX source is available within the 9 GB cap). Primary metric:
   A's FLUX AUROC at `jpeg75_resize05`, as in the idea file. Keep the 6 Synthbuster generators as secondary; report Firefly separately
   (shared-JPEG caveat, F2).
2. **Reals beyond VISION.** Add RAISE PNG reals (PNG → PNG round-trips) to *train* and, from disjoint images, to *test*.
   - A must be scored against non-VISION reals at a fixed FPR.
   - Add a pre-registered out-of-domain FPR check (the F1 RAID result shows the in-domain AUROC does not travel).
3. **More train reals and devices.** At least 30 devices, and ≥ 500 reals → ≥ 1,000 round-trips. The 10-device training set is the
   biggest unmodelled variance (F6).
   - Add a train-device bootstrap, or 5 device-disjoint refits, to the CI.
4. **SDXL VAE on CPU for the round-trips only.** About 11 s per pass, so 594 passes take ≈ 1.8 h per arm. The GPU guard issue is then avoided under the existing rule.
   - Do not relax D-008 post hoc. If the owner prefers GPU, pre-register a ranking-level guard for VAE outputs (Spearman ≥ 0.999 and
     max |Δlogit| ≤ 1% of the logit SD), stated before any result.
5. **Pair-trace diagnostic as a gate.** Report the out-of-fold, device-grouped AUROC of reals vs their own round-trips at each condition.
   - The pilot values are 0.645 clean and 0.546 laundered. If it is ≤ 0.65 at a condition, label A's test AUROC at that condition "not attributable to the VAE trace" (F1).
6. **Laundering-aware A.** Pre-register an arm that also trains on `jpeg75_resize05` copies of both pair members (JM-style). Today A is clean-trained and its pair signal dies under laundering.
7. **Arm.** Declare the label arm up front in light of D-013. The fov512 A is 0.681 laundered (F2), so the arm choice can flip the label.
8. **#7: do not re-test in its current form.** It is a clean, own-decoder-only, no-resize detector (F4). If kept at all, use it only as a
   "provenance hint" feature for SD-family decoders on un-resized images, and record the MJ-v5 observation as [unverified] in the KB.
9. **Hygiene.** Record the bank change in `results.json` (`bank`, `pre_registered_vaes`). Update the stale `notes.md` status and config strings (F3).
