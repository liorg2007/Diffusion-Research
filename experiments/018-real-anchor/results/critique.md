# E9 (018-real-anchor) independent critique

Critic: opus/high, CPU only, no downloads. The only edits are this file and `results/critic_checks/`.
Inputs read: Amendment 18 and its addendum (`docs/04-phase4-plan.md`), D-016/D-017 (`DECISIONS.md`), `run.py`, `config.json`, `selftest.py`, `apply_to_new_source.py`, `notes.md`, `results/*`, and the E8 critique with its `c5_fewshot*` files.

| Check | Files (`results/critic_checks/`) | What it does |
|---|---|---|
| c1 | `c1_leak.{py,out,json}` | Leak controls for A-off (fov512, F_b8, jpeg75_resize05, n = 50, the 20 E9 draws, LOGO macro) |
| c2 | `c2_c5_logo.{py,out,json}` | The E8 critic's c5 recipe re-run under the E7/E9 convention (LOGO macro), 7 anchor sets per source |
| c3 | `c3_aproj.{py,out,json}` | Paired bootstrap of A-proj vs E7; A-proj with an anchor threshold; full vs LOGO macro; scan of all 260 label-agnostic cells in `results.json` against the DISCOVERY bar |

**Sanity anchors.**
- c1 reproduces E9's A-off AUROCs to every printed digit: VISION 0.9566, RAISE 0.9854, OI 0.9514. It also reproduces E7: 0.9167 / 0.6892 / 0.8085.
- c2 reproduces the E8 critic's c5 full-model numbers exactly: RAISE 0.9328 / 0.9185, OI 0.9247 / 0.9292.
- `config.json` sha256 `f31b4e729eb3` matches the freeze recorded in `choice.json`.
- No code bug found. The problem is in the design, not the code.

## Q1. Is the literal A-off PASS a label leak? **Yes. Confirmed.**

**Mechanism (code).**
- `run.py` `eval_task`, branch `A-off`: `sc[:, :, ctx.rr[S]] += shift`. Only the held-out source's test **reals** are shifted. The fakes (`ctx.ff_all`) are not.
- The probe is linear, so the anchors enter only through one scalar per model, `shift = W @ delta`.
- A-off is therefore exactly "subtract a constant from every real's score and leave every fake alone".
- For any positive constant this raises AUROC, with or without anchors. The pipeline information in the anchors decides only how large the constant is.

**Evidence (c1; LOGO macro, mean over 20 draws [2.5, 97.5 percentile]).**

| Held-out | E7 | A-off (reals only) | A-off-all (every image) | Same shift on fakes only | "Anchors" = 50 random train **fakes** | "Anchors" = 50 training-source reals |
|---|---|---|---|---|---|---|
| VISION | 0.917 | 0.957 [0.945, 0.967] | **0.917** | 0.957 | **0.9996** | 0.914 |
| RAISE | 0.689 | 0.985 [0.979, 0.990] | **0.689** | 0.985 | **0.999** | 0.691 |
| OI | 0.809 | 0.951 [0.940, 0.961] | **0.809** | 0.951 | **0.998** | 0.806 |

- **Anchors made of fakes "work" better than real anchors.** They contain no information about the new pipeline. This control alone shows the gain comes from knowing which test images to shift (the label), not from what the anchors contain.
- **Applying the same shift to every image** gives exactly E7's AUROC. Moving it onto the fakes instead of the reals gives exactly A-off's AUROC. Only the relative real-vs-fake move matters.
- **Size of the RAISE shift.** It is 2.6 SD of the real scores (full model: −5.95), larger than the whole gap between the fake and real score means (2.46). In effect, A-off moves the RAISE reals past the fakes.
- **No anchors at all.** Subtracting a constant c (in real-score SDs) from the reals only gives:

| c (real-score SDs) | 0 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| RAISE AUROC | 0.689 | 0.796 | 0.877 | 0.932 | 0.966 | 0.993 |

  The OI and VISION curves are just as monotonic.
- **Consistent with the lead's evidence.**
  - A-off is flat in n (RAISE 0.984 at n = 10).
  - RAISE 0.985 is above the in-source RAISE probe (0.935, E8 critic).

**Was the pre-registration ambiguous?** Yes, and it was flawed as well as ambiguous.
- Amendment 18 says "shift the held-out source's features". In a leave-one-real-source-out fold, "source" is defined only for reals. The fakes belong to generators and have no source.
- So "the source's images" silently meant "the reals", and the conditioning variable is the label itself.
- **Why "only images from pipeline S get corrected" cannot be deployed.** If the pipeline is "photos taken by camera/phone X", then being from X already means being real. A system that knew this would have provenance and need no detector.
- **The only deployable reading** is "every image arriving through the channel gets the same correction" (A-off-all). With a linear probe and a constant shift, that is a pure threshold move: AUROC is unchanged by construction.
- **This was knowable before any held-out result.** Before the freeze:
  - `config.json` already described A-off-all as "AUROC unchanged, only the operating point moves";
  - `selftest.py` (lines 133-136) checks exactly that;
  - the synthetic self-test (line 102) validated A-off with the reals-only shift, so the self-test certified the leak instead of catching it.
- **The leak also decided the method choice.** On the VISION fold, A-off beat A-proj 0.957 vs 0.946 (`choice.json`) only because of the leak. Under the deployable reading, A-off-all (0.917) would have lost to A-proj (0.946), and A-proj's label is KILL.

**Correct labels for E9.**
- A-off (literal): PASS, **INVALID** (label leak).
- A-off-all (deployable): **KILL**. It shows no gain over A-thr on any source and only trades TPR for FPR (TPR 0.25 / 0.01 / 0.26 vs 0.46 / 0.57 / 0.76 unadapted, `tables.md`).
- Any threshold-only method sits on E7's ROC. Even with an oracle threshold set on the test reals, the full-model macro TPR is only 0.19 on RAISE and 0.23 on OI at 1% FPR, and 0.30 and 0.42 at 5% FPR (c1).
- The pre-registered E9 outcome is **KILL**. I agree with the lead.

## Q2. Did the E8 critic's c5 few-shot check have the same leak? **No. It has a different inflation.**

- **No leak.** c5 projects the U_k + anchor direction out of **every** row (`R.project_out(..., Uo)` on all of `A.X`, reals and fakes) and then retrains the probe. That is label-agnostic, essentially A-proj plus E8's subspace.
- **But the numbers were full-model macros.** That model saw 7 of the 8 test generators in training, so it is not the E7/E9 LOGO convention.
- **The quoted lifts were not few-shot.** "RAISE 0.764→0.936, OI 0.856→0.933" are c4 oracle numbers, using all of the target's train-role reals. Both ends are full-model.

**Like-for-like recompute (c2 and c3; jpeg75_resize05, n = 50).**

| | RAISE | OI |
|---|---|---|
| E7 baseline, LOGO (full-model) | 0.689 (0.764) | 0.809 (0.856) |
| c5 recipe, full model (as the E8 critic reported) | 0.912–0.933 | 0.908–0.929 |
| c5 recipe, **LOGO**, 7 anchor sets | **0.864–0.894** (mean ≈ 0.879) | **0.874–0.898** (mean ≈ 0.886) |
| E9 A-proj, LOGO / full, 20 draws | 0.877 / 0.922 | 0.893 / 0.922 |
| c5 recipe, FPR (calib threshold) | 2.7–7.7% | 7.0–14.0% |

- The full-model convention inflates the result by about 0.045 AUROC.
- **Corrected E8 F5 statement.** With 50 anchors, the label-agnostic lift is RAISE 0.69 → ≈0.88 and OI 0.81 → ≈0.89 (LOGO), not "≈0.93".
- Adding the E8 subspace gives nothing over plain A-proj.
- The E8 critique's caveat line did say "full model", but its headline table and its verdict ("AUROC about 0.93") overstate the result. The real-anchor artifact page should cite the LOGO numbers.

## Q3. Is A-proj's gain real, and does any label-agnostic variant meet DISCOVERY? Real but too small. **No variant meets the pre-specified bar.**

**Paired cluster bootstrap, A-proj − E7** (c3; same Plan replicates; fov512, F_b8, jpeg75_resize05):

| Source | n | A-proj AUROC [95% CI] | Gain [95% CI] | Share of replicates ≥ 0.90 |
|---|---|---|---|---|
| VISION | 50 | 0.946 [0.927, 0.961] | +0.029 [+0.022, +0.036] | 100% |
| RAISE | 50 | 0.877 [0.852, 0.899] | +0.188 [+0.158, +0.219] | 1.6% |
| OI | 50 | 0.893 [0.867, 0.916] | +0.084 [+0.069, +0.101] | 29% |
| RAISE | 100 | 0.894 [0.871, 0.915] | +0.205 [+0.174, +0.240] | 31% |
| OI | 100 | 0.903 [0.878, 0.926] | +0.094 [+0.077, +0.113] | 63% |

- The CIs cover cluster resampling only, not anchor-draw variation (they use the draw-mean). Draw spread is in `tables.md`.
- The gain over unadapted E7 is clearly non-zero on all three sources.
- At n = 50, the RAISE CI lies wholly below 0.90.
- A-proj's FPR at the calibration threshold is 4.5% [2.3, 7.1] on RAISE and 8.3% [5.3, 11.7] on OI (`tables.md`). The OI CI is wholly above the 5% DISCOVERY bar.

**Label-agnostic variants that use saved data only.**
- **A-proj ranking with an anchor-set threshold** (deployable, c3, n = 50): mean FPR 4.5% on RAISE and 3.4% on OI, but the 97.5th percentile over draws is 15.7% and 9.0%. TPR is 0.62 and 0.50.
  - This lowers OI's mean FPR, but AUROC still fails on RAISE and OI, and the threshold is unstable at n = 50.
- **Scan of all 260 label-agnostic cells** in `results.json` (E7, A-thr, A-off-all, A-proj × arm × feature × n × condition):
  - **None** meets AUROC ≥ 0.90 and FPR ≤ 5% on all three sources at jpeg75_resize05.
  - The best is fov512 / F_b8 / A-proj / n = 100: min AUROC 0.894, max FPR 7.6%.
- **The single cell that meets the numbers** is orig / F_b8 / A-proj / n = 100 / **clean** (min AUROC 0.938, max FPR 4.0%). It is off-protocol on three counts:
  - the arm is not fov512;
  - n is not 50;
  - the condition is clean, and laundered results are the ones that count.
  
  It is one cell out of 260, a multiple-comparison artefact. Reported only. Not a DISCOVERY.
- **Pre-registered DISCOVERY label (frozen A-off): void**, because A-off is invalid.
- **Re-applied to A-proj** (not pre-registered as the DISCOVERY method): **fails** on RAISE AUROC 0.877 and on OI AUROC 0.893 with FPR 8.3%.

## Q4. Verdict for the owner (plain language)

- **What E9 tried.** E9 tested a simple idea: give the detector about 50 real photos from a new camera or website, and let it correct for that pipeline's "look".
- **Why the reported PASS is invalid.** The version that "passed" corrected only the real test photos and never the AI images. That is like grading a test after secretly telling the grader which answers are right. Even a "correction" computed from AI images instead of real photos "works" (AUROC 0.999), which proves the score came from the answer key, not from the method.
- **What happens in real use.** You do not know which incoming images are real, so every image gets the same correction. That only moves the alarm threshold. The detector cannot rank images any better than before (E7: 0.92 / 0.69 / 0.81).
- **The honest variant.** The variant that does not peek, A-proj, really helps: +0.19 AUROC on RAISE and +0.08 on OI. It still falls short: AUROC about 0.88–0.89 and 4–8% false alarms at 50 photos, and it does not reach the relaxed DISCOVERY bar (≥ 0.90 and ≤ 5%) on every source even at 100 photos.
- **The E8 headline.** The earlier "fixes most of the problem, AUROC about 0.93" was not a leak, but it used a lenient scoring convention. Like for like, it is about 0.88.
- **Recommendation: close the real-anchor line.**
  - Record E9 as KILL, with the A-off PASS marked invalid (label leak in the pre-registration).
  - Correct E8 F5 and the artifact page to the LOGO numbers.
  - Change `apply_to_new_source.py`'s frozen choice: A-off is threshold-only in deployment. If the owner's phone photos arrive, the one honest confirmatory test is A-proj with an anchor threshold, run once, pre-registered as exploratory.
  - Do not raise the limit again for this line.
- **Lesson for future pre-registrations.** Any per-source correction must be stated as applied to every test image of a channel. It should be checked with a "fake anchors" control before results are seen.
