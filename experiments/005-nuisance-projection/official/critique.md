# Critique: OFFICIAL H1 (laundering-nuisance projection), experiment 005

**Critic:** opus/high, independent. **Date:** 2026-10-25.
**Inputs read:**
- the pre-registration: `docs/04-phase4-plan.md` (H1 section, Amendment 1, Amendment 3, Amendment 4 and Amendment 13);
- the results: `official/{orig,fov512,orig_twin_disjoint,fov512_twin_disjoint}/{results.json,tables.md,run.log,cache/}`, `official/gate_robustness.md` and `official/launch.sh`;
- the code: `git diff ab63196 32767e6 -- run.py`, plus the full `run.py`;
- context: the pilot critique, the LOG tail and D-016.

**Checks** are in `official/critic_checks/`. They are CPU only, read-only towards `official/`, and any scratch cache went to the session scratchpad.
- `c1_integrity.py`: leakage, the twin mask, and a refit of the cached scores.
- `c2_caption_share.py`: a second caption-twin route, between fakes.

## Verdict

**FAIL stands and is the right reading.** No bug or deviation could change it.

Projecting out the real-only "laundering" subspace makes the probe worse, not better. Under laundering it loses AUROC on FLUX and on the LOGO macro. It loses clean AUROC too.

## 1. Bugs and deviations that could change the label

| Item | Finding | Could it change the label? |
|---|---|---|
| FLUX held out everywhere | FLUX has 150 test rows, 0 rows in `b0_train`, and 0 calib/test rows in `b0_train`. The 437 rows used for standardization and for the projector deltas (WA/FBH pairs plus the 9-grid) are all train reals. k\* uses only those reals, so FLUX cannot touch the k choice. Recomputed k\* = 182 equals the logged value. (c1) | No |
| k choice | No k reaches the +0.03 bar. At jpeg75_resize05 the best is k=16: macro +0.001, FLUX 0.909 vs A1 0.914 (−0.005). k=64 and k=256 are also ≤ 0 (orig/tables.md). | No |
| Killed first launch (pools including RAISE) and cache reuse | The orig cache files A0–A2c date from 09:28–10:01, from the first launch, before Amendment 13 (09:58) and before the twin code existed. Fitting never reads `PRIMARY_POOLS`: `run_arm` uses only `b0_train`, `TWIN_DROP` and the generator. The cache key matches (`opts.json`). Refitting from features reproduces the official cache to ≤ 5e-7 for A1/A2 `full` (the FLUX endpoint) and `logo_sd3_large` (c1). The first launch died before any evaluation: evaluation runs only after A4, and the A4 history is from 10:06, written by the second launch. So no official H1 metric existed before Amendment 13. | No |
| Twin-disjoint mask | It drops 217 of 435 train fakes, as intended by Amendment 13. Caveat: H1's primary reals are VISION natives, and the caption-twin leak concerns RAISE reals, so for the primary endpoint this row mostly measures "half the fakes" rather than "leak removed". It is not "more valid" here. The label is FAIL in both versions, so the Amendment 13 tie-break does not apply. | No |
| Second twin route (fakes ↔ fakes), not covered by Amendment 13 | 25–66% of each held-out generator's fakes share a caption id with a train fake of another generator. FLUX: 57%. Restricting each held-out set to unshared captions leaves the gain unchanged. orig: macro −0.014, FLUX −0.038. fov512: +0.004 / +0.007. Sharing lifts A1 by about 0.01, equally for both arms. (c2) | No |
| Cluster bootstrap | Reals are resampled by device (25 clusters, `bootstrap_plan`), and fakes by image within each generator. One plan is shared by all arms, so the differences are paired. The A2−A1 CIs are fully negative in orig: macro [−0.023, −0.007], FLUX [−0.045, −0.020]. In fov512 the CI upper bounds are 0.008 (macro) and 0.018 (FLUX), both below +0.03. | No |
| Label key | `results.json` stores the official label under `PILOT_verdict` (a legacy key name), and the tables header says "Not a Phase 4 verdict". This is cosmetic, but the report should quote `gate_robustness.md` and this critique, not the header. | No |
| Process gap | The killed first launch is not in LOG.md. Its run.log lines are gone: orig/run.log starts at 10:05 although `launch.sh` appends. Log it, with the pools it used. | No |

I also re-derived the primary numbers independently from the caches (c2, "all" row): macro −0.0149 and FLUX −0.0319 in orig, +0.0019 and +0.0053 in fov512. These equal `results.json`.

## 2. Is FAIL the right reading?

Yes, with two refinements for the report.
- **It is robust.** It holds in all 4 runs and for every single-generator drop (`gate_robustness.md`). The CIs exclude the +0.03 pass bar everywhere, and no k rescues it.
- **The one criterion met, drift −67% (orig), is not a win.**
  - A2 cuts the WA/FBH gap only by raising every FPR. At the threshold set for 5% FPR on the calib natives, the test-native FPR rises from 8.1% (A1) to 16.5% (A2); WA rises from 16.5% to 19.3% (orig/tables.md). So the threshold transfers worse, not better.
  - In fov512, A1's drift is already 0.0014. Platform drift was a zoom artifact that the field-of-view match removes.
- **What the result does say:** the subspace that real-photo laundering moves overlaps the subspace the fake signal lives in. This confirms the falsification risk the pilot named, now with headroom (A1 0.927 ≤ 0.96).
- **A2 does help with one thing.** It spreads false positives more evenly across phones: device D17 carries 8.4 of 19.9 FPs under A2, vs 19.6 of 25.1 under A0 (007 critic_checks/c3). But it buys that with AUROC.

## 3. What this means for the Phase-5 FPR ≤ 1% target

- H1 was an AUROC hypothesis. Its threshold evidence points the wrong way: a 5%-calibrated threshold lands at 8–17% FPR on test natives (orig).
- The projection does not provide a phone-independent operating point.
- The FPR question is handled in the 007 critique.

## 4. Plain-language takeaway for the owner

We tried to teach the detector to ignore the changes that WhatsApp, Facebook and resizing make to real photos. We measured those changes on real photos only, then deleted those directions from the detector's view.

It did not work. The directions that laundering changes are partly the same directions that tell fakes from reals, so deleting them hurt detection:
- about −1.5 AUROC points on average;
- about −3 points on the never-seen generator FLUX;
- worse on clean images too.

The result is solid. It holds with any number of deleted directions, with any one generator removed, on both image framings, and with the caption-twin fix. No code error or leak could have changed it.

The one "improvement", more consistent false-alarm rates across platforms, came from making false alarms worse everywhere. It also disappears once the images are framed fairly (fov512).

This idea is closed. The useful lesson is that "real-photo nuisance" and "fake signal" are entangled in DINOv2's final-layer features.
