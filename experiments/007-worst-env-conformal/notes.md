# 007 worst-env-conformal (H2) -- PILOT

Status: PILOT (Amendment 2 of docs/04-phase4-plan.md). Partial data (VISION + 6 Synthbuster generators, 720 fakes; no RAISE/SD3.5/FLUX, B-Free weights unavailable). The verdicts below are pilot verdicts, not Phase 4 verdicts, and do not count toward the 3-failed-hypotheses stop rule. Pre-registered primary detector is B0 = arm A0 of 005; it is NOT yet analysed here (only the secondary pretrained detectors are).

## Files
- `score_pretrained.py` (Part A), `analyze.py` (Part B), `config.json`.
- `pilot/scores_pretrained.csv.gz` (19,980 rows = 3 detectors x 3 conditions x 2,220 crops; columns arm, model="pretrained", condition, sha256, score; score = raw logit, higher = more fake, same orientation as 003).
- `pilot/results_scores_pretrained.json`, `pilot/tables_scores_pretrained.md`, `pilot/score.log`. `cache/` (git-ignored) holds the resumable per-detector/condition npz scores, the CPU reference logits and `d008_guard.json`.

## Commands
```
env/.venv/bin/python -I experiments/007-worst-env-conformal/score_pretrained.py          # Part A (resumable; ~35 min total on the 4060 Ti, RINE dominates)
env/.venv/bin/python -I experiments/007-worst-env-conformal/analyze.py --scores experiments/007-worst-env-conformal/pilot/scores_pretrained.csv.gz
# later, on 005's file (arms A0/A1/A2, models full + logo_<g>):
env/.venv/bin/python -I experiments/007-worst-env-conformal/analyze.py --scores experiments/005-nuisance-projection/pilot/scores_long.csv.gz
```
The second run writes `pilot/results_scores_long.json` and `pilot/tables_scores_long.md` (tag = file stem). `--splits` and `--out-dir` are optional.

## Part A design
- Data: `splits_pilot.csv`, every row (2,220), crops at `df.path`. Conditions via `perturb.apply_condition(img, cond, key=sha256, seed=0)` exactly as 003's `load_cond`. Note 003's third condition is `social_sim`; here it is `screenshot_sim` as instructed.
- Scorer functions are imported from `experiments/002-baselines/scorers` unchanged (CommFor `score`, RINE `score` (non-tencrop, same as 003), ClipDet `score`), batch 20 per call (the scorers sub-batch at 8 internally, as in 003).
- D-008 guard: CPU reference (batch 4, `DD_DEVICE=cpu` subprocess) vs CUDA on a fixed 32-row subset (16 real + 16 fake, seed 0, drawn from split==test), per detector and condition; abort if max |dlogit| > 1e-3. TF32 disabled explicitly (`torch.backends.cuda.matmul.allow_tf32 = cudnn.allow_tf32 = False`). All 9 checks passed; maxima per detector/condition (source `cache/d008_guard.json`, also `pilot/score.log`):

  | detector | clean | jpeg75_resize05 | screenshot_sim |
  |---|---|---|---|
  | CommFor | 8.8e-05 | 9.0e-05 | 7.8e-05 |
  | RINE | 3.2e-05 | 5.1e-06 | 1.6e-05 |
  | ClipDet | 3.9e-05 | 1.3e-04 | 3.2e-05 |
  Overall max 1.31e-04 (ClipDet, jpeg75_resize05).

## Part B design and judgment calls
1. **Strict flagging.** A sample is flagged fake iff `score > threshold` (conformal convention, so the k-th smallest calibration score itself is not flagged). 003/lib use `>=`; with continuous logits the difference is only through ties (RINE has ties: M1 = M2 = 0.568 on all 25 devices).
2. **Reals.** Test = split=='test' reals of pools vision_native / vision_whatsapp / vision_facebook_hq (25 devices x 3 platforms, 1,074 images = 358 photos x 3 versions). Calib devices = split=='calib' (10 other devices, 129 images, only M1 uses them). "Device" groups are the `device` column pooled across the 3 platform versions.
3. **M3 degeneracy (important).** `np.quantile(..., 0.99, method="higher")` over a group of <= 100 images returns the group MAXIMUM. Each device has 42-45 calibration images (<= 3 x 15 in LOPO), so M3 here = max over all calibration reals (the overall max). It is therefore always >= M2 and the TPR ratio is <= 1 by construction; the "worst device quantile" and "pooled maximum" coincide at this sample size. This is exactly the pre-registered rule; I did not change it.
4. **LODO.** Per test device d: calibrate M2-M4 on the clean reals (3 platforms) of the other 24; test on d's clean reals; decisions pooled over the 25 folds (n = 1,074). M0 and M1 are fixed thresholds evaluated on the same decisions. CP 95% interval on pooled k/n; per-device max FPR reported with the device id. Supplementary (not used for any verdict): a photo-cluster bootstrap CI (2,000 resamples of the 358-per-device photos, seed 0), since the 3 versions of one photo are dependent and CP on 1,074 treats them as independent.
5. **LOPO.** native/WA/FBH: calibrate on the other two platforms' clean reals of all 25 devices, test on the held-out platform (n = 358). jpeg75_resize05/screenshot_sim: calibrate on all 1,074 clean reals, test on the 358 natives transformed in that condition (same perturb call, key = sha256). Caveat: the calibration data contain other versions of the SAME photos that are tested (VISION shares scenes across platforms), so held-out-platform FPR is optimistic w.r.t. scene shift; held-out-device LODO does not have this leak.
6. **TPR.** Thresholds are fit on all 25 test devices' clean reals (M1: the 10 calib devices) and applied unchanged to fakes under each condition (clean, jpeg75_resize05, screenshot_sim reported). Pretrained detectors: TPR over all 720 fakes (train + test split, since nothing is trained) plus per generator (120 each). Ratio = TPR(M3)/TPR(M2), clean and jpeg75_resize05; NaN if TPR(M2) = 0 (counts as not >= 0.7, and does not trigger the < 0.5 fail).
7. **Verdict logic** (`analyze.verdict`): FAIL if M3 LODO FPR > 2% or any ratio < 0.5 (checked first); else PASS if M3 LODO FPR <= 1.5% with CP upper <= 2.0% AND M3 LOPO FPR <= 3% on every one of the 5 held-out platforms AND both ratios >= 0.7 AND at least one of M0/M1/M2 has FPR > 2% on some split (pooled LODO or any LOPO platform); else INCONCLUSIVE. "Some split" excludes per-device FPRs.
8. **005-style files.** Per arm: FPR/LODO/LOPO use model `full` (the verdict uses these, with CP intervals); a LOGO-averaged version (mean of pooled FPR over the logo_<g> models, plus the max over models) is reported alongside and a secondary check (avg M3 LODO <= 1.5%, avg M3 LOPO <= 3%) is added to the verdict record, not to the verdict. TPR: for each g, model `logo_<g>` supplies BOTH the real-score thresholds and g's fake scores (every fake of generator g present in the file; 120 if train and test fakes are included, as for the pretrained scorers; the code does not filter by split); macro mean over g. Ratio = macro-mean TPR(M3) / macro-mean TPR(M2); per-generator ratios are also given. A0 is flagged `primary`. If 005's file uses a condition name other than `screenshot_sim`/`jpeg75_resize05`, that platform is skipped silently (LOPO) so check the table.
9. Arms with incomplete clean real scores (missing test or calib shas) are skipped with a printed message.

## PILOT results (pretrained detectors; alpha = 1%; n = 1,074 reals in LODO, 358 per LOPO platform)
Source: `pilot/results_scores_pretrained.json`, `pilot/tables_scores_pretrained.md`.

| detector | LODO FPR M0 / M1 / M2 / M3 (M4) | M3 LODO CP upper | worst LOPO FPR at M3 | TPR ratio M3/M2 clean / jpeg75_resize05 | verdict |
|---|---|---|---|---|---|
| CommFor | 2.23 / 1.49 / 1.12 / 0.37 % (0.74) | 0.95 % | 1.68 % (native) | 0.780 / 0.295 | PILOT FAIL (ratio < 0.5 on jpeg75_resize05) |
| RINE | 1.12 / 0.84 / 1.02 / 0.09 % (0.65) | 0.52 % | 0.84 % (native) | 0.871 / 0.522 | PILOT INCONCLUSIVE (jpeg ratio 0.52 < 0.7) |
| ClipDet | 6.42 / 2.51 / 1.12 / 0.09 % (0.56) | 0.52 % | 3.35 % (jpeg75_resize05) | 0.237 / 0.219 | PILOT FAIL (ratios < 0.5; LOPO > 3%) |

- M3 does what it promises on FPR (all three detectors pass the LODO criterion, 0.1-0.4% pooled), but the price is large: at the pooled-max threshold, jpeg75_resize05 TPR falls to 12.9% (CommFor), 21.2% (RINE), 5.7% (ClipDet) from 43.8 / 40.7 / 26.0% at M2. On clean fakes CommFor loses 92.6 -> 72.2%, RINE 61.5 -> 53.6%, ClipDet 16.4 -> 3.9%.
- M2 (plain conformal) already holds LODO FPR at ~1.0-1.1% for all three, i.e. the "standard rules exceed 2% somewhere" clause is met mainly by M0/M1 and by LOPO-native (RINE M2 3.35%, ClipDet M2 2.51%), plus M2 under synthetic platforms for ClipDet (jpeg 27.1%, screenshot 6.4%) and CommFor (screenshot 2.2%).
- Per-device tails are heavy for the standard rules (e.g. CommFor device D30: 21.4% at M2, 9.5% at M3; ClipDet D21: 9.5% at M2).
- The pretrained detectors have very low TPR at 1% FPR on these fakes (RINE 61.5%, ClipDet 16.4% at M2), so the TPR ratios are noisy for weak detectors.

### Sanity check (CommFor, M0 = logit 0, clean)
VISION test natives (25 devices): FPR = 15/358 = **4.19%** (CP 95% CI per `results` json); 003 reports 4.2% on its full test set. WhatsApp 4/358 = 1.12%, Facebook-HQ 5/358 = 1.40%. Consistent.

## Caveats
- Pilot data only; real pool = VISION only, so "unseen phone" means unseen VISION device. Fakes are the 6 Synthbuster generators.
- Pooled CP intervals ignore the dependence between the 3 versions of one photo; see the cluster-bootstrap column.
- Pretrained detectors were not trained on these fakes; TPR numbers are not comparable with probe LOGO TPR in 005.
- Primary detector (005 arm A0 = B0) has not been analysed; run the 005 command above once `scores_long.csv.gz` exists.
