# 018-real-anchor (E9, Amendment 18): per-pipeline real-anchor adaptation of the E7 detector

Exploratory (D-016). Constants in `config.json` (written before any result). Run: `env/.venv/bin/python -I experiments/018-real-anchor/run.py all --workers 14` (stages `probes`, `choose`, `proj`, `eval`, `report`; resumable via `cache/`, git-ignored). Self-test: `selftest.py` -> `results/selftest.txt` (ALL PASS). Every number below is in `results/results.json` / `results/tables.md`; per-row scores (draw 0, n = 50) in `results/scores_long.csv.gz`; n-curve in `results/ncurve.png`. Run log with timestamps: `results/run.log`. CPU only, no downloads, no new features (D-008 guard not applicable). Runtime about 75 min wall on 14 workers (probes 1 min, VISION choice 2.5 min, A-proj retraining 940 probe sets x 8 models about 70 min, eval + report under 1 min). OWNER photos: PENDING.

## Label (Amendment 18 rule; fov512, jpeg75_resize05, block 8, n = 50)
- **Chosen method: A-off** (chosen on the VISION fold only, frozen in `results/choice.json` at 07:59:13, before any RAISE/OI adapted result; the code asserts that no RAISE/OI adapted cache exists at that moment, and the report asserts every RAISE/OI adapted cache file is later than the freeze). VISION fold, n = 50: A-off AUROC 0.957, A-proj 0.946 (FPR 0.47% vs 0.46%).
- **Label for A-off: PASS** (AUROC VISION 0.957 / RAISE 0.985 / OI 0.951; FPR 0.5% / 0.0% / 1.6%; gain over A-thr at matched TPR with CI lower bound > 0 on VISION and OI, RAISE lower bound = 0.00, so 2 of 3).
- **This PASS must NOT be taken at face value.** It rests on the literal Amendment reading of A-off ("shift the held-out source's features"), which in the E7 protocol moves only the held-out source's REAL test images while all test fakes stay put. See "Critical caveat".
- A-proj (label-agnostic, projects every image): **KILL** (OI FPR 8.3% > 8%; RAISE AUROC 0.877, OI AUROC 0.893, both below the 0.90 PASS bar), all drop-one subsets except one INCONCLUSIVE are KILL.
- A-off-all (sensitivity: the same shift applied to reals AND fakes, i.e. what a deployed correction does to unlabeled images): **KILL** (no gain over A-thr on any source; AUROC identical to E7: 0.917 / 0.689 / 0.809).
- orig arm: A-off INCONCLUSIVE, A-proj KILL, A-off-all KILL.

## Design (Amendment 18 as implemented; interpretations marked I)
- Folds, probes, splits, roles, groups, T-aug, probe recipe, cluster-bootstrap Plan: E7 (`016-loso`) unchanged. The E7 probes were refit (8 models per arm x feature x held-out source, raw-space weights stored) and reproduce E7's cached scores to <= 2.5e-6; the report asserts that the E7 AUROC/FPR numbers are reproduced (fov512 F_b8 jpeg75_resize05: VISION 0.9167 / 0.6%, RAISE 0.6892 / 19.1%, OI 0.8085 / 16.0%).
- Anchors: held-out source's train + calib-role reals in its train pools (VISION natives 99 + 43, RAISE 84 + 53, OI 300 + 100), row- and group-disjoint from its test reals (self-test). Per (source, draw) one seeded permutation; n anchors = first n (nested over n = 10, 25, 50, 100); 20 draws; same rows in both arms. I: VISION anchors are natives only (as in E7 training); test reals are natives + WA + FBH.
- Offset (I): in raw feature space, mean over the three T-aug conditions of the per-condition mean; training reals = train-role reals of the two training sources. One offset is applied at every evaluation condition (sensitivity A-off-pc: offset per condition).
- A-off: S's test reals' features + (mean training reals - mean anchors), i.e. the score of every S real moves by w.delta (exact for the linear probe, self-test); fakes unchanged; threshold from the training sources' calib reals (unshifted).
- A-proj: unit vector of the standardized offset (standardization = training reals' T-aug mean/std, E8 convention) projected out of ALL features (training, calib, test, fakes), probe retrained with the E7 recipe (full CV for C), threshold from the training calib reals.
- A-thr: E7 probe unchanged, threshold = 1% FPR of the anchors' own scores (reading A: clean scores; n = 10..100 means "max of the anchors" or the second largest at n = 100).
- FPR: 'full' model, reading A (clean-calibration threshold applied at the evaluated condition, E7 headline); reading B (same-condition) in `tables.md`. AUROC: macro over 8 generators, LOGO model per generator, 'full' for FLUX (E7 convention).
- Gain over A-thr (I, "FPR at matched TPR"): per draw, A-thr's operating point gives (macro TPR, FPR); the method's FPR is read at the highest threshold whose macro TPR (full model, equal weight per generator) reaches A-thr's TPR; gain = FPR(A-thr) - FPR(method). Mean over draws; CI = cluster bootstrap (E7's 1,000-replicate plan, draw-mean per replicate). "No gain" (I) = no source with CI lower bound > 0.
- Method choice rule (I, not specified in the Amendment; written into `config.json` before the run): higher mean macro AUROC on the VISION fold at fov512 / F_b8 / jpeg75_resize05 / n = 50; if within 0.002, lower FPR.
- Drop-one-generator robustness: macros recomputed without generator g; FPR and gain from the LOGO model that never saw g (as E7).

## Results, primary arm (fov512, F_b8, jpeg75_resize05, n = 50; mean over 20 draws)
| held-out | E7 unadapted AUROC / FPR | A-thr AUROC / FPR | A-off AUROC / FPR | A-proj AUROC / FPR | A-off gain vs A-thr (pt FPR, CI) |
|---|---|---|---|---|---|
| VISION | 0.917 / 0.6% | 0.917 / 1.3% | 0.957 / 0.5% | 0.946 / 0.5% | +0.63 [+0.27, +1.19] |
| RAISE | 0.689 / 19.1% | 0.689 / 0.2% | 0.985 / 0.0% | 0.877 / 4.5% | +0.16 [+0.00, +0.47] |
| OI | 0.809 / 16.0% | 0.809 / 2.4% | 0.951 / 1.6% | 0.893 / 8.3% | +2.35 [+0.93, +3.98] |

Block 8 vs last layer (F_last, same setting): A-off AUROC 0.966 / 0.978 / 0.928, FPR 0.0% / 0.0% / 0.9%, label PASS under the same reals-only reading, `tables.md`; A-proj 0.941 / 0.728 / 0.847 with RAISE FPR 8.9% (KILL). E7 unadapted F_last: 0.885 / 0.600 / 0.758, FPR 0.3% / 10.4% / 4.5%. Block 8 is better than the last layer for A-proj (+0.005 / +0.149 / +0.046 AUROC), A-off is similar (last layer +0.009 on VISION, -0.007 on RAISE, -0.023 on OI).

## n-curve (fov512 F_b8 jpeg75_resize05; AUROC / FPR, mean over draws)
| held-out | method | n = 10 | n = 25 | n = 50 | n = 100 |
|---|---|---|---|---|---|
| VISION | A-thr | 0.917 / 7.3% | 0.917 / 2.7% | 0.917 / 1.3% | 0.917 / 1.2% |
| VISION | A-off | 0.958 / 0.4% | 0.958 / 0.5% | 0.957 / 0.5% | 0.958 / 0.5% |
| VISION | A-proj | 0.933 / 0.5% | 0.942 / 0.5% | 0.946 / 0.5% | 0.950 / 0.4% |
| RAISE | A-thr | 0.689 / 1.6% | 0.689 / 0.7% | 0.689 / 0.2% | 0.689 / 0.2% |
| RAISE | A-off | 0.984 / 0.0% | 0.984 / 0.0% | 0.985 / 0.0% | 0.986 / 0.0% |
| RAISE | A-proj | 0.797 / 10.1% | 0.845 / 6.0% | 0.877 / 4.5% | 0.894 / 3.7% |
| OI | A-thr | 0.809 / 13.2% | 0.809 / 5.7% | 0.809 / 2.4% | 0.809 / 2.3% |
| OI | A-off | 0.955 / 1.6% | 0.955 / 1.4% | 0.951 / 1.6% | 0.952 / 1.5% |
| OI | A-proj | 0.854 / 12.0% | 0.881 / 9.8% | 0.893 / 8.3% | 0.903 / 7.6% |

A-off is flat in n (already at n = 10); A-proj improves with n, and at n = 100 passes the AUROC bar on VISION and nearly on OI (0.903) but not the FPR bar on OI (7.6%) or RAISE (3.7%). A-thr's FPR is the usual 1/(n+1) behaviour of a threshold set by the maximum of n anchors, with a large spread over draws at small n; its TPR is very low (e.g. 0.064 on RAISE at n = 50).

## Held-out launderings (A-off, n = 50, F_b8, fov512; AUROC / FPR; E7 unadapted in `tables.md`)
- clean / social_sim / screenshot_sim / mavali_medium, VISION: 0.957 / 0.954 / 0.964 / 0.943 (FPR 0.2 / 0.2 / 0.3 / 0.3%); RAISE: 0.971 / 0.976 / 0.969 / 0.975 (FPR 0.0 / 0.0 / 0.3 / 0.0%); OI: 0.970 / 0.961 / 0.972 / 0.953 (FPR 0.0 / 0.6 / 0.0 / 2.0%). Unadapted E7 for comparison: VISION 0.920 / 0.911 / 0.930 / 0.894, RAISE 0.651 / 0.668 / 0.662 / 0.663, OI 0.845 / 0.807 / 0.847 / 0.794.
- FLUX (unseen generator) at jpeg75_resize05: A-off 0.942 / 0.982 / 0.915 vs E7 0.880 / 0.627 / 0.689.
- A-proj and the A-off-all variants for every condition: `results.json` (`results.fov512.F_b8.<S>.<method>.50.<cond>`).

## orig arm (reported; F_b8, jpeg75_resize05, n = 50)
A-off AUROC / FPR: VISION 0.970 / 0.1%, RAISE 0.986 / 0.0%, OI 0.965 / 3.1% (INCONCLUSIVE: OI FPR above 3%); A-proj 0.920 / 0.852 / 0.883 with FPR 0.3% / 4.0% / 11.3% (KILL); unadapted E7 0.851 / 0.624 / 0.753 with FPR 2.3% / 34.4% / 35.0%. Full tables in `tables.md`.

## Robustness (drop-one-generator, fov512 F_b8 n = 50, `tables.md`)
- A-off: PASS in 7 of 10 subsets (all, no FLUX, drop dalle3, firefly, midjourney-v5, SDXL, FLUX), INCONCLUSIVE in 3 (drop dalle2, SD2, SD3: max FPR 3.7-3.8% > 3%); never KILL; min AUROC 0.945-0.961. The RAISE gain lower bound is 0.00 in every subset, so the "2 of 3" count rests on VISION and OI.
- A-proj: KILL in 9 of 10 (INCONCLUSIVE when SDXL is dropped); max FPR 7.6-10.5%.

## Critical caveat (read before using the PASS)
1. **The literal A-off shifts only the real images.** Shifting the held-out reals by (train-real mean - anchor mean) while the fakes stay unshifted moves them toward the training reals and away from the (RAISE-content) fakes. The RAISE AUROC then reaches 0.985, higher than the in-source RAISE probe's AUROC (0.935, E8 critic c2 table), and it is independent of n (0.984 at n = 10). That is the signature of an over-correction/leak of "this image is a real from S", not of a better detector. In deployment the images arrive unlabeled and every image of the new pipeline gets the same correction: that is A-off-all, whose AUROC is unchanged (0.917 / 0.689 / 0.809), whose gain over A-thr is 0 by construction, and whose label is KILL. A-off-all only moves the operating point (FPR 0.5 / 0.0 / 1.6% but TPR 0.25 / 0.01 / 0.26, against 0.46 / 0.57 / 0.76 unadapted).
2. A-proj is the only variant that is label-agnostic and still changes the ranking (it retrains without the offset direction). It reproduces the critic's oracle direction of effect (RAISE 0.689 -> 0.877, OI 0.809 -> 0.893 at n = 50) but misses the pre-registered bars on FPR and AUROC; it needs n of about 100 to approach 0.90.
3. The gain-over-A-thr criterion is weak where A-thr is very conservative (A-thr FPR 0.2% at TPR 0.064 on RAISE leaves almost no FPR to gain: +0.16 pt, lower bound 0.00).
4. Test sets are small (RAISE 183 reals, OI 200: FPR resolution 0.5%), RAISE is content-confounded (E7 critique), 8 OI Flickr authors overlap train and test (E7 critique). Anchor draws are nested and not independent over n.
5. No phone photos yet: the owner's photos are the confirmatory test (4th pipeline, all choices frozen; use `apply_to_new_source.py`, which refuses to run if `config.json` changed after the freeze, and which uses the deployment convention, all images corrected).

## Deviations / interpretations to review
1. Method-choice metric not specified in Amendment 18 (I): VISION-fold mean AUROC at the criterion setting, tie -> FPR; frozen in `config.json` and `choice.json`.
2. A-off literal reading (reals only) used for the label as pre-registered; A-off-all and A-off-pc reported as sensitivities (not pre-registered).
3. Offset from the T-aug mean (one offset for all conditions); VISION anchors from natives only.
4. "Gain at matched TPR" and "no gain" operationalized as described above; A-thr threshold reading A (clean anchor scores).
5. The selftest synthetic cases use a custom generator: pure offset (partly along the diffusion direction) -> A-off 0.975 / A-proj 0.969 vs in-source 0.975 and unadapted 0.723; offset + 90 degree rotation of the real variability -> A-off 0.827 / A-proj 0.823 vs unadapted 0.644 and in-source 0.975 (partial).
6. `apply_to_new_source.py` was smoke-tested only on already-cached OI features (plumbing, outputs in the scratchpad, not stored); it was not run on anything new.

## Files
`config.json`, `run.py`, `selftest.py`, `apply_to_new_source.py`, `results/` (`choice.json`, `results.json`, `tables.md`, `scores_long.csv.gz`, `ncurve.png`, `selftest.txt`, `run.log`), `cache/` (git-ignored, 531 MB: probes, A-proj score tensors, per-method eval pickles).
