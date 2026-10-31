# Earlier README status blocks

Moved from README.md on 2026-10-31, unchanged.

## Current status — 2026-10-31: final run (D-020) reported (`docs/12-final-run-report.md`); no work in progress
- **Holds:** mid layers beat the last layer on all 5 backbones (+0.148 to +0.295 bacc; already published, Li et al. 2026). Closed-set attribution reaches 0.839-0.896 after jpeg75_resize05. The signal does not need image layout and is not a tone curve (after retraining).
- **Retracted:** transfer to unseen launderings (new mild blur or unsharp: about 0.52, exploratory); generic features beating detectors; the 0.92-0.94 data extrapolation.
- **Biggest open hole:** each generator class is one configuration from one operator, so we cannot separate generator from settings. No real photos in this run, so it does not bear on the FPR targets.
- **Owner open items:** modern-phone photos; D-013; future-research list in docs/12.

## Current status — 2026-10-28: attribution cycle (D-018) reported (`docs/10-attribution-report.md`)
- **Question:** can we fingerprint the source of an image (which generator; for real photos, which camera or platform)?
- **H-A1 PASS:** mid-level DINOv2 + linear probe names 1 of 8 known generators at 0.855 bacc after jpeg75_resize05 (chance 0.125). It holds 0.81–0.87 on unseen launderings and only works if trained on laundered images. Near-twins (SD1.3/1.4) are inseparable; open-set rejection fails.
- **H-A2 PASS** (device arm only; a replication): PRNU identifies 14 same-size 2010s phones at 0.985, and via WhatsApp/Facebook copies at 0.93 / 0.87. The generator residual arm is uninformative.
- **Fails:** H-A3 (platform history is only readable from file metadata), H-A4 (unseen-source novelty), H-A5 (published detectors' embeddings); counted 3/3, so the stop rule is reached (D-019).
- **H-A6 diagnostic:** the VAE decoder signature scores 0.97 on clean files and chance after one laundering.
- **Owner open items:** modern-phone photos (data/raw/owner_phone/, 150–250); limit raise for any next step in docs/10 §9.

## Current status — 2026-10-25: Phase 5 report written (`docs/03-final-report.md`)
| Phase | State |
|---|---|
| 0 Setup | ✅ Mac + GPU PC (D-012); Phase 3 reproduced on GPU (max \|ΔAUROC\| 0.0019) |
| 1 Survey | ✅ 47 papers in `kb/`; headline numbers PDF-checked |
| 2 Hypotheses | ✅ `docs/02-hypotheses.md`; diffusion-specific set `docs/06-diffusion-hypotheses.md` (D-014) |
| 3 Data & baselines | ✅ |
| 4 Hypothesis tests | ✅ Official: H4 REFUTED, H1 FAIL, H2-v2 NOT_SUPPORTED, so the stop rule was hit. Owner override D-016; exploratory E1–E7 |
| 5 Final report | ✅ `docs/03-final-report.md`: findings and a recommended (not validated) method |

### Bottom line
- None of the three Phase 5 targets is met.
- The real-vs-diffusion difference survives laundering (E5). But every detector partly learns what its training real-photo sources look like, so unseen real pipelines (polished photos) get 16–19% FPR (E7). This is the central obstacle.
- Details, numbers and sources are in `docs/03-final-report.md`.

### Open items for the owner
1. 150–250 photos from a 2021+ phone, plus WhatsApp copies, in `data/raw/owner_phone/` (git-ignored): the acceptance test for any detector (H3).
2. D-013: whether fov512 is primary. The report recommends fov512.

