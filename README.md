# Diffusion-Generated Image Detection — Research Project

**Goal.** Find a new method, or a measurable improvement over existing ones, for detecting diffusion-generated images, measured on
**unseen generators** and **in-the-wild processing** (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count as progress.

**Acceptance targets for a full-scale validation:** FPR ≤ 1% on fresh phone photos; AUROC ≥ 0.95 after social-media laundering; evaluation against transfer attacks.

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

## Hardware (verified 2026-10-21)
Intel Core i9-9880H (8C), 32 GB RAM, AMD Radeon Pro 5500M 4 GB (Metal 3), macOS 15.6. **Not Apple Silicon.**
Consequence: last PyTorch x86_64 macOS wheel is 2.2.2 (Python ≤ 3.12); MPS on AMD is available but limited; plan for CPU-feasible experiments (frozen backbones + light heads, ≤5k images).

## Layout
- `LOG.md` append-only timestamped log · `DECISIONS.md` decisions + rationale
- `kb/` knowledge base (`kb/SCHEMA.md` defines entry formats, `kb/index.md` lists all entries)
- `docs/` explainers for the project owner · `experiments/NNN-name/` one folder per experiment
- `env/` environment + `env/SOURCES.md` provenance · `third_party/` pinned clones (git-ignored) · `data/` datasets (git-ignored)

## How to reproduce
```
cd env
/usr/local/bin/python3.12 -m venv .bootstrap && .bootstrap/bin/pip install uv==0.13.4
UV_PROJECT_ENVIRONMENT=.venv .bootstrap/bin/uv sync --frozen --python /usr/local/bin/python3.12
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```
