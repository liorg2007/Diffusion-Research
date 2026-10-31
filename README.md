# Diffusion-Generated Image Detection: Research Project

**Can we tell diffusion-generated images from real photos, and identify where an image came from, when it has been compressed, resized or shared, and when its source is unfamiliar?**

**Short answer.** We found no new detection method, and the acceptance targets were not met. The diffusion signal is real and survives laundering, and known generators can be named at 84–90% accuracy (chance 12.5%). Reliable detection on unfamiliar real photos remains unsolved. The main value is a tested map of where these methods work and fail, a few possibly-new measurements, and specific experiments for future work.

| Where | What |
|---|---|
| [`findings/`](findings/README.md) | Results that may add something new, and claims we retracted |
| [`establishments/`](establishments/README.md) | Results that are solid but known, or replications |
| `docs/03`, `docs/10`, `docs/12` | The three full reports |
| `poc/fingerprint-demo/` | Fingerprint Lab, a local visual demo |

## Current status (2026-10-31)
No work is in progress. **Open owner items:** modern-phone photos (`data/raw/owner_phone/`, 150–250), the D-013 decision, and 8.8 GB of regenerable final-run caches (`env/STORAGE.md`). Earlier status blocks are in `docs/00-status-history.md`.

## The research in brief
All numbers are after JPEG-75 + half-size resize unless stated.

**1. Baselines (E1–E4, Phase 3–4).** We reproduced published detectors and built a controlled test set. Clean-image scores were high (Community Forensics AUROC 0.992), but no detector reached 0.95 laundered; our DINOv2 baseline was best at 0.875. The three planned hypotheses (H1, H2, H4) all failed.

**2. The statistical field (E5–E6).** Diffusion images do form a different population, even with content controlled, and a mid-level probe reaches about 0.95 on familiar real sources.

**3. The unfamiliar-source wall (E7–E10).** On real sources the detector never trained on, it fails. Removing source information, adapting with a few real photos, and adding more sources all fell short.

**4. Fingerprinting the source (D-018).** Naming known generators works; flagging new ones does not. Sensor noise identifies 2010s phones. Decoder traces exist only on untouched files.

**5. The final challenge (D-020).** We attacked our best claim. The measurements held, but much of the interpretation did not.

## Interesting points
**Possibly new** (none confirmed; details in [`findings/`](findings/README.md)):
- **Two equal penalties.** An unseen real source and an unseen generator each cost about 0.07 AUROC (0.926 with both seen, 0.767 with both unseen). [F1](findings/F1-two-equal-penalties.md)
- **Variety fixes false alarms, not separation.** Going from 2 to 5 real sources cuts mean FPR from 16.4% to 6.9%, but TPR at matched FPR gains only +0.074. [F2](findings/F2-diversity-recalibrates.md)
- **Polished and DSLR photos are the hard ones.** Unsplash and RAISE reach 14–17% FPR, older phone photos 0.5–1%. Modern phones are untested. [F3](findings/F3-hard-real-photos.md)
- **The generator signal needs neither layout nor tone**, once the classifier is retrained: 94% kept after tile shuffling, 90% after equalisation. A normally trained probe does not cope. [F4](findings/F4-signal-without-layout-or-tone.md)
- **A leak to watch for.** Adapting only to a new source's images scores near-perfect by leaking the label. We made this error and caught it. [F5](findings/F5-one-sided-adaptation-leak.md)

**Solid but known or replicated** (details in [`establishments/`](establishments/README.md)):
- The signal survives laundering, at 0.927–0.953 with content controlled. [S1](establishments/S1-signal-survives-laundering.md)
- No method passed on unseen real sources (FPR 0.5–17%). [S2](establishments/S2-unseen-source-limit.md)
- Naming 1 of 8 known generators reaches 0.839–0.896. [S3](establishments/S3-closed-set-attribution.md)
- Mid layers beat the last layer on all 5 backbones (+0.148 to +0.295). [S4](establishments/S4-mid-layers-beat-last.md)
- Limits: SD1.3 vs SD1.4 at chance, new-generator flagging 0.768, a new mild blur drops accuracy to about 0.52. [S5](establishments/S5-attribution-limits.md)
- Sensor noise names the phone at 0.985 on 2010s phones. [S6](establishments/S6-prnu-phone-fingerprint.md)
- Decoder traces score 0.967 on pristine files and chance after laundering. [S7](establishments/S7-decoder-trace-pristine-only.md)
- Metadata and threshold pitfalls can inflate results. [S8](establishments/S8-evaluation-pitfalls.md)

**Retracted along the way:** see [`findings/retracted.md`](findings/retracted.md).

## Acceptance targets
| Target | Status |
|---|---|
| FPR ≤ 1% on fresh phone photos | **Not tested** on fresh phones (owner photos pending); older phones reached 0.5–1% at the 1% threshold |
| AUROC ≥ 0.95 after laundering | **Not met** on unseen sources; about 0.95 only on familiar ones |
| Robustness to transfer attacks (RAID) | **Untested** in proper form; the pilot was uninformative |

## Limits of all of this
Small data (120–150 images per generator, 600 per new real source). No modern phones. Each generator class is one configuration from one operator, so we cannot separate generator from settings. Literature checks were partial. Several datasets and weights are non-commercial.

## What future research should do
1. **Generator vs settings:** 3–4 models × about 5 configurations, about 500 images each. Fails if holding out a configuration costs more than 20 points.
2. **At least 30 generators,** with error bars over generators.
3. **Real laundering chains,** whole families held out. Fails below 60%.
4. **Measure** 1,000 images per generator instead of extrapolating.
5. **Modern-phone photos** for the false-alarm target and PRNU.

More backbones or finer JPEG/resize grids on these 8 generators would be a waste. Details: `docs/12-final-run-report.md` §6.

## Layout
- `findings/`, `establishments/`: conclusions, each pointing to its evidence
- `docs/`: plans and reports · `experiments/NNN-name/`: one folder per experiment (config, code, results, critique)
- `kb/`: knowledge base (`kb/SCHEMA.md`, `kb/index.md`) · `poc/fingerprint-demo/`: visual demo
- `LOG.md`: append-only log · `DECISIONS.md`: decisions and rationale
- `env/`: environment, `SOURCES.md`, `STORAGE.md` · `third_party/`, `data/`: git-ignored, re-created with `env/bootstrap_data.py`

## Hardware
Experiments ran on a WSL2 PC with an RTX 4060 Ti 8 GB (D-012; see `env/SETUP_WSL.md`). Phase 1–3 work began on an Intel Mac.

## How to reproduce
```
cd env
/usr/local/bin/python3.12 -m venv .bootstrap && .bootstrap/bin/pip install uv==0.13.4
UV_PROJECT_ENVIRONMENT=.venv .bootstrap/bin/uv sync --frozen --python /usr/local/bin/python3.12
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```
