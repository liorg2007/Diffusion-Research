# Diffusion-Generated Image Detection — Research Project

**Goal.** Find a new method, or a measurable improvement over existing ones, for detecting diffusion-generated images, measured on
**unseen generators** and **in-the-wild processing** (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count as progress.

**Acceptance targets for a full-scale validation:** FPR ≤ 1% on fresh phone photos; AUROC ≥ 0.95 after social-media laundering; evaluation against transfer attacks.

## Current status — 2026-10-24: Phase 4 pilots done; official run blocked by a data server outage
| Phase | State |
|---|---|
| 0 Setup | ✅ Mac (D-001…D-004). GPU PC: WSL Ubuntu 22.04, uv-managed Python 3.12.15 (D-012), torch 2.8.0+cu128, RTX 4060 Ti |
| 1 Survey | ✅ done; headline numbers re-checked against PDFs (`kb/verification-pass-2026-10-24.md`, 17/17) |
| 2 Gaps & hypotheses | ✅ done. Phase 4 plan in `docs/04-phase4-plan.md` (Amendments 1–10 + clarifications) |
| 3 Data & baselines | ✅ done on the Mac. ⏸ GPU-PC reproduction (SETUP_WSL step 6) waits for grip.unina.it |
| 4 Hypothesis tests | 🟡 PILOTS done on VISION + 6 Synthbuster (+ a RAID subset), each with an opus/high critic. None counts toward the stop rule. Official H4/H1/H2 code is ready and reviewed |
| 5 Final report | ⏸ not started |

### Blocker
- grip.unina.it (RAISE reals, SD3.5/FLUX fakes, B-Free weights) has been unreachable since 2026-10-23 ~09:40. There is no other official host.
- When it returns, run `env/.venv/bin/python -I env/bootstrap_data.py`, then SETUP_WSL step 6, then the official sequence in `experiments/006-format-shortcut/notes.md`.

### Pilot findings (pilot data only; summary in `docs/05-phase4-pilot-synthesis.md`)
| Pilot | Label | Main reading (critic) |
|---|---|---|
| H1 nuisance projection (005) | uninformative | no headroom: baseline already 0.988 |
| H2 worst-env threshold (007) | uninformative | ~43 reals per device; plain real-photo calibration already ~1% across VISION devices |
| H4 drift format shortcut (006) | uninformative | RIGID-style score is below chance (0.29) and tracks texture |
| E1 zoom sensitivity (008) | exploratory | published detectors move up to 0.23 AUROC with crop vs resize-first; FPR inflates ≥5× when the input scale changes |
| E1b zoom vs resampling (008/pilot_du2) | confounded | du2 adds differential blur; only "orig-trained B0 is zoom-fragile" survives |
| E2 #6 aligned-VAE probe (009) | inconclusive | 0.82 clean / 0.78 laundered on 6 unseen generators without seeing any, but not via the VAE trace |
| E2 #7 VAE ratio (009) | fail | loses to a texture baseline; dropped |
| E3 #5 RAID canonicalization (010) | uninformative | B0 fails the gate on RAID (0.64); JPEG+resize cancels attacks at equal clean cost |

### Owner decisions pending
1. D-013: make `fov512` primary for our probes, or keep it as an extra arm. E1/E1b evidence favours primary.
2. ~250 photos from a 2021+ phone (H3), the main FPR ≤ 1% test.
3. Over the 9 GB cap (≈8.8 GB used): a larger backbone (DINOv2-L / DINOv3), and/or ~0.9 GB of RAID originals for a proper idea-#5 test.

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
