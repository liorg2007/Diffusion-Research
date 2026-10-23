# Diffusion-Generated Image Detection — Research Project

**Goal.** Find a new method, or a measurable improvement over existing ones, for detecting diffusion-generated images, measured on
**unseen generators** and **in-the-wild processing** (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count as progress.

**Acceptance targets for a full-scale validation:** FPR ≤ 1% on fresh phone photos; AUROC ≥ 0.95 after social-media laundering; evaluation against transfer attacks.

## Current status — 2026-10-23: Phase 4 in progress (pilot done; official run blocked by a data server outage)
| Phase | State |
|---|---|
| 0 Setup | ✅ Mac (D-001…D-004). GPU PC: WSL Ubuntu 22.04, uv-managed Python 3.12.15 (D-012), torch 2.8.0+cu128, RTX 4060 Ti |
| 1 Survey | ✅ done |
| 2 Gaps & hypotheses | ✅ done. Phase 4 plan in `docs/04-phase4-plan.md` (Amendments 1–4) |
| 3 Data & baselines | ✅ done on the Mac. ⏸ GPU-PC reproduction (SETUP_WSL step 6) waits for grip.unina.it |
| 4 Hypothesis tests | 🟡 PILOT done on VISION + 6 Synthbuster: H1 and H2 both "uninformative" per critics; not counted toward the stop rule. Official run code ready |
| 5 Final report | ⏸ not started |

### Blocker
- grip.unina.it (RAISE reals, SD3.5/FLUX fakes, B-Free weights) has been unreachable since 2026-10-23 ~09:40, from both WSL and Windows.
- When it returns: `env/.venv/bin/python -I env/bootstrap_data.py`, then SETUP_WSL step 6, then the official Phase 4 run (H4 first, D-013).

### Key pilot findings (pilot data only)
- **FOV confound** (D-013): real crops are ~2.5–3× more zoomed than fake crops. `fov512` removes the native-vs-WA/FBH gaps.
- **Default thresholds are unusable:** B0 flags 17% of real photos at logit 0. Real-photo calibration transfers across VISION devices (pooled 1% → ~1% LODO); WhatsApp is the weak spot.
- **H4 pilot** (`experiments/006-format-shortcut/pilot/`): the training-free RIGID-style drift score is below chance (AUROC 0.29) and mostly measures input texture, so the label is uninformative. The official-run fixes are in plan Amendment 6.

### Owner decisions pending
1. D-013: make `fov512` primary, or keep it as an extra arm.
2. ~250 photos from a 2021+ phone (H3), now the main FPR ≤ 1% test.
3. Larger backbone (DINOv2-L / DINOv3): exceeds the 9 GB cap (D-007).

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
