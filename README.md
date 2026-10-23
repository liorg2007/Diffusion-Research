# Diffusion-Generated Image Detection — Research Project

**Goal.** Find a new method, or a measurable improvement over existing ones, for detecting diffusion-generated images, measured on
**unseen generators** and **in-the-wild processing** (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count as progress.

**Acceptance targets for a full-scale validation:** FPR ≤ 1% on fresh phone photos; AUROC ≥ 0.95 after social-media laundering; evaluation against transfer attacks.

## Current status — 2026-10-23: Phase 3 complete, waiting for the owner's go-ahead for Phase 4
| Phase | State |
|---|---|
| 0 Setup | ✅ done — Python 3.12 / torch 2.2.2 / MPS verified (D-001, D-003, D-004) |
| 1 Survey | ✅ done — 34 papers + 8 datasets in kb/, `docs/01-survey-explainer.md`, fact-check `kb/verification-log.md` |
| 2 Gaps & hypotheses | ✅ done — `docs/02-hypotheses.md`, 7 ideas in `kb/ideas/` |
| 3 Data & baselines | ✅ done — eval set (`experiments/001-eval-set`), 4 baselines (`experiments/002-baselines`), harness (`experiments/lib`, 26 tests), baseline table `experiments/003-baselines-eval/{tables.md,results.json,notes.md}`; D-010: B0 runs on CPU |
| 4 Hypothesis tests | ⏸ **gated**: owner must approve after seeing the baseline table |
| 5 Final report | ⏸ not started |

### Next steps
1. The owner reviews `experiments/003-baselines-eval/tables.md` + `notes.md` and approves Phase 4.
2. Open questions for the owner: (a) ~250 photos from a 2021+ phone (neural-isp-fpr-audit); (b) DINOv3 gated access (D-006); (c) optional novelty-check + KB clean-up agents.
3. Phase 4: reconcile the Phase 2 plan with the actual data (DINOv2, 6 Synthbuster + SD3.5 + FLUX). Test laundering-nuisance-projection and drift-format-shortcut against B0, with an opus/high critic per result.

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
