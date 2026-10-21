# Diffusion-Generated Image Detection — Research Project

**Goal.** Find a new method, or a measurable improvement over existing ones, for detecting diffusion-generated images, measured on
**unseen generators** and **in-the-wild processing** (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count as progress.

**Acceptance targets for a full-scale validation:** FPR ≤ 1% on fresh phone photos; AUROC ≥ 0.95 after social-media laundering; evaluation against transfer attacks.

## Status
| Phase | State |
|---|---|
| 0 Setup | in progress |
| 1 Survey | in progress |
| 2 Gaps & hypotheses | pending |
| 3 Data & baselines | pending |
| 4 Hypothesis tests | pending |
| 5 Final method & report | pending |

## Hardware (verified 2026-10-21)
Intel Core i9-9880H (8C), 32 GB RAM, AMD Radeon Pro 5500M 4 GB (Metal 3), macOS 15.6. **Not Apple Silicon.**
Consequence: last PyTorch x86_64 macOS wheel is 2.2.2 (Python ≤ 3.12); MPS on AMD is available but limited; plan for CPU-feasible experiments (frozen backbones + light heads, ≤5k images).

## Layout
- `LOG.md` append-only timestamped log · `DECISIONS.md` decisions + rationale
- `kb/` knowledge base (`kb/SCHEMA.md` defines entry formats, `kb/index.md` lists all entries)
- `docs/` explainers for the project owner · `experiments/NNN-name/` one folder per experiment
- `env/` environment + `env/SOURCES.md` provenance · `third_party/` pinned clones (git-ignored) · `data/` datasets (git-ignored)

## How to reproduce
(filled in after Phase 0 env is built)
