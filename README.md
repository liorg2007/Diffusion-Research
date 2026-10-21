# Diffusion-Generated Image Detection — Research Project

**Goal.** Find a new method, or a measurable improvement over existing ones, for detecting diffusion-generated images, measured on
**unseen generators** and **in-the-wild processing** (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count as progress.

**Acceptance targets for a full-scale validation:** FPR ≤ 1% on fresh phone photos; AUROC ≥ 0.95 after social-media laundering; evaluation against transfer attacks.

## Current status — PAUSED 2026-10-21 (checkpoint commit "checkpoint: paused")
| Phase | State |
|---|---|
| 0 Setup | ✅ done — env built, MPS verified (LOG.md, DECISIONS D-001, D-003, D-004) |
| 1 Survey | 🟡 partial — 5/5 literature agents finished (34 paper + 8 dataset entries, index merged). **Not done:** opus/high survey explainer `docs/01-survey-explainer.md` (never started) |
| 2 Gaps & hypotheses | ⏸ not started (was to be launched in background before presenting the survey) |
| 3 Data & baselines | ⏸ not started |
| 4 Hypothesis tests | ⏸ not started |
| 5 Final method & report | ⏸ not started |

**Interrupted / partial work:** no experiments have been run, so none needs re-running. No subagent was running at pause time.
All 34 `kb/papers/*.md` files are complete but most are status `partially-verified` (~83 `[unverified]` tags total), because
several agents read arXiv HTML through a summarising fetch tool rather than the PDF tables. `kb/_index_rows/` holds the raw per-family rows (already merged into `kb/index.md`).

**Known reconciliation item:** "WebAIG-25" is not a standalone benchmark; it is the internal eval set of arXiv 2509.12995 (see `kb/papers/vfm-baseline-arxiv2025.md`); the data-bias agent found no separate source.

### Exact next steps to resume
1. (Optional, cheap) Spot-check the `[unverified]` numbers that the survey will lean on: B-Free Tables 5/7/11, SPAI Table 1, 2509.12995 Table 4 — `grep -rn "unverified" kb/papers`.
2. Launch Phase 2 (opus/high) in the background: gap analysis → `kb/ideas/*.md` + `docs/02-hypotheses.md`.
3. Launch the Phase 1 survey writer (opus/high) → `docs/01-survey-explainer.md` (overview, per-method intuition, comparison table, glossary, open problems).
4. Present the 15-line survey summary to the owner; answer questions while Phase 2 runs.
5. Phase 3 needs owner approval before downloads: candidate sources are VISION (CC BY-SA 4.0 phone photos) for reals and Synthbuster / B-Free extended Synthbuster (non-commercial, partial download via range requests) for fakes — see `kb/datasets/`.

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
