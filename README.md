# Diffusion-Generated Image Detection — Research Project

**Goal.** Find a new method, or a measurable improvement over existing ones, for detecting diffusion-generated images, measured on
**unseen generators** and **in-the-wild processing** (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count as progress.

**Acceptance targets for a full-scale validation:** FPR ≤ 1% on fresh phone photos; AUROC ≥ 0.95 after social-media laundering; evaluation against transfer attacks.

## Current status — PAUSED 2026-10-21, mid-Phase 3 (checkpoint commit "checkpoint: paused mid-Phase 3")
| Phase | State |
|---|---|
| 0 Setup | ✅ done — Python 3.12 / torch 2.2.2 / MPS verified (D-001, D-003, D-004) |
| 1 Survey | ✅ done — 34 papers + 8 datasets in kb/, `docs/01-survey-explainer.md`, fact-check `kb/verification-log.md` (3 corrections applied) |
| 2 Gaps & hypotheses | ✅ done — `docs/02-hypotheses.md`, 7 ideas in `kb/ideas/` (top: laundering-nuisance-projection, worst-env-conformal-fpr, neural-isp-fpr-audit) |
| 3 Data & baselines | 🟡 **partial** — eval set ✅ (`experiments/001-eval-set`, 3,220 imgs), 4 baselines set up ✅ (`experiments/002-baselines`), harness ✅ (`experiments/lib`, 26 tests pass), **baseline eval INTERRUPTED** (`experiments/003-baselines-eval`) |
| 4 Hypothesis tests | ⏸ not started — needs owner go-ahead after the baseline table |
| 5 Final report | ⏸ not started |

### What was interrupted (experiments/003-baselines-eval) — must be resumed, not re-run from scratch
Stopped by owner request at 23:41 while scoring. On disk (all writes atomic; resumable):
- ✅ prep: `data/processed/crop512/` (2,919 images; 300 latent-diffusion excluded at 256 px; 1 duplicate RAISE image dropped), `splits.csv`, `cache/prep_report.json`
- ✅ Community Forensics scores: clean 1961/1961, jpeg75_resize05 1961/1961 (D-008 CPU check passed, max |Δlogit| ≈ 6e-5)
- 🟡 Community Forensics social_sim: **432/1961 (partial)** — resumes from cache
- ⏸ B-Free, RINE, ClipDet scores; B0 (DINOv2-base weights already downloaded to data/hf_cache, SHA256 in env/SOURCES.md); report → `results.json`, `tables.md`, `notes.md`, `config.json` not yet written
- ⚠ `run.py` (written by the eval agent) has not yet been reviewed by the lead; review the split assertions in `stage_prep` and `stage_b0` before trusting results.

### Exact next steps to resume
1. `cd ~/projects/diffusion-detect-research && env/.venv/bin/python -I experiments/003-baselines-eval/run.py score` (resumes; ~40–60 min: B-Free is slowest at ~5 img/s). Optionally run in the background.
2. `env/.venv/bin/python -I experiments/003-baselines-eval/run.py b0` then `... run.py report`.
3. Lead checks `tables.md`, format-bias AUROCs, D-008 guard (`cache/d008_guard.json`), writes `notes.md`/`config.json` if missing, commits.
4. **STOP and show the owner the baseline table** (required gate before Phase 4). Also ask: will the owner supply ~250 photos from a 2021+ phone for hypothesis neural-isp-fpr-audit? And DINOv3 (gated) access for Phase 4 (D-006).
5. Phase 4: test top hypotheses vs B0 with an opus/high critic review per result.

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
