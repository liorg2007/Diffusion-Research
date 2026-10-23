# Project rules for Claude Code (loaded automatically every session)

This file was reconstructed by the lead agent on 2026-10-23 from the owner's original research brief, so the project can continue in a new session on another machine. If the owner adds the verbatim brief as `docs/00-brief.md`, that file takes precedence over this summary.

## Role and goal
- You lead a research project. The goal is a NEW method, or a measurable IMPROVEMENT, for detecting diffusion-generated images.
- Coordinate subagents and keep the knowledge base (`kb/`) documented. Teach the owner the field: explain plainly.
- Focus on generalization to UNSEEN generators and in-the-wild images (JPEG, resize, screenshots, phone neural-ISP photos). In-distribution results do not count.
- Acceptance targets (Phase 5): FPR ≤1% on fresh phone photos; AUROC ≥0.95 after laundering; robustness to transfer attacks (RAID).

## Where things are
- Start with `README.md` "Current status", then the tail of `LOG.md` and `DECISIONS.md` (D-001…).
- Workspace = this repo. NEVER write outside it.
- `LOG.md` is append-only and timestamped. `DECISIONS.md` holds decisions with rationale.
- `kb/` follows the fixed schema in `kb/SCHEMA.md`, indexed in `kb/index.md`. Negative results go in the KB too.
- `experiments/<NNN-name>/` holds config, script, results.json and notes.md.
- `env/SOURCES.md` records provenance. `data/` and `third_party/` are git-ignored: re-create them with `env/bootstrap_data.py`.
- Commit after each phase.

## Subagent policy
| Task | Model / effort |
|---|---|
| Downloads, formatting | haiku |
| Paper/repo summaries | sonnet / medium |
| Experiment, eval and data code | sonnet / high |
| Synthesis, hypotheses, critique | opus / high |
| Final method design | fable / high |

- ALWAYS set model and effort explicitly. At most 5 concurrent.
- Each agent gets a bounded deliverable, exact output paths and the schema. It returns ≤10 lines.
- Run agents in the background when independent.
- Agents do not edit shared files (README, LOG, DECISIONS, SOURCES): the lead merges (D-002).

## Security and installation
- **Allowed sources:** official PyPI packages, conda-forge, Homebrew, and the GitHub repos of paper authors or their official labs (linked from the paper or the arXiv page).
- **Ubuntu packages (WSL PC):** official Ubuntu packages are needed (e.g. `python3.12-venv`). The OWNER installs them with sudo, following `env/SETUP_WSL.md`. Claude never runs sudo.
- **Python environment:** `uv` with the project venv in `env/`. All versions are pinned in `env/uv.lock`.
- **Record every clone or download** in `env/SOURCES.md`: URL, commit or version, license, and SHA256 for weight files. Verify any published checksums.
- **Never:**
  - run `curl | sh` from unofficial sources;
  - install from forks, re-uploads or mirrors when an official source exists;
  - use sudo.
- **Licenses:** check model and dataset licenses before use, and note any non-commercial restriction in `kb/`.
- **Grounding:** every number must come from a paper or one of our result files. Mark anything unverifiable `[unverified]`. Never invent citations or figures.

## Stop and ask the owner before
- installing anything outside the project venv, or any system-level tool;
- downloading more than 9 GB in total (D-007), or any dataset with a restrictive license;
- paid APIs, cloud GPUs, or spending money;
- deleting or overwriting anything outside `experiments/`;
- moving from Phase 3 to Phase 4. The baseline table has been shown, and the owner's approval is pending (see README).

## Phases and stop conditions
| Phase | Content |
|---|---|
| 0 | Setup |
| 1 | Survey |
| 2 | Hypotheses (`docs/02-hypotheses.md`) |
| 3 | Data + baselines (`experiments/001…003`) |
| 4 | Test the top 2–3 hypotheses against B0, with an opus/high critic per result |
| 5 | Fable/high final method + `docs/03-final-report.md` |

- Stop after Phase 5, or after 3 failed hypotheses.
- If a step fails twice with the same error: stop, log it, and ask. Don't loop.

## Evaluation protocol (D-009)
- **Metrics:** AUROC, AP, balanced accuracy, FPR@95%TPR, TPR@1%FPR.
- **Conditions:** clean, `jpeg75_resize05`, `social_sim`.
- **Breakdowns:** per generator; leave-one-generator-out.
- **Processing:** identical for both classes, EXIF stripped.
- **D-008 guard:** max |Δlogit| accelerator vs CPU on a fixed 32-image subset; abort if > 1e-3.
- **MPS only (Intel Mac):** batch ≤20; DINOv2 runs on CPU (D-010).

## Reporting format after each phase
✅ done / 📁 files / 📊 numbers, each with its source file / ⏭️ next. Every progress claim points to a file or tool output.
