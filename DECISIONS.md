# DECISIONS

Format: **D-NNN — title** (date) · Decision · Rationale · Alternatives considered

**D-001 — Target torch 2.2.2 on Python 3.12 (Intel Mac)** (2026-10-21 09:45)
- Decision: env uses Python 3.12 (Homebrew 3.12.15 already present), torch==2.2.2, torchvision==0.17.2, numpy<2.
- Rationale: PyTorch stopped publishing macOS x86_64 wheels after 2.2.x; machine is Intel + AMD GPU, not Apple Silicon.
- Alternatives: build torch from source (hours, fragile); cloud GPU (needs owner approval, costs money).
- Consequence: newer libs that require torch>=2.3 must be pinned to older compatible versions; DINOv3 via transformers must be checked for compatibility.

**D-002 — Concurrent subagents never write shared files** (2026-10-21 09:45)
- Decision: Phase-1 agents write their index rows to kb/_index_rows/<family>.md; the lead merges them into kb/index.md and writes LOG.md.
- Rationale: avoid write races with ≤5 concurrent agents.

**D-003 — uv installed from PyPI into env/.bootstrap, not Homebrew** (2026-10-21 10:07)
- Decision: aborted `brew install uv` (owner-approved) because Homebrew had no Intel bottle and began compiling rustc 1.99.1 from source (~1h+, adds a system Rust toolchain). Used owner's offered option 2 instead: `python3.12 -m venv env/.bootstrap && pip install uv` (prebuilt x86_64 wheel). Nothing installed system-wide; rust/llvm/uv not present in `brew list`.
- Rationale: approval covered a quick bottle install, not a toolchain build.

**D-004 — Use MPS (AMD Radeon Pro 5500M) for inference** (2026-10-21 10:07)
- Evidence: ResNet-50 forward 141.5 img/s on MPS vs 18.9 img/s on CPU; DINOv3ViTModel (random init) forward OK on cpu+mps under torch 2.2.2 / transformers 4.57.6. 4 GB VRAM → small batches; fall back to CPU on op errors.

**D-005 — Phase 3 eval set (owner-approved 2026-10-21 22:30)**
- Reals: VISION native phone photos 500 (CC BY-SA 4.0) + their WhatsApp and Facebook-HQ shared copies (real-side laundering / FPR calibration); RAISE-1k resized PNG 400 (B-Free extended; NON-COMMERCIAL) as format-matched control.
- Fakes: Synthbuster (CC BY-NC-SA 4.0) 120 each of DALL-E 2, DALL-E 3, Firefly, Midjourney v5, SDXL, SD2; B-Free extended (NON-COMMERCIAL) SD3.5 150, latent-diffusion 300, **FLUX 150 held out entirely (test only)**.
- Budget ≈4.3 GB (hard cap 4.6 GB). Partial download via HTTP range requests + per-entry CRC32 check against zip central directory.
- Rationale: only licence-clean phone-photo source found is VISION; no licence-clean diffusion fake source exists → NC fakes accepted for research use. Format bias (JPEG reals vs PNG fakes) controlled via RAISE PNG reals, both-class JPEG+resize condition, and a format-bias probe.

**D-006 — Baselines and backbones (owner-approved 2026-10-21 22:30)**
- Run 4 pretrained baselines: Community Forensics (MIT), B-Free (NON-COMMERCIAL), RINE (Apache-2.0), CLIP few-shot / Cozzolino (Apache-2.0).
- Use DINOv2 (Apache-2.0, ungated) now; DINOv3 (gated, Meta licence) only when Phase 4 needs it — ask owner then.

**D-007 — Total download cap raised to 9 GB (owner-approved 2026-10-21 22:37)**
- Data ≈4.3 GB (D-005) + model weights ≤4.5 GB. Each weight file's size checked before download; SHA256 recorded in env/SOURCES.md.

**D-008 — MPS silent-error guard (2026-10-21 23:21)**
- Finding (baselines agent): MPS on AMD 5500M returned wrong ViT-L logits (up to 0.4 off) for a 40-image batch WITHOUT raising; ≤30 was correct.
- Decision: all forwards on MPS use batch ≤20; every eval run cross-checks a fixed 32-image subset on CPU and aborts if max |Δlogit| > 1e-3. Re-run experiments/002-baselines/smoke.py before changing batch sizes.

**D-009 — Phase 3 baseline protocol (2026-10-21 23:21)**
- Preprocess: EXIF-orientation, strip metadata, center crop 512 with NO resize, saved as PNG (harness default), same for both classes. Conditions: clean, jpeg75_resize05 (resize 0.5× then JPEG q75), plus social_sim as an extra.
- latent-diffusion (256 px) cannot give a 512 crop → excluded from Phase 3 tables (kept for possible 256-crop ablation). Generators in tables: dalle2, dalle3, firefly, midjourney-v5, SDXL, SD2, SD3.5, FLUX (held out).
- Real pools reported separately: VISION-native (JPEG phone), RAISE (PNG DSLR, format-matched control), VISION-WhatsApp and VISION-Facebook-HQ (platform-shared reals; FPR rows only). Firefly is JPEG (q≈95) under .png names → detect format by content, never extension.
- Splits: VISION by device group (10 train / 25 test devices; WA/FB copies follow their native's device); RAISE 50/50 random by image; each fake generator 50/50, FLUX test-only. Calib reals = held-out slice of train-side reals.
- Pretrained baselines are scored on the TEST split only (no training). Added trained baseline B0: frozen DINOv2 ViT-B/14 (Apache-2.0, ~350 MB; keeps total downloads ≈8.3 GB < 9 GB cap) + logistic probe, with leave-one-generator-out.
- Threshold: native (logit 0) AND calibrated to 5% FPR on calib reals; headline metrics AUROC, AP, bal.acc, FPR@95TPR, TPR@1%FPR with bootstrap CIs. Format-bias probe reported per run. D-008 MPS guard applies.

**D-010 — DINOv2 (and any backbone that fails on MPS) runs on CPU only; MPS fallback made sticky (2026-10-23 07:28)**
- Finding: B0 aborted on the D-008 guard (max |Δlogit| 5.8, max |Δfeature| 102). Diagnosis (scratchpad tests): DINOv2's bicubic position-embedding interpolation raises NotImplementedError on MPS in torch 2.2.2, so every batch went through the harness's MPS→CPU fallback. Pure CPU is bit-exact across batch sizes 1–32, SDPA and eager attention, and repeats; a failed-MPS round-trip of weights is lossless in isolation; yet the fallback path inside the full B0 run produced features differing from CPU by up to 102 and differing between runs. Exact root cause NOT identified (appears order/state-dependent: MPS attempted before any CPU forward in the process).
- Decision: run.py stage_b0 extracts DINOv2 features with device="cpu"; lib/features.embed_images now pins a backbone to CPU permanently after its first MPS failure (backbone.force_cpu). Corrupt cache deleted. D-008 guard stays mandatory. Pretrained scorers unaffected (own code path; all 12 guards passed).
- Cost: B0 feature extraction on CPU (~15 min).

## D-011 — 2026-10-23 — Move compute to the owner's GPU PC (Windows + NVIDIA, WSL2 Ubuntu)
**Decision.** The project continues on the owner's GPU PC under WSL2 Ubuntu 24.04. The Intel Mac stays a valid environment.

**Code and git**
- Code goes through a private GitHub repo (owner-created).
- `data/` and `third_party/` are NOT copied. They are re-downloaded from the original sources by `env/bootstrap_data.py`:
  - every weight is SHA256-checked against env/SOURCES.md;
  - every eval image is checked against the committed `experiments/001-eval-set/manifest_raw.csv`;
  - repos are checked out at pinned commits.
  - Re-acquisition is about 7.96 GB of the same, already-approved files (the D-007 cap is about new material, not re-downloads).

**Environment**
- One `env/uv.lock` with platform markers:
  - darwin x86_64: torch 2.2.2 / torchvision 0.17.2 (unchanged; `uv sync --frozen --dry-run` on the Mac reports no changes);
  - linux x86_64: torch 2.8.0 / torchvision 0.23.0 from PyPI (bundles CUDA 12.8).
- numpy<2 stays on both.

**Code changes**
- Device order is cuda > mps > cpu, in `experiments/lib/features.get_device` and `scorers/_common.pick_device`.
- TF32 is disabled on CUDA, so the D-008 guard compares fp32 to fp32.
- run.py uses the detected accelerator instead of hard-coded "mps". B0 stays on CPU only when the accelerator is MPS (D-010).
- `splits.csv` paths are now repo-relative (prefix stripped; no other change).
- `fetch.py` derives ROOT from `__file__` and gained a `from-manifest` mode.
- `CLAUDE.md` holds the project rules (reconstructed from the brief) for new sessions.

**Rationale**
- MPS on the AMD card caused two silent-error incidents (D-008, D-010).
- B0 on CPU took about 30 min per pass.
- Phase 4 needs many feature passes and possibly DINOv3 / larger backbones.

**Check.** Before Phase 4, Phase 3 is re-run on the PC:
- splits must be byte-identical;
- metrics are compared with the Mac `tables.md` (expected to agree within about ±0.005 AUROC).
