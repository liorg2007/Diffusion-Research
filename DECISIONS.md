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
