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
