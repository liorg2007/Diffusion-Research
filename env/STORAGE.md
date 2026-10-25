# Storage ledger (D-015)

Large items on the GPU PC (> 100 MB), why each is kept, and when it can be deleted. Provenance for downloads is in `env/SOURCES.md`.
Sizes were measured with `du -sh` on 2026-10-24. Cumulative downloads are about 9.9 GB (incl. VAEs, RAID subset, Open Images subset), against the 20 GB working ceiling (D-015).

| Item | Size | Source | Needed by | Status |
|---|---|---|---|---|
| env/.venv (+ env/.uv-cache, mostly hardlinks) | 6.9 GB | PyPI via uv.lock | everything | keep |
| data/raw/vision | 1.8 GB | VISION (CC BY-SA) | all pilots, official run | keep |
| data/raw/synthbuster subset | 1.5 GB | Synthbuster (NC) | all pilots, official run | keep |
| data/raw/raid | 159 MB | RAID subset (MIT) | E3 / idea #5, paused (D-014) | keep while paused (re-download costs budget) |
| data/weights/laion-commonpool-vitl14 | 1.6 GB | open_clip (ClipDet backbone) | ClipDet scoring | keep |
| data/weights/openai-clip-vitl14 | 890 MB | OpenAI CLIP (RINE/UnivFD backbone) | RINE scoring | keep |
| data/weights/rine, community-forensics | 190 MB | official repos/HF | scoring | keep |
| data/hf_cache: dinov2-base | ~350 MB | HF facebook/dinov2-base | B0 and all probes | keep |
| data/hf_cache: sd-vae-ft-mse | 335 MB | HF stabilityai | E2, E4 | keep |
| data/hf_cache: sdxl-vae | 335 MB | HF stabilityai | official #6 design (CPU round-trips) | keep until the #6 decision; delete if #6 is dropped |
| data/processed/crop512, fov512, fov224 | 1.9 GB | derived (regenerable) | pilots, official run | keep |
| experiments/004-phase4-features/cache | 967 MB | derived | pilots | keep |
| experiments/009-vae-aligned/cache | 262 MB | derived | E4 reuses train round-trips | keep until E4 is critiqued |
| third_party | 263 MB | author GitHub clones | scorers | keep |

| data/raw/openimages | 186 MB | Open Images V7 val, CC BY 2.0 | E7 | keep until E7 critiqued |

Deleted:
- 2026-10-24: data/processed/du2512 (587 MB) and 004 cache pilot_du2 (81 MB), the confounded E1b arm. Results are committed in experiments/008-scale-sensitivity/pilot_du2/.
