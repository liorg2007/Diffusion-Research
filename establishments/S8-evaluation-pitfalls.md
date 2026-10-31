# S8: evaluation pitfalls we measured

- **Metadata gives away attribution:** file size and format alone name the generator at 0.688 (chance 0.125). All pixel tests must control for it (`experiments/020-attr-foundation/results/tables.md`).
- **Platform "signal" was metadata:** see `experiments/022-attr-platform/results/critique.md`.
- **Zoom changes results:** changing only the crop rule moves laundered AUROC by +0.062 (CommFor), +0.228 (RINE), −0.217 (ClipDet). Thresholds set at one scale inflate FPR at least 5× at another (`docs/03-final-report.md`).
- **Published thresholds fail:** at its default threshold, B-Free flags 17.3% of clean VISION photos (`experiments/003-baselines-eval/notes.md`).
- **Saturated checks:** same-caption pairwise accuracy of 1.000 could not fail, since unmatched pairs reach 0.991 (`experiments/020-attr-foundation/results/critique.md`).
- **Guard tolerance:** D-008's 1e-3 is below sdxl-vae's own CPU run-to-run variation of 1.1e-2 on those features (`experiments/024-attr-vae/results/critique.md`).
