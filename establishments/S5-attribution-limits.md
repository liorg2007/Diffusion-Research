# S5: attribution limits

- **Near-identical versions:** SD1.3 vs SD1.4 best 0.527 [0.490, 0.562], inside the label-swap null (`experiments/020-attr-foundation/secondary/tables.md`).
- **New generators:** "unknown" rejection mean AUROC 0.768, min 0.703, below the 0.80 bar. A shared 16-source space flags unseen sources at 0.531 (H-A4 KILL; `experiments/020-attr-foundation/results/`).
- **New edits:** mild blur σ=2 or unsharp 150%, absent from training, drop accuracy to about 0.52; adding them to training recovers 0.85–0.86. Exploratory (`experiments/026-final-run/critique.md`).
- **Platform history:** native / WhatsApp / Facebook is readable from file details, not from laundered pixels (H-A3; `experiments/022-attr-platform/results/critique.md`).

**Practical meaning.** A suite must be retrained as generators and processing change; it will not flag what it has not seen.
