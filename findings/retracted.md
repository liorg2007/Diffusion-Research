# Claims that did not survive

| Claim | What happened | Source |
|---|---|---|
| Mid-layer advantage for attribution is new | Already published (Li et al. arXiv 2607.12052; Cioni et al. 2024). Moved to establishments | `experiments/026-final-run/r5_novelty.md` |
| Training on laundered images transfers to unseen laundering | The "unseen" launderings were close to training (a clean-only probe gets 0.875 on screenshot_sim). New mild blur or unsharp drops to about 0.52 (exploratory) | `experiments/026-final-run/critique.md` §1 |
| Generic features beat published detectors for attribution | Detectors were read at their last layer only. At a mid block CommFor beats DINOv2-base by +0.026 [0.006, 0.047]. H-A5's KILL label stands | `experiments/026-final-run/r1r4/results/tables.md` |
| 0.92–0.94 accuracy at 10,000 images per generator | Equally good fits give 0.860, 0.923 and 0.980 | `experiments/026-final-run/critique.md` §4 |
| A few real photos fix a new source (E9 PASS) | Label leak; see [F5](F5-one-sided-adaptation-leak.md). Honest version about 0.88 AUROC, 4–8% FPR | `experiments/018-real-anchor/results/critique.md` |
| Native / WhatsApp / Facebook readable from pixels (0.929) | Read file size and JPEG grid. With a common size: 0.690 clean, chance laundered | `experiments/022-attr-platform/results/critique.md` |
