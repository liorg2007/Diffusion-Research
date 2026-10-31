# S1: a real-vs-diffusion difference survives laundering

With content, scale and format controlled, block-8 DINOv2 features separate RAISE photos from fakes made from their own captions at AUROC **0.927–0.953** after laundering. Trained with laundering and tested on familiar real sources, the block-8 probe reaches 0.947–0.953.

**Evidence.** `experiments/014-statistical-field/results/critique.md` (F3), `experiments/015-midlayer-probe/results/critique.md`, `docs/03-final-report.md`.

**Limit.** This is about familiar real sources; see [S2](S2-unseen-source-limit.md).
