# F5: one-sided adaptation fakes near-perfect scores

**Finding.** In leave-one-real-source-out tests, "adapting to the new source" by shifting only that source's images shifts only the real test images, because the fakes do not belong to the source. That hands the label to the model.

| held-out source | literal (leaky) AUROC | deployable (shift all images) |
|---|---|---|
| VISION | 0.957 | 0.917 |
| RAISE | 0.985 | 0.689 |
| OI | 0.951 | 0.809 |

**The control that exposes it.** Using 50 *fake* images as the "real anchors" scores even higher (about 0.999), though they carry no information about the pipeline.

**Why it matters.** It is an easy mistake in few-shot and domain-adaptation evaluations. This one was our own pre-registration error, caught in review.

**Evidence.** `experiments/018-real-anchor/results/critique.md`, `experiments/018-real-anchor/notes.md`.

**Limits.** A methods warning, not a detection result. Whether published work makes this error has not been surveyed.
