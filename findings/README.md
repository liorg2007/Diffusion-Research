# Findings: what this project may add to the field

These are results we measured and had reviewed, and that we did not find stated in this form in the papers we read. **None is confirmed as new.** F1–F3 have had no targeted literature check; F4's specific form is still unchecked. All rest on small data (120–150 images per generator, 600 per new real source).

| # | Finding | Key number | Novelty status |
|---|---|---|---|
| [F1](F1-two-equal-penalties.md) | An unseen real source costs as much as an unseen generator | 0.926 → 0.858 / 0.855 → 0.767 AUROC | not checked |
| [F2](F2-diversity-recalibrates.md) | More kinds of real photos mainly fix false alarms, not separation | FPR 16.4% → 6.9%; TPR at matched FPR only +0.074 | not checked |
| [F3](F3-hard-real-photos.md) | Polished and DSLR photos false-alarm more than phone photos | 14–17% vs 0.5–1% FPR | not checked |
| [F4](F4-signal-without-layout-or-tone.md) | Generator identity is recoverable without layout or tone | 0.944 / 0.904 of the signal kept, after retraining | check pending |
| [F5](F5-one-sided-adaptation-leak.md) | One-sided adaptation fakes near-perfect scores (a methods warning) | 0.985 literal vs 0.689 deployable | not checked |

Claims that did not survive are in [retracted.md](retracted.md). Results that are solid but already known are in [`../establishments/`](../establishments/README.md).
