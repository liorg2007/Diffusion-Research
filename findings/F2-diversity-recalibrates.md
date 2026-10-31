# F2: more kinds of real photos mainly fix false alarms, not separation

**Finding.** Mean over 6 held-out real sources, jpeg75_resize05:

| training sources | AUROC | FPR |
|---|---|---|
| 1 | 0.692 | 40.3% |
| 2 | 0.727 | 16.4% |
| 5 | 0.767 | 6.9% |

From 2 to 5 sources, TPR at the threshold also falls (0.437 → 0.308). At matched FPR the real gain is only +0.074 TPR. A more flexible classifier (MLP) adds about +0.009 AUROC, so the linear head is not the cap. Adding Unsplash lowers some other sources, so more sources are not always better.

**Pre-registered labels.** DIVERSITY HELPS true; DIVERSITY NOT QUANTITY false; DISCOVERY false; line KILL.

**Why it matters.** Diversity is a cheap way to get a sane threshold, but it should not be expected to make fakes easier to separate.

**Evidence.** `experiments/019-source-diversity/results/{tables.md,critique.md}`.

**Limits.** 6 sources; the size-matched control's FPR part is not clean; HDR+ test comes from few shooters.
