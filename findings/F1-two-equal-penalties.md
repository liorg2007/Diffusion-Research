# F1: an unseen real source costs as much as an unseen generator

**Finding.** Mean AUROC at jpeg75_resize05, block-8 probe (post-hoc analysis by the E10 critic):

| | generator seen | generator held out |
|---|---|---|
| real source in training | 0.926 | 0.855 |
| real source held out | 0.858 | **0.767** |

**Why it matters.** Most generalisation work tests unseen generators against familiar real photos, which measures only one of these two costs.

**Evidence.** `experiments/019-source-diversity/results/critique.md`.

**Limits.** Post hoc, one backbone, 6 real sources, 8 generators.

**What would falsify it.** Repeating on a much larger set of real sources and generators, with error bars over both, and finding one penalty clearly larger.
