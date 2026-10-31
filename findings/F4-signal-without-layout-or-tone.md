# F4: generator identity is recoverable without layout or tone

**Finding.** DINOv2-base mid block, jpeg75_resize05, classifier **retrained** on the changed images (share of above-chance accuracy kept):

| change | kept | status |
|---|---|---|
| 32×32 tile shuffle | 0.944 [0.918, 0.970] | pre-registered |
| Gaussian blur σ=1.5 | 0.972 [0.952, 0.991] | pre-registered (prediction refuted) |
| high-pass residual | 0.854 | descriptive |
| grayscale | 0.917 | descriptive |
| histogram equalisation | 0.904 [0.873, 0.935] | exploratory (critic) |

Tone histograms alone reach only 0.466–0.516, against about 0.867 for the block. The best description is several redundant cues plus a sharpness axis.

**Crucial caveat.** This shows the information is *sufficient after retraining*, not what a normal probe uses. A frozen probe falls to 0.50 under the same shuffle and 0.527 under equalisation.

**Evidence.** `experiments/026-final-run/r2r3/results/tables.md`, `experiments/026-final-run/critique.md`, `docs/12-final-run-report.md`.

**Limits.** Literature check of this specific form is pending. Tile-shuffle analyses appear in detection papers (seen in search snippets only). Each class is one configuration from one operator.
